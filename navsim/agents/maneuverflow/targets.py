from typing import Any, Dict, List, Optional
import numpy.typing as npt
import torch
import numpy as np
from torchvision import transforms
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.planning.training.abstract_feature_target_builder import (
    AbstractTargetBuilder,
)
from navsim.common.dataclasses import Scene
import cv2
from navsim.common.enums import BoundingBoxIndex, LidarIndex
from navsim.planning.scenario_builder.navsim_scenario_utils import tracked_object_types
from enum import IntEnum

from shapely import affinity
from shapely.geometry import Polygon, LineString

from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import StateSE2

from .cache_schema import MANEUVERFLOW_TARGET_CACHE_KEY


class ManeuverFlowTargetBuilder(AbstractTargetBuilder):
    """Build cached supervision and online future-observation targets."""

    def __init__(
        self,
        trajectory_sampling: TrajectorySampling,
        slice_indices: Optional[List[int]] = None,
        config=None,
    ):
        self._config = config
        self._trajectory_sampling = trajectory_sampling
        self.slice_indices = [3] if slice_indices is None else list(slice_indices)

    def get_unique_name(self) -> str:
        return MANEUVERFLOW_TARGET_CACHE_KEY

    @property
    def requires_online_target_construction(self) -> bool:
        """Joint training reads future observations from the source scene."""

        return self._config.training_phase == "joint_finetuning"

    def compute_targets(self, scene: Scene) -> Dict[str, torch.Tensor]:
        result = {}
        future_trajectory_list = []

        assert len(self.slice_indices) == 1, "Slice indices must be of length 1"
        index = self.slice_indices[0]
        current_index = scene.scene_metadata.num_history_frames - 1
        if index != current_index:
            raise ValueError(
                "ManeuverFlow target construction requires the current "
                f"NAVSIM frame index {current_index}, got {index}."
            )
        future_trajectory = scene.get_future_trajectory(
            num_trajectory_frames=self._trajectory_sampling.num_poses,
        )
        future_trajectory = torch.tensor(future_trajectory.poses, dtype=torch.float32)
        future_trajectory_list.append(future_trajectory)

        future_trajectory_stacked = torch.stack(future_trajectory_list)
        result['gt_trajectory'] = future_trajectory_stacked

        cur_agent_index = self.slice_indices[0]
        annotations = scene.frames[cur_agent_index].annotations
        agent_states, agent_labels = self._compute_agent_targets(annotations)
        result["agent_states"] = agent_states
        result["agent_labels"] = agent_labels

        cur_map_index = self.slice_indices[0]
        annotations = scene.frames[cur_map_index].annotations
        ego_pose = StateSE2(*scene.frames[cur_map_index].ego_status.ego_pose)

        bev_semantic_map = self._compute_bev_semantic_map(
            annotations,
            scene.map_api,
            ego_pose,
        )
        cur_ego_box = [0.0, 0.0, 0.0, 4.0, 2.0, 1.8, 0.0]
        bev_semantic_map = self._add_ego_box_to_bev_map(
            bev_semantic_map,
            cur_ego_box,
        )
        result["bev_semantic_map"] = bev_semantic_map

        return result

    def compute_online_targets(self, scene: Scene) -> Dict[str, torch.Tensor]:
        """Load future observations for the stop-gradient EMA target pass.

        Raw future sensors come from SceneLoader and are omitted from the
        reusable current-frame cache built by :meth:`compute_targets`.
        """

        if not self.requires_online_target_construction:
            return {}

        # NAVSIM scenes are sampled at 2 Hz. Relative to current index 3,
        # 1/2/3/4 seconds correspond to frame indices 5/7/9/11.
        cur_idx = int(self.slice_indices[0])
        future_indices = [
            cur_idx + 2,
            cur_idx + 4,
            cur_idx + 6,
            cur_idx + 8,
        ]
        current_frame = scene.frames[cur_idx]
        zero_camera_feature = torch.zeros_like(
            self._camera_feature_from_cameras(current_frame.cameras)
        )
        zero_lidar_feature = torch.zeros_like(
            self._lidar_feature_from_lidar(current_frame.lidar)
        )

        future_camera_features = []
        future_lidar_features = []
        future_valid = []
        for future_idx in future_indices:
            frame_exists = future_idx < len(scene.frames)
            future_frame = scene.frames[
                min(future_idx, len(scene.frames) - 1)
            ]
            has_future_sensor = (
                frame_exists
                and self._has_future_tfu_sensors(future_frame)
            )
            if has_future_sensor:
                future_camera_features.append(
                    self._camera_feature_from_cameras(future_frame.cameras)
                )
                future_lidar_features.append(
                    self._lidar_feature_from_lidar(future_frame.lidar)
                )
            else:
                future_camera_features.append(zero_camera_feature.clone())
                future_lidar_features.append(zero_lidar_feature.clone())
            future_valid.append(bool(has_future_sensor))

        self._require_complete_future_sensor_supervision(future_valid)
        result = {
            "future_camera_feature": torch.stack(future_camera_features),
            "future_bev_feature_valid_mask": torch.tensor(
                future_valid,
                dtype=torch.bool,
            ),
        }
        result["future_lidar_feature"] = torch.stack(future_lidar_features)
        return result

    @staticmethod
    def _require_complete_future_sensor_supervision(
        future_valid: List[bool],
    ) -> None:
        """Fail when the four required target observations are unavailable."""

        if len(future_valid) == 4 and all(future_valid):
            return
        missing_seconds = [
            index + 1
            for index, is_valid in enumerate(future_valid)
            if not is_valid
        ]
        raise FileNotFoundError(
            "Joint Future-BEV supervision requires "
            "front/left/right cameras and LiDAR at 1/2/3/4 seconds. "
            "Missing future sensor data at "
            f"{missing_seconds} second(s). The reduced navtrain "
            "current/history download is insufficient; install full "
            "trainval sensor blobs or an equivalent future-frame-complete "
            "navtrain subset."
        )

    @staticmethod
    def _has_future_tfu_sensors(frame) -> bool:
        cameras = frame.cameras
        has_cameras = (
            cameras.cam_l0.image is not None
            and cameras.cam_f0.image is not None
            and cameras.cam_r0.image is not None
        )
        if not has_cameras:
            return False
        return frame.lidar.lidar_pc is not None

    def _camera_feature_from_cameras(self, cameras) -> torch.Tensor:
        l0 = cameras.cam_l0.image[28:-28, 416:-416]
        f0 = cameras.cam_f0.image[28:-28]
        r0 = cameras.cam_r0.image[28:-28, 416:-416]

        stitched_image = np.concatenate([l0, f0, r0], axis=1)
        resized_image = cv2.resize(stitched_image, (1024, 256))
        return transforms.ToTensor()(resized_image)

    def _lidar_feature_from_lidar(self, lidar) -> torch.Tensor:
        lidar_pc = lidar.lidar_pc[LidarIndex.POSITION].T

        def splat_points(point_cloud):
            xbins = np.linspace(
                self._config.bev_minimum_x,
                self._config.bev_maximum_x,
                int((self._config.bev_maximum_x - self._config.bev_minimum_x)
                * int(self._config.pixels_per_meter)
                + 1),
            )
            ybins = np.linspace(
                self._config.bev_minimum_y,
                self._config.bev_maximum_y,
                int((self._config.bev_maximum_y - self._config.bev_minimum_y)
                * int(self._config.pixels_per_meter)
                + 1),
            )
            hist = np.histogramdd(point_cloud[:, :2], bins=(xbins, ybins))[0]
            hist[hist > self._config.hist_max_per_pixel] = self._config.hist_max_per_pixel
            overhead_splat = hist / self._config.hist_max_per_pixel
            return overhead_splat

        lidar_pc = lidar_pc[lidar_pc[..., 2] < self._config.max_height_lidar]
        above = lidar_pc[lidar_pc[..., 2] > self._config.lidar_split_height]
        above_features = splat_points(above)
        features = np.stack([above_features], axis=-1)
        features = np.transpose(features, (2, 0, 1)).astype(np.float32)
        return torch.tensor(features)

    def _compute_agent_targets(self, annotations):
        max_agents = self._config.num_bounding_boxes
        agent_states_list: List[npt.NDArray[np.float32]] = []

        def _xy_in_lidar(x, y, config) -> bool:
            return (config.bev_minimum_x <= x <= config.bev_maximum_x) and (
                config.bev_minimum_y <= y <= config.bev_maximum_y
            )

        for box, name in zip(annotations.boxes, annotations.names):
            box_x, box_y, box_heading, box_length, box_width = (
                box[BoundingBoxIndex.X],
                box[BoundingBoxIndex.Y],
                box[BoundingBoxIndex.HEADING],
                box[BoundingBoxIndex.LENGTH],
                box[BoundingBoxIndex.WIDTH],
            )

            if (
                name == "vehicle"
                and _xy_in_lidar(box_x, box_y, self._config)
            ):
                agent_states_list.append(
                    np.array([box_x, box_y, box_heading, box_length, box_width], dtype=np.float32)
                )

        agents_states_arr = np.array(agent_states_list)

        agent_states = np.zeros((max_agents, BoundingBox2DIndex.size()), dtype=np.float32)
        agent_labels = np.zeros(max_agents, dtype=bool)

        if len(agents_states_arr) > 0:
            distances = np.linalg.norm(agents_states_arr[..., BoundingBox2DIndex.POINT], axis=-1)
            argsort = np.argsort(distances)[:max_agents]

            agents_states_arr = agents_states_arr[argsort]
            agent_states[: len(agents_states_arr)] = agents_states_arr
            agent_labels[: len(agents_states_arr)] = True

        return np.array(agent_states), np.array(agent_labels)

    def _compute_bev_semantic_map(
            self, annotations, map_api, ego_pose
        ) -> torch.Tensor:
        bev_semantic_map = np.zeros(self._config.bev_semantic_frame, dtype=np.int64)
        for label, (entity_type, layers) in self._config.bev_semantic_classes.items():
            if entity_type == "polygon":
                entity_mask = self._compute_map_polygon_mask(map_api, ego_pose, layers)
            elif entity_type == "linestring":
                entity_mask = self._compute_map_linestring_mask(map_api, ego_pose, layers)
            elif entity_type == 'box':
                entity_mask = self._compute_box_mask(annotations, layers)
            elif entity_type == "ego_box":
                # Ego is added separately after the semantic layers are rendered.
                continue
            else:
                raise ValueError(f"Unsupported BEV semantic entity type: {entity_type!r}")
            bev_semantic_map[entity_mask] = label

        return torch.Tensor(bev_semantic_map)

    def _add_ego_box_to_bev_map(self, bev_semantic_map, ego_box) -> torch.Tensor:
        entity_mask = self._compute_ego_box_mask(ego_box)
        bev_semantic_map[entity_mask] = self._config.ego_box_map_idx
        return bev_semantic_map

    def _compute_ego_box_mask(
        self, box_value
    ) -> npt.NDArray[np.bool_]:
        box_polygon_mask = np.zeros(self._config.bev_semantic_frame[::-1], dtype=np.uint8)
        x, y, heading = box_value[0], box_value[1], box_value[-1]
        box_length, box_width, box_height = box_value[3], box_value[4], box_value[5]
        agent_box = OrientedBox(StateSE2(x, y, heading), box_length, box_width, box_height)
        exterior = np.array(agent_box.geometry.exterior.coords).reshape((-1, 1, 2))
        exterior = self._coords_to_pixel(exterior)
        cv2.fillPoly(box_polygon_mask, [exterior], color=255)
        box_polygon_mask = np.rot90(box_polygon_mask)[::-1]
        return box_polygon_mask > 0

    def _compute_map_polygon_mask(
        self, map_api, ego_pose, layers
    ) -> npt.NDArray[np.bool_]:
        map_object_dict = map_api.get_proximal_map_objects(
            point=ego_pose.point, radius=self._config.bev_radius, layers=layers
        )
        map_polygon_mask = np.zeros(self._config.bev_semantic_frame[::-1], dtype=np.uint8)
        for layer in layers:
            for map_object in map_object_dict[layer]:
                polygon: Polygon = self._geometry_local_coords(map_object.polygon, ego_pose)
                exterior = np.array(polygon.exterior.coords).reshape((-1, 1, 2))
                exterior = self._coords_to_pixel(exterior)
                cv2.fillPoly(map_polygon_mask, [exterior], color=255)
        map_polygon_mask = np.rot90(map_polygon_mask)[::-1]
        return map_polygon_mask > 0

    def _compute_map_linestring_mask(
        self, map_api, ego_pose, layers
    ) -> npt.NDArray[np.bool_]:
        map_object_dict = map_api.get_proximal_map_objects(
            point=ego_pose.point, radius=self._config.bev_radius, layers=layers
        )
        map_linestring_mask = np.zeros(self._config.bev_semantic_frame[::-1], dtype=np.uint8)
        for layer in layers:
            for map_object in map_object_dict[layer]:
                linestring: LineString = self._geometry_local_coords(
                    map_object.baseline_path.linestring, ego_pose
                )
                points = np.array(linestring.coords).reshape((-1, 1, 2))
                points = self._coords_to_pixel(points)
                cv2.polylines(map_linestring_mask, [points], isClosed=False, color=255, thickness=2)
        map_linestring_mask = np.rot90(map_linestring_mask)[::-1]
        return map_linestring_mask > 0

    def _compute_box_mask(
        self, annotations, layers
    ) -> npt.NDArray[np.bool_]:
        box_polygon_mask = np.zeros(self._config.bev_semantic_frame[::-1], dtype=np.uint8)
        for name_value, box_value in zip(annotations.names, annotations.boxes):
            agent_type = tracked_object_types[name_value]
            if agent_type in layers:
                x, y, heading = box_value[0], box_value[1], box_value[-1]
                box_length, box_width, box_height = box_value[3], box_value[4], box_value[5]
                agent_box = OrientedBox(StateSE2(x, y, heading), box_length, box_width, box_height)
                exterior = np.array(agent_box.geometry.exterior.coords).reshape((-1, 1, 2))
                exterior = self._coords_to_pixel(exterior)
                cv2.fillPoly(box_polygon_mask, [exterior], color=255)
        box_polygon_mask = np.rot90(box_polygon_mask)[::-1]
        return box_polygon_mask > 0

    @staticmethod
    def _geometry_local_coords(geometry: Any, origin: StateSE2) -> Any:
        a = np.cos(origin.heading)
        b = np.sin(origin.heading)
        d = -np.sin(origin.heading)
        e = np.cos(origin.heading)
        xoff = -origin.x
        yoff = -origin.y

        translated_geometry = affinity.affine_transform(geometry, [1, 0, 0, 1, xoff, yoff])
        rotated_geometry = affinity.affine_transform(translated_geometry, [a, b, d, e, 0, 0])

        return rotated_geometry

    def _coords_to_pixel(self, coords):
        pixel_center = np.array([[0, self._config.bev_pixel_width / 2.0]])
        coords_idcs = (coords / self._config.bev_pixel_size) + pixel_center
        return coords_idcs.astype(np.int32)


class BoundingBox2DIndex(IntEnum):

    _X = 0
    _Y = 1
    _HEADING = 2
    _LENGTH = 3
    _WIDTH = 4

    @classmethod
    def size(cls):
        valid_attributes = [
            attribute
            for attribute in dir(cls)
            if attribute.startswith("_")
            and not attribute.startswith("__")
            and not callable(getattr(cls, attribute))
        ]
        return len(valid_attributes)

    @classmethod
    @property
    def X(cls):
        return cls._X

    @classmethod
    @property
    def Y(cls):
        return cls._Y

    @classmethod
    @property
    def HEADING(cls):
        return cls._HEADING

    @classmethod
    @property
    def LENGTH(cls):
        return cls._LENGTH

    @classmethod
    @property
    def WIDTH(cls):
        return cls._WIDTH

    @classmethod
    @property
    def POINT(cls):
        return slice(cls._X, cls._Y + 1)

    @classmethod
    @property
    def STATE_SE2(cls):
        return slice(cls._X, cls._HEADING + 1)
