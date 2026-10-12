"""NAVSIM sensor feature builder for ManeuverFlow."""
from typing import Dict
import torch
import numpy as np
from torchvision import transforms

from navsim.common.enums import LidarIndex

from navsim.common.dataclasses import AgentInput
from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder,
)
import cv2

from .cache_schema import MANEUVERFLOW_FEATURE_CACHE_KEY


class ManeuverFlowFeatureBuilder(AbstractFeatureBuilder):
    """Build image, physical LiDAR, and planning-state features."""

    def __init__(self, config):
        self._config = config
        self._num_driving_commands = 4

    def get_unique_name(self) -> str:
        return MANEUVERFLOW_FEATURE_CACHE_KEY

    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        features = {}

        features["camera_feature"] = self._get_camera_feature(agent_input)
        features["lidar_feature"] = self._get_lidar_feature(agent_input)
        features["status_feature"] = self._get_status_feature(agent_input)

        return features

    def _get_status_feature(self, agent_input: AgentInput) -> torch.Tensor:
        """
        Construct the shared planning-state feature.
        Dimension: 8 = 4 (one-hot driving_command) + 2 (velocity) + 2 (acceleration)
        """
        ego_status = agent_input.ego_statuses[-1]

        driving_command_raw = ego_status.driving_command
        if isinstance(driving_command_raw, (list, np.ndarray, torch.Tensor)):
            if isinstance(driving_command_raw, torch.Tensor) and driving_command_raw.ndim > 0 and driving_command_raw.shape[0] > 1:
                 cmd_idx = int(torch.argmax(driving_command_raw).item())
            elif isinstance(driving_command_raw, (list, np.ndarray)) and len(driving_command_raw) > 1 and np.sum(driving_command_raw) <= 1.01:
                 cmd_idx = int(np.argmax(driving_command_raw))
            else:
                 cmd_idx = int(driving_command_raw[0]) if len(driving_command_raw) > 0 else 0
        else:
            cmd_idx = int(driving_command_raw)

        cmd_idx = max(0, min(cmd_idx, self._num_driving_commands - 1))
        driving_command_onehot = torch.zeros(self._num_driving_commands, dtype=torch.float32)
        driving_command_onehot[cmd_idx] = 1.0

        ego_velocity = torch.tensor(ego_status.ego_velocity, dtype=torch.float32)
        ego_acceleration = torch.tensor(ego_status.ego_acceleration, dtype=torch.float32)

        status_feature = torch.cat([
            driving_command_onehot,
            ego_velocity,
            ego_acceleration,
        ])

        return status_feature

    def _get_camera_feature(self, agent_input: AgentInput) -> torch.Tensor:
        cameras = agent_input.cameras[-1]

        l0 = cameras.cam_l0.image[28:-28, 416:-416]
        f0 = cameras.cam_f0.image[28:-28]
        r0 = cameras.cam_r0.image[28:-28, 416:-416]

        stitched_image = np.concatenate([l0, f0, r0], axis=1)
        resized_image = cv2.resize(stitched_image, (1024, 256))
        tensor_image = transforms.ToTensor()(resized_image)

        return tensor_image

    def _get_lidar_feature(self, agent_input: AgentInput) -> torch.Tensor:
        lidar_pc = agent_input.lidars[-1].lidar_pc[LidarIndex.POSITION].T

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
