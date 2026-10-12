"""Strict Maneuver Gaussian Mixture Prior construction and load contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


MGMP_PRIOR_FILENAMES = {
    "displacement_mean": "maneuver_displacement_mean.npy",
    "displacement_std": "maneuver_displacement_std.npy",
    "heading_mean": "maneuver_heading_mean.npy",
    "heading_std": "maneuver_heading_std.npy",
    "component_counts": "maneuver_component_counts.npy",
}

# The paper distance uses the fitted positive scales directly.
MGMP_DISPLACEMENT_STANDARD_DEVIATION_FLOOR = np.float32(0.03)
MGMP_HEADING_STANDARD_DEVIATION_FLOOR = np.float32(0.05)
MGMP_PRIOR_METADATA_FILENAME = "maneuver_mgmp_metadata.json"

MGMP_FAMILY_NAMES = (
    "Stop/Crawl",
    "Cruise",
    "Accelerating",
    "Decelerating",
    "Left lane change",
    "Left turn",
    "Right lane change",
    "Right turn",
)

MGMP_CONSTRUCTION_VERSION = (
    "maneuverflow.mgmp.eight_interval_symmetric_mean.v1"
)
MGMP_NUM_COMPONENTS = 8
MGMP_TRAJECTORY_NUM_POSES = 8
MGMP_STOP_DISTANCE_METERS = 5.0
MGMP_LANE_CHANGE_THRESHOLD_DEGREES = 4.5
MGMP_TURN_THRESHOLD_DEGREES = 13.0
MGMP_SPEED_TREND_THRESHOLD_METERS_PER_INTERVAL = 0.25
MGMP_WAYPOINT_INTERVAL_SECONDS = 0.5

# Fixed affine normalization shared with ManeuverFlow's trajectory codec.
MGMP_DISPLACEMENT_X_MEAN = 2.950225591659546
MGMP_DISPLACEMENT_X_SCALE = 4.52533745765686
MGMP_DISPLACEMENT_Y_MEAN = 0.0607292577624321
MGMP_DISPLACEMENT_Y_SCALE = 5.0728104040026665

_CIRCULAR_RESULTANT_FLOOR = 1e-6


class ManeuverGMPPriorError(ValueError):
    """Raised when an M-GMP prior violates the released schema."""


def _validate_expert_trajectories(trajectories: np.ndarray) -> np.ndarray:
    """Return finite expert trajectories as float64 ``[N, 8, 3]``."""

    array = np.asarray(trajectories)
    expected_tail = (MGMP_TRAJECTORY_NUM_POSES, 3)
    if array.ndim != 3 or array.shape[1:] != expected_tail:
        raise ManeuverGMPPriorError(
            "M-GMP construction expects expert trajectories with shape "
            f"[N, {expected_tail[0]}, {expected_tail[1]}], got {array.shape}."
        )
    if array.shape[0] == 0:
        raise ManeuverGMPPriorError(
            "M-GMP construction requires at least one expert trajectory."
        )
    if not np.issubdtype(array.dtype, np.number):
        raise ManeuverGMPPriorError(
            f"M-GMP expert trajectories must be numeric, got {array.dtype}."
        )
    array = array.astype(np.float64, copy=False)
    if not np.isfinite(array).all():
        raise ManeuverGMPPriorError(
            "M-GMP expert trajectories must contain only finite values."
        )
    return array


def compute_mgmp_motion_descriptors(
    trajectories: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute the released rule descriptors for ``[N, 8, 3]`` experts.

    Headings are input in radians. The rule uses their arithmetic mean in
    degrees, not a circular mean. The longitudinal descriptor includes the
    ego-to-first-waypoint increment and compares the mean lengths of equal
    four-interval late and early windows.
    """

    trajectories = _validate_expert_trajectories(trajectories)
    xy = trajectories[..., :2]
    heading = trajectories[..., 2]

    first_to_last_displacement = np.linalg.norm(
        xy[:, -1] - xy[:, 0],
        axis=-1,
    )
    arithmetic_mean_heading_degrees = np.rad2deg(heading.mean(axis=1))

    ego_origin = np.zeros((xy.shape[0], 1, 2), dtype=xy.dtype)
    displacement_steps = np.diff(xy, axis=1, prepend=ego_origin)
    step_lengths = np.linalg.norm(displacement_steps, axis=-1)
    step_length_mean_contrast = (
        step_lengths[:, 4:8].sum(axis=1)
        - step_lengths[:, 0:4].sum(axis=1)
    ) / 4.0

    return {
        "first_to_last_displacement": first_to_last_displacement,
        "arithmetic_mean_heading_degrees": (
            arithmetic_mean_heading_degrees
        ),
        "step_lengths": step_lengths,
        "step_length_mean_contrast": step_length_mean_contrast,
    }


def classify_mgmp_families(
    trajectories: np.ndarray,
) -> np.ndarray:
    """Assign the eight hierarchical M-GMP motion families."""

    descriptors = compute_mgmp_motion_descriptors(trajectories)
    displacement = descriptors["first_to_last_displacement"]
    heading_degrees = descriptors["arithmetic_mean_heading_degrees"]
    speed_trend = descriptors["step_length_mean_contrast"]
    absolute_heading = np.abs(heading_degrees)

    labels = np.full(displacement.shape, -1, dtype=np.int64)
    stop = displacement < MGMP_STOP_DISTANCE_METERS
    straight = absolute_heading <= MGMP_LANE_CHANGE_THRESHOLD_DEGREES
    non_stop = ~stop

    labels[stop] = 0
    labels[
        non_stop
        & straight
        & (
            np.abs(speed_trend)
            <= MGMP_SPEED_TREND_THRESHOLD_METERS_PER_INTERVAL
        )
    ] = 1
    labels[
        non_stop
        & straight
        & (
            speed_trend
            > MGMP_SPEED_TREND_THRESHOLD_METERS_PER_INTERVAL
        )
    ] = 2
    labels[
        non_stop
        & straight
        & (
            speed_trend
            < -MGMP_SPEED_TREND_THRESHOLD_METERS_PER_INTERVAL
        )
    ] = 3
    labels[
        non_stop
        & (heading_degrees > MGMP_LANE_CHANGE_THRESHOLD_DEGREES)
        & (heading_degrees <= MGMP_TURN_THRESHOLD_DEGREES)
    ] = 4
    labels[
        non_stop & (heading_degrees > MGMP_TURN_THRESHOLD_DEGREES)
    ] = 5
    labels[
        non_stop
        & (heading_degrees >= -MGMP_TURN_THRESHOLD_DEGREES)
        & (heading_degrees < -MGMP_LANE_CHANGE_THRESHOLD_DEGREES)
    ] = 6
    labels[
        non_stop & (heading_degrees < -MGMP_TURN_THRESHOLD_DEGREES)
    ] = 7

    if np.any(labels < 0):
        raise RuntimeError(
            "The complete hierarchical M-GMP rule left trajectories "
            "unassigned."
        )
    return labels


def encode_mgmp_diff_space(
    trajectories: np.ndarray,
) -> np.ndarray:
    """Encode ``[x, y, yaw]`` experts as normalized ``[dx, dy, sin, cos]``."""

    trajectories = _validate_expert_trajectories(trajectories)
    xy = trajectories[..., :2]
    heading = trajectories[..., 2]
    origin = np.zeros((xy.shape[0], 1, 2), dtype=xy.dtype)
    displacement = np.diff(xy, axis=1, prepend=origin)

    displacement[..., 0] = (
        displacement[..., 0] - MGMP_DISPLACEMENT_X_MEAN
    ) / MGMP_DISPLACEMENT_X_SCALE
    displacement[..., 1] = (
        displacement[..., 1] - MGMP_DISPLACEMENT_Y_MEAN
    ) / MGMP_DISPLACEMENT_Y_SCALE

    return np.concatenate(
        (
            displacement,
            np.sin(heading)[..., None],
            np.cos(heading)[..., None],
        ),
        axis=-1,
    ).astype(np.float32)


def _circular_mean_std(angles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-pose circular mean and standard deviation in radians."""

    sine_mean = np.sin(angles).mean(axis=0)
    cosine_mean = np.cos(angles).mean(axis=0)
    heading_mean = np.arctan2(sine_mean, cosine_mean)
    resultant_length = np.hypot(sine_mean, cosine_mean)
    resultant_length = np.clip(
        resultant_length,
        _CIRCULAR_RESULTANT_FLOOR,
        1.0,
    )
    heading_std = np.sqrt(
        np.maximum(-2.0 * np.log(resultant_length), 0.0)
    )
    return heading_mean, heading_std


def fit_maneuver_gmp_prior(
    trajectories: np.ndarray,
) -> dict[str, np.ndarray]:
    """Fit diagonal displacement and circular heading moments by family."""

    trajectories = _validate_expert_trajectories(trajectories)
    labels = classify_mgmp_families(trajectories)
    diff_space = encode_mgmp_diff_space(trajectories)

    displacement_mean = np.empty(
        (MGMP_NUM_COMPONENTS, MGMP_TRAJECTORY_NUM_POSES, 2),
        dtype=np.float32,
    )
    displacement_std = np.empty_like(displacement_mean)
    heading_mean = np.empty(
        (MGMP_NUM_COMPONENTS, MGMP_TRAJECTORY_NUM_POSES),
        dtype=np.float32,
    )
    heading_std = np.empty_like(heading_mean)
    component_counts = np.bincount(
        labels,
        minlength=MGMP_NUM_COMPONENTS,
    ).astype(np.int64)

    empty_families = np.flatnonzero(component_counts == 0)
    if empty_families.size:
        names = ", ".join(
            f"{family_id} ({MGMP_FAMILY_NAMES[family_id]})"
            for family_id in empty_families
        )
        raise ManeuverGMPPriorError(
            "Cannot fit M-GMP moments because these families have no "
            f"expert trajectories: {names}."
        )

    for family_id in range(MGMP_NUM_COMPONENTS):
        family_mask = labels == family_id
        family_displacement = diff_space[family_mask, ..., :2].astype(
            np.float64
        )
        displacement_mean[family_id] = family_displacement.mean(
            axis=0
        ).astype(np.float32)
        displacement_std[family_id] = np.maximum(
            family_displacement.std(axis=0, ddof=0),
            MGMP_DISPLACEMENT_STANDARD_DEVIATION_FLOOR,
        ).astype(np.float32)

        family_heading = trajectories[family_mask, ..., 2]
        circular_mean, circular_std = _circular_mean_std(family_heading)
        heading_mean[family_id] = circular_mean.astype(np.float32)
        heading_std[family_id] = np.maximum(
            circular_std,
            MGMP_HEADING_STANDARD_DEVIATION_FLOOR,
        ).astype(np.float32)

    return {
        "displacement_mean": displacement_mean,
        "displacement_std": displacement_std,
        "heading_mean": heading_mean,
        "heading_std": heading_std,
        "component_counts": component_counts,
    }


def build_mgmp_metadata(
    *,
    trajectories: np.ndarray,
    component_counts: np.ndarray,
    training_population: Mapping[str, object],
) -> dict[str, object]:
    """Build deterministic provenance metadata for one prior construction."""

    array = np.asarray(trajectories)
    counts = np.asarray(component_counts, dtype=np.int64)
    if counts.shape != (MGMP_NUM_COMPONENTS,) or int(counts.sum()) != len(array):
        raise ManeuverGMPPriorError(
            "M-GMP metadata requires eight component counts matching the trajectory population."
        )
    if training_population.get("num_scenes") != len(array):
        raise ManeuverGMPPriorError("M-GMP training population and trajectory count differ.")
    return {
        "schema_name": "maneuverflow.mgmp_prior",
        "schema_version": 2,
        "construction_version": MGMP_CONSTRUCTION_VERSION,
        "training_data": {"split": "navtrain", "partition": "train", **dict(training_population)},
        "input_trajectories_sha256": hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest(),
        "input_array_dtype": str(array.dtype),
        "input_array_shape": list(array.shape),
        "waypoint_interval_seconds": MGMP_WAYPOINT_INTERVAL_SECONDS,
        "family_names": list(MGMP_FAMILY_NAMES),
        "component_counts": counts.tolist(),
        "rules": {
            "assignment": "hierarchical_stop_heading_then_speed_trend",
            "first_to_last_displacement": "norm(p8 - p1)",
            "stop": "D < 5 m",
            "heading": "arithmetic mean of 8 yaw values, converted to degrees",
            "straight": "abs(mean_heading_deg) <= 4.5",
            "left_lane_change": "4.5 < mean_heading_deg <= 13",
            "left_turn": "mean_heading_deg > 13",
            "right_lane_change": "-13 <= mean_heading_deg < -4.5",
            "right_turn": "mean_heading_deg < -13",
            "speed_trend": (
                "(sum(step_lengths[4:8]) - "
                "sum(step_lengths[0:4])) / 4"
            ),
            "speed_trend_threshold_meters_per_interval": (
                MGMP_SPEED_TREND_THRESHOLD_METERS_PER_INTERVAL
            ),
            "classification_intervals": (
                "eight intervals from the ego origin through waypoint 8"
            ),
            "ego_origin_interval": (
                "included in the early four-interval classification window"
            ),
        },
        "diff_space": {
            "channels": [
                "normalized_dx",
                "normalized_dy",
                "sin_yaw",
                "cos_yaw",
            ],
            "x_mean": MGMP_DISPLACEMENT_X_MEAN,
            "x_scale": MGMP_DISPLACEMENT_X_SCALE,
            "y_mean": MGMP_DISPLACEMENT_Y_MEAN,
            "y_scale": MGMP_DISPLACEMENT_Y_SCALE,
        },
        "moments": {
            "displacement": "diagonal population mean/std per family and pose",
            "displacement_std_floor": float(
                MGMP_DISPLACEMENT_STANDARD_DEVIATION_FLOOR
            ),
            "heading": "circular mean/std per family and pose, radians",
            "heading_std_floor": float(
                MGMP_HEADING_STANDARD_DEVIATION_FLOOR
            ),
            "circular_resultant_floor": _CIRCULAR_RESULTANT_FLOOR,
        },
        "output_files": dict(MGMP_PRIOR_FILENAMES),
    }


def save_maneuver_gmp_prior(
    prior_dir: Path,
    arrays: Mapping[str, np.ndarray],
    *,
    metadata: Mapping[str, object] | None = None,
    overwrite: bool = False,
) -> None:
    """Write the canonical five-array prior and optional deterministic JSON."""

    prior_dir = Path(prior_dir).expanduser()
    output_paths = [
        prior_dir / filename for filename in MGMP_PRIOR_FILENAMES.values()
    ]
    if metadata is not None:
        output_paths.append(prior_dir / MGMP_PRIOR_METADATA_FILENAME)
    existing = [path for path in output_paths if path.exists()]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            "Refusing to overwrite existing M-GMP prior outputs: "
            f"{joined}."
        )

    expected_names = set(MGMP_PRIOR_FILENAMES)
    if set(arrays) != expected_names:
        raise ManeuverGMPPriorError(
            "M-GMP save contract requires exactly "
            f"{sorted(expected_names)}, got {sorted(arrays)}."
        )

    prior_dir.mkdir(parents=True, exist_ok=True)
    for name, filename in MGMP_PRIOR_FILENAMES.items():
        np.save(
            prior_dir / filename,
            np.asarray(arrays[name]),
            allow_pickle=False,
        )
    if metadata is not None:
        metadata_path = prior_dir / MGMP_PRIOR_METADATA_FILENAME
        metadata_path.write_text(
            json.dumps(
                dict(metadata),
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
            )
            + "\n",
            encoding="utf-8",
        )

    # Reuse the public loader as the final on-disk schema check.
    load_maneuver_gmp_prior(prior_dir)


def load_maneuver_gmp_prior(
    prior_dir: Path,
    *,
    num_maneuver_components: int = 8,
    trajectory_num_poses: int = 8,
) -> dict[str, np.ndarray]:
    """Load and validate the complete Stage-1 M-GMP prior."""

    prior_dir = Path(prior_dir).expanduser()
    if not prior_dir.is_dir():
        raise FileNotFoundError(
            f"M-GMP prior directory does not exist: {prior_dir}"
        )

    arrays = {}
    for name, filename in MGMP_PRIOR_FILENAMES.items():
        path = prior_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"M-GMP prior file does not exist: {path}")
        try:
            array = np.load(path, allow_pickle=False)
        except Exception as error:
            raise ManeuverGMPPriorError(
                f"Failed to load M-GMP prior file {path}: {error}"
            ) from error
        if not isinstance(array, np.ndarray):
            raise ManeuverGMPPriorError(
                f"M-GMP prior file must contain one NumPy array: {path}"
            )
        arrays[name] = array

    expected_shapes = {
        "displacement_mean": (
            int(num_maneuver_components),
            int(trajectory_num_poses),
            2,
        ),
        "displacement_std": (
            int(num_maneuver_components),
            int(trajectory_num_poses),
            2,
        ),
        "heading_mean": (
            int(num_maneuver_components),
            int(trajectory_num_poses),
        ),
        "heading_std": (
            int(num_maneuver_components),
            int(trajectory_num_poses),
        ),
        "component_counts": (int(num_maneuver_components),),
    }
    expected_dtypes = {
        "displacement_mean": np.dtype(np.float32),
        "displacement_std": np.dtype(np.float32),
        "heading_mean": np.dtype(np.float32),
        "heading_std": np.dtype(np.float32),
        "component_counts": np.dtype(np.int64),
    }
    for name, array in arrays.items():
        if array.shape != expected_shapes[name]:
            raise ManeuverGMPPriorError(
                f"M-GMP {name} shape mismatch: "
                f"{array.shape} vs {expected_shapes[name]}."
            )
        if array.dtype != expected_dtypes[name]:
            raise ManeuverGMPPriorError(
                f"M-GMP {name} dtype mismatch: "
                f"{array.dtype} vs {expected_dtypes[name]}."
            )

    for name in (
        "displacement_mean",
        "displacement_std",
        "heading_mean",
        "heading_std",
    ):
        if not np.isfinite(arrays[name]).all():
            raise ManeuverGMPPriorError(
                f"M-GMP {name} must contain only finite values."
            )
    scale_floors = {
        "displacement_std": MGMP_DISPLACEMENT_STANDARD_DEVIATION_FLOOR,
        "heading_std": MGMP_HEADING_STANDARD_DEVIATION_FLOOR,
    }
    for name, scale_floor in scale_floors.items():
        if np.any(arrays[name] < 0.0):
            raise ManeuverGMPPriorError(
                "M-GMP "
                f"{name} must be non-negative before the "
                f"{float(scale_floor):.2f} fitting floor."
            )
        arrays[name] = np.maximum(
            arrays[name],
            scale_floor,
        )

    if np.any(arrays["component_counts"] <= 0):
        raise ManeuverGMPPriorError(
            "M-GMP component_counts must be strictly positive for all "
            f"{num_maneuver_components} maneuver components."
        )
    return arrays


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the released five-file ManeuverFlow M-GMP prior."
        )
    )
    parser.add_argument(
        "--prior-dir",
        required=True,
        type=Path,
        help="Directory containing the five canonical .npy files.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    arrays = load_maneuver_gmp_prior(args.prior_dir)
    print(
        "Validated ManeuverFlow M-GMP prior: "
        f"components={arrays['component_counts'].shape[0]}, "
        f"poses={arrays['heading_mean'].shape[1]}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
