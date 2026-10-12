#!/usr/bin/env python3
"""Fit the M-GMP prior directly from the navtrain training log partition."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from navsim.agents.maneuverflow.mgmp_prior import (  # noqa: E402
    build_mgmp_metadata,
    fit_maneuver_gmp_prior,
    save_maneuver_gmp_prior,
)
from navsim.planning.training.data_split import navtrain_scene_filters, population_metadata  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--navsim-log-path", required=True, type=Path, help="OpenScene navsim_logs/trainval directory.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for five prior arrays and metadata.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing prior outputs.")
    return parser.parse_args(argv)


def load_training_trajectories(log_path: Path):
    from nuplan.common.actor_state.state_representation import StateSE2
    from pyquaternion import Quaternion

    from navsim.common.dataloader import SceneLoader
    from navsim.common.dataclasses import SensorConfig
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
        convert_absolute_to_relative_se2_array,
    )

    scene_filter, _ = navtrain_scene_filters()
    loader = SceneLoader(
        data_path=log_path, sensor_blobs_path=None, scene_filter=scene_filter,
        sensor_config=SensorConfig.build_no_sensors(),
    )
    tokens = sorted(loader.tokens)
    population = population_metadata(tokens)
    trajectories = []
    current = scene_filter.num_history_frames - 1
    for token in tokens:
        frames = loader.scene_frames_dicts[token][current:current + 9]
        global_poses = np.array([
            [*frame["ego2global_translation"][:2], Quaternion(*frame["ego2global_rotation"]).yaw_pitch_roll[0]]
            for frame in frames
        ], dtype=np.float64)
        trajectories.append(convert_absolute_to_relative_se2_array(StateSE2(*global_poses[0]), global_poses[1:]))
    return np.stack(trajectories).astype(np.float32), population


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    trajectories, population = load_training_trajectories(args.navsim_log_path.expanduser())
    prior = fit_maneuver_gmp_prior(trajectories)
    metadata = build_mgmp_metadata(
        trajectories=trajectories, component_counts=prior["component_counts"], training_population=population,
    )
    save_maneuver_gmp_prior(args.output_dir, prior, metadata=metadata, overwrite=args.overwrite)
    print(f"Built M-GMP prior from {len(trajectories)} navtrain training scenes.")
    print(f"Output directory: {args.output_dir.expanduser()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
