import numpy as np

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import BBCoordsIndex, EgoAreaIndex


class PDMReferenceScorer(PDMScorer):
    """NAVSIM v1.0 scoring for selecting cached reference trajectories."""

    def __init__(self, proposal_sampling: TrajectorySampling):
        super().__init__(proposal_sampling, config=PDMScorerConfig(progress_distance_threshold=0.1))

    def _calculate_driving_direction_compliance(self) -> None:
        # Preserve the v1.0 proposal aggregation used during cache preparation.
        center_coordinates = self._ego_coords[:, :, BBCoordsIndex.CENTER]
        oncoming_progress = np.zeros(
            (self._num_proposals, self.proposal_sampling.num_poses + 1), dtype=np.float64,
        )
        oncoming_progress[:, 1:] = (
            (center_coordinates[:, 1:] - center_coordinates[:, :-1]) ** 2.0
        ).sum(axis=-1) ** 0.5
        oncoming_progress[~self._ego_areas[:, :, EgoAreaIndex.ONCOMING_TRAFFIC]] = 0.0

        horizon = int(self._config.driving_direction_horizon / self.proposal_sampling.interval_length)
        oncoming_progress_over_horizon = np.array(
            [
                sum(oncoming_progress[max(0, time_idx - horizon) : time_idx + 1])
                for time_idx in range(len(oncoming_progress))
            ],
            dtype=np.float64,
        )
        direction_scores = np.ones(self._num_proposals, dtype=np.float64)
        for proposal_idx, progress in enumerate(oncoming_progress_over_horizon.max(axis=-1)):
            if progress < self._config.driving_direction_compliance_threshold:
                direction_scores[proposal_idx] = 1.0
            elif progress < self._config.driving_direction_violation_threshold:
                direction_scores[proposal_idx] = 0.5
            else:
                direction_scores[proposal_idx] = 0.0

        self._multi_metrics = np.concatenate((self._multi_metrics, direction_scores[None, :]), axis=0)
