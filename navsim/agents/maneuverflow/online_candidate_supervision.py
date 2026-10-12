"""Official NAVSIM labels for freshly generated Stage-2 trajectories.

Only static scene metric caches are read. Candidate trajectories and their
labels are recomputed on every training forward from the current online model.
The simulator/scorer dependencies are imported lazily so inference never needs
to construct a training evaluator.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import torch

from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
    MultiMetricIndex,
    StateIndex,
    WeightedMetricIndex,
)

from .refinement_supervision import build_same_component_teacher_supervision


def interpolate_relative_hypotheses_to_state_array(
    hypotheses: np.ndarray,
    *,
    initial_ego_state: Any,
    expert_states: np.ndarray,
    proposal_sampling: Any,
) -> np.ndarray:
    """Vectorize the trajectory interpolation used by official evaluation."""

    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (
        ego_states_to_state_array,
    )

    hypotheses = np.asarray(hypotheses, dtype=np.float64)
    if hypotheses.ndim != 3 or hypotheses.shape[1:] != (8, 3):
        raise ValueError(f"PDM hypotheses must be [N, 8, 3], got {hypotheses.shape}.")
    if not np.isfinite(hypotheses).all():
        raise ValueError("PDM hypotheses contain non-finite values.")
    hypothesis_count, input_pose_count = hypotheses.shape[:2]
    target_pose_count = int(proposal_sampling.num_poses) + 1
    horizon_seconds = float(proposal_sampling.time_horizon)
    ego_pose = initial_ego_state.rear_axle
    cosine, sine = math.cos(ego_pose.heading), math.sin(ego_pose.heading)
    world_poses = np.empty_like(hypotheses)
    world_poses[..., 0] = hypotheses[..., 0] * cosine - hypotheses[..., 1] * sine + ego_pose.x
    world_poses[..., 1] = hypotheses[..., 0] * sine + hypotheses[..., 1] * cosine + ego_pose.y
    world_poses[..., 2] = (hypotheses[..., 2] + ego_pose.heading + np.pi) % (2 * np.pi) - np.pi

    source_poses = np.empty((hypothesis_count, input_pose_count + 1, 3), dtype=np.float64)
    source_poses[:, 0] = (ego_pose.x, ego_pose.y, ego_pose.heading)
    source_poses[:, 1:] = world_poses
    source_poses[..., 2] = np.unwrap(source_poses[..., 2], axis=1)
    source_times = np.linspace(0.0, horizon_seconds, input_pose_count + 1)
    target_times = np.minimum(
        np.arange(target_pose_count, dtype=np.float64) * float(proposal_sampling.interval_length),
        horizon_seconds,
    )
    interval_indices = np.clip(
        np.searchsorted(source_times, target_times, side="right") - 1,
        0,
        input_pose_count - 1,
    )
    interval_widths = source_times[interval_indices + 1] - source_times[interval_indices]
    interpolation_weights = (target_times - source_times[interval_indices]) / interval_widths
    interpolated = source_poses[:, interval_indices] + (
        source_poses[:, interval_indices + 1] - source_poses[:, interval_indices]
    ) * interpolation_weights[None, :, None]
    interpolated[..., 2] = (interpolated[..., 2] + np.pi) % (2 * np.pi) - np.pi

    all_states = np.zeros(
        (hypothesis_count + 1, target_pose_count, StateIndex.size()), dtype=np.float64
    )
    all_states[0] = expert_states
    all_states[1:, :, :3] = interpolated
    all_states[1:, 0] = ego_states_to_state_array([initial_ego_state])[0]
    return all_states


def extract_pairwise_pdm_factors(scorer: Any) -> np.ndarray:
    """Extract five factors with each candidate's EP relative to the expert.

    NAVSIM's scorer also computes a pool-wide normalized progress value. That
    value cannot label an online candidate independently of its 63 neighbors;
    recomputing the expert/candidate pair normalization preserves the official
    two-proposal evaluation semantics without changing the simulator/scorer.
    """

    multi_metrics = scorer._multi_metrics
    weighted_metrics = scorer._weighted_metrics
    raw_progress = scorer._progress_raw
    if multi_metrics is None or weighted_metrics is None or raw_progress is None:
        raise RuntimeError("PDM scorer did not expose completed metric arrays.")
    multiplicative = (
        multi_metrics[MultiMetricIndex.NO_COLLISION]
        * multi_metrics[MultiMetricIndex.DRIVABLE_AREA]
    )
    expert_progress = raw_progress[0] * multiplicative[0]
    candidate_progress = raw_progress[1:] * multiplicative[1:]
    pair_maximum = np.maximum(expert_progress, candidate_progress)
    threshold = float(scorer._config.progress_distance_threshold)
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized_progress = np.where(
            pair_maximum > threshold,
            candidate_progress / (pair_maximum + 1e-6),
            np.where(multiplicative[1:] == 0.0, 0.0, 1.0),
        )
    factors = np.stack(
        [
            multi_metrics[MultiMetricIndex.NO_COLLISION, 1:],
            multi_metrics[MultiMetricIndex.DRIVABLE_AREA, 1:],
            normalized_progress,
            weighted_metrics[WeightedMetricIndex.TTC, 1:],
            weighted_metrics[WeightedMetricIndex.COMFORTABLE, 1:],
        ],
        axis=0,
    ).astype(np.float32)
    if not np.isfinite(factors).all():
        raise RuntimeError("Official PDM evaluation produced non-finite factors.")
    if (factors < -1e-6).any() or (factors > 1.0 + 1e-6).any():
        raise RuntimeError("Official PDM factors fall outside [0, 1].")
    return np.clip(factors, 0.0, 1.0)


class OfficialPDMFactorScorer:
    """Evaluate online candidates using NAVSIM's unmodified simulator/scorer."""

    def __init__(self, metric_cache_loader: Any, *, chunk_size: int = 64) -> None:
        from navsim.evaluate.pdm_score import get_trajectory_as_array
        from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import (
            PDMScorer,
            PDMScorerConfig,
        )
        from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
        from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

        if int(chunk_size) <= 0:
            raise ValueError("PDM scoring chunk_size must be positive.")
        self._metric_cache_loader = metric_cache_loader
        self._chunk_size = int(chunk_size)
        self._get_trajectory_as_array = get_trajectory_as_array
        self.proposal_sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
        self._simulator = PDMSimulator(proposal_sampling=self.proposal_sampling)
        self._scorer = PDMScorer(
            proposal_sampling=self.proposal_sampling,
            config=PDMScorerConfig(
                progress_weight=5.0,
                ttc_weight=5.0,
                comfortable_weight=2.0,
                driving_direction_weight=0.0,
                driving_direction_horizon=1.0,
                driving_direction_compliance_threshold=2.0,
                driving_direction_violation_threshold=6.0,
                stopped_speed_threshold=5e-3,
                progress_distance_threshold=5.0,
            ),
        )

    def score(self, token: str, hypotheses: np.ndarray) -> np.ndarray:
        metric_cache = self._metric_cache_loader.get_from_token(token)
        initial_ego_state = metric_cache.ego_state
        expert_states = self._get_trajectory_as_array(
            metric_cache.trajectory, self.proposal_sampling, initial_ego_state.time_point
        )
        factor_chunks = []
        for start in range(0, len(hypotheses), self._chunk_size):
            proposals = interpolate_relative_hypotheses_to_state_array(
                hypotheses[start : start + self._chunk_size],
                initial_ego_state=initial_ego_state,
                expert_states=expert_states,
                proposal_sampling=self.proposal_sampling,
            )
            simulated_states = self._simulator.simulate_proposals(proposals, initial_ego_state)
            self._scorer.score_proposals(
                simulated_states,
                metric_cache.observation,
                metric_cache.centerline,
                metric_cache.route_lane_ids,
                metric_cache.drivable_area_map,
            )
            factor_chunks.append(extract_pairwise_pdm_factors(self._scorer))
        factors = np.concatenate(factor_chunks, axis=1)
        if factors.shape != (5, len(hypotheses)):
            raise RuntimeError(f"PDM factor cardinality mismatch: {factors.shape}.")
        return factors


class OnlineCandidateSupervision:
    """Lazy scene-token adapter returning detached labels on the model device."""

    def __init__(self, metric_cache_path: str | Path, *, scorer: Any = None) -> None:
        self.metric_cache_path = Path(metric_cache_path).expanduser()
        self._scorer = scorer

    def _get_scorer(self) -> Any:
        if self._scorer is None:
            from navsim.common.dataloader import MetricCacheLoader

            loader = MetricCacheLoader(self.metric_cache_path)
            self._scorer = OfficialPDMFactorScorer(loader)
        return self._scorer

    @torch.no_grad()
    def score_batch(
        self, tokens: Sequence[str] | str, hypotheses: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        if hypotheses.dim() != 4 or tuple(hypotheses.shape[-2:]) != (8, 3):
            raise ValueError(f"Online hypotheses must be [B, N, 8, 3], got {hypotheses.shape}.")
        token_list = [tokens] if isinstance(tokens, str) else [str(token) for token in tokens]
        if len(token_list) != hypotheses.shape[0]:
            raise ValueError("Scene token count must match the online hypothesis batch.")
        detached_hypotheses = hypotheses.detach().float().cpu().numpy()
        if not np.isfinite(detached_hypotheses).all():
            raise ValueError("Online hypotheses contain non-finite values.")
        labels = []
        scorer = self._get_scorer()
        for token, row in zip(token_list, detached_hypotheses):
            factors = np.asarray(scorer.score(token, row), dtype=np.float32)
            if factors.shape != (5, hypotheses.shape[1]):
                raise ValueError(f"Online PDM factors must be [5, N], got {factors.shape}.")
            if not np.isfinite(factors).all():
                raise ValueError("Online PDM factors contain non-finite values.")
            if (factors < 0).any() or (factors > 1).any():
                raise ValueError("Online PDM factors must lie in [0, 1].")
            labels.append(factors)
        factor_labels = torch.from_numpy(np.stack(labels)).to(device=hypotheses.device)
        nc, dac, ep, ttc, comfort = factor_labels.unbind(dim=1)
        pdm_scores = nc * dac * (5.0 * ep + 5.0 * ttc + 2.0 * comfort) / 12.0
        return {"pdm_factor_labels": factor_labels, "online_pdm_scores": pdm_scores}


def build_online_refinement_supervision(
    *,
    hypotheses: torch.Tensor,
    pdm_scores: torch.Tensor,
    maneuver_component_indices: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Use the same online set for bases and same-component argmax targets."""

    return build_same_component_teacher_supervision(
        base_hypotheses=hypotheses,
        base_pdm_scores=pdm_scores,
        base_maneuver_component_indices=maneuver_component_indices,
        candidate_hypotheses=hypotheses,
        candidate_pdm_scores=pdm_scores,
        candidate_maneuver_component_indices=maneuver_component_indices,
    )
