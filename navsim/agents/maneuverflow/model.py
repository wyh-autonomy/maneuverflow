from collections.abc import Mapping
from typing import Dict, Optional, Any, List, Tuple
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager, nullcontext

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except Exception:
    SDPBackend = None
    sdpa_kernel = None

from .transfuser_backbone import TransfuserBackbone
from .targets import BoundingBox2DIndex
from .architecture import ManeuverFlowHypothesizer, ManeuverFlowImprover
from .config import (
    DEFAULT_ECFR_FINAL_POOL_COMPOSITION,
    ECFR_FINAL_POOL_COMPOSITIONS,
    ECFRFinalPoolComposition,
    MANEUVERFLOW_INFERENCE_POLICY,
    ManeuverFlowConfig,
)
from .ecfr import EvidenceCoupledFrenetRefiner
from .ema_target_encoder import EMATargetEncoder
from .inference_api import (
    ManeuverFlowInferenceOutput,
    validate_maneuverflow_inference_output,
)
from .m3f import ManeuverMixtureMeanFlowField
from .mgmp_prior import (
    load_maneuver_gmp_prior,
)
from .trajectory_codec import (
    M3FTrajectoryTokenizer,
    M3FFieldDecoder,
    trajectory_to_flow_state,
    flow_state_to_trajectory,
)
from .verifier import ManeuverFlowVerifier
from .online_candidate_supervision import (
    OnlineCandidateSupervision,
    build_online_refinement_supervision,
)

@contextmanager
def _temporary_evaluation_mode(module: nn.Module):
    """Disable stochastic/state updates for a temporary stop-gradient pass."""

    training_states = {
        child: bool(child.training)
        for child in module.modules()
    }
    module.eval()
    try:
        yield
    finally:
        for child, was_training in training_states.items():
            child.training = was_training


def _select_ecfr_sources_from_preliminary_ranking(
    base_scores: torch.Tensor,
    *,
    primary_family_counts: torch.Tensor,
    refined_hypothesis_count: int,
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    """Select the highest-scoring bases inside the routed Top-1 family.

    ManeuverFlow generates the ranked maneuver families as contiguous blocks. ECFR
    sources come only from the first (highest-ranked) block.
    """

    if base_scores.dim() != 2:
        raise ValueError(
            "ECFR source selection expects base_scores [B, N], got "
            f"{tuple(base_scores.shape)}"
        )

    num_candidates = int(base_scores.shape[1])
    if num_candidates <= 0:
        raise ValueError("ECFR needs at least one base hypothesis.")

    requested = int(refined_hypothesis_count)
    if requested <= 0:
        raise ValueError(
            "inference_num_refined_hypotheses must be positive when ECFR is enabled."
        )

    primary_counts = primary_family_counts.to(
        device=base_scores.device,
        dtype=torch.long,
    ).reshape(-1)
    if primary_counts.numel() == 1:
        primary_counts = primary_counts.expand(base_scores.shape[0])
    if primary_counts.numel() != base_scores.shape[0]:
        raise ValueError(
            "primary_family_counts must provide one value per batch row, got "
            f"{primary_counts.numel()} for batch size {base_scores.shape[0]}."
        )
    if torch.any(primary_counts <= 0) or torch.any(
        primary_counts > num_candidates
    ):
        raise ValueError(
            "primary_family_counts must lie in [1, N] for every batch row."
        )

    effective = min(requested, int(primary_counts.min().item()))
    positions = torch.arange(
        num_candidates,
        device=base_scores.device,
    ).view(1, -1)
    primary_mask = positions < primary_counts.view(-1, 1)
    masked_scores = base_scores.masked_fill(
        ~primary_mask,
        torch.finfo(base_scores.dtype).min,
    )
    indices = torch.topk(
        masked_scores,
        k=effective,
        dim=1,
        largest=True,
    ).indices
    mask = torch.zeros_like(base_scores, dtype=torch.bool)
    mask.scatter_(1, indices, True)
    return indices, mask, requested, effective


def _allocate_probability_proportional_counts(
    probabilities: torch.Tensor,
    *,
    total_count: int,
    min_per_family: int = 0,
) -> torch.Tensor:
    """Apportion an exact integer budget with largest remainders."""
    if probabilities.dim() != 2 or probabilities.shape[1] <= 0:
        raise ValueError(f"probabilities must be [B, K], got {tuple(probabilities.shape)}")
    total_count = int(total_count)
    min_per_family = max(0, int(min_per_family))
    reserved = min_per_family * int(probabilities.shape[1])
    if total_count < reserved:
        raise ValueError(
            f"total_count={total_count} cannot reserve {min_per_family} for "
            f"each of {probabilities.shape[1]} families."
        )

    probs = torch.nan_to_num(probabilities.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    denom = probs.sum(dim=1, keepdim=True)
    uniform = torch.full_like(probs, 1.0 / float(probs.shape[1]))
    probs = torch.where(denom > 0.0, probs / denom.clamp_min(1e-12), uniform)

    distributable = total_count - reserved
    quotas = probs * float(distributable)
    counts = torch.floor(quotas).long() + min_per_family
    remainder_count = total_count - counts.sum(dim=1)
    remainder_order = torch.argsort(quotas - torch.floor(quotas), dim=1, descending=True, stable=True)
    remainder_rank = torch.arange(probabilities.shape[1], device=probabilities.device).view(1, -1)
    add_mask = remainder_rank < remainder_count.view(-1, 1)
    additions = torch.zeros_like(counts).scatter(1, remainder_order, add_mask.long())
    counts = counts + additions
    if not torch.all(counts.sum(dim=1) == total_count):
        raise RuntimeError(
            "Probability allocation failed to preserve the exact proposal budget."
        )
    return counts


def _expand_selected_families_by_counts(
    selected_families: torch.Tensor,
    family_counts: torch.Tensor,
) -> torch.Tensor:
    """Expand ranked family ids into contiguous per-row proposal family ids."""
    if selected_families.shape != family_counts.shape or selected_families.dim() != 2:
        raise ValueError(
            "selected_families and family_counts must share [B, K], got "
            f"{tuple(selected_families.shape)} and {tuple(family_counts.shape)}."
        )
    row_totals = family_counts.sum(dim=1)
    if torch.any(family_counts < 0):
        raise ValueError("family_counts must be non-negative.")
    if row_totals.numel() == 0 or not torch.all(row_totals == row_totals[0]):
        raise ValueError("All batch rows must have the same total proposal count.")
    total = int(row_totals[0].item())
    if total <= 0:
        raise ValueError("family_counts must keep at least one proposal.")

    boundaries = family_counts.cumsum(dim=1)[:, :-1]
    positions = torch.arange(total, device=selected_families.device).view(1, -1, 1)
    family_slots = (positions >= boundaries.unsqueeze(1)).sum(dim=-1)
    return torch.gather(selected_families, 1, family_slots)


def _build_maneuverflow_retained_base_indices(
    base_scores: torch.Tensor,
    *,
    retained_hypothesis_count: int,
    ecfr_source_indices: Optional[torch.Tensor] = None,
    final_pool_composition: ECFRFinalPoolComposition = (
        DEFAULT_ECFR_FINAL_POOL_COMPOSITION
    ),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compose the retained-base portion of the final ManeuverFlow candidate pool."""

    if base_scores.dim() != 2:
        raise ValueError(
            "Retained-base selection expects base_scores [B, N], got "
            f"{tuple(base_scores.shape)}."
        )
    num_candidates = int(base_scores.shape[1])
    keep_count = int(retained_hypothesis_count)
    if keep_count <= 0 or keep_count > num_candidates:
        raise ValueError(
            f"retained_hypothesis_count must lie in [1, {num_candidates}], "
            f"got {keep_count}."
        )
    if final_pool_composition not in ECFR_FINAL_POOL_COMPOSITIONS:
        raise ValueError(
            "Unknown ECFR final-pool composition: "
            f"{final_pool_composition!r}."
        )

    if final_pool_composition == "source_replacement":
        if ecfr_source_indices is None:
            raise ValueError(
                "source_replacement requires ecfr_source_indices."
            )
        if (
            ecfr_source_indices.dim() != 2
            or ecfr_source_indices.shape[0] != base_scores.shape[0]
        ):
            raise ValueError(
                "ecfr_source_indices must have shape [B, R], got "
                f"{tuple(ecfr_source_indices.shape)}."
            )
        source_count = num_candidates - keep_count
        if int(ecfr_source_indices.shape[1]) != source_count:
            raise ValueError(
                "source_replacement requires exactly N - retained_count "
                f"ECFR sources ({source_count}), got "
                f"{ecfr_source_indices.shape[1]}."
            )
        if ecfr_source_indices.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }:
            raise ValueError("ecfr_source_indices must use an integer dtype.")

        source_indices = ecfr_source_indices.to(
            device=base_scores.device,
            dtype=torch.long,
        )
        if torch.any(source_indices < 0) or torch.any(
            source_indices >= num_candidates
        ):
            raise ValueError(
                "ecfr_source_indices must lie in [0, N) for every batch row."
            )
        if source_count > 1:
            ordered_sources = torch.sort(source_indices, dim=1).values
            if torch.any(ordered_sources[:, 1:] == ordered_sources[:, :-1]):
                raise ValueError(
                    "source_replacement requires distinct ECFR sources "
                    "within each batch row."
                )

        source_mask = torch.zeros_like(base_scores, dtype=torch.bool)
        source_mask.scatter_(1, source_indices, True)
        retained_mask = ~source_mask
    else:
        topk_indices = torch.topk(
            base_scores,
            k=keep_count,
            dim=1,
            largest=True,
        ).indices
        retained_mask = torch.zeros_like(base_scores, dtype=torch.bool)
        retained_mask.scatter_(1, topk_indices, True)

    positions = torch.arange(
        num_candidates,
        device=base_scores.device,
    ).view(1, -1)
    retained_indices = positions.expand(
        base_scores.shape[0],
        -1,
    ).masked_select(retained_mask)
    return (
        retained_indices.view(base_scores.shape[0], keep_count),
        retained_mask,
    )


def _compose_log_pdm_score(
    nc_r: torch.Tensor,
    dac_r: torch.Tensor,
    ep_r: torch.Tensor,
    ttc_r: torch.Tensor,
    comfort_r: torch.Tensor,
    ddc_r: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    score = torch.log(nc_r + eps) + torch.log(dac_r + eps)
    if ddc_r is not None:
        score = score + torch.log(ddc_r + eps)
    score = score + torch.log(5.0 * ep_r + 5.0 * ttc_r + 2.0 * comfort_r + eps)
    return score


def _compute_ecfr_residual_cost(
    trajectory_residuals: torch.Tensor,
    *,
    config: Optional[Any] = None,
) -> torch.Tensor:
    """Apply explicitly configured refinement penalties when ranking.

    The EMA recipe sets all coefficients to zero for raw log-PDM ranking. The
    released-checkpoint configuration retains its original coefficients.
    """

    if trajectory_residuals.dim() != 4 or trajectory_residuals.shape[-1] < 2:
        raise ValueError(
            "ECFR trajectory residuals must have shape [B, R, T, >=2], got "
            f"{tuple(trajectory_residuals.shape)}."
        )

    residuals = trajectory_residuals.float()
    policy = MANEUVERFLOW_INFERENCE_POLICY if config is None else config
    xy_cost = residuals[..., :2].pow(2).sum(dim=-1).mean(dim=-1)
    cost = float(policy.refined_score_delta_cost_xy) * xy_cost

    if residuals.shape[-1] >= 3:
        wrapped_yaw = torch.atan2(
            torch.sin(residuals[..., 2]),
            torch.cos(residuals[..., 2]),
        )
        yaw_cost = wrapped_yaw.pow(2).mean(dim=-1)
        cost = (
            cost
            + float(policy.refined_score_delta_cost_yaw) * yaw_cost
        )

    if residuals.shape[-2] >= 3:
        xy_second_difference = (
            residuals[..., 2:, :2]
            - 2.0 * residuals[..., 1:-1, :2]
            + residuals[..., :-2, :2]
        )
        smooth_cost = (
            xy_second_difference.pow(2).sum(dim=-1).mean(dim=-1)
        )
        cost = (
            cost
            + float(policy.refined_score_delta_cost_smooth)
            * smooth_cost
        )

    return cost.to(dtype=trajectory_residuals.dtype)


def _build_maneuver_anchor_indices_from_anchor_count(
    total_anchors: int,
    num_maneuver_components: int,
) -> torch.Tensor:
    """Build maneuver-component anchor ids as ``[M, anchors_per_component]``."""
    total_anchors = int(total_anchors)
    num_maneuver_components = int(num_maneuver_components)
    if total_anchors <= 0:
        raise ValueError(f"total_anchors must be positive, got {total_anchors}.")
    if num_maneuver_components <= 0:
        raise ValueError(
            "num_maneuver_components must be positive, got "
            f"{num_maneuver_components}."
        )
    if total_anchors % num_maneuver_components != 0:
        raise ValueError(
            f"Cannot split {total_anchors} verifier anchors evenly across "
            f"{num_maneuver_components} maneuver components."
        )
    anchors_per_maneuver = total_anchors // num_maneuver_components
    return torch.arange(total_anchors, dtype=torch.long).view(
        num_maneuver_components,
        anchors_per_maneuver,
    )


def _nc_score_from_logits(nc_logits: torch.Tensor, expected_score: bool = False) -> torch.Tensor:
    if nc_logits.shape[-1] == 1:
        return torch.sigmoid(nc_logits.squeeze(-1))
    if nc_logits.shape[-1] != 3:
        raise ValueError(f"Expected NC logits with last dim 1 or 3, got {tuple(nc_logits.shape)}")
    nc_probs = F.softmax(nc_logits, dim=-1)
    if expected_score:
        return 0.5 * nc_probs[..., 1] + nc_probs[..., 2]
    return nc_probs[..., 1] + nc_probs[..., 2]


def _future_target_pose_offsets_from_targets(
    targets: Dict[str, torch.Tensor],
    *,
    batch_size: int,
    time_steps: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    trajectory = targets.get("gt_trajectory", None)
    if not torch.is_tensor(trajectory):
        return None
    trajectory = trajectory.to(device=device, dtype=dtype)
    if trajectory.dim() == 2:
        trajectory = trajectory.unsqueeze(0)
    elif trajectory.dim() >= 4:
        trajectory = trajectory.reshape(trajectory.shape[0], -1, trajectory.shape[-2], trajectory.shape[-1])[:, 0]
    if trajectory.dim() != 3 or trajectory.shape[-1] < 3 or trajectory.shape[1] <= 0:
        return None
    if trajectory.shape[0] == 1 and batch_size > 1:
        trajectory = trajectory.expand(batch_size, -1, -1)
    if trajectory.shape[0] != batch_size:
        return None

    # Future BEV targets are emitted at cur+2/4/6/8 frames (1s/2s/3s/4s),
    # while gt_trajectory is sampled at 0.5s intervals.
    indices = torch.arange(time_steps, device=device, dtype=torch.long) * 2 + 1
    indices = indices.clamp(max=trajectory.shape[1] - 1)
    return trajectory.index_select(1, indices)[..., :3]


def _warp_future_bev_target_to_current_frame(
    future_bev: torch.Tensor,
    ego_offsets: torch.Tensor,
    config: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if future_bev.dim() != 5:
        raise ValueError(f"future_bev must be [B, T, C, H, W], got {tuple(future_bev.shape)}")
    if ego_offsets.dim() != 3 or ego_offsets.shape[-1] < 3:
        raise ValueError(f"ego_offsets must be [B, T, 3], got {tuple(ego_offsets.shape)}")

    batch_size, time_steps, channels, height, width = future_bev.shape
    if ego_offsets.shape[0] != batch_size or ego_offsets.shape[1] < time_steps:
        raise ValueError(
            f"ego_offsets must match future_bev batch/time, got {tuple(ego_offsets.shape)} for {tuple(future_bev.shape)}"
        )
    ego_offsets = ego_offsets[:, :time_steps].to(device=future_bev.device, dtype=future_bev.dtype)

    min_x = float(config.bev_minimum_x)
    max_x = float(config.bev_maximum_x)
    min_y = float(config.bev_minimum_y)
    max_y = float(config.bev_maximum_y)
    device = future_bev.device
    dtype = future_bev.dtype

    legacy_geometry = config.bev_coordinate_convention == "legacy_descending_x"
    if legacy_geometry:
        row_norm = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        col_norm = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        x_coords = min_x + (1.0 - row_norm) * 0.5 * (max_x - min_x)
        y_coords = min_y + (col_norm + 1.0) * 0.5 * (max_y - min_y)
    else:
        x_coords = min_x + (torch.arange(height, device=device, dtype=dtype) + 0.5) * (max_x - min_x) / height
        y_coords = min_y + (torch.arange(width, device=device, dtype=dtype) + 0.5) * (max_y - min_y) / width
    x0, y0 = torch.meshgrid(x_coords, y_coords, indexing="ij")

    dx = ego_offsets[..., 0].view(batch_size, time_steps, 1, 1)
    dy = ego_offsets[..., 1].view(batch_size, time_steps, 1, 1)
    theta = ego_offsets[..., 2].view(batch_size, time_steps, 1, 1)
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    rel_x = x0.view(1, 1, height, width) - dx
    rel_y = y0.view(1, 1, height, width) - dy
    future_x = cos_theta * rel_x + sin_theta * rel_y
    future_y = -sin_theta * rel_x + cos_theta * rel_y

    grid_x = 2.0 * (future_y - min_y) / max(max_y - min_y, 1e-6) - 1.0
    grid_y = 1.0 - 2.0 * (future_x - min_x) / max(max_x - min_x, 1e-6)
    if not legacy_geometry:
        grid_y = -grid_y
    sample_grid = torch.stack([grid_x, grid_y], dim=-1)
    valid = (
        (sample_grid[..., 0] >= -1.0)
        & (sample_grid[..., 0] <= 1.0)
        & (sample_grid[..., 1] >= -1.0)
        & (sample_grid[..., 1] <= 1.0)
    ).unsqueeze(2)

    warped = F.grid_sample(
        future_bev.reshape(batch_size * time_steps, channels, height, width),
        sample_grid.reshape(batch_size * time_steps, height, width, 2),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=legacy_geometry,
    )
    warped = warped.view(batch_size, time_steps, channels, height, width)
    warped = warped * valid.to(dtype=warped.dtype)
    return warped, valid


def _as_future_spatial_mask(mask: Optional[torch.Tensor], reference: torch.Tensor) -> Optional[torch.Tensor]:
    if not torch.is_tensor(mask):
        return None
    mask = mask.to(device=reference.device, dtype=torch.bool)
    if mask.dim() == 3:
        mask = mask.unsqueeze(0).unsqueeze(2)
    elif mask.dim() == 4:
        if mask.shape[0] == reference.shape[1]:
            mask = mask.unsqueeze(0)
        if mask.shape[2] != 1:
            mask = mask.unsqueeze(2)
    if mask.dim() != 5:
        return None
    if mask.shape[0] == 1 and reference.shape[0] > 1:
        mask = mask.expand(reference.shape[0], -1, -1, -1, -1)
    return mask[:, : reference.shape[1]]


def _add_future_spatial_masks_to_feature_targets(
    feature_targets: Dict[str, torch.Tensor],
    spatial_mask: Optional[torch.Tensor],
    reference: torch.Tensor,
) -> None:
    spatial_mask = _as_future_spatial_mask(spatial_mask, reference)
    if spatial_mask is None:
        return
    feature_targets["future_bev_feature_spatial_valid_mask"] = spatial_mask
    res_spatial_mask = spatial_mask.clone()
    if res_spatial_mask.shape[1] > 1:
        res_spatial_mask[:, 1:] = spatial_mask[:, 1:] & spatial_mask[:, :-1]
    feature_targets["future_bev_residual_feature_spatial_valid_mask"] = (
        res_spatial_mask
    )


class IntervalSinusoidalEmbedding(nn.Module):
    """Continuous scalar -> sinusoidal embedding."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = np.log(10000) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=x.dtype) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

class ManeuverFlowModel(nn.Module):
    """ManeuverFlow's Hypothesize--Verify--Improve computation graph.

    ``hypothesizer`` combines the TransFuser planning context, M-GMP and M³F
    to produce maneuver-indexed base hypotheses.  ``verifier`` predicts one
    shared Future BEV Field, retrieves Trajectory Evidence Memory, and applies
    Metric Prism. ``improver`` contains ECFR; its refinements join a
    retained base subset before the fixed-size pool is re-queried
    and jointly re-scored.

    Joint training scores and refines one fresh 64-candidate online set.
    """
    def __init__(self, config: ManeuverFlowConfig):
        super().__init__()
        self._config = config
        self._online_candidate_supervision: Optional[OnlineCandidateSupervision] = None
        self.hypothesizer = ManeuverFlowHypothesizer()
        self.verifier = ManeuverFlowVerifier(config)
        self.improver = ManeuverFlowImprover()

        # One maneuver-context query plus the fixed set of agent queries.
        self._query_splits = [1, config.num_bounding_boxes]

        # Hypothesize: TransFuser perception and planning-context encoding.
        self.hypothesizer.perception.transfuser_backbone = TransfuserBackbone(config)
        self.ema_target_encoder = (
            EMATargetEncoder(
                self.hypothesizer.perception.transfuser_backbone,
                decay=float(config.ema_target_decay),
            )
            if config.training_phase == "joint_finetuning"
            else None
        )
        self.hypothesizer.perception.bev_memory_embedding = nn.Embedding(
            8**2 + 1,
            config.planning_context_dimension,
        )
        self.hypothesizer.perception.query_embedding = nn.Embedding(
            sum(self._query_splits),
            config.planning_context_dimension,
        )
        self.hypothesizer.perception.bev_projection = nn.Conv2d(
            512,
            config.planning_context_dimension,
            kernel_size=1,
        )
        self.hypothesizer.perception.state_navigation_encoder = nn.Linear(
            4 + 2 + 2,
            config.planning_context_dimension,
        )

        self.hypothesizer.perception.bev_semantic_head = nn.Sequential(
            nn.Conv2d(
                config.current_bev_feature_channels,
                config.current_bev_feature_channels,
                (3, 3),
                1,
                (1, 1),
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                config.current_bev_feature_channels,
                config.num_bev_classes,
                (1, 1),
                1,
                0,
                bias=True,
            ),
            nn.Upsample(size=(config.lidar_resolution_height // 2, config.lidar_resolution_width), mode="bilinear", align_corners=False),
        )

        tf_decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.planning_context_dimension,
            nhead=config.perception_query_num_heads,
            dim_feedforward=config.perception_query_feed_forward_dimension,
            dropout=config.perception_query_dropout,
            batch_first=True,
        )
        self.hypothesizer.perception.query_decoder = nn.TransformerDecoder(
            tf_decoder_layer,
            config.perception_query_num_layers,
        )

        self.hypothesizer.perception.agent_prediction_head = AgentPredictionHead(
            config.num_bounding_boxes,
            config.perception_query_feed_forward_dimension,
            config.planning_context_dimension,
            dropout=0.0,
            use_norm=False,
        )
        self.hypothesizer.maneuver_decoder = StatusModulatedManeuverDecoder(
            config.planning_context_dimension,
            nav_dim=4,
            kin_dim=4,
            num_classes=config.num_maneuver_components,
            dropout=config.maneuver_head_dropout,
        )

        # Hypothesize: M-GMP assets and M³F source trajectories.
        checkpoint_assets = bool(
            getattr(config, "mgmp_assets_from_checkpoint", False)
        )
        proposal_anchor_count = int(config.proposal_anchor_count)
        trajectory_num_poses = int(getattr(config, "trajectory_num_poses", 8))
        if proposal_anchor_count <= 0 or trajectory_num_poses <= 0:
            raise ValueError(
                "Checkpoint tensor geometry requires positive "
                "proposal_anchor_count and trajectory_num_poses."
            )
        if (
            int(config.num_maneuver_components) <= 0
            or proposal_anchor_count
            % int(config.num_maneuver_components)
            != 0
        ):
            raise ValueError(
                "Checkpoint proposal-anchor geometry must divide evenly "
                "across maneuver components."
            )

        # The canonical inference state_dict contains four fixed anchor tensors.
        # They are retained only as immutable schema-compatible assets; the
        # paper's M-GMP/M³F training and inference paths never consume their
        # values and therefore require no external trajectory-anchor file.
        self.hypothesizer.proposal_anchor_bank = nn.Parameter(
            torch.zeros(proposal_anchor_count, trajectory_num_poses, 3),
            requires_grad=False,
        )
        self.verifier.proposal_anchor_bank = nn.Parameter(
            torch.zeros(proposal_anchor_count, trajectory_num_poses, 3),
            requires_grad=False,
        )
        maneuver_anchor_indices = _build_maneuver_anchor_indices_from_anchor_count(
            total_anchors=proposal_anchor_count,
            num_maneuver_components=int(config.num_maneuver_components),
        )
        self.verifier.maneuver_anchor_indices = maneuver_anchor_indices
        self.hypothesizer.maneuver_anchor_bank = torch.zeros(
            int(config.num_maneuver_components),
            proposal_anchor_count // int(config.num_maneuver_components),
            trajectory_num_poses,
            3,
        )

        # Eight maneuver-conditioned temporal Gaussians in displacement space.
        # Buffers remain under the immutable released-checkpoint namespace.
        if not checkpoint_assets:
            prior = load_maneuver_gmp_prior(
                Path(config.mgmp_prior_dir),
                num_maneuver_components=config.num_maneuver_components,
                trajectory_num_poses=trajectory_num_poses,
            )
            self.hypothesizer.maneuver_gmp.displacement_mean = (
                torch.from_numpy(prior["displacement_mean"])
            )
            self.hypothesizer.maneuver_gmp.displacement_std = (
                torch.from_numpy(prior["displacement_std"])
            )
            self.hypothesizer.maneuver_gmp.heading_mean = torch.from_numpy(
                prior["heading_mean"]
            )
            self.hypothesizer.maneuver_gmp.heading_std = torch.from_numpy(
                prior["heading_std"]
            )
            self.hypothesizer.maneuver_gmp.component_counts = (
                torch.from_numpy(prior["component_counts"])
            )
            print(
                "[ManeuverFlow] Loaded M-GMP prior: "
                f"displacement_mean "
                f"{tuple(self.hypothesizer.maneuver_gmp.displacement_mean.shape)}"
            )

        if checkpoint_assets:
            num_maneuver_components = int(config.num_maneuver_components)
            self.hypothesizer.maneuver_gmp.displacement_mean = torch.zeros(
                num_maneuver_components,
                trajectory_num_poses,
                2,
                dtype=torch.float32,
            )
            self.hypothesizer.maneuver_gmp.displacement_std = torch.zeros(
                num_maneuver_components,
                trajectory_num_poses,
                2,
                dtype=torch.float32,
            )
            self.hypothesizer.maneuver_gmp.heading_mean = torch.zeros(
                num_maneuver_components,
                trajectory_num_poses,
                dtype=torch.float32,
            )
            self.hypothesizer.maneuver_gmp.heading_std = torch.zeros(
                num_maneuver_components,
                trajectory_num_poses,
                dtype=torch.float32,
            )
            self.hypothesizer.maneuver_gmp.component_counts = torch.zeros(
                num_maneuver_components,
                dtype=torch.long,
            )

        # M³F network components.
        self.hypothesizer.meanflow_generator.start_time_embedding = (
            IntervalSinusoidalEmbedding(config.m3f_hidden_dim)
        )
        self.hypothesizer.meanflow_generator.interval_duration_embedding = (
            IntervalSinusoidalEmbedding(config.m3f_hidden_dim)
        )
        self.hypothesizer.meanflow_generator.interval_encoder = nn.Sequential(
            # Time representation for adaLN-Zero:
            #   SinEmb(t) + SinEmb(delta=t-r) are concatenated and fused by a small MLP.
            nn.Linear(2 * config.m3f_hidden_dim, config.m3f_hidden_dim),
            nn.SiLU(),
            nn.Linear(config.m3f_hidden_dim, config.m3f_hidden_dim),
        )
        self.hypothesizer.meanflow_generator.trajectory_tokenizer = M3FTrajectoryTokenizer(config, input_dim=4)
        self.hypothesizer.meanflow_generator.field = ManeuverMixtureMeanFlowField(config)
        self.hypothesizer.meanflow_generator.field_decoder = M3FFieldDecoder(config, output_dim=4)

        # M³F BEV position embedding.
        self.hypothesizer.meanflow_generator.bev_position_embedding = (
            nn.Embedding(
                config.bev_memory_token_count,
                config.m3f_hidden_dim,
            )
        )

        # Trajectory Evidence Memory is produced once and shared by verification
        # and evidence-coupled refinement.
        self.improver.ecfr = EvidenceCoupledFrenetRefiner(config)

        # Apply the release lifecycle's trainability contract.
        if self._config.training_phase == "generation_pretraining":
            self._freeze_for_generation_pretraining()
        elif self._config.training_phase == "joint_finetuning":
            self._freeze_for_joint_finetuning()
        elif self._config.training_phase == "inference":
            self._freeze_perception()
            self._set_m3f_trainable(False)
            self._set_verifier_trainable(False)
            self._set_ecfr_trainable(False)
        else:
            raise ValueError(
                f"Unsupported ManeuverFlow training_phase={self._config.training_phase!r}."
            )

    def _get_perception_modules(self) -> List[nn.Module]:
        return [
            self.hypothesizer.perception.transfuser_backbone,
            self.hypothesizer.perception.bev_memory_embedding,
            self.hypothesizer.perception.query_embedding,
            self.hypothesizer.perception.bev_projection,
            self.hypothesizer.perception.state_navigation_encoder,
            self.hypothesizer.perception.bev_semantic_head,
            self.hypothesizer.perception.query_decoder,
            self.hypothesizer.perception.agent_prediction_head,
            self.hypothesizer.maneuver_decoder,
        ]

    def _get_m3f_modules(self) -> List[nn.Module]:
        return [
            self.hypothesizer.meanflow_generator.interval_encoder,
            self.hypothesizer.meanflow_generator.trajectory_tokenizer,
            self.hypothesizer.meanflow_generator.field,
            self.hypothesizer.meanflow_generator.field_decoder,
            self.hypothesizer.meanflow_generator.bev_position_embedding,
        ]

    def _set_modules_trainable(self, modules: List[nn.Module], enabled: bool) -> None:
        for module in modules:
            if module is None:
                continue
            for param in module.parameters():
                param.requires_grad = enabled
            module.train(enabled)

    def _freeze_perception(self):
        """Freeze the perception modules."""
        self._set_modules_trainable(self._get_perception_modules(), False)

    def _set_perception_trainable(self, enabled: bool):
        self._set_modules_trainable(self._get_perception_modules(), enabled)

    def _set_m3f_trainable(self, enabled: bool):
        self._set_modules_trainable(self._get_m3f_modules(), enabled)

    def _set_verifier_trainable(self, enabled: bool):
        self._set_modules_trainable([self.verifier, self.verifier.trajectory_evidence_probe], enabled)
        # The released schema contains this unused compatibility tensor.
        # It is never part of the released computation graph or optimizer.
        if self.verifier.proposal_anchor_bank is not None:
            self.verifier.proposal_anchor_bank.requires_grad_(False)


    def _set_ecfr_trainable(self, enabled: bool):
        self._set_modules_trainable([self.improver.ecfr], enabled)

    def _freeze_for_generation_pretraining(self):
        """Generation pretraining: optimize perception and M³F."""
        self._set_perception_trainable(True)
        self._set_m3f_trainable(True)
        self._set_verifier_trainable(False)
        self._set_ecfr_trainable(False)


    def _freeze_for_joint_finetuning(self):
        """Apply module-level trainability for Joint fine-tuning."""
        self._set_perception_trainable(True)
        self._set_m3f_trainable(True)
        # ECFR reuses the live verifier evidence graph so its loss can update
        # the shared evidence/world/perception path during joint fine-tuning.
        self._set_verifier_trainable(True)
        self._set_ecfr_trainable(True)

    def train(self, mode: bool = True):
        """Keep frozen modules in evaluation mode.

        PyTorch Lightning calls model.train() at each epoch start,
        which would otherwise reset ALL modules to train mode, including frozen ones.
        This causes BatchNorm to use batch statistics during training but running
        statistics during validation, leading to train/val metric mismatch.
        """
        super().train(mode)

        if mode:
            if self._config.training_phase == "generation_pretraining":
                self._freeze_for_generation_pretraining()
            elif self._config.training_phase == "joint_finetuning":
                self._freeze_for_joint_finetuning()
            elif self._config.training_phase == "inference":
                self._freeze_perception()
                self._set_m3f_trainable(False)
                self._set_verifier_trainable(False)
                self._set_ecfr_trainable(False)

        return self

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        targets: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run one of the two optimization graphs.

        Inference is intentionally exposed through :meth:`inference`; this
        training forward accepts only ``generation_pretraining`` and
        ``joint_finetuning`` so unsupported single-module modes cannot be
        enabled accidentally.
        """
        phase = self._config.training_phase
        if phase not in {"generation_pretraining", "joint_finetuning"}:
            raise RuntimeError(
                "ManeuverFlowModel.forward is a training API. "
                "Use model.inference(...) when training_phase='inference'."
            )

        if targets is None:
            raise KeyError(
                "ManeuverFlow training requires targets; call "
                "agent.forward(features, targets)."
            )
        perception_output, context = self._forward_perception(features)
        output = perception_output
        maneuver_rank_output = self._compute_maneuver_mgmp_rank_targets(
            targets,
            device=perception_output["maneuver_component_logits"].device,
            dtype=perception_output["maneuver_component_logits"].dtype,
        )
        if maneuver_rank_output is not None:
            perception_output.update(maneuver_rank_output)
            self._add_maneuver_rank_prediction_monitors(
                perception_output
            )

        output.update(
            self._forward_m3f_training(
                features,
                context,
                perception_output["maneuver_component_logits"],
                targets=targets,
            )
        )

        if phase == "joint_finetuning":
            self._validate_joint_training_inputs()

            verifier_output = self._forward_verifier_training(
                features,
                context,
                perception_output.get("maneuver_component_logits"),
                targets=targets,
            )
            verifier_reuse = {
                "candidate_batch": verifier_output.pop(
                    "__verifier_candidate_batch_for_ecfr__",
                    None,
                ),
                "ecfr_reuse": verifier_output.pop(
                    "__verifier_evidence_reuse_for_ecfr__",
                    None,
                ),
            }
            if any(value is None for value in verifier_reuse.values()):
                raise RuntimeError(
                    "The online candidate set did not produce the shared "
                    "Metric-Prism/ECFR training batch."
                )
            output.update(verifier_output)

            output.update(
                self._forward_joint_finetuning(
                    features,
                    verifier_reuse=verifier_reuse,
                )
            )

        return output

    # ------------------------------------------------------------------
    # Hypothesize: perception and maneuver context.
    # ------------------------------------------------------------------
    def _forward_perception(self, features):
        """Encode observations, planning state, agents, and maneuver logits."""
        camera_feature = features["camera_feature"]
        lidar_feature = features.get("lidar_feature", None)
        if not torch.is_tensor(lidar_feature):
            raise KeyError(
                "features['lidar_feature'] is required by ManeuverFlow-TF."
            )
        status_feature = features["status_feature"]
        batch_size = status_feature.shape[0]

        # The high-resolution branch feeds semantics; the low-resolution
        # branch feeds planning-context attention.
        bev_feature_upscale, bev_feature, _ = self.hypothesizer.perception.transfuser_backbone(camera_feature, lidar_feature)

        # Flatten BEV features for Transformer: [B, H*W, C]
        bev_feature_world = self.hypothesizer.perception.bev_projection(bev_feature)
        bev_feature_flat = bev_feature_world.flatten(-2, -1).permute(0, 2, 1)

        # Encode Ego Status (Velocity, Acceleration, Command): [B, D]
        status_encoding = self.hypothesizer.perception.state_navigation_encoder(status_feature)

        # Assemble BEV and status key/value tokens.
        keyval = torch.concatenate([bev_feature_flat, status_encoding[:, None]], dim=1)
        keyval = keyval + self.hypothesizer.perception.bev_memory_embedding.weight[None, ...]

        # One maneuver-context query plus the fixed agent queries.
        agent_queries = self.hypothesizer.perception.query_embedding.weight[1:][None, ...].repeat(batch_size, 1, 1) # [B, num_agents, D]
        nav_command = status_feature[:, :4]
        nav_command_for_maneuver = nav_command
        nav_mask_prob = float(
            self._config.maneuver_navigation_mask_probability
        )
        if self.training and self._config.training_phase != "inference" and nav_mask_prob > 0.0:
            keep_mask = (
                torch.rand(batch_size, 1, device=nav_command.device)
                >= nav_mask_prob
            ).to(dtype=nav_command.dtype)
            nav_command_for_maneuver = nav_command * keep_mask

        maneuver_query = self.hypothesizer.maneuver_decoder.forward_query(
            nav_command_for_maneuver
        ).unsqueeze(1)
        query = torch.cat([maneuver_query, agent_queries], dim=1)

        # Transformer Decoder: Query interacts with BEV context
        query_out = self.hypothesizer.perception.query_decoder(query, keyval)

        # BEV semantics, maneuver mixture weights, and agent detections.
        output = {}
        if bev_feature_upscale is None:
            raise ValueError(
                "The TransFuser backbone did not produce the required "
                "semantic BEV features."
            )
        # Front-half metric BEV aligns with the 128x256 semantic target.
        semantic_feat = (
            bev_feature_upscale[:, :, :32, :]
            if self._config.bev_coordinate_convention == "legacy_descending_x"
            else bev_feature_upscale[:, :, 32:, :]
        )
        output["bev_semantic_map"] = self.hypothesizer.perception.bev_semantic_head(semantic_feat)

        maneuver_query_out, agents_query_out = query_out.split(
            self._query_splits,
            dim=1,
        )
        kinematic_state = status_feature[:, 4:8]
        maneuver_component_logits = (
            self.hypothesizer.maneuver_decoder.forward_classify(
                maneuver_query_out.squeeze(1),
                kinematic_state,
            )
        )
        output["maneuver_component_logits"] = maneuver_component_logits

        # Agent Detection Outputs
        output.update(self.hypothesizer.perception.agent_prediction_head(agents_query_out))

        # Shared planning context C_n used by M³F, Metric Prism, and ECFR.
        context = {
            "bev_feat": bev_feature_flat,    # [B, 64, 256]
            "status_feat": status_encoding,  # [B, 256]
            "agent_feat": agents_query_out,  # [B, 30, 256] Agent Queries
            "bev_high": bev_feature_upscale, # [B, 64, 64, 64] High-res map prior
            "bev_low": bev_feature,  # [B, 512, 8, 8] low-resolution prior
            "bev_world": bev_feature_world,  # [B, 256, 8, 8] M³F world prior
        }
        return output, context

    def _encode_future_bev_feature_targets(
        self,
        targets: Optional[Dict[str, torch.Tensor]],
        context: Dict[str, torch.Tensor],
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Build EMA stop-gradient targets for Future-BEV self-supervision.

        The target builder provides future raw camera/lidar tensors. The frozen
        EMA perception backbone encodes them before
        comparing the world rollout's absolute/residual BEV feature grids.
        """
        if targets is None:
            return None
        device = context["status_feat"].device

        def _future_mask(batch_size: int, time_steps: int) -> Tuple[torch.Tensor, torch.Tensor]:
            valid_mask = targets.get("future_bev_feature_valid_mask", None)
            if torch.is_tensor(valid_mask):
                valid_mask = valid_mask.to(device=device, dtype=torch.bool)
                if valid_mask.dim() == 1:
                    valid_mask = valid_mask.unsqueeze(0)
                valid_mask = valid_mask[:, :time_steps]
            else:
                valid_mask = torch.ones(batch_size, time_steps, device=device, dtype=torch.bool)
            res_valid_mask = valid_mask.clone()
            if time_steps > 1:
                res_valid_mask[:, 1:] = valid_mask[:, 1:] & valid_mask[:, :-1]
            return valid_mask, res_valid_mask

        future_camera = targets.get("future_camera_feature", None)
        future_lidar = targets.get("future_lidar_feature", None)
        has_future_target = torch.is_tensor(future_camera) and torch.is_tensor(future_lidar)
        if not has_future_target:
            if self._config.training_phase == "joint_finetuning":
                available = ", ".join(sorted(str(key) for key in targets.keys()))
                raise KeyError(
                    "[ManeuverFlow Future BEV] Missing online target "
                    "observations. Expected future_camera_feature plus "
                    "future_lidar_feature from SceneLoader for TF; "
                    f"available target keys: {available}"
                )
            return None

        future_camera = future_camera.to(device=device)
        future_lidar = future_lidar.to(device=device)
        if future_camera.dim() == 4:
            future_camera = future_camera.unsqueeze(0)
        if future_lidar.dim() == 4:
            future_lidar = future_lidar.unsqueeze(0)
        if future_camera.dim() != 5 or future_lidar.dim() != 5:
            raise ValueError(
                "future_camera_feature/future_lidar_feature must be [B, T, ...], got "
                f"{tuple(future_camera.shape)} and "
                f"{tuple(future_lidar.shape)}"
            )

        batch_size, time_steps = future_camera.shape[:2]
        flat_camera = future_camera.reshape(batch_size * time_steps, *future_camera.shape[2:]).contiguous()
        flat_lidar = future_lidar.reshape(
            batch_size * time_steps,
            *future_lidar.shape[2:],
        ).contiguous()

        target_backbone = self.ema_target_encoder
        if target_backbone is None:
            raise RuntimeError("Future BEV targets require the joint-training EMA encoder.")
        with torch.no_grad(), _temporary_evaluation_mode(target_backbone):
            _, future_bev_low, _ = target_backbone(
                flat_camera,
                flat_lidar,
            )
        if future_bev_low is None:
            return None

        future_bev_dimension = self.verifier.future_bev_model.feature_dimension
        if int(future_bev_low.shape[1]) != int(future_bev_dimension):
            if int(future_bev_low.shape[1]) == int(self.hypothesizer.perception.bev_projection.in_channels) and int(
                future_bev_dimension
            ) == int(self.hypothesizer.perception.bev_projection.out_channels):
                with torch.no_grad():
                    future_bev_low = self.hypothesizer.perception.bev_projection(future_bev_low)
            else:
                raise ValueError(
                    "Online future BEV target channel mismatch: "
                    f"target={future_bev_low.shape[1]} "
                    f"future_bev={future_bev_dimension}."
                )

        future_bev_low = future_bev_low.view(batch_size, time_steps, *future_bev_low.shape[1:]).detach()
        valid_mask, res_valid_mask = _future_mask(batch_size, time_steps)
        pose_offsets = _future_target_pose_offsets_from_targets(
            targets,
            batch_size=batch_size,
            time_steps=time_steps,
            device=device,
            dtype=torch.float32,
        )
        if pose_offsets is None:
            raise KeyError(
                "[ManeuverFlow Future BEV] Teacher warping requires "
                "gt_trajectory in targets to align online future BEV targets."
            )
        with torch.no_grad():
            future_bev_low, spatial_mask = _warp_future_bev_target_to_current_frame(
                future_bev_low,
                pose_offsets,
                self._config,
            )
        with torch.no_grad():
            feature_targets = self.verifier.encode_future_bev_targets(
                future_bev_high=None,
                future_bev_low=future_bev_low,
                current_bev_high=None,
                current_bev_low=(
                    self.verifier.future_bev_model.select_current_bev(context).detach()
                ),
            )
        feature_targets["future_bev_feature_valid_mask"] = valid_mask
        feature_targets["future_bev_residual_feature_valid_mask"] = res_valid_mask
        _add_future_spatial_masks_to_feature_targets(
            feature_targets,
            spatial_mask,
            future_bev_low,
        )
        return feature_targets

    # ------------------------------------------------------------------
    # Maneuver-component routing.
    # ------------------------------------------------------------------
    def _build_maneuver_route(
        self,
        maneuver_logits: torch.Tensor,
        num_anchors: int,
        proposal_selected_families: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Build maneuver-mixture routing weights.
        - proposal-aware path: use one-hot family ids per proposal
        - fallback path: use the maneuver classifier softmax for all proposals
        """
        if proposal_selected_families is not None:
            if proposal_selected_families.shape[1] != num_anchors:
                raise ValueError(
                    "proposal_selected_families and num_anchors mismatch: "
                    f"{proposal_selected_families.shape[1]} vs {num_anchors}"
                )
            return F.one_hot(
                proposal_selected_families.long(),
                num_classes=int(self._config.num_maneuver_components),
            ).to(device=maneuver_logits.device, dtype=maneuver_logits.dtype)

        return F.softmax(maneuver_logits, dim=-1).unsqueeze(1).expand(-1, num_anchors, -1)

    @staticmethod
    def _wrap_angle(angle: torch.Tensor) -> torch.Tensor:
        return torch.atan2(torch.sin(angle), torch.cos(angle))

    @staticmethod
    def _mgmp_iid_random_codes(
        batch_size: int,
        num_families: int,
        count: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict[str, torch.Tensor]:
        xy = torch.randn(
            (batch_size, num_families, count, seq_len, 2),
            device=device,
            dtype=dtype,
        )
        heading = torch.randn(
            (batch_size, num_families, count, seq_len),
            device=device,
            dtype=dtype,
        )
        return {
            "xy": xy,
            "heading": heading,
        }

    def _has_mgmp_prior(self) -> bool:
        prior = self.hypothesizer.maneuver_gmp
        return all(
            value is not None
            for value in (
                prior.displacement_mean,
                prior.displacement_std,
                prior.heading_mean,
                prior.heading_std,
                prior.component_counts,
            )
        )

    def _sample_mgmp(
        self,
        selected_families: torch.Tensor,
        proposals_per_family: int,
        seq_len: int = 8,
        dtype: Optional[torch.dtype] = None,
    ) -> Optional[torch.Tensor]:
        """Sample maneuver-conditioned M-GMP states [B, F*K, L, 4]."""
        if not self._has_mgmp_prior():
            return None

        device = selected_families.device
        B, F = selected_families.shape
        K = max(1, int(proposals_per_family))
        L = max(1, int(seq_len))
        if L != int(self.hypothesizer.maneuver_gmp.displacement_mean.shape[1]):
            raise ValueError(
                "M-GMP sequence-length mismatch: "
                f"requested {L}, prior has "
                f"{self.hypothesizer.maneuver_gmp.displacement_mean.shape[1]}"
            )

        out_dtype = dtype if dtype is not None else self.hypothesizer.maneuver_gmp.displacement_mean.dtype
        prior = self.hypothesizer.maneuver_gmp
        xy_mean = prior.displacement_mean[selected_families].to(
            device=device, dtype=out_dtype
        )
        xy_std = prior.displacement_std[selected_families].to(
            device=device, dtype=out_dtype
        )
        heading_mean = prior.heading_mean[selected_families].to(
            device=device, dtype=out_dtype
        )
        heading_std = prior.heading_std[selected_families].to(
            device=device, dtype=out_dtype
        )

        xy_mean = xy_mean.unsqueeze(2).expand(B, F, K, L, 2)
        xy_std = xy_std.unsqueeze(2).expand(B, F, K, L, 2)
        heading_mean = heading_mean.unsqueeze(2).expand(B, F, K, L)
        heading_std = heading_std.unsqueeze(2).expand(B, F, K, L)

        codes = self._mgmp_iid_random_codes(
            B,
            F,
            K,
            L,
            device,
            out_dtype,
        )
        sampled_xy = xy_mean + xy_std * codes["xy"]
        sampled_heading = heading_mean + heading_std * codes["heading"]

        samples = torch.empty((B, F, K, L, 4), device=device, dtype=out_dtype)
        samples[..., :2] = sampled_xy
        samples[..., 2] = torch.sin(sampled_heading)
        samples[..., 3] = torch.cos(sampled_heading)
        return samples.view(B, F * K, L, 4)

    def _get_mgmp_sequence_length(self) -> int:
        """Return the fixed waypoint count used by M-GMP."""

        return int(self._config.trajectory_num_poses)

    def _select_predicted_maneuver_components(
        self,
        maneuver_logits: torch.Tensor,
        topk: int,
    ) -> Optional[torch.Tensor]:
        """Select the top-k predicted maneuver components without GT forcing."""
        num_classes = int(self._config.num_maneuver_components)
        topk = max(1, min(int(topk), num_classes))
        return torch.topk(maneuver_logits, k=topk, dim=-1).indices


    def _build_probability_mgmp_batch(
        self,
        maneuver_logits: torch.Tensor,
        *,
        total_count: int,
        topk: int,
        min_per_family: int = 0,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Sample a probability-apportioned M-GMP pool from predicted maneuvers."""

        if not self._has_mgmp_prior():
            return None
        total_count = int(total_count)
        if total_count <= 0:
            raise ValueError("M-GMP probability pool size must be positive.")

        selected_maneuvers = self._select_predicted_maneuver_components(
            maneuver_logits,
            topk=int(topk),
        )
        if selected_maneuvers is None:
            return None

        selected_logits = torch.gather(
            maneuver_logits,
            1,
            selected_maneuvers,
        )
        selected_probabilities = torch.softmax(
            selected_logits.float(),
            dim=1,
        )
        maneuver_counts = _allocate_probability_proportional_counts(
            selected_probabilities,
            total_count=total_count,
            min_per_family=min_per_family,
        )
        proposal_maneuvers = _expand_selected_families_by_counts(
            selected_maneuvers,
            maneuver_counts,
        )
        proposal_state = self._sample_mgmp(
            selected_families=proposal_maneuvers,
            proposals_per_family=1,
            seq_len=self._get_mgmp_sequence_length(),
            dtype=maneuver_logits.dtype,
        )
        if proposal_state is None:
            return None

        return {
            "selected_families": selected_maneuvers,
            "selected_family_probabilities": selected_probabilities,
            "proposal_selected_families": proposal_maneuvers,
            "proposal_family_counts": maneuver_counts,
            "proposal_component_indices": None,
            "proposal_e": proposal_state,
        }

    def _build_inference_mgmp_batch(
        self,
        maneuver_logits: torch.Tensor,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Sample the fixed N=64 M-GMP proposal pool from predicted maneuvers."""

        return self._build_probability_mgmp_batch(
            maneuver_logits,
            total_count=MANEUVERFLOW_INFERENCE_POLICY.num_base_hypotheses,
            topk=MANEUVERFLOW_INFERENCE_POLICY.maneuver_top_k,
            min_per_family=(
                MANEUVERFLOW_INFERENCE_POLICY.minimum_per_selected_component
            ),
        )

    def _build_iid_probability_mgmp_batch(
        self,
        maneuver_logits: torch.Tensor,
        *,
        total_count: int,
        topk: int,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Draw i.i.d. M-GMP routes from Top-k-renormalized probabilities."""

        if not self._has_mgmp_prior():
            return None
        total_count = int(total_count)
        if total_count <= 0:
            raise ValueError("M-GMP probability pool size must be positive.")

        selected_maneuvers = self._select_predicted_maneuver_components(
            maneuver_logits,
            topk=int(topk),
        )
        if selected_maneuvers is None:
            return None
        selected_logits = torch.gather(
            maneuver_logits,
            1,
            selected_maneuvers,
        )
        selected_probabilities = torch.softmax(
            selected_logits.float(),
            dim=1,
        )
        route_slots = torch.multinomial(
            selected_probabilities,
            num_samples=total_count,
            replacement=True,
        )
        proposal_maneuvers = torch.gather(
            selected_maneuvers,
            1,
            route_slots,
        )
        maneuver_counts = torch.zeros_like(
            selected_maneuvers,
            dtype=torch.long,
        ).scatter_add(
            1,
            route_slots,
            torch.ones_like(route_slots, dtype=torch.long),
        )
        proposal_state = self._sample_mgmp(
            selected_families=proposal_maneuvers,
            proposals_per_family=1,
            seq_len=self._get_mgmp_sequence_length(),
            dtype=maneuver_logits.dtype,
        )
        if proposal_state is None:
            return None
        return {
            "selected_families": selected_maneuvers,
            "selected_family_probabilities": selected_probabilities,
            "proposal_selected_families": proposal_maneuvers,
            "proposal_family_counts": maneuver_counts,
            "proposal_component_indices": route_slots,
            "proposal_e": proposal_state,
        }

    @torch.no_grad()
    def _generate_online_probability_batch(
        self,
        context: Dict[str, torch.Tensor],
        maneuver_logits: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Sample 64 current-model Top-3 IID routes and one-step M³F solves.

        Only the sampling pass uses eval mode and detached context. The caller
        retains its original live context for the verifier/evidence graph.
        """

        total_count = int(self._config.joint_probability_source_count)
        if total_count != 64:
            raise ValueError("Online joint training requires exactly 64 candidates.")
        generator = self.hypothesizer.meanflow_generator
        with _temporary_evaluation_mode(generator):
            detached_logits = maneuver_logits.detach()
            mgmp_batch = self._build_iid_probability_mgmp_batch(
                detached_logits,
                total_count=total_count,
                topk=int(self._config.inference_mgmp_top_k),
            )
            if mgmp_batch is None:
                raise RuntimeError("Online candidate generation requires loaded M-GMP priors.")
            proposal_state = mgmp_batch["proposal_e"]
            proposal_maneuvers = mgmp_batch["proposal_selected_families"]
            maneuver_weights = self._build_maneuver_route(
                detached_logits,
                num_anchors=total_count,
                proposal_selected_families=proposal_maneuvers,
            )
            batch_size = int(detached_logits.shape[0])
            start_time = torch.ones(batch_size, device=detached_logits.device)
            end_time = torch.zeros_like(start_time)
            velocity, _, _ = self._evaluate_m3f(
                z=proposal_state,
                r=end_time,
                t=start_time,
                c_state=context["status_feat"].detach().unsqueeze(1),
                c_bev=(
                    context["bev_feat"].detach()
                    + generator.bev_position_embedding.weight[None, ...]
                ),
                c_agent=context["agent_feat"].detach(),
                maneuver_component_weights=maneuver_weights,
            )
            flow_state = proposal_state - velocity
            hypotheses = self._flow_features_to_physical_trajectory(flow_state)
        if hypotheses.shape != (batch_size, total_count, 8, 3):
            raise RuntimeError(
                "Online candidate generation changed the fixed trajectory geometry: "
                f"{tuple(hypotheses.shape)}."
            )
        if not torch.isfinite(hypotheses).all():
            raise ValueError("Online candidate generation produced non-finite hypotheses.")
        return {
            "hypotheses": hypotheses.detach(),
            "hypotheses_flow_state": flow_state.detach(),
            "hypothesis_maneuver_component_indices": proposal_maneuvers.detach(),
            "selected_maneuver_component_indices": mgmp_batch["selected_families"].detach(),
            "selected_maneuver_component_probabilities": mgmp_batch["selected_family_probabilities"].detach(),
            "mgmp_hypothesis_counts": mgmp_batch["proposal_family_counts"].detach(),
        }

    def _build_online_candidate_batch(
        self,
        features: Dict[str, Any],
        context: Dict[str, torch.Tensor],
        maneuver_logits: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Evaluate a fresh candidate set against the scene's static cache."""

        tokens = features.get("token")
        if tokens is None:
            raise ValueError("Online candidate supervision requires features['token'].")
        candidate_batch = self._generate_online_probability_batch(context, maneuver_logits)
        if self._online_candidate_supervision is None:
            self._online_candidate_supervision = OnlineCandidateSupervision(
                self._config.metric_cache_path
            )
        candidate_batch.update(
            self._online_candidate_supervision.score_batch(tokens, candidate_batch["hypotheses"])
        )
        return self._detach_tensor_mapping(candidate_batch)


    def _trajectory_to_flow_features(
        self,
        trajectory: torch.Tensor,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Convert [x, y, heading] to the released displacement flow state."""
        out_dtype = dtype if dtype is not None else trajectory.dtype
        return trajectory_to_flow_state(trajectory, self._config).to(
            dtype=out_dtype
        )

    def _flow_features_to_physical_trajectory(self, flow_features: torch.Tensor) -> torch.Tensor:
        """Decode the released displacement flow state to [x, y, heading]."""
        return flow_state_to_trajectory(flow_features, self._config)

    def _resolve_generation_teacher_trajectory(
        self,
        targets: Optional[Dict[str, torch.Tensor]],
        *,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> Optional[torch.Tensor]:
        """Return the Stage-1 ground-truth generation teacher."""
        if targets is None:
            return None

        raw_gt = targets.get("gt_trajectory", None)
        if not torch.is_tensor(raw_gt):
            return None
        raw_gt = raw_gt.to(device=device, dtype=dtype)
        if raw_gt.dim() == 4 and raw_gt.shape[1] == 1:
            raw_gt = raw_gt[:, 0]
        elif raw_gt.dim() != 3:
            return None
        return raw_gt

    def _compute_maneuver_mgmp_rank_targets(
        self,
        targets: Optional[Dict[str, torch.Tensor]],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[Dict[str, torch.Tensor]]:
        if targets is None:
            return None

        prior = self.hypothesizer.maneuver_gmp
        xy_mean = prior.displacement_mean
        xy_std = prior.displacement_std
        heading_mean = prior.heading_mean
        heading_std = prior.heading_std

        if (
            xy_mean is None
            or xy_std is None
            or heading_mean is None
            or heading_std is None
            or xy_mean.ndim != 3
            or xy_std.ndim != 3
            or heading_mean.ndim != 2
            or heading_std.ndim != 2
        ):
            return None

        num_classes = int(xy_mean.shape[0])
        if num_classes < 2:
            return None

        gt_traj = self._resolve_generation_teacher_trajectory(
            targets,
            device=device,
            dtype=torch.float32,
        )
        if gt_traj is None:
            return None
        if gt_traj.shape[-1] < 3 or gt_traj.shape[1] <= 0:
            return None

        seq_len = min(int(gt_traj.shape[1]), int(xy_mean.shape[1]), int(heading_mean.shape[1]))
        if seq_len <= 0:
            return None

        gt_traj = gt_traj[:, :seq_len, :3]
        gt_flow = self._trajectory_to_flow_features(gt_traj, dtype=torch.float32)
        gt_xy = gt_flow[..., :2]
        gt_heading = gt_traj[..., 2]

        xy_mean = xy_mean[:num_classes, :seq_len].to(device=device, dtype=torch.float32)
        xy_std = xy_std[:num_classes, :seq_len].to(device=device, dtype=torch.float32)
        heading_mean = heading_mean[:num_classes, :seq_len].to(device=device, dtype=torch.float32)
        heading_std = heading_std[:num_classes, :seq_len].to(device=device, dtype=torch.float32)
        if (
            not bool(torch.isfinite(xy_std).all())
            or not bool(torch.isfinite(heading_std).all())
            or bool((xy_std <= 0.0).any())
            or bool((heading_std <= 0.0).any())
        ):
            raise ValueError(
                "M-GMP soft association requires finite, strictly positive "
                "fitted component standard deviations."
            )

        xy_z = (gt_xy[:, None] - xy_mean[None]) / xy_std[None]
        heading_z = self._wrap_angle(gt_heading[:, None] - heading_mean[None]) / heading_std[None]
        distance = xy_z.pow(2).sum(dim=-1).mean(dim=-1) + heading_z.pow(2).mean(dim=-1)

        rank_target = F.softmax(-distance, dim=-1)
        rank_topk = max(
            1,
            min(int(self._config.inference_mgmp_top_k), num_classes),
        )
        top_values, top_indices = torch.topk(
            distance,
            k=rank_topk,
            dim=-1,
            largest=False,
        )
        top_probs = rank_target.gather(1, top_indices)

        out: Dict[str, torch.Tensor] = {
            "maneuver_rank_target": rank_target.to(dtype=dtype).detach(),
            "mgmp_component_distance": distance.detach(),
            "mgmp_topk_component_indices": top_indices.detach(),
            "mgmp_top1_component_index": top_indices[:, 0].detach(),
            "maneuver_rank_q1_monitor": top_probs[:, 0].mean().detach(),
            "maneuver_rank_q2_monitor": (
                top_probs[:, 1].mean().detach() if rank_topk > 1 else top_probs.new_zeros(())
            ),
            "maneuver_rank_q3_monitor": (
                top_probs[:, 2].mean().detach() if rank_topk > 2 else top_probs.new_zeros(())
            ),
        }
        entropy = -(
            rank_target.clamp_min(1e-8)
            * rank_target.clamp_min(1e-8).log()
        ).sum(dim=-1)
        out["maneuver_rank_effective_k_monitor"] = entropy.exp().mean().detach()
        if rank_topk > 1:
            out["maneuver_rank_gap12_monitor"] = (top_values[:, 1] - top_values[:, 0]).mean().detach()
        return out

    @staticmethod
    def _add_maneuver_rank_prediction_monitors(
        predictions: Dict[str, torch.Tensor],
    ) -> None:
        maneuver_logits = predictions.get("maneuver_component_logits", None)
        target_topk = predictions.get("mgmp_topk_component_indices", None)
        if (
            not torch.is_tensor(maneuver_logits)
            or not torch.is_tensor(target_topk)
            or target_topk.ndim != 2
        ):
            return

        topk = min(
            int(target_topk.shape[1]),
            int(maneuver_logits.shape[-1]),
        )
        if topk <= 0:
            return
        pred_topk = torch.topk(maneuver_logits, k=topk, dim=-1).indices
        target_topk = target_topk[:, :topk].to(
            device=maneuver_logits.device,
            dtype=torch.long,
        )
        target_top1 = target_topk[:, 0]

        predictions["maneuver_rank_top1_acc_monitor"] = pred_topk[:, 0].eq(target_top1).float().mean().detach()
        predictions["maneuver_rank_top3_recall_monitor"] = (
            pred_topk.eq(target_top1[:, None]).any(dim=1).float().mean().detach()
        )
        intersection = pred_topk[:, :, None].eq(target_topk[:, None, :]).any(dim=2).float().sum(dim=1)
        union = (2.0 * float(topk) - intersection).clamp_min(1.0)
        predictions["maneuver_rank_top3_jaccard_monitor"] = (intersection / union).mean().detach()

    def _resolve_generation_mgmp_component_indices(
        self,
        targets: Optional[Dict[str, torch.Tensor]],
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """Resolve the GT-matched M-GMP component used for generation."""
        rank_output = self._compute_maneuver_mgmp_rank_targets(
            targets,
            device=device,
            dtype=torch.float32,
        )
        if rank_output is not None:
            component_index = rank_output.get(
                "mgmp_top1_component_index",
                None,
            )
            if torch.is_tensor(component_index):
                return component_index.to(
                    device=device,
                    dtype=torch.long,
                ).reshape(-1)
        return None

    def _build_generation_mgmp_supervision(
        self,
        maneuver_logits: torch.Tensor,
        targets: Optional[Dict[str, torch.Tensor]],
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Build the released GT-only M-GMP component supervision."""
        if targets is None or "gt_trajectory" not in targets:
            return None

        device = maneuver_logits.device
        selected_components = (
            self._resolve_generation_mgmp_component_indices(targets, device)
        )
        gt_trajectory = self._resolve_generation_teacher_trajectory(
            targets,
            device=device,
            dtype=torch.float32,
        )
        if (
            selected_components is None
            or gt_trajectory is None
        ):
            return None

        return {
            "teacher_traj": gt_trajectory[:, None],
            "selected_families": selected_components.view(-1, 1),
        }

    @staticmethod
    def _repeat_maneuver_tensor(
        tensor: Optional[torch.Tensor],
        proposal_count: int,
    ) -> Optional[torch.Tensor]:
        if tensor is None or proposal_count <= 1:
            return tensor
        return tensor.repeat_interleave(
            proposal_count,
            dim=1,
        )

    def _build_maneuver_mgmp_training_batch(
        self,
        maneuver_logits: torch.Tensor,
        targets: Optional[Dict[str, torch.Tensor]],
        proposals_per_family: int,
        dtype: torch.dtype,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Sample M-GMP hypotheses for the GT-matched maneuver component."""
        supervision = self._build_generation_mgmp_supervision(
            maneuver_logits,
            targets,
        )
        if supervision is None:
            return None

        proposals_per_family = max(1, int(proposals_per_family))
        selected_families = supervision["selected_families"]
        hypotheses = self._sample_mgmp(
            selected_families=selected_families,
            proposals_per_family=proposals_per_family,
            seq_len=int(supervision["teacher_traj"].shape[2]),
            dtype=dtype,
        )
        if hypotheses is None:
            raise ValueError(
                "Maneuver-conditioned M-GMP sampling failed."
            )

        repeat = self._repeat_maneuver_tensor
        return {
            "teacher_traj": repeat(
                supervision["teacher_traj"],
                proposals_per_family,
            ),
            "proposal_selected_families": repeat(
                selected_families,
                proposals_per_family,
            ),
            "proposal_e": hypotheses,
        }

    @staticmethod
    def _clamp_ratio(value: float) -> float:
        return max(0.0, min(float(value), 1.0))

    def _get_m3f_time_mixture(self) -> Tuple[float, float, float]:
        """Return the fixed M³F full/diagonal/general interval mixture."""
        full_ratio = self._clamp_ratio(
            self._config.m3f_full_interval_ratio
        )
        remainder = 1.0 - full_ratio
        general_ratio = remainder * self._clamp_ratio(
            self._config.m3f_unequal_interval_ratio
        )
        diagonal_ratio = remainder - general_ratio
        return full_ratio, diagonal_ratio, general_ratio

    def _sample_m3f_time_intervals(
        self,
        batch_size: int,
        device: torch.device,
        full_step_ratio: Optional[float] = None,
        diag_ratio: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample (r, t) with r <= t for M³F training.

        The sampler mixes three regimes:
        1. Diagonal regime: r=t, used to keep the model calibrated near the identity path.
        2. Full-step regime: (r, t)=(0, 1), explicitly matching one-step inference.
        3. General regime: random interval samples covering the rest of the M³F path.
        """
        sampler = self._config.m3f_time_sampler
        mu = self._config.m3f_time_mu
        sigma = self._config.m3f_time_sigma
        ratio_r_not_equal_t = self._config.m3f_unequal_interval_ratio
        if full_step_ratio is None:
            full_step_ratio = self._config.m3f_full_interval_ratio
        full_step_ratio = self._clamp_ratio(full_step_ratio)
        if diag_ratio is None:
            diag_ratio = (1.0 - full_step_ratio) * (
                1.0 - self._clamp_ratio(ratio_r_not_equal_t)
            )
        else:
            diag_ratio = self._clamp_ratio(diag_ratio)
            diag_ratio = min(diag_ratio, 1.0 - full_step_ratio)

        if sampler == "uniform":
            time_samples = torch.rand(batch_size, 2, device=device)
        else:
            normal_samples = torch.randn(batch_size, 2, device=device) * sigma + mu
            time_samples = torch.sigmoid(normal_samples)

        sorted_samples, _ = torch.sort(time_samples, dim=1)
        r, t = sorted_samples[:, 0], sorted_samples[:, 1]

        eps = 1e-6
        t = t.clamp(eps, 1.0 - eps)
        r = r.clamp(0.0, 1.0 - eps)
        r = torch.minimum(r, t)

        draw = torch.rand(batch_size, device=device)
        full_mask = draw < full_step_ratio
        diag_mask = (draw >= full_step_ratio) & (draw < (full_step_ratio + diag_ratio))

        # A diagonal time is one draw from the configured distribution. Using
        # the sorted maximum of two draws biases diagonal supervision toward 1.
        diagonal_time = time_samples[:, 0].clamp(eps, 1.0 - eps)
        r = torch.where(diag_mask, diagonal_time, r)
        t = torch.where(diag_mask, diagonal_time, t)

        # Full-step samples mirror the exact inference regime instead of relying on extrapolation.
        r = torch.where(full_mask, torch.zeros_like(r), r)
        t = torch.where(full_mask, torch.ones_like(t), t)
        return r, t

    def _encode_m3f_interval(
        self, r: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode the M³F interval to DiT time embedding.
        """
        delta = (t - r).clamp(min=0.0)
        t_emb = self.hypothesizer.meanflow_generator.start_time_embedding(t)
        delta_emb = self.hypothesizer.meanflow_generator.interval_duration_embedding(delta)
        time_emb = torch.cat([t_emb, delta_emb], dim=-1)
        return self.hypothesizer.meanflow_generator.interval_encoder(time_emb)

    def _evaluate_m3f(
        self,
        z: torch.Tensor,
        r: torch.Tensor,
        t: torch.Tensor,
        c_state: torch.Tensor,
        c_bev: torch.Tensor,
        c_agent: torch.Tensor,
        maneuver_component_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, None]:
        """Evaluate M³F and decode the one-step M³F velocity."""
        t_emb = self._encode_m3f_interval(r, t)
        tokens_in = self.hypothesizer.meanflow_generator.trajectory_tokenizer(z)
        tokens_out = self.hypothesizer.meanflow_generator.field(
            x=tokens_in,
            t_emb=t_emb,
            c_state=c_state,
            c_bev=c_bev,
            c_agent=c_agent,
            maneuver_component_weights=maneuver_component_weights,
        )

        u = self.hypothesizer.meanflow_generator.field_decoder(tokens_out)
        return u, tokens_out, None

    @staticmethod
    def _disabled_autocast_context(device: torch.device):
        if device.type in {"cuda", "cpu"}:
            return torch.autocast(device_type=device.type, enabled=False)
        return nullcontext()

    def _run_m3f_jvp(
        self,
        z_t: torch.Tensor,
        tangent_v: torch.Tensor,
        r: torch.Tensor,
        t: torch.Tensor,
        c_state: torch.Tensor,
        c_bev: torch.Tensor,
        c_agent: torch.Tensor,
        maneuver_component_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run the M³F JVP path in FP32 for numerical stability.
        """
        z_run = z_t.float()
        tangent_v_run = tangent_v.float()
        r_run = r.float()
        t_run = t.float()
        c_state_run = c_state.float()
        c_bev_run = c_bev.float()
        c_agent_run = c_agent.float()
        maneuver_component_weights_run = maneuver_component_weights.float()

        drdt = torch.zeros_like(r_run)
        dtdt = torch.ones_like(t_run)

        def fn_current(z_cur: torch.Tensor, r_cur: torch.Tensor, t_cur: torch.Tensor) -> torch.Tensor:
            u_cur, _, _ = self._evaluate_m3f(
                z=z_cur,
                r=r_cur,
                t=t_cur,
                c_state=c_state_run,
                c_bev=c_bev_run,
                c_agent=c_agent_run,
                maneuver_component_weights=maneuver_component_weights_run,
            )
            return u_cur

        attention_ctx = nullcontext()
        if sdpa_kernel is not None and SDPBackend is not None:
            attention_ctx = sdpa_kernel([SDPBackend.MATH])

        autocast_ctx = self._disabled_autocast_context(z_run.device)
        with autocast_ctx:
            with attention_ctx:
                u_run, dudt_run = torch.autograd.functional.jvp(
                    fn_current,
                    (z_run, r_run, t_run),
                    (tangent_v_run, drdt, dtdt),
                    create_graph=self.training,
                )

        return u_run, dudt_run


    def _future_bev_feature_supervision_enabled(self) -> bool:
        return self._config.training_phase == "joint_finetuning"


    def _build_trajectory_evidence_memory(
        self,
        trajectory_hypotheses: torch.Tensor,
        state_navigation_feature: torch.Tensor,
        evidence_field: Dict[str, torch.Tensor],
        maneuver_component_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        return self.verifier.trajectory_evidence_probe(
            trajectory_hypotheses,
            state_navigation_feature,
            evidence_field,
            maneuver_component_weights=maneuver_component_weights,
        )

    def _run_ecfr(
        self,
        base_hypotheses: torch.Tensor,
        base_hypotheses_flow_state: torch.Tensor,
        evidence_memory_outputs: Dict[str, torch.Tensor],
        residual_scale: float = 1.0,
    ) -> Dict[str, Any]:
        return self.improver.ecfr(
            base_hypotheses_flow_state=base_hypotheses_flow_state,
            evidence_memory_outputs=evidence_memory_outputs,
            base_hypotheses=base_hypotheses,
            residual_scale=residual_scale,
        )

    @staticmethod
    def _gather_hypotheses(tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        index_shape = [indices.shape[0], indices.shape[1]] + [1] * (tensor.dim() - 2)
        gather_index = indices.view(*index_shape).expand(-1, -1, *tensor.shape[2:])
        return torch.gather(tensor, dim=1, index=gather_index)

    @staticmethod
    def _merge_inference_candidate_sets(
        *,
        base_trajs: torch.Tensor,
        base_scores: torch.Tensor,
        refined_trajs: torch.Tensor,
        refined_scores: torch.Tensor,
        base_raw_scores: torch.Tensor,
        refined_raw_scores: torch.Tensor,
        base_source_indices: torch.Tensor,
        refined_source_indices: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if base_source_indices.shape != base_scores.shape:
            raise ValueError("base_source_indices must match base_scores [B, N].")
        if refined_source_indices.shape != refined_scores.shape:
            raise ValueError("refined_source_indices must match refined_scores [B, R].")
        return {
            "final_hypotheses": torch.cat([base_trajs, refined_trajs], dim=1),
            "final_hypothesis_scores": torch.cat([base_scores, refined_scores], dim=1),
            "final_hypothesis_scores_raw": torch.cat([base_raw_scores, refined_raw_scores], dim=1),
            "final_hypothesis_is_refined": torch.cat([
                torch.zeros_like(base_scores, dtype=torch.bool),
                torch.ones_like(refined_scores, dtype=torch.bool),
            ], dim=1),
            "final_hypothesis_source_base_indices": torch.cat([
                base_source_indices, refined_source_indices,
            ], dim=1),
        }


    @staticmethod
    def _proposal_invalid_mask(tensor: torch.Tensor) -> torch.Tensor:
        if tensor is None:
            raise ValueError("tensor must not be None")
        if tensor.dim() < 2:
            raise ValueError(f"Expected proposal tensor with dim >= 2, got {tuple(tensor.shape)}")
        return ~torch.isfinite(tensor).reshape(tensor.shape[0], tensor.shape[1], -1).all(dim=-1)

    @staticmethod
    def _invalid_monitor(mask: torch.Tensor) -> torch.Tensor:
        return mask.float().sum().detach()

    @staticmethod
    def _proposal_mask_from_eval_tensor(
        tensor: Optional[torch.Tensor],
        num_proposals: int,
    ) -> Optional[torch.Tensor]:
        if tensor is None or not torch.is_tensor(tensor) or tensor.dim() < 2:
            return None

        if tensor.shape[1] == num_proposals:
            finite = torch.isfinite(tensor).reshape(tensor.shape[0], num_proposals, -1).all(dim=-1)
            return ~finite

        if tensor.shape[-1] == num_proposals:
            finite = torch.isfinite(tensor).movedim(-1, 1).reshape(tensor.shape[0], num_proposals, -1).all(dim=-1)
            return ~finite

        return None

    @staticmethod
    def _apply_proposal_mask_fill(
        tensor: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
        fill_value: float = 0.0,
        clamp: Optional[Tuple[float, float]] = None,
    ) -> Optional[torch.Tensor]:
        if tensor is None or not torch.is_tensor(tensor):
            return tensor

        safe = torch.nan_to_num(tensor, nan=fill_value, posinf=fill_value, neginf=fill_value)
        if clamp is not None:
            safe = safe.clamp(min=clamp[0], max=clamp[1])

        if mask is None or safe.dim() < 2:
            return safe

        num_proposals = mask.shape[1]
        if safe.shape[1] == num_proposals:
            expand_mask = mask.reshape(mask.shape[0], num_proposals, *([1] * (safe.dim() - 2)))
        elif safe.shape[-1] == num_proposals:
            expand_mask = mask.reshape(mask.shape[0], *([1] * (safe.dim() - 2)), num_proposals)
        else:
            return safe

        fill_tensor = torch.full_like(safe, fill_value)
        return torch.where(expand_mask, fill_tensor, safe)


    def _guard_eval_outputs(
        self,
        eval_outputs: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        pdm_factor_scores = eval_outputs.get("pdm_factor_scores", None)
        if (
            pdm_factor_scores is None
            or not torch.is_tensor(pdm_factor_scores)
            or pdm_factor_scores.dim() < 3
        ):
            return eval_outputs, next(value for value in eval_outputs.values() if torch.is_tensor(value)).new_zeros((0, 0), dtype=torch.bool)

        num_proposals = pdm_factor_scores.shape[-1]
        invalid_mask = torch.zeros(
            (pdm_factor_scores.shape[0], num_proposals),
            device=pdm_factor_scores.device,
            dtype=torch.bool,
        )

        for key in (
            "nc_logits",
            "binary_pdm_factor_logits",
            "ep_pred",
            "pdm_factor_scores",
            "alpha_time_shared",
        ):
            invalid_mask_cur = self._proposal_mask_from_eval_tensor(eval_outputs.get(key, None), num_proposals)
            if invalid_mask_cur is not None:
                invalid_mask |= invalid_mask_cur


        safe_outputs = dict(eval_outputs)
        safe_outputs["nc_logits"] = self._apply_proposal_mask_fill(
            safe_outputs.get("nc_logits", None),
            invalid_mask,
            fill_value=0.0,
        )
        safe_outputs["binary_pdm_factor_logits"] = (
            self._apply_proposal_mask_fill(
                safe_outputs.get("binary_pdm_factor_logits", None),
                invalid_mask,
                fill_value=0.0,
            )
        )
        safe_outputs["ep_pred"] = self._apply_proposal_mask_fill(
            safe_outputs.get("ep_pred", None),
            invalid_mask,
            fill_value=0.0,
            clamp=(0.0, 1.0),
        )
        safe_outputs["alpha_time_shared"] = self._apply_proposal_mask_fill(
            safe_outputs.get("alpha_time_shared", None),
            invalid_mask,
            fill_value=0.0,
        )

        if (
            safe_outputs.get("nc_logits", None) is not None
            and safe_outputs.get("binary_pdm_factor_logits", None) is not None
            and safe_outputs.get("ep_pred", None) is not None
        ):
            binary_pdm_factor_scores = torch.sigmoid(
                safe_outputs["binary_pdm_factor_logits"]
            )
            nc_score = _nc_score_from_logits(safe_outputs["nc_logits"], expected_score=True)
            recomputed_factors = torch.stack(
                [
                    nc_score,
                    binary_pdm_factor_scores[:, 0, :],
                    safe_outputs["ep_pred"],
                    binary_pdm_factor_scores[:, 1, :],
                    binary_pdm_factor_scores[:, 2, :],
                ],
                dim=1,
            )
            safe_outputs["pdm_factor_scores"] = self._apply_proposal_mask_fill(
                recomputed_factors,
                invalid_mask,
                fill_value=0.0,
                clamp=(0.0, 1.0),
            )
        else:
            safe_outputs["pdm_factor_scores"] = self._apply_proposal_mask_fill(
                safe_outputs.get("pdm_factor_scores", None),
                invalid_mask,
                fill_value=0.0,
                clamp=(0.0, 1.0),
            )

        if safe_outputs.get("alpha_time_shared", None) is not None:
            denom = safe_outputs["alpha_time_shared"].sum(dim=-1, keepdim=True).clamp_min(1e-6)
            safe_outputs["alpha_time_shared"] = safe_outputs["alpha_time_shared"] / denom

        return safe_outputs, invalid_mask


    # ------------------------------------------------------------------
    # M³F generation training.
    # ------------------------------------------------------------------
    def _forward_m3f_training(
        self,
        features,
        context,
        maneuver_logits,
        targets=None,
    ):
        """Train M3F with the boundary-tangent target."""
        output = {}
        batch_size = maneuver_logits.shape[0]
        device = maneuver_logits.device
        if not self._has_mgmp_prior():
            raise ValueError(
                "Maneuver-conditioned M-GMP supervision requires a valid "
                "M-GMP prior. Check config.mgmp_prior_dir."
            )
        training_proposal_count = max(
            1,
            int(self._config.m3f_training_proposal_count),
        )
        family_supervision = self._build_maneuver_mgmp_training_batch(
            maneuver_logits=maneuver_logits,
            targets=targets,
            proposals_per_family=training_proposal_count,
            dtype=maneuver_logits.dtype,
        )
        if family_supervision is None:
            raise ValueError(
                "GT-matched M-GMP supervision could not be constructed."
            )

        teacher_trajectory = family_supervision["teacher_traj"]
        proposal_maneuver_components = family_supervision[
            "proposal_selected_families"
        ]
        source_hypotheses = family_supervision["proposal_e"]
        num_anchors = source_hypotheses.shape[1]
        target_flow_state = trajectory_to_flow_state(
            teacher_trajectory,
            self._config,
        ).to(device=device, dtype=maneuver_logits.dtype)

        output["m3f_training_proposal_count_monitor"] = torch.tensor(
            float(training_proposal_count), device=device,
        )

        # Sample (r, t), r <= t
        full_step_ratio, diag_ratio, general_ratio = self._get_m3f_time_mixture()
        r, t = self._sample_m3f_time_intervals(
            batch_size,
            device,
            full_step_ratio=full_step_ratio,
            diag_ratio=diag_ratio,
        )
        r_expanded = r.view(batch_size, 1, 1, 1)
        t_expanded = t.view(batch_size, 1, 1, 1)

        z_t = (
            (1 - t_expanded) * target_flow_state
            + t_expanded * source_hypotheses
        )
        v_hat = source_hypotheses - target_flow_state

        # Inject planning context and maneuver routing.
        w_route = self._build_maneuver_route(
            maneuver_logits,
            num_anchors=num_anchors,
            proposal_selected_families=proposal_maneuver_components,
        )
        c_state = context["status_feat"].unsqueeze(1)
        c_agent = context["agent_feat"]
        c_bev = context["bev_feat"] + self.hypothesizer.meanflow_generator.bev_position_embedding.weight[None, ...]
        r_diag = t
        t_diag = t
        # The frozen boundary field v_theta(z_t, t, t) is the data tangent
        # for the JVP, while the optimization target remains v_hat = e - x.
        with torch.no_grad():
            v_theta_tangent, _, _ = self._evaluate_m3f(
                z=z_t,
                r=r_diag,
                t=t_diag,
                c_state=c_state,
                c_bev=c_bev,
                c_agent=c_agent,
                maneuver_component_weights=w_route,
            )
        u, dudt = self._run_m3f_jvp(
            z_t=z_t,
            tangent_v=v_theta_tangent.detach(),
            r=r,
            t=t,
            c_state=c_state,
            c_bev=c_bev,
            c_agent=c_agent,
            maneuver_component_weights=w_route,
        )

        interval = (t - r).view(batch_size, 1, 1, 1).to(dtype=u.dtype)
        dudt = dudt.detach()
        flow_v_pred = u + interval * dudt
        target_v = v_hat.to(dtype=u.dtype)

        output["m3f_velocity_prediction"] = flow_v_pred
        output["m3f_target_velocity"] = target_v
        output["m3f_boundary_tangent_l1_monitor"] = (
            v_theta_tangent.detach().to(dtype=target_v.dtype) - target_v
        ).float().abs().mean()

        output["m3f_velocity_invalid_hypotheses_monitor"] = self._invalid_monitor(
            self._proposal_invalid_mask(flow_v_pred)
        )
        output["m3f_target_velocity_invalid_hypotheses_monitor"] = self._invalid_monitor(
            self._proposal_invalid_mask(target_v)
        )
        output["m3f_time_interval_monitor"] = interval.detach().mean()
        output["m3f_full_step_ratio_monitor"] = torch.tensor(
            full_step_ratio,
            device=device,
            dtype=u.dtype,
        )
        output["m3f_diagonal_time_ratio_monitor"] = torch.tensor(
            diag_ratio,
            device=device,
            dtype=u.dtype,
        )
        output["m3f_general_time_ratio_monitor"] = torch.tensor(
            general_ratio,
            device=device,
            dtype=u.dtype,
        )
        return output

    # ------------------------------------------------------------------
    # Verifier training: Future BEV rollout and Metric Prism calibration.
    # ------------------------------------------------------------------

    @staticmethod
    def _detach_tensor_mapping(mapping: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            key: value.detach() if torch.is_tensor(value) else value
            for key, value in mapping.items()
        }


    def _validate_joint_training_inputs(self) -> None:
        if self._config.training_phase != "joint_finetuning":
            return
        if not self._config.metric_cache_path:
            raise ValueError(
                "Joint fine-tuning requires config.metric_cache_path for "
                "official online candidate supervision."
            )
        metric_cache_path = Path(self._config.metric_cache_path).expanduser()
        if not metric_cache_path.is_dir():
            raise FileNotFoundError(f"Metric cache directory does not exist: {metric_cache_path}")
        if int(self._config.joint_probability_source_count) != 64:
            raise ValueError("Joint fine-tuning requires 64 online candidates.")

    def _forward_verifier_training(self, features, context, maneuver_logits=None, targets=None):
        """Train Metric Prism and retain live evidence for all 64 online bases."""
        results: Dict[str, torch.Tensor] = {}
        device = context["status_feat"].device
        if maneuver_logits is None:
            raise ValueError("Online joint training requires current maneuver logits.")
        candidate_batch = self._build_online_candidate_batch(
            features,
            context,
            maneuver_logits,
        )
        base_hypotheses = candidate_batch["hypotheses"]
        batch_size, candidate_count = base_hypotheses.shape[:2]
        teacher_sim = candidate_batch["pdm_factor_labels"]
        teacher_pdms = candidate_batch["online_pdm_scores"]
        proposal_selected_families = candidate_batch[
            "hypothesis_maneuver_component_indices"
        ]
        base_indices = torch.arange(
            candidate_count,
            device=device,
            dtype=torch.long,
        ).unsqueeze(0).expand(batch_size, -1)

        maneuver_component_weights = self._build_maneuver_route(
            maneuver_logits,
            num_anchors=candidate_count,
            proposal_selected_families=proposal_selected_families,
        )

        future_bev_feature_targets = None
        if self._future_bev_feature_supervision_enabled():
            future_bev_feature_targets = (
                self._encode_future_bev_feature_targets(targets, context)
            )
        _, _, recon_outputs, bridge_state = self.verifier.rollout_world(
            context,
            future_bev_targets=future_bev_feature_targets,
        )
        reader_outputs = self.verifier.trajectory_evidence_probe(
            base_hypotheses,
            features["status_feature"],
            bridge_state,
            maneuver_component_weights=maneuver_component_weights,
        )
        eval_outputs = self.verifier.score_evidence(reader_outputs)
        eval_outputs, eval_invalid = self._guard_eval_outputs(
            eval_outputs,
        )

        if not torch.is_tensor(maneuver_component_weights):
            raise RuntimeError(
                "Joint training requires maneuver-indexed routing weights."
            )
        results["__verifier_candidate_batch_for_ecfr__"] = candidate_batch
        results["__verifier_evidence_reuse_for_ecfr__"] = {
            "maneuver_component_weights": maneuver_component_weights.detach(),
            # These tensors intentionally retain the live verifier, world, and
            # perception graph for refinement loss.
            "evidence_memory": reader_outputs,
            "source_indices": base_indices,
        }

        results.update(recon_outputs)
        results.update(eval_outputs)
        results["verifier_online_pool_base_indices"] = base_indices
        results["verifier_probability_pool_size_monitor"] = torch.tensor(
            float(candidate_count),
            device=device,
            dtype=torch.float32,
        )
        results["verifier_online_pool_size_monitor"] = torch.tensor(
            float(candidate_count),
            device=device,
            dtype=torch.float32,
        )
        results["verifier_training_subset_hypothesis_count_monitor"] = torch.tensor(
            float(base_hypotheses.shape[1]),
            device=device,
            dtype=torch.float32,
        )
        results["verifier_online_pool_eval_invalid_hypotheses_monitor"] = (
            self._invalid_monitor(eval_invalid)
            if eval_invalid.numel() > 0
            else torch.tensor(0.0, device=device)
        )
        results["verifier_online_pool_pdm_factor_labels"] = teacher_sim
        results["verifier_online_pool_pdm_scores"] = teacher_pdms
        results[
            "verifier_online_pool_hypothesis_maneuver_component_indices"
        ] = proposal_selected_families
        results["verifier_online_pool_pdm_mean_monitor"] = (
            teacher_pdms.detach().mean()
        )
        results["verifier_online_pool_pdm_max_monitor"] = (
            teacher_pdms.detach().max()
        )
        results["verifier_online_pool_pdm_min_monitor"] = (
            teacher_pdms.detach().min()
        )
        return results

    # ------------------------------------------------------------------
    # Joint ECFR fine-tuning.
    # ------------------------------------------------------------------

    def _forward_joint_finetuning(
        self,
        features: Dict[str, torch.Tensor],
        verifier_reuse: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Refine all online bases toward their same-set component argmax."""
        if not isinstance(verifier_reuse, Mapping):
            raise ValueError(
                "Joint fine-tuning requires shared verifier/ECFR candidate data."
            )
        candidate_batch = verifier_reuse.get("candidate_batch")
        ecfr_reuse = verifier_reuse.get("ecfr_reuse")
        if not isinstance(candidate_batch, Mapping) or not isinstance(
            ecfr_reuse,
            Mapping,
        ):
            raise ValueError(
                "Joint fine-tuning requires both the online candidate set and "
                "its live ECFR evidence memory."
            )

        required_refiner_keys = {
            "maneuver_component_weights",
            "evidence_memory",
        }
        missing_refiner = sorted(
            required_refiner_keys.difference(ecfr_reuse)
        )
        if missing_refiner:
            raise KeyError(
                f"ECFR reuse batch is missing keys: {missing_refiner}."
            )
        required_candidate_keys = {
            "hypotheses",
            "hypotheses_flow_state",
            "online_pdm_scores",
            "pdm_factor_labels",
            "hypothesis_maneuver_component_indices",
        }
        missing_candidates = sorted(required_candidate_keys.difference(candidate_batch))
        if missing_candidates:
            raise KeyError(
                f"Online candidate set is missing keys: {missing_candidates}."
            )

        device = features["status_feature"].device
        base_hypotheses = candidate_batch["hypotheses"].to(
            device=device
        ).detach()
        base_flow_state = candidate_batch["hypotheses_flow_state"].to(
            device=device
        ).detach()
        base_teacher_scores = candidate_batch[
            "online_pdm_scores"
        ].to(
            device=device,
            dtype=torch.float32,
        ).detach()
        base_maneuvers = candidate_batch[
            "hypothesis_maneuver_component_indices"
        ].to(
            device=device,
            dtype=torch.long,
        ).detach()
        maneuver_weights = ecfr_reuse[
            "maneuver_component_weights"
        ].to(device=device).detach()
        evidence_memory = {
            key: (
                value.to(device=device)
                if torch.is_tensor(value)
                else value
            )
            for key, value in ecfr_reuse["evidence_memory"].items()
        }
        expected_count = int(
            self._config.joint_probability_source_count
        )
        if int(base_hypotheses.shape[1]) != expected_count:
            raise ValueError(
                "Shared verifier/ECFR online set must contain "
                f"{expected_count} hypotheses, got "
                f"{base_hypotheses.shape[1]}."
            )
        if maneuver_weights.shape[:2] != base_hypotheses.shape[:2]:
            raise ValueError(
                "ECFR maneuver weights must align with base hypotheses."
            )

        refinement = self._run_ecfr(
            base_hypotheses=base_hypotheses,
            base_hypotheses_flow_state=base_flow_state,
            evidence_memory_outputs=evidence_memory,
        )
        refined_hypotheses = refinement["refined_hypotheses"]
        if (
            self._proposal_invalid_mask(base_hypotheses).any()
            or self._proposal_invalid_mask(refined_hypotheses).any()
        ):
            raise ValueError(
                "Non-finite hypotheses encountered in ECFR fine-tuning."
            )

        supervision = build_online_refinement_supervision(
            hypotheses=base_hypotheses,
            pdm_scores=base_teacher_scores,
            maneuver_component_indices=base_maneuvers,
        )
        return {
            "joint_finetuning_delta_traj_pred_frenet": refinement["frenet_residuals"],
            "joint_finetuning_teacher_delta_frenet": supervision["target_frenet_residuals"],
            "joint_finetuning_refine_supervision_weight": supervision["supervision_weight"],
        }

    def _finalize_inference_output(
        self,
        output: Mapping[str, torch.Tensor],
    ) -> ManeuverFlowInferenceOutput:
        """Enforce the configured fixed-cardinality paper-facing contract."""

        base_count = int(self._config.inference_num_base_hypotheses)
        expected_counts: Dict[str, int] = {
            "generated_base_hypotheses": base_count,
            "final_hypotheses": base_count,
        }
        expected_counts.update(
            {
                "retained_base_hypotheses": int(
                    self._config.inference_num_retained_base_hypotheses
                ),
                "refined_hypotheses": int(
                    self._config.inference_num_refined_hypotheses
                ),
            }
        )
        public_fields = ManeuverFlowInferenceOutput.__annotations__
        public_output = {
            name: value
            for name, value in output.items()
            if name in public_fields
        }
        return validate_maneuverflow_inference_output(
            public_output,
            expected_hypothesis_counts=expected_counts,
        )


    # ------------------------------------------------------------------
    # Fixed-budget ManeuverFlow inference.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def inference(
        self,
        features: Dict[str, torch.Tensor],
    ) -> ManeuverFlowInferenceOutput:
        """Run the released fixed-budget Hypothesize--Verify--Improve graph."""
        if self._config.training_phase != "inference":
            raise RuntimeError(
                "ManeuverFlowModel.inference requires "
                "training_phase='inference'; training recipes must use "
                "ManeuverFlowModel.forward."
            )
        self._config.validate_maneuverflow_inference_contract(
            require_deployment_bounds=True
        )
        actual_ecfr_bounds = (
            float(self.improver.ecfr.max_ds),
            float(self.improver.ecfr.max_dl),
            float(self.improver.ecfr.max_dpsi),
        )
        expected_ecfr_bounds = (
            MANEUVERFLOW_INFERENCE_POLICY.ecfr_max_ds,
            MANEUVERFLOW_INFERENCE_POLICY.ecfr_max_dl,
            MANEUVERFLOW_INFERENCE_POLICY.ecfr_max_dpsi,
        )
        if actual_ecfr_bounds != expected_ecfr_bounds:
            raise RuntimeError(
                "ECFR module bounds do not match the ManeuverFlow inference "
                f"policy: expected {expected_ecfr_bounds}, got "
                f"{actual_ecfr_bounds}."
            )

        output: Dict[str, torch.Tensor] = {}

        perception_output, context = self._forward_perception(features)
        maneuver_logits = perception_output["maneuver_component_logits"]
        output.update(perception_output)
        batch_size = int(maneuver_logits.shape[0])
        device = maneuver_logits.device

        mgmp_batch = self._build_inference_mgmp_batch(maneuver_logits)
        if mgmp_batch is None:
            raise RuntimeError(
                "The released inference graph requires checkpoint-backed "
                "M-GMP statistics."
            )
        proposal_state = mgmp_batch["proposal_e"]
        proposal_maneuvers = mgmp_batch["proposal_selected_families"]
        maneuver_counts = mgmp_batch["proposal_family_counts"].to(
            device=device,
            dtype=torch.long,
        )
        selected_probabilities = mgmp_batch["selected_family_probabilities"]
        num_base = int(proposal_state.shape[1])
        if num_base != int(self._config.inference_num_base_hypotheses):
            raise RuntimeError(
                "M-GMP proposal count does not match the release contract: "
                f"{num_base}."
            )

        output["selected_maneuver_component_indices"] = mgmp_batch["selected_families"]
        output["base_hypothesis_maneuver_component_indices"] = proposal_maneuvers
        output["mgmp_hypothesis_counts"] = maneuver_counts
        output["selected_maneuver_component_probabilities"] = selected_probabilities
        output["mgmp_source_states"] = proposal_state

        maneuver_weights = self._build_maneuver_route(
            maneuver_logits,
            num_anchors=num_base,
            proposal_selected_families=proposal_maneuvers,
        )
        state_context = context["status_feat"].unsqueeze(1)
        agent_context = context["agent_feat"]
        bev_context = (
            context["bev_feat"]
            + self.hypothesizer.meanflow_generator.bev_position_embedding.weight[
                None, ...
            ]
        )

        # One shared Future-BEV field conditions every verifier/refiner query.
        _, _, _, evidence_field = self.verifier.rollout_world(context)

        # The released M³F solver is the one-step r=0, t=1 M³F update.
        start_time = torch.ones(batch_size, device=device)
        end_time = torch.zeros(batch_size, device=device)
        velocity, _, _ = self._evaluate_m3f(
            z=proposal_state,
            r=end_time,
            t=start_time,
            c_state=state_context,
            c_bev=bev_context,
            c_agent=agent_context,
            maneuver_component_weights=maneuver_weights,
        )
        generated_flow_state = proposal_state - velocity
        generated_hypotheses = self._flow_features_to_physical_trajectory(
            generated_flow_state
        )

        def fixed_log_pdm(
            evaluation: Dict[str, torch.Tensor],
        ) -> torch.Tensor:
            factors = evaluation["pdm_factor_scores"]
            return _compose_log_pdm_score(
                factors[:, 0, :],
                factors[:, 1, :],
                factors[:, 2, :],
                factors[:, 3, :],
                factors[:, 4, :],
                eps=1e-6,
            )

        base_evaluation = self.verifier.evaluate_hypotheses(
            trajectory_hypotheses=generated_hypotheses,
            state_navigation_feature=features["status_feature"],
            evidence_field=evidence_field,
            maneuver_component_weights=maneuver_weights,
        )
        base_evaluation, _ = self._guard_eval_outputs(
            base_evaluation,
        )
        base_prescores = fixed_log_pdm(base_evaluation)

        (
            refiner_indices,
            refiner_mask,
            _,
            effective_refiner_count,
        ) = _select_ecfr_sources_from_preliminary_ranking(
            base_prescores,
            primary_family_counts=maneuver_counts[:, 0],
            refined_hypothesis_count=int(
                self._config.inference_num_refined_hypotheses
            ),
        )
        expected_refiner_count = int(
            self._config.inference_num_refined_hypotheses
        )
        if effective_refiner_count != expected_refiner_count:
            raise RuntimeError(
                "ECFR source count does not match the release contract: "
                f"{effective_refiner_count}."
            )

        retained_indices, retained_mask = _build_maneuverflow_retained_base_indices(
            base_prescores,
            retained_hypothesis_count=int(
                self._config.inference_num_retained_base_hypotheses
            ),
            ecfr_source_indices=refiner_indices,
            final_pool_composition=(
                self._config.ecfr_final_pool_composition
            ),
        )
        keep_count = int(retained_indices.shape[1])
        expected_keep_count = int(
            self._config.inference_num_retained_base_hypotheses
        )
        if keep_count != expected_keep_count:
            raise RuntimeError(
                "Retained-base count does not match the release contract: "
                f"{keep_count} vs {expected_keep_count}."
            )
        retained_hypotheses = self._gather_hypotheses(
            generated_hypotheses,
            retained_indices,
        )
        retained_flow_state = self._gather_hypotheses(
            generated_flow_state,
            retained_indices,
        )
        retained_maneuver_weights = self._gather_hypotheses(
            maneuver_weights,
            retained_indices,
        )

        source_hypotheses = self._gather_hypotheses(
            generated_hypotheses,
            refiner_indices,
        )
        source_flow_state = self._gather_hypotheses(
            generated_flow_state,
            refiner_indices,
        )
        source_maneuver_weights = self._gather_hypotheses(
            maneuver_weights,
            refiner_indices,
        )
        evidence_memory = self._build_trajectory_evidence_memory(
            trajectory_hypotheses=source_hypotheses,
            state_navigation_feature=features["status_feature"],
            evidence_field=evidence_field,
            maneuver_component_weights=source_maneuver_weights,
        )
        refinement = self._run_ecfr(
            base_hypotheses=source_hypotheses,
            base_hypotheses_flow_state=source_flow_state,
            evidence_memory_outputs=evidence_memory,
        )
        refined_hypotheses = refinement["refined_hypotheses"]
        refined_flow_state = refinement["refined_hypotheses_flow_space"]

        joint_hypotheses = torch.cat(
            [retained_hypotheses, refined_hypotheses],
            dim=1,
        )
        joint_maneuver_weights = torch.cat(
            [retained_maneuver_weights, source_maneuver_weights],
            dim=1,
        )
        final_evaluation = self.verifier.evaluate_hypotheses(
            trajectory_hypotheses=joint_hypotheses,
            state_navigation_feature=features["status_feature"],
            evidence_field=evidence_field,
            maneuver_component_weights=joint_maneuver_weights,
        )
        final_evaluation, _ = self._guard_eval_outputs(
            final_evaluation,
        )
        final_raw_scores = fixed_log_pdm(final_evaluation)
        retained_raw_scores = final_raw_scores[:, :keep_count]
        refined_raw_scores = final_raw_scores[:, keep_count:]
        ecfr_delta_cost = _compute_ecfr_residual_cost(
            refinement["trajectory_residuals"],
            config=self._config,
        ).to(
            device=refined_raw_scores.device,
            dtype=refined_raw_scores.dtype,
        )
        refined_scores = refined_raw_scores - ecfr_delta_cost

        output.update(
            self._merge_inference_candidate_sets(
                base_trajs=retained_hypotheses,
                base_scores=retained_raw_scores,
                refined_trajs=refined_hypotheses,
                refined_scores=refined_scores,
                base_raw_scores=retained_raw_scores,
                refined_raw_scores=refined_raw_scores,
                base_source_indices=retained_indices,
                refined_source_indices=refiner_indices,
            )
        )

        output.update(
            {
                "generated_base_hypotheses": generated_hypotheses,
                "generated_base_hypotheses_flow_state": generated_flow_state,
                "retained_base_hypotheses": retained_hypotheses,
                "retained_base_hypotheses_flow_state": retained_flow_state,
                "refined_hypotheses": refined_hypotheses,
                "refined_hypotheses_flow_state": refined_flow_state,
                "ecfr_frenet_residuals": refinement[
                    "frenet_residuals"
                ],
                "retained_base_hypothesis_scores": retained_raw_scores,
                "retained_base_hypothesis_scores_raw": retained_raw_scores,
                "refined_hypothesis_scores": refined_scores,
                "refined_hypothesis_scores_raw": refined_raw_scores,
                "ecfr_delta_cost": ecfr_delta_cost,
                "retained_base_hypothesis_indices": retained_indices,
                "retained_base_hypothesis_mask": retained_mask,
                "ecfr_source_hypothesis_indices": refiner_indices,
                "ecfr_source_hypothesis_mask": refiner_mask,
                "final_pdm_factor_scores": final_evaluation[
                    "pdm_factor_scores"
                ],
                "base_pdm_factor_scores": final_evaluation[
                    "pdm_factor_scores"
                ][:, :, :keep_count],
                "refined_pdm_factor_scores": final_evaluation[
                    "pdm_factor_scores"
                ][:, :, keep_count:],
            }
        )
        return self._finalize_inference_output(output)

class StatusModulatedManeuverDecoder(nn.Module):
    """Predict M-GMP maneuver-component logits from planning context.

    The navigation command forms the maneuver query and ego kinematics apply
    FiLM modulation before classification.
    """
    def __init__(self, d_model, nav_dim=4, kin_dim=4, num_classes=5, dropout=0.1):
        super().__init__()
        # Navigation command -> dynamic maneuver query.
        self.dynamic_query_proj = nn.Sequential(
            nn.Linear(nav_dim, d_model),
            nn.LayerNorm(d_model)
        )

        # Ego kinematics -> FiLM scale and shift.
        self.kinematic_modulation = nn.Sequential(
            nn.Linear(kin_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model * 2)
        )

        # Maneuver-component classifier.
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

    def forward_query(self, nav_command):
        return self.dynamic_query_proj(nav_command)

    def forward_classify(self, maneuver_context_query, kinematic_state):
        scale, shift = self.kinematic_modulation(kinematic_state).chunk(2, dim=-1)
        modulated_feat = maneuver_context_query * (1 + scale) + shift
        logits = self.classifier(modulated_feat)
        return logits

    def forward(self, maneuver_context_query, kinematic_state):
        return self.forward_classify(maneuver_context_query, kinematic_state)

class AgentPredictionHead(nn.Module):
    def __init__(self, num_agents, d_ffn, d_model, dropout: float = 0.0, use_norm: bool = False):
        super().__init__()
        dropout = max(0.0, float(dropout))
        self._input_norm = (
            nn.LayerNorm(d_model, elementwise_affine=False)
            if use_norm
            else nn.Identity()
        )
        self._state_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self._label_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self._mlp_states = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.ReLU(),
            nn.Linear(d_ffn, BoundingBox2DIndex.size()),
        )
        self._mlp_label = nn.Sequential(nn.Linear(d_model, 1))
    def forward(self, agent_queries):
        agent_queries = self._input_norm(agent_queries)
        state_features = self._mlp_states[0](agent_queries)
        state_features = self._mlp_states[1](state_features)
        state_features = self._state_dropout(state_features)
        agent_states = self._mlp_states[2](state_features)
        point_xy = agent_states[..., BoundingBox2DIndex.POINT].tanh() * 32
        heading = agent_states[..., BoundingBox2DIndex.HEADING:BoundingBox2DIndex.HEADING + 1].tanh() * np.pi
        other_dims = agent_states[..., 3:]
        agent_states = torch.cat([point_xy, heading, other_dims], dim=-1)
        agent_labels = self._mlp_label(self._label_dropout(agent_queries)).squeeze(dim=-1)
        return {"agent_states": agent_states, "agent_labels": agent_labels}
