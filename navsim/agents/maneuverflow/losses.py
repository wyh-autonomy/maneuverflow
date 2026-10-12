import math
from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from .mgmp_prior import (
    MGMP_LANE_CHANGE_THRESHOLD_DEGREES,
    MGMP_SPEED_TREND_THRESHOLD_METERS_PER_INTERVAL,
    MGMP_STOP_DISTANCE_METERS,
    MGMP_TURN_THRESHOLD_DEGREES,
)


_METRIC_PRISM_NC_CLASS_WEIGHTS = (1.7, 3.1, 1.0)
_METRIC_PRISM_FACTOR_LOSS_WEIGHTS = {
    "nc": 5.0,
    "dac": 5.0,
    "ep": 5.0,
    "ttc": 5.0,
    "comfort": 2.0,
}
_METRIC_PRISM_DAC_NEGATIVE_WEIGHT = 1.8
_METRIC_PRISM_TTC_NEGATIVE_WEIGHT = 1.75
_METRIC_PRISM_COMFORT_NEGATIVE_WEIGHT = 4.5
_METRIC_PRISM_BINARY_LABEL_SMOOTHING = 0.01
_METRIC_PRISM_NC_LABEL_SMOOTHING = 0.02
_METRIC_PRISM_LOGIT_L2_WEIGHT = 1e-5

def build_dense_maneuver_target(
    component_distances: torch.Tensor,
) -> torch.Tensor:
    """Return the paper target ``Softmax(-d)`` over every component."""

    if component_distances.dim() != 2:
        raise ValueError(
            "component_distances must be [B, M], got "
            f"{tuple(component_distances.shape)}."
        )
    distances = component_distances.detach().float()
    finite_rows = torch.isfinite(distances).all(dim=1)
    if not bool(finite_rows.all()):
        bad_rows = torch.nonzero(
            ~finite_rows,
            as_tuple=False,
        ).flatten().detach().cpu().tolist()
        raise ValueError(
            "Dense maneuver supervision requires finite distances for all "
            f"components; invalid rows {bad_rows}."
        )
    target = F.softmax(-distances, dim=-1)
    return target.to(dtype=component_distances.dtype)


def classify_hard_maneuver_targets(
    trajectories: torch.Tensor,
) -> torch.Tensor:
    """Apply the appendix-aligned eight-family rule to trajectories."""

    if trajectories.dim() == 4 and int(trajectories.shape[1]) == 1:
        trajectories = trajectories[:, 0]
    if (
        trajectories.dim() != 3
        or trajectories.shape[-2:] != (8, 3)
    ):
        raise ValueError(
            "trajectories must be [B, 8, 3] or [B, 1, 8, 3], got "
            f"{tuple(trajectories.shape)}."
        )
    if not bool(torch.isfinite(trajectories).all()):
        raise ValueError(
            "Hard maneuver classification requires finite trajectories."
        )

    trajectory = trajectories.detach().float()
    xy = trajectory[..., :2]
    heading = trajectory[..., 2]
    displacement = torch.linalg.norm(xy[:, -1] - xy[:, 0], dim=-1)
    mean_heading_degrees = (
        heading.mean(dim=-1).to(torch.float64)
        * (180.0 / math.pi)
    )
    absolute_heading = mean_heading_degrees.abs()
    displacement_steps = torch.cat(
        (xy[:, :1], xy[:, 1:] - xy[:, :-1]),
        dim=1,
    )
    step_lengths = torch.linalg.norm(displacement_steps, dim=-1)
    speed_trend = (
        step_lengths[:, 4:8].sum(dim=-1)
        - step_lengths[:, 0:4].sum(dim=-1)
    ) / 4.0

    labels = torch.ones_like(displacement, dtype=torch.long)
    labels = torch.where(
        speed_trend
        > MGMP_SPEED_TREND_THRESHOLD_METERS_PER_INTERVAL,
        2,
        labels,
    )
    labels = torch.where(
        speed_trend
        < -MGMP_SPEED_TREND_THRESHOLD_METERS_PER_INTERVAL,
        3,
        labels,
    )
    labels = torch.where(
        absolute_heading > MGMP_LANE_CHANGE_THRESHOLD_DEGREES,
        torch.where(mean_heading_degrees > 0.0, 4, 6),
        labels,
    )
    labels = torch.where(
        absolute_heading > MGMP_TURN_THRESHOLD_DEGREES,
        torch.where(mean_heading_degrees > 0.0, 5, 7),
        labels,
    )
    return torch.where(
        displacement < MGMP_STOP_DISTANCE_METERS,
        0,
        labels,
    )

def _smooth_binary_target(target: torch.Tensor, eps: float) -> torch.Tensor:
    eps = max(0.0, min(float(eps), 0.49))
    if eps <= 0.0:
        return target
    return target * (1.0 - eps) + 0.5 * eps


def compute_maneuverflow_loss(
    targets: Dict[str, torch.Tensor],
    predictions: Dict[str, torch.Tensor],
    config: Any,
) -> Dict[str, torch.Tensor]:
    """Compose the Stage-1 and Joint training objective."""
    phase = str(config.training_phase)
    if phase not in {"generation_pretraining", "joint_finetuning"}:
        raise ValueError(
            "ManeuverFlow losses are defined only for "
            "'generation_pretraining' and 'joint_finetuning'."
        )

    perception_components = compute_perception_loss(
        targets,
        predictions,
        config,
    )
    required_auxiliary_keys = (
        "agent_cls_loss",
        "agent_box_loss",
        "map_loss",
        "maneuver_auxiliary_loss",
    )
    missing_auxiliary = [
        key for key in required_auxiliary_keys if key not in perception_components
    ]
    if missing_auxiliary:
        raise RuntimeError(
            "L_aux requires map, agent, and maneuver supervision; missing "
            + ", ".join(missing_auxiliary)
        )
    auxiliary_loss = sum(
        (
            perception_components.pop(key)
            for key in required_auxiliary_keys[1:]
        ),
        perception_components.pop(required_auxiliary_keys[0]),
    )

    m3f_components = compute_m3f_loss(
        targets,
        predictions,
        config,
    )
    m3f_loss = m3f_components.pop("m3f_loss")

    verification_loss = auxiliary_loss.new_zeros(())
    refinement_loss = auxiliary_loss.new_zeros(())
    verifier_components: Dict[str, torch.Tensor] = {}
    refinement_components: Dict[str, torch.Tensor] = {}
    if phase == "joint_finetuning":
        verifier_components = compute_verifier_loss(
            targets,
            predictions,
            config,
        )
        required_verification_keys = (
            "future_bev_feature_loss",
            "metric_prism_pdm_factor_loss",
        )
        missing_verification = [
            key
            for key in required_verification_keys
            if key not in verifier_components
        ]
        if missing_verification:
            raise RuntimeError(
                "L_verify requires Future-BEV and factor-score calibration; "
                "missing "
                + ", ".join(missing_verification)
            )
        verification_loss = (
            verifier_components.pop(required_verification_keys[0])
            + verifier_components.pop(required_verification_keys[1])
        )

        refinement_components = compute_joint_finetuning_loss(
            targets,
            predictions,
            config,
        )
        refinement_loss = refinement_components.pop("joint_finetuning_delta_traj_loss")

    weighted_auxiliary = (
        float(config.auxiliary_loss_weight) * auxiliary_loss
    )
    weighted_m3f = float(config.m3f_loss_weight) * m3f_loss
    total_loss = weighted_auxiliary + weighted_m3f
    weighted_verification = (
        float(config.verification_loss_weight) * verification_loss
    )
    weighted_refinement = (
        float(config.refinement_loss_weight) * refinement_loss
    )
    if phase == "joint_finetuning":
        total_loss = (
            total_loss + weighted_verification + weighted_refinement
        )

    loss_dict: Dict[str, torch.Tensor] = {"loss": total_loss}
    component_groups = (
        perception_components,
        m3f_components,
        verifier_components,
        refinement_components,
    )
    for components in component_groups:
        for key, value in components.items():
            if not torch.is_tensor(value):
                continue
            monitor_key = key if key.endswith("_monitor") else f"{key}_monitor"
            loss_dict[monitor_key] = value.detach()

    loss_dict.update(
        {
            "auxiliary_loss_monitor": auxiliary_loss.detach(),
            "m3f_loss_monitor": m3f_loss.detach(),
            "weighted_auxiliary_loss_monitor": weighted_auxiliary.detach(),
            "weighted_m3f_loss_monitor": weighted_m3f.detach(),
        }
    )
    if phase == "joint_finetuning":
        loss_dict.update(
            {
                "verification_loss_monitor": verification_loss.detach(),
                "refinement_loss_monitor": refinement_loss.detach(),
                "weighted_verification_loss_monitor": (
                    weighted_verification.detach()
                ),
                "weighted_refinement_loss_monitor": (
                    weighted_refinement.detach()
                ),
            }
        )

    return loss_dict

# ==============================================================================
# ManeuverFlow task losses (kept local to make the release self-contained)
# ==============================================================================

def compute_perception_loss(
    targets: Dict[str, torch.Tensor],
    predictions: Dict[str, torch.Tensor],
    config: Any
) -> Dict[str, torch.Tensor]:
    losses = {}

    # 1. Agent Detection Loss (Hungarian Matching).
    agent_cls_loss, agent_box_loss = compute_agent_detection_loss(
        targets["agent_states"],
        targets["agent_labels"],
        predictions["agent_states"],
        predictions["agent_labels"],
        config,
    )
    agent_class_weight = config.agent_class_weight
    agent_box_weight = config.agent_box_weight

    losses["agent_cls_loss"] = agent_cls_loss * agent_class_weight
    losses["agent_box_loss"] = agent_box_loss * agent_box_weight

    # 2. BEV Semantic Map Loss.
    if predictions.get("bev_semantic_map") is None:
        raise RuntimeError("L_aux requires predictions['bev_semantic_map'].")
    focal_loss_fn = FocalLoss(
        alpha=config.focal_loss_alpha,
        gamma=config.focal_loss_gamma,
    )
    map_loss = focal_loss_fn(
        predictions["bev_semantic_map"],
        targets["bev_semantic_map"].long(),
    )
    map_weight = config.bev_semantic_weight
    losses["map_loss"] = map_loss * map_weight

    # 3. Paper M-GMP association target: CE(Softmax(-d), pi_man).
    if "maneuver_component_logits" in predictions:
        maneuver_logits = predictions["maneuver_component_logits"]
        component_distances = predictions.get(
            "mgmp_component_distance",
            None,
        )
        if torch.is_tensor(component_distances):
            rank_target = build_dense_maneuver_target(
                component_distances.to(device=maneuver_logits.device),
            )
        else:
            raw_rank_target = predictions.get(
                "maneuver_rank_target",
                None,
            )
            if not torch.is_tensor(raw_rank_target):
                raise RuntimeError(
                    "M-GMP supervision requires "
                    "predictions['mgmp_component_distance'] or an already "
                    "normalized predictions['maneuver_rank_target']."
                )
            rank_target = raw_rank_target.to(
                device=maneuver_logits.device,
                dtype=maneuver_logits.dtype,
            )
        if rank_target.shape != maneuver_logits.shape:
            raise ValueError(
                "maneuver rank target shape must match "
                "maneuver_component_logits: "
                f"{tuple(rank_target.shape)} vs "
                f"{tuple(maneuver_logits.shape)}"
            )
        rank_target = rank_target.to(
            device=maneuver_logits.device,
            dtype=maneuver_logits.dtype,
        )
        if (
            not bool(torch.isfinite(rank_target).all())
            or bool((rank_target < 0.0).any())
        ):
            raise RuntimeError(
                "The maneuver rank target must contain finite, non-negative "
                "weights for all components."
            )
        rank_mass = rank_target.sum(dim=-1, keepdim=True)
        if bool((rank_mass <= 0.0).any()):
            raise RuntimeError(
                "The maneuver rank target must have positive row mass."
            )
        normalized_rank_target = rank_target / rank_mass
        maneuver_rank_loss = -(
            normalized_rank_target * F.log_softmax(maneuver_logits, dim=-1)
        ).sum(dim=-1).mean()

        losses["maneuver_rank_loss_monitor"] = maneuver_rank_loss.detach()
        losses["maneuver_auxiliary_loss"] = maneuver_rank_loss

    return losses

def compute_agent_detection_loss(gt_states, gt_valid, pred_states, pred_logits, config):
    batch_dim, num_instances = pred_states.shape[:2]
    num_gt_instances = gt_valid.sum()
    num_gt_instances = num_gt_instances if num_gt_instances > 0 else num_gt_instances + 1

    ce_cost = _get_ce_cost(gt_valid, pred_logits)
    l1_cost = _get_l1_cost(gt_states, pred_states, gt_valid)

    agent_class_weight = config.agent_class_weight
    agent_box_weight = config.agent_box_weight

    cost = agent_class_weight * ce_cost + agent_box_weight * l1_cost
    cost = cost.cpu()

    if torch.isnan(cost).any() or torch.isinf(cost).any():
        cost = torch.nan_to_num(cost, nan=1e4, posinf=1e4, neginf=1e4)

    indices = [linear_sum_assignment(c) for i, c in enumerate(cost)]
    matching = [
        (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
        for i, j in indices
    ]
    idx = _get_src_permutation_idx(matching)

    pred_states_idx = pred_states[idx]
    gt_states_idx = torch.cat([t[i] for t, (_, i) in zip(gt_states, indices)], dim=0)

    pred_valid_idx = pred_logits[idx]
    gt_valid_idx = torch.cat([t[i] for t, (_, i) in zip(gt_valid, indices)], dim=0).float()

    l1_loss = F.l1_loss(pred_states_idx, gt_states_idx, reduction="none")
    l1_loss = l1_loss.sum(-1) * gt_valid_idx
    l1_loss = l1_loss.view(batch_dim, -1).sum() / num_gt_instances

    ce_loss = F.binary_cross_entropy_with_logits(pred_valid_idx, gt_valid_idx, reduction="none")
    ce_loss = ce_loss.view(batch_dim, -1).mean()

    return ce_loss, l1_loss

def compute_m3f_loss(
    targets: Dict[str, torch.Tensor],
    predictions: Dict[str, torch.Tensor],
    config: Any,
) -> Dict[str, torch.Tensor]:
    """L1 MeanFlow objective in normalized displacement/sin/cos space."""
    prediction = predictions["m3f_velocity_prediction"]
    target = predictions["m3f_target_velocity"]
    if prediction.ndim != 4 or prediction.shape[-2:] != (8, 4) or target.shape != prediction.shape:
        raise ValueError("M3F velocity predictions and targets must share shape [B, N, 8, 4].")
    per_hypothesis_loss = (prediction - target.detach()).float().abs().mean(dim=(-1, -2))
    loss = per_hypothesis_loss.sum() / per_hypothesis_loss.numel()
    losses = {"m3f_loss": loss, "m3f_velocity_loss_raw_monitor": loss.detach()}
    for key in (
        "m3f_time_interval_monitor",
        "m3f_full_step_ratio_monitor",
        "m3f_diagonal_time_ratio_monitor",
        "m3f_general_time_ratio_monitor",
        "m3f_velocity_invalid_hypotheses_monitor",
        "m3f_target_velocity_invalid_hypotheses_monitor",
        "m3f_training_proposal_count_monitor",
        "m3f_boundary_tangent_l1_monitor",
    ):
        if torch.is_tensor(predictions.get(key)):
            losses[key] = predictions[key].detach()
    return losses


def _compose_pdm_scores_from_factors(
    pdm_factor_labels: torch.Tensor,
) -> torch.Tensor:
    nc = pdm_factor_labels[:, 0]
    dac = pdm_factor_labels[:, 1]
    ep = pdm_factor_labels[:, 2]
    ttc = pdm_factor_labels[:, 3]
    comfort = pdm_factor_labels[:, 4]
    return nc * dac * (5.0 * ep + 5.0 * ttc + 2.0 * comfort) / 12.0


def compute_joint_finetuning_loss(
    targets: Dict[str, torch.Tensor],
    predictions: Dict[str, torch.Tensor],
    config: Any,
) -> Dict[str, torch.Tensor]:
    """Average the main ECFR Frenet objective over valid reference pairs."""
    required = (
        "joint_finetuning_delta_traj_pred_frenet",
        "joint_finetuning_teacher_delta_frenet",
        "joint_finetuning_refine_supervision_weight",
    )
    missing = [key for key in required if not torch.is_tensor(predictions.get(key))]
    if missing:
        raise KeyError(f"ECFR Frenet supervision is missing tensors: {missing}.")
    prediction, reference, supervision_weight = (predictions[key] for key in required)
    if prediction.ndim != 4 or prediction.shape[-2:] != (8, 3):
        raise ValueError("ECFR Frenet predictions must have shape [B, N, 8, 3].")
    if reference.shape != prediction.shape:
        raise ValueError("ECFR Frenet reference shape must match its predictions.")
    proposal_weight = supervision_weight.detach().to(device=prediction.device, dtype=torch.float32)
    if proposal_weight.shape != prediction.shape[:2]:
        raise ValueError("ECFR supervision weight must match the teacher bases [B, N].")
    if not bool(torch.isfinite(proposal_weight).all()):
        raise ValueError("ECFR supervision weights must be finite.")
    if not bool((proposal_weight.eq(0.0) | proposal_weight.eq(1.0)).all()):
        raise ValueError("ECFR supervision weights must be the binary valid-teacher mask.")
    reference = reference.detach().to(device=prediction.device, dtype=torch.float32)
    delta_error = F.smooth_l1_loss(
        prediction.float(), reference, reduction="none",
    ).mean(dim=(-1, -2))
    delta_loss = (delta_error * proposal_weight).sum() / proposal_weight.sum().clamp_min(1e-6)
    delta_weight = float(config.refinement_delta_loss_weight)
    if not math.isfinite(delta_weight) or delta_weight < 0.0:
        raise ValueError("ECFR objective weight must be finite and non-negative.")
    losses = {
        "joint_finetuning_delta_traj_loss": delta_weight * delta_loss,
        "joint_finetuning_valid_teacher_rate_monitor": proposal_weight.mean(),
    }
    losses["joint_finetuning_nonfinite_loss_terms_monitor"] = torch.stack([
        ~torch.isfinite(value.detach()) for value in losses.values()
    ]).sum().to(dtype=prediction.dtype)
    return losses


def compute_metric_prism_factor_loss(
    pdm_factor_labels: torch.Tensor,
    predictions: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    if not torch.is_tensor(pdm_factor_labels) or pdm_factor_labels.ndim != 3 or pdm_factor_labels.shape[1] != 5:
        raise ValueError("Metric Prism labels must have shape [B, 5, N] for the current candidate set.")
    pdm_factor_labels = torch.nan_to_num(
        pdm_factor_labels.float(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).clamp(0.0, 1.0)
    nc_class_weights = _METRIC_PRISM_NC_CLASS_WEIGHTS
    factor_loss_weights = _METRIC_PRISM_FACTOR_LOSS_WEIGHTS
    binary_label_smoothing = _METRIC_PRISM_BINARY_LABEL_SMOOTHING
    nc_label_smoothing = _METRIC_PRISM_NC_LABEL_SMOOTHING

    dac_negative_weight = _METRIC_PRISM_DAC_NEGATIVE_WEIGHT
    ttc_negative_weight = _METRIC_PRISM_TTC_NEGATIVE_WEIGHT
    comfort_negative_weight = _METRIC_PRISM_COMFORT_NEGATIVE_WEIGHT

    nc_logits = predictions["nc_logits"]
    nc_target_raw = pdm_factor_labels[:, 0, :].float()
    if nc_logits.shape[-1] != 1:
        raise ValueError(
            "Metric Prism uses one binary non-collision logit per hypothesis; "
            f"got {tuple(nc_logits.shape)}"
        )
    nc_logit = nc_logits.squeeze(-1)
    # This recipe converts ternary NAVSIM NC labels to a binary target:
    # only the fully collision-free value is positive.
    nc_target = nc_target_raw.gt(0.5).to(dtype=nc_logit.dtype)
    nc_target_smooth = _smooth_binary_target(
        nc_target,
        nc_label_smoothing,
    )
    nc_loss_raw = F.binary_cross_entropy_with_logits(
        nc_logit,
        nc_target_smooth,
        reduction="none",
    )
    class_weights = torch.tensor(
        nc_class_weights,
        device=nc_logits.device,
        dtype=nc_logits.dtype,
    )
    nc_sample_w = torch.where(
        nc_target_raw.gt(0.5),
        class_weights[2],
        torch.where(
            nc_target_raw.gt(0.0),
            class_weights[1],
            class_weights[0],
        ),
    )
    nc_loss = (nc_loss_raw * nc_sample_w).mean()
    nc_pred_score = torch.sigmoid(nc_logit)

    dac_logit = predictions["binary_pdm_factor_logits"][:, 0, :]
    dac_target = pdm_factor_labels[:, 1, :].float()
    dac_target_smooth = _smooth_binary_target(dac_target, binary_label_smoothing)
    dac_loss_raw = F.binary_cross_entropy_with_logits(dac_logit, dac_target_smooth, reduction='none')
    dac_sample_w = (
        dac_target * 1.0 + (1 - dac_target) * dac_negative_weight
    )
    dac_loss = (dac_loss_raw * dac_sample_w).mean()

    ttc_logit = predictions["binary_pdm_factor_logits"][:, 1, :]
    ttc_target = pdm_factor_labels[:, 3, :].float()
    ttc_target_smooth = _smooth_binary_target(ttc_target, binary_label_smoothing)
    ttc_loss_raw = F.binary_cross_entropy_with_logits(ttc_logit, ttc_target_smooth, reduction='none')
    ttc_sample_w = (
        ttc_target * 1.0 + (1 - ttc_target) * ttc_negative_weight
    )
    ttc_loss = (ttc_loss_raw * ttc_sample_w).mean()

    comf_logit = predictions["binary_pdm_factor_logits"][:, 2, :]
    comf_target = pdm_factor_labels[:, 4, :].float()
    comf_target_smooth = _smooth_binary_target(comf_target, binary_label_smoothing)
    comf_loss_raw = F.binary_cross_entropy_with_logits(comf_logit, comf_target_smooth, reduction='none')
    comf_sample_w = (
        comf_target * 1.0 + (1 - comf_target) * comfort_negative_weight
    )
    comf_loss = (comf_loss_raw * comf_sample_w).mean()

    ep_target = pdm_factor_labels[:, 2, :]
    ep_logit = predictions.get("ep_logits")
    if not torch.is_tensor(ep_logit) or ep_logit.ndim != 2:
        raise ValueError(
            "Metric Prism uses one continuous ego-progress logit per "
            f"hypothesis; got {getattr(ep_logit, 'shape', None)}"
        )
    ep_pred = torch.sigmoid(ep_logit)
    ep_target_smooth = _smooth_binary_target(
        ep_target.float(),
        binary_label_smoothing,
    )
    ep_loss = F.binary_cross_entropy_with_logits(
        ep_logit,
        ep_target_smooth,
    )

    nc_weight = float(factor_loss_weights["nc"])
    dac_weight = float(factor_loss_weights["dac"])
    ep_weight = float(factor_loss_weights["ep"])
    ttc_weight = float(factor_loss_weights["ttc"])
    comfort_weight = float(factor_loss_weights["comfort"])
    weighted_nc_loss = nc_weight * nc_loss
    weighted_dac_loss = dac_weight * dac_loss
    weighted_ep_loss = ep_weight * ep_loss
    weighted_ttc_loss = ttc_weight * ttc_loss
    weighted_comfort_loss = comfort_weight * comf_loss

    logit_l2_loss = torch.stack(
        (
            nc_logits.float().square().mean(),
            dac_logit.float().square().mean(),
            ep_logit.float().square().mean(),
            ttc_logit.float().square().mean(),
            comf_logit.float().square().mean(),
        )
    ).mean().to(dtype=nc_logits.dtype)
    weighted_logit_l2_loss = (
        _METRIC_PRISM_LOGIT_L2_WEIGHT * logit_l2_loss
    )
    total_loss = (
        weighted_nc_loss +
        weighted_dac_loss +
        weighted_ep_loss +
        weighted_ttc_loss +
        weighted_comfort_loss +
        weighted_logit_l2_loss
    )

    components = {
        "metric_prism_nc_loss_raw_monitor": nc_loss.detach(),
        "metric_prism_dac_loss_raw_monitor": dac_loss.detach(),
        "metric_prism_ep_loss_raw_monitor": ep_loss.detach(),
        "metric_prism_ttc_loss_raw_monitor": ttc_loss.detach(),
        "metric_prism_comfort_loss_raw_monitor": comf_loss.detach(),
        "metric_prism_nc_loss_contribution_monitor": (
            weighted_nc_loss.detach()
        ),
        "metric_prism_dac_loss_contribution_monitor": (
            weighted_dac_loss.detach()
        ),
        "metric_prism_ep_loss_contribution_monitor": (
            weighted_ep_loss.detach()
        ),
        "metric_prism_ttc_loss_contribution_monitor": (
            weighted_ttc_loss.detach()
        ),
        "metric_prism_comfort_loss_contribution_monitor": (
            weighted_comfort_loss.detach()
        ),
        "metric_prism_logit_l2_loss_raw_monitor": (
            logit_l2_loss.detach()
        ),
        "metric_prism_logit_l2_loss_contribution_monitor": (
            weighted_logit_l2_loss.detach()
        ),
        "metric_prism_loss_unweighted_total_monitor": total_loss.detach(),
        "metric_prism_nc_target_mean_monitor": nc_target_raw.mean().detach(),
        "metric_prism_dac_target_mean_monitor": dac_target.mean().detach(),
        "metric_prism_ep_target_mean_monitor": (
            ep_target.float().mean().detach()
        ),
        "metric_prism_ttc_target_mean_monitor": ttc_target.mean().detach(),
        "metric_prism_comfort_target_mean_monitor": (
            comf_target.mean().detach()
        ),
        "metric_prism_nc_prediction_mean_monitor": (
            nc_pred_score.mean().detach()
        ),
        "metric_prism_dac_prediction_mean_monitor": (
            torch.sigmoid(dac_logit).mean().detach()
        ),
        "metric_prism_ep_prediction_mean_monitor": ep_pred.mean().detach(),
        "metric_prism_ttc_prediction_mean_monitor": (
            torch.sigmoid(ttc_logit).mean().detach()
        ),
        "metric_prism_comfort_prediction_mean_monitor": (
            torch.sigmoid(comf_logit).mean().detach()
        ),
    }
    return total_loss, components


@torch.no_grad()
def _get_ce_cost(gt_valid: torch.Tensor, pred_logits: torch.Tensor) -> torch.Tensor:
    gt_valid_expanded = gt_valid[:, :, None].detach().float()
    pred_logits_expanded = pred_logits[:, None, :].detach()
    max_val = torch.relu(-pred_logits_expanded)
    helper_term = max_val + torch.log(
        torch.exp(-max_val) + torch.exp(-pred_logits_expanded - max_val)
    )
    ce_cost = (1 - gt_valid_expanded) * pred_logits_expanded + helper_term
    ce_cost = ce_cost.permute(0, 2, 1)
    return ce_cost

@torch.no_grad()
def _get_l1_cost(gt_states, pred_states, gt_valid):
    gt_states_expanded = gt_states[:, :, None, :2].detach()
    pred_states_expanded = pred_states[:, None, :, :2].detach()
    l1_cost = gt_valid[..., None].float() * (gt_states_expanded - pred_states_expanded).abs().sum(dim=-1)
    l1_cost = l1_cost.permute(0, 2, 1)
    return l1_cost

def _get_src_permutation_idx(indices):
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx


def _align_future_bev_feature_pair(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Align predicted and target Future-BEV features on common support."""
    if pred.dim() != target.dim():
        raise ValueError(
            "Future-BEV feature rank mismatch: "
            f"pred={tuple(pred.shape)}, target={tuple(target.shape)}"
        )
    if pred.dim() < 3:
        raise ValueError(
            "Future-BEV feature tensors must include [B, T, C, ...], "
            f"got {tuple(pred.shape)}"
        )
    if pred.shape[0] != target.shape[0] or pred.shape[2] != target.shape[2]:
        raise ValueError(
            "Future-BEV feature batch/channel mismatch: "
            f"pred={tuple(pred.shape)}, target={tuple(target.shape)}"
        )

    time_steps = min(int(pred.shape[1]), int(target.shape[1]))
    pred = pred[:, :time_steps]
    target = target[:, :time_steps].to(device=pred.device, dtype=pred.dtype)

    if pred.dim() == 5 and pred.shape[-2:] != target.shape[-2:]:
        bsz, steps = target.shape[:2]
        target = F.interpolate(
            target.reshape(bsz * steps, *target.shape[2:]),
            size=pred.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).view(bsz, steps, target.shape[2], *pred.shape[-2:])
    return pred, target


def _align_future_bev_valid_mask(
    valid_mask: Optional[torch.Tensor],
    pred: torch.Tensor,
) -> torch.Tensor:
    if not torch.is_tensor(valid_mask):
        return torch.ones(pred.shape[:2], device=pred.device, dtype=pred.dtype)
    valid_mask = valid_mask.to(device=pred.device, dtype=pred.dtype)
    if valid_mask.dim() == 1:
        valid_mask = valid_mask.unsqueeze(0)
    return valid_mask[:, : pred.shape[1]]


def _align_future_bev_spatial_mask(
    spatial_mask: Optional[torch.Tensor],
    pred: torch.Tensor,
) -> Optional[torch.Tensor]:
    if not torch.is_tensor(spatial_mask):
        return None
    spatial_mask = spatial_mask.to(device=pred.device, dtype=pred.dtype)
    if spatial_mask.dim() == 3:
        spatial_mask = spatial_mask.unsqueeze(0).unsqueeze(2)
    elif spatial_mask.dim() == 4:
        if spatial_mask.shape[0] == pred.shape[1]:
            spatial_mask = spatial_mask.unsqueeze(0)
        if spatial_mask.shape[2] != 1:
            spatial_mask = spatial_mask.unsqueeze(2)
    if spatial_mask.dim() != 5:
        return None
    if spatial_mask.shape[0] == 1 and pred.shape[0] > 1:
        spatial_mask = spatial_mask.expand(pred.shape[0], -1, -1, -1, -1)
    spatial_mask = spatial_mask[:, : pred.shape[1]]
    if spatial_mask.shape[-2:] != pred.shape[-2:]:
        bsz, steps = spatial_mask.shape[:2]
        spatial_mask = F.interpolate(
            spatial_mask.reshape(bsz * steps, 1, *spatial_mask.shape[-2:]),
            size=pred.shape[-2:],
            mode="nearest",
        ).view(bsz, steps, 1, *pred.shape[-2:])
    if spatial_mask.shape[2] != 1:
        spatial_mask = spatial_mask.amax(dim=2, keepdim=True)
    return spatial_mask


def _apply_spatial_coverage_to_step_mask(
    valid_mask: torch.Tensor,
    spatial_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    if spatial_mask is None:
        return valid_mask
    coverage = spatial_mask.flatten(2).sum(dim=-1) > 0
    return valid_mask * coverage.to(device=valid_mask.device, dtype=valid_mask.dtype)


def _masked_step_mean(values: torch.Tensor, valid_mask: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    weight = valid_mask.to(device=values.device, dtype=values.dtype)
    denom = weight.sum()
    if float(denom.detach().item()) <= 0.0:
        return ref.sum() * 0.0
    return (values * weight).sum() / denom.clamp_min(1.0)


def _masked_smooth_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    spatial_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    loss = F.smooth_l1_loss(pred.float(), target.float(), reduction="none")
    if spatial_mask is not None:
        spatial_weight = spatial_mask.to(device=loss.device, dtype=loss.dtype)
        if spatial_weight.shape[2] == 1 and loss.shape[2] != 1:
            spatial_weight = spatial_weight.expand(
                loss.shape[0],
                loss.shape[1],
                loss.shape[2],
                *loss.shape[3:],
            )
        loss_sum = (loss * spatial_weight).flatten(2).sum(dim=-1)
        denom = spatial_weight.flatten(2).sum(dim=-1).clamp_min(1.0)
        step_loss = loss_sum / denom
        step_mask = _apply_spatial_coverage_to_step_mask(valid_mask, spatial_mask)
        return _masked_step_mean(step_loss, step_mask, pred)
    reduce_dims = tuple(range(2, loss.dim()))
    if reduce_dims:
        loss = loss.mean(dim=reduce_dims)
    return _masked_step_mean(loss, valid_mask, pred)


def _future_bev_cosine_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    spatial_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if spatial_mask is not None:
        spatial_weight = spatial_mask.to(device=pred.device, dtype=pred.dtype)
        if spatial_weight.shape[2] == 1 and pred.shape[2] != 1:
            spatial_weight = spatial_weight.expand(
                pred.shape[0],
                pred.shape[1],
                pred.shape[2],
                *pred.shape[3:],
            )
        pred = pred * spatial_weight
        target = target * spatial_weight
        valid_mask = _apply_spatial_coverage_to_step_mask(valid_mask, spatial_mask)
    pred_vec = pred.float().reshape(pred.shape[0], pred.shape[1], -1)
    target_vec = target.float().reshape(target.shape[0], target.shape[1], -1)
    cos = (1.0 - F.cosine_similarity(pred_vec, target_vec, dim=-1, eps=1e-6)).clamp_min(0.0)
    return _masked_step_mean(cos, valid_mask, pred)


def _future_bev_feature_pair_loss(
    predictions: Dict[str, torch.Tensor],
    pred_key: str,
    target_key: str,
    res_pred_key: str,
    res_target_key: str,
    res_weight: float,
    cos_weight: float,
    valid_mask: Optional[torch.Tensor] = None,
    res_valid_mask: Optional[torch.Tensor] = None,
    spatial_mask: Optional[torch.Tensor] = None,
    res_spatial_mask: Optional[torch.Tensor] = None,
) -> Tuple[Optional[torch.Tensor], Dict[str, torch.Tensor]]:
    pred = predictions.get(pred_key, None)
    target = predictions.get(target_key, None)
    if not (torch.is_tensor(pred) and torch.is_tensor(target)):
        return None, {}

    pred, target = _align_future_bev_feature_pair(pred, target)
    valid_mask = _align_future_bev_valid_mask(valid_mask, pred)
    spatial_mask = _align_future_bev_spatial_mask(spatial_mask, pred)
    abs_loss = _masked_smooth_l1_loss(pred, target, valid_mask, spatial_mask)
    cos_loss = (
        _future_bev_cosine_loss(pred, target, valid_mask, spatial_mask)
        if cos_weight > 0.0
        else pred.new_zeros(())
    )
    total = abs_loss + float(cos_weight) * cos_loss

    res_pred = predictions.get(res_pred_key, None)
    res_target = predictions.get(res_target_key, None)
    if torch.is_tensor(res_pred) and torch.is_tensor(res_target) and float(res_weight) > 0.0:
        res_pred, res_target = _align_future_bev_feature_pair(
            res_pred,
            res_target,
        )
        res_valid_mask = _align_future_bev_valid_mask(
            res_valid_mask,
            res_pred,
        )
        res_spatial_mask = _align_future_bev_spatial_mask(
            res_spatial_mask,
            res_pred,
        )
        res_loss = _masked_smooth_l1_loss(res_pred, res_target, res_valid_mask, res_spatial_mask)
        total = total + float(res_weight) * res_loss
    else:
        res_loss = pred.new_zeros(())
        res_valid_mask = valid_mask

    monitors = {
        "abs": abs_loss.detach(),
        "res": res_loss.detach(),
        "cos": cos_loss.detach(),
        "valid_steps": valid_mask.detach().float().sum(),
        "res_valid_steps": res_valid_mask.detach().float().sum(),
    }
    return total, monitors


def compute_future_bev_feature_loss(
    targets: Dict[str, torch.Tensor],
    predictions: Dict[str, torch.Tensor],
    config: Any,
) -> Dict[str, torch.Tensor]:
    """Align the future field to detached targets throughout joint training."""
    del targets
    res_weight = float(config.future_residual_loss_weight)
    cos_weight = float(
        getattr(config, "future_cosine_loss_weight", 0.02)
    )
    valid_mask = predictions.get("future_bev_feature_valid_mask", None)
    res_valid_mask = predictions.get(
        "future_bev_residual_feature_valid_mask",
        valid_mask,
    )
    spatial_mask = predictions.get(
        "future_bev_feature_spatial_valid_mask",
        None,
    )
    res_spatial_mask = predictions.get(
        "future_bev_residual_feature_spatial_valid_mask",
        spatial_mask,
    )

    feature_loss, feature_monitors = _future_bev_feature_pair_loss(
        predictions,
        "future_bev_absolute_feature_prediction",
        "future_bev_absolute_feature_target",
        "future_bev_residual_feature_prediction",
        "future_bev_residual_feature_target",
        res_weight=res_weight,
        cos_weight=cos_weight,
        valid_mask=valid_mask,
        res_valid_mask=res_valid_mask,
        spatial_mask=spatial_mask,
        res_spatial_mask=res_spatial_mask,
    )

    if feature_loss is None:
        return {}

    base_weight = float(
        getattr(config, "future_bev_loss_weight", 0.5)
    )
    weighted_feature_loss = base_weight * feature_loss
    losses = {
        "future_bev_feature_loss": weighted_feature_loss,
        "future_bev_feature_loss_raw_monitor": feature_loss.detach(),
        "future_bev_feature_loss_weight_monitor": feature_loss.new_tensor(
            base_weight
        ),
        "future_bev_absolute_feature_loss_monitor": feature_monitors["abs"],
        "future_bev_residual_feature_loss_monitor": feature_monitors["res"],
        "future_bev_cosine_loss_monitor": feature_monitors["cos"],
        "future_bev_residual_valid_steps_monitor": feature_monitors["res_valid_steps"],
        "future_bev_valid_steps_monitor": feature_monitors["valid_steps"],
    }
    return losses


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean', epsilon=1e-6):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.epsilon = epsilon

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        ce_loss = ce_loss.clamp(max=10.0)
        pt = torch.exp(-ce_loss).clamp(min=self.epsilon, max=1.0 - self.epsilon)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = torch.full_like(targets, fill_value=self.alpha, dtype=torch.float32)
            else:
                alpha_t = torch.tensor(self.alpha, dtype=torch.float32).to(inputs.device)[targets]
            focal_loss = alpha_t * focal_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def compute_verifier_loss(
    targets: Dict[str, torch.Tensor],
    predictions: Dict[str, torch.Tensor],
    config: Any,
) -> Dict[str, torch.Tensor]:
    """
    Train the Future BEV / evidence-probe / Metric Prism verifier.
    """
    losses = {}

    # Labels are evaluated on this step's current-model candidate set.
    online_factor_labels = predictions.get(
        "verifier_online_pool_pdm_factor_labels",
        None,
    )
    if torch.is_tensor(online_factor_labels):
        teacher_factor_labels = torch.nan_to_num(
            online_factor_labels.detach().float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        teacher_factor_labels = teacher_factor_labels.clamp(0.0, 1.0)
        online_pdm_scores = predictions.get(
            "verifier_online_pool_pdm_scores",
            None,
        )
        if torch.is_tensor(online_pdm_scores):
            formula_pdms = _compose_pdm_scores_from_factors(
                teacher_factor_labels
            ).detach()
            online_pdm_scores = torch.nan_to_num(
                online_pdm_scores.detach().float(),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).clamp(0.0, 1.0)
            losses["verifier_online_pool_pdm_mean_monitor"] = (
                online_pdm_scores.mean().detach()
            )
            losses["verifier_online_pool_pdm_formula_absdiff_monitor"] = (
                formula_pdms - online_pdm_scores
            ).abs().mean().detach()
    else:
        teacher_factor_labels = None

    if teacher_factor_labels is not None:
        if "nc_logits" in predictions:
            metric_prism_loss, metric_prism_monitors = (
                compute_metric_prism_factor_loss(
                    teacher_factor_labels,
                    predictions,
                )
            )
            losses.update(metric_prism_monitors)
            metric_prism_weight = float(
                getattr(config, "metric_prism_loss_weight", 10.0)
            )
            if metric_prism_weight < 0.0:
                raise ValueError(
                    "metric_prism_loss_weight must be non-negative."
                )
            losses["metric_prism_pdm_factor_loss"] = (
                metric_prism_weight * metric_prism_loss
            )
            losses["metric_prism_pdm_factor_loss_monitor"] = (
                metric_prism_loss.detach()
            )
            losses["metric_prism_loss_weight_monitor"] = (
                metric_prism_loss.new_tensor(metric_prism_weight)
            )

    losses.update(
        compute_future_bev_feature_loss(
            targets,
            predictions,
            config,
        )
    )

    return losses
