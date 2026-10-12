"""Same-component supervision for joint ECFR training."""

from typing import Dict

import torch

from .frenet_geometry import trajectory_to_frenet_residual


def build_same_component_teacher_supervision(
    *,
    base_hypotheses: torch.Tensor,
    base_pdm_scores: torch.Tensor,
    base_maneuver_component_indices: torch.Tensor,
    candidate_hypotheses: torch.Tensor,
    candidate_pdm_scores: torch.Tensor,
    candidate_maneuver_component_indices: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Pair each base with its best finite same-component candidate teacher.

    Teacher lookup is independent for every base. A base with invalid data or
    without a finite candidate in its maneuver component receives a zero
    target and zero validity weight instead of aborting the full batch.
    """

    if base_hypotheses.dim() != 4 or base_hypotheses.shape[-2:] != (8, 3):
        raise ValueError(
            "base_hypotheses must be [B, R, 8, 3], got "
            f"{tuple(base_hypotheses.shape)}."
        )
    if candidate_hypotheses.dim() != 4 or candidate_hypotheses.shape[-2:] != (8, 3):
        raise ValueError(
            "candidate_hypotheses must be [B, P, 8, 3], got "
            f"{tuple(candidate_hypotheses.shape)}."
        )

    batch_size, base_count = base_hypotheses.shape[:2]
    pool_batch_size, pool_count = candidate_hypotheses.shape[:2]
    if int(pool_batch_size) != int(batch_size):
        raise ValueError(
            "base_hypotheses and candidate_hypotheses must share batch size."
        )
    if base_pdm_scores.shape != (batch_size, base_count):
        raise ValueError(
            "base_pdm_scores must be [B, R], got "
            f"{tuple(base_pdm_scores.shape)}."
        )
    if base_maneuver_component_indices.shape != (batch_size, base_count):
        raise ValueError(
            "base_maneuver_component_indices must be [B, R], got "
            f"{tuple(base_maneuver_component_indices.shape)}."
        )
    if candidate_pdm_scores.shape != (batch_size, pool_count):
        raise ValueError(
            "candidate_pdm_scores must be [B, P], got "
            f"{tuple(candidate_pdm_scores.shape)}."
        )
    if candidate_maneuver_component_indices.shape != (batch_size, pool_count):
        raise ValueError(
            "candidate_maneuver_component_indices must be [B, P], got "
            f"{tuple(candidate_maneuver_component_indices.shape)}."
        )

    device = base_hypotheses.device
    dtype = base_hypotheses.dtype
    detached_bases = base_hypotheses.detach()
    detached_base_scores = base_pdm_scores.detach().to(device=device)
    base_components = base_maneuver_component_indices.detach().to(
        device=device,
        dtype=torch.long,
    )
    candidate_trajectories = candidate_hypotheses.detach().to(
        device=device,
        dtype=dtype,
    )
    candidate_scores = candidate_pdm_scores.detach().to(
        device=device,
        dtype=detached_base_scores.dtype,
    )
    candidate_components = candidate_maneuver_component_indices.detach().to(
        device=device,
        dtype=torch.long,
    )

    finite_base = (
        torch.isfinite(detached_bases).flatten(start_dim=-2).all(dim=-1)
        & torch.isfinite(detached_base_scores)
    )
    finite_candidates = (
        torch.isfinite(candidate_trajectories)
        .flatten(start_dim=-2)
        .all(dim=-1)
        & torch.isfinite(candidate_scores)
    )
    component_match = candidate_components[:, None, :].eq(
        base_components[:, :, None]
    )
    eligible = finite_candidates[:, None, :] & component_match
    has_teacher = eligible.any(dim=-1)
    valid_teacher = finite_base & has_teacher

    ranked_scores = candidate_scores[:, None, :].expand(
        -1,
        base_count,
        -1,
    ).masked_fill(~eligible, -float("inf"))
    selected_indices = ranked_scores.argmax(dim=-1)
    batch_indices = torch.arange(batch_size, device=device)[:, None]
    gathered_hypotheses = candidate_trajectories[
        batch_indices,
        selected_indices,
    ]
    gathered_scores = candidate_scores[batch_indices, selected_indices]

    safe_bases = torch.where(
        finite_base[..., None, None],
        detached_bases,
        torch.zeros_like(detached_bases),
    )
    teacher_hypotheses = torch.where(
        valid_teacher[..., None, None],
        gathered_hypotheses,
        safe_bases,
    )
    safe_base_scores = torch.where(
        torch.isfinite(detached_base_scores),
        detached_base_scores,
        torch.zeros_like(detached_base_scores),
    )
    teacher_scores = torch.where(
        valid_teacher,
        gathered_scores,
        safe_base_scores,
    )
    teacher_indices = torch.where(
        valid_teacher,
        selected_indices,
        torch.full_like(selected_indices, -1),
    )

    target_frenet_residuals = trajectory_to_frenet_residual(
        safe_bases,
        teacher_hypotheses,
    )
    supervision_weight = valid_teacher.to(dtype=dtype)

    return {
        "teacher_hypotheses": teacher_hypotheses,
        "target_frenet_residuals": target_frenet_residuals,
        "teacher_pdm_scores": teacher_scores,
        "base_pdm_scores": safe_base_scores,
        "supervision_weight": supervision_weight,
        "teacher_hypothesis_indices": teacher_indices,
        "valid_teacher_mask": valid_teacher,
        "base_maneuver_component_indices": base_components,
    }


__all__ = ["build_same_component_teacher_supervision"]
