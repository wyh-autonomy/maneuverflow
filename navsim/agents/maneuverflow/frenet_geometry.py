"""Frenet-frame geometry used by Evidence-Coupled Frenet Refinement (ECFR).

The functions in this module define the paper-facing trajectory refinement
space.  ECFR predicts bounded longitudinal, lateral, and heading residuals
around each generated hypothesis while all inputs and outputs remain physical
ego-frame trajectories with layout ``[..., time, (x, y, heading)]``.
"""

from typing import Tuple

import torch


def _wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    """Wrap angles to ``[-pi, pi]`` without changing tensor shape."""
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def compute_point_frenet_frame(
    trajectory_phys: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build pointwise tangent, normal, and tangent heading for a trajectory.

    When planar motion is nearly zero, the trajectory heading supplies the
    tangent so that the local Frenet frame remains well-defined for stopped or
    slowly moving hypotheses.

    Args:
        trajectory_phys: Physical trajectory with final dimension
            ``(x, y, heading)``.

    Returns:
        Pointwise unit tangent, unit normal, and tangent heading tensors.
    """
    if trajectory_phys.shape[-1] < 3:
        raise ValueError(
            "compute_point_frenet_frame expects last dim >= 3, "
            f"got {tuple(trajectory_phys.shape)}"
        )

    trajectory_xy = trajectory_phys[..., :2]
    trajectory_heading = trajectory_phys[..., 2]

    previous_xy = torch.cat(
        [trajectory_xy[..., :1, :], trajectory_xy[..., :-1, :]],
        dim=-2,
    )
    next_xy = torch.cat(
        [trajectory_xy[..., 1:, :], trajectory_xy[..., -1:, :]],
        dim=-2,
    )
    tangent_raw = next_xy - previous_xy
    tangent_norm = tangent_raw.norm(dim=-1, keepdim=True)
    tangent = tangent_raw / tangent_norm.clamp_min(1e-6)

    heading_vector = torch.stack(
        [torch.cos(trajectory_heading), torch.sin(trajectory_heading)],
        dim=-1,
    )
    use_heading_fallback = tangent_norm < 1e-3
    tangent = torch.where(
        use_heading_fallback.expand_as(tangent),
        heading_vector,
        tangent,
    )
    tangent = tangent / tangent.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    normal = torch.stack([-tangent[..., 1], tangent[..., 0]], dim=-1)
    tangent_heading = torch.atan2(tangent[..., 1], tangent[..., 0])
    return tangent, normal, tangent_heading


def trajectory_to_frenet_residual(
    base_trajectory_phys: torch.Tensor,
    target_trajectory_phys: torch.Tensor,
) -> torch.Tensor:
    """Project a target trajectory into base-hypothesis Frenet residuals.

    The output channels are ``(ds, dl, dpsi)``.  Heading residual ``dpsi`` is
    measured relative to the base hypothesis heading rather than its geometric
    tangent.  Consequently, a zero residual is an exact physical no-op even
    when the predicted heading is not perfectly tangent-aligned.
    """
    base_xy = base_trajectory_phys[..., :2]
    target_xy = target_trajectory_phys[..., :2]
    tangent, normal, _ = compute_point_frenet_frame(base_trajectory_phys)

    position_residual = target_xy - base_xy
    longitudinal_residual = (position_residual * tangent).sum(dim=-1)
    lateral_residual = (position_residual * normal).sum(dim=-1)
    heading_residual = _wrap_angle(
        target_trajectory_phys[..., 2] - base_trajectory_phys[..., 2]
    )
    return torch.stack(
        [longitudinal_residual, lateral_residual, heading_residual],
        dim=-1,
    )


def frenet_residual_to_trajectory(
    base_trajectory_phys: torch.Tensor,
    frenet_residual: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply ``(ds, dl, dpsi)`` residuals to a base trajectory hypothesis.

    Returns:
        The refined physical trajectory, its physical-space residual relative
        to the base hypothesis, and the refined planar tangent heading.  The
        last value is exposed for diagnostics and does not alter refinement.
    """
    if frenet_residual.shape[-1] != 3:
        raise ValueError(
            "frenet_residual_to_trajectory expects last dim == 3, "
            f"got {tuple(frenet_residual.shape)}"
        )

    base_xy = base_trajectory_phys[..., :2]
    base_heading = base_trajectory_phys[..., 2]
    tangent, normal, _ = compute_point_frenet_frame(base_trajectory_phys)

    longitudinal_residual = frenet_residual[..., 0:1]
    lateral_residual = frenet_residual[..., 1:2]
    heading_residual = frenet_residual[..., 2]

    refined_xy = (
        base_xy
        + longitudinal_residual * tangent
        + lateral_residual * normal
    )
    refined_heading = _wrap_angle(base_heading + heading_residual)
    refined_trajectory = torch.cat(
        [refined_xy, refined_heading.unsqueeze(-1)],
        dim=-1,
    )
    _, _, refined_tangent_heading = compute_point_frenet_frame(
        refined_trajectory
    )

    position_residual = refined_xy - base_xy
    physical_heading_residual = _wrap_angle(refined_heading - base_heading)
    trajectory_residual = torch.cat(
        [position_residual, physical_heading_residual.unsqueeze(-1)],
        dim=-1,
    )
    return refined_trajectory, trajectory_residual, refined_tangent_heading
