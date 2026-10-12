"""Typed, paper-facing output contract for ManeuverFlow inference."""

from __future__ import annotations

from typing import Dict, Mapping, Optional, TypedDict, cast

import torch


class ManeuverFlowInferenceOutput(TypedDict, total=False):
    """Stable tensors returned directly by :meth:`ManeuverFlowModel.inference`."""

    maneuver_component_logits: torch.Tensor
    selected_maneuver_component_indices: torch.Tensor
    base_hypothesis_maneuver_component_indices: torch.Tensor
    selected_maneuver_component_probabilities: torch.Tensor
    mgmp_hypothesis_counts: torch.Tensor
    mgmp_source_states: torch.Tensor
    generated_base_hypotheses: torch.Tensor
    generated_base_hypotheses_flow_state: torch.Tensor
    retained_base_hypotheses: torch.Tensor
    retained_base_hypotheses_flow_state: torch.Tensor
    refined_hypotheses: torch.Tensor
    refined_hypotheses_flow_state: torch.Tensor
    final_hypotheses: torch.Tensor
    final_hypothesis_scores: torch.Tensor
    final_hypothesis_scores_raw: torch.Tensor
    final_hypothesis_is_refined: torch.Tensor
    final_hypothesis_source_base_indices: torch.Tensor
    retained_base_hypothesis_scores: torch.Tensor
    retained_base_hypothesis_scores_raw: torch.Tensor
    refined_hypothesis_scores: torch.Tensor
    refined_hypothesis_scores_raw: torch.Tensor
    base_pdm_factor_scores: torch.Tensor
    refined_pdm_factor_scores: torch.Tensor
    final_pdm_factor_scores: torch.Tensor
    ecfr_frenet_residuals: torch.Tensor
    ecfr_delta_cost: torch.Tensor
    retained_base_hypothesis_indices: torch.Tensor
    retained_base_hypothesis_mask: torch.Tensor
    ecfr_source_hypothesis_indices: torch.Tensor
    ecfr_source_hypothesis_mask: torch.Tensor


def validate_maneuverflow_inference_output(
    outputs: Dict[str, torch.Tensor],
    *,
    expected_hypothesis_counts: Optional[Mapping[str, int]] = None,
) -> ManeuverFlowInferenceOutput:
    """Validate the direct paper-facing output dictionary without renaming it."""

    required = ("final_hypotheses", "final_hypothesis_scores")
    missing = [name for name in required if name not in outputs]
    if missing:
        raise KeyError(
            "ManeuverFlow inference did not produce required contract fields: "
            f"{missing}."
        )

    hypotheses = outputs["final_hypotheses"]
    scores = outputs["final_hypothesis_scores"]
    if (
        not torch.is_tensor(hypotheses)
        or hypotheses.ndim != 4
        or hypotheses.shape[-1] != 3
    ):
        raise ValueError(
            "final_hypotheses must have shape [B, N, T, 3], got "
            f"{getattr(hypotheses, 'shape', None)}."
        )
    if (
        not torch.is_tensor(scores)
        or scores.ndim != 2
        or scores.shape[:2] != hypotheses.shape[:2]
    ):
        raise ValueError(
            "final_hypothesis_scores must have shape [B, N] aligned with "
            f"final_hypotheses, got {getattr(scores, 'shape', None)}."
        )
    if not bool(torch.isfinite(hypotheses).all()):
        raise ValueError("final_hypotheses must contain only finite values.")
    if not bool(torch.isfinite(scores).all()):
        raise ValueError(
            "final_hypothesis_scores must contain only finite values."
        )

    if expected_hypothesis_counts is not None:
        for name, expected_count in expected_hypothesis_counts.items():
            value = outputs.get(name)
            if not torch.is_tensor(value):
                raise KeyError(
                    f"ManeuverFlow inference contract requires {name!r}."
                )
            if value.ndim < 2:
                raise ValueError(
                    f"{name} must carry a hypothesis axis at dimension 1, "
                    f"got shape {tuple(value.shape)}."
                )
            actual_count = int(value.shape[1])
            if actual_count != int(expected_count):
                raise ValueError(
                    f"{name} must contain {int(expected_count)} hypotheses, "
                    f"got {actual_count}."
                )

    return cast(ManeuverFlowInferenceOutput, outputs)


__all__ = [
    "ManeuverFlowInferenceOutput",
    "validate_maneuverflow_inference_output",
]
