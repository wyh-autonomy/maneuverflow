"""ManeuverFlow verification orchestration."""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from .future_bev_model import FutureBEVModel
from .metric_prism import (
    MetricPrism,
    non_collision_score,
)
from .trajectory_evidence_probe import TrajectoryGuidedBEVFeatureProbe


class ManeuverFlowVerifier(nn.Module):
    """Predict shared future evidence and score trajectory hypotheses."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.future_bev_model = FutureBEVModel(config)
        self.trajectory_evidence_probe = (
            TrajectoryGuidedBEVFeatureProbe(config)
        )
        self.metric_prism = self._build_metric_prism(config)
        self.register_parameter("proposal_anchor_bank", None)
        self.register_buffer("maneuver_anchor_indices", None)

    @staticmethod
    def _build_metric_prism(config: Any) -> nn.Module:
        return MetricPrism(config)

    def encode_future_bev_targets(
        self,
        future_bev_high: Optional[torch.Tensor],
        future_bev_low: torch.Tensor,
        current_bev_high: Optional[torch.Tensor] = None,
        current_bev_low: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Delegate construction of stop-gradient future-BEV targets."""

        return self.future_bev_model.encode_future_bev_targets(
            future_bev_high,
            future_bev_low,
            current_bev_high=current_bev_high,
            current_bev_low=current_bev_low,
        )

    def rollout_world(
        self,
        planning_context: Dict[str, torch.Tensor],
        future_bev_targets: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
    ]:
        """Predict the shared future-BEV evidence field once."""

        return self.future_bev_model.rollout_world(
            planning_context,
            future_bev_targets=future_bev_targets,
        )

    def score_evidence(
        self,
        evidence_outputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Estimate the five PDM-aligned factors from trajectory evidence."""

        factor_logits = self.metric_prism(evidence_outputs)
        if factor_logits.shape[-1] != 5:
            raise ValueError(
                "Metric Prism must emit five continuous factor logits, "
                f"got {tuple(factor_logits.shape)}"
            )
        non_collision_logits = factor_logits[:, :, 0:1]
        drivable_area_compliance_logit = factor_logits[:, :, 1]
        ego_progress_output_logits = factor_logits[:, :, 2]
        ego_progress_prediction = torch.sigmoid(
            ego_progress_output_logits
        )
        time_to_collision_logit = factor_logits[:, :, 3]
        comfort_logit = factor_logits[:, :, 4]
        binary_factor_logits = torch.stack(
            [
                drivable_area_compliance_logit,
                time_to_collision_logit,
                comfort_logit,
            ],
            dim=1,
        )
        binary_factor_scores = torch.sigmoid(binary_factor_logits)
        pdm_factor_scores = torch.stack(
            [
                non_collision_score(non_collision_logits),
                binary_factor_scores[:, 0, :],
                ego_progress_prediction,
                binary_factor_scores[:, 1, :],
                binary_factor_scores[:, 2, :],
            ],
            dim=1,
        )

        return {
            "pdm_factor_scores": pdm_factor_scores,
            "alpha_time_shared": evidence_outputs["alpha_time_shared"],
            "nc_logits": non_collision_logits,
            "binary_pdm_factor_logits": binary_factor_logits,
            "ep_logits": ego_progress_output_logits,
            "ep_pred": ego_progress_prediction,
        }

    def evaluate_hypotheses(
        self,
        trajectory_hypotheses: torch.Tensor,
        state_navigation_feature: torch.Tensor,
        evidence_field: Dict[str, torch.Tensor],
        maneuver_component_weights: Optional[torch.Tensor] = None,
        prepared_evidence_field: Optional[
            Dict[str, torch.Tensor]
        ] = None,
    ) -> Dict[str, torch.Tensor]:
        """Probe and jointly score a set of trajectory hypotheses."""

        evidence_outputs = self.trajectory_evidence_probe(
            trajectory_hypotheses,
            state_navigation_feature,
            evidence_field,
            maneuver_component_weights=maneuver_component_weights,
            prepared_evidence_field=prepared_evidence_field,
        )
        return self.score_evidence(evidence_outputs)
