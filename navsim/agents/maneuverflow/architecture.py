"""Paper-aligned module hierarchy for ManeuverFlow.

The paper describes one policy with three semantic stages:

``Hypothesize -> Verify -> Improve``.

These containers make that structure explicit in both Python attribute access
and the serialized ``state_dict``. They intentionally contain no compatibility
aliases. ``checkpoint_schema.py`` validates the already-canonical
``maneuverflow.paper.v1`` envelope and performs no namespace conversion.
"""

from __future__ import annotations

import torch.nn as nn


class PlanningContextEncoder(nn.Module):
    """TransFuser perception and planning-context tokenization."""

    def __init__(self) -> None:
        super().__init__()


class ManeuverGaussianMixturePrior(nn.Module):
    """Maneuver Gaussian Mixture Prior (M-GMP) statistics."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("displacement_mean", None)
        self.register_buffer("displacement_std", None)
        self.register_buffer("heading_mean", None)
        self.register_buffer("heading_std", None)
        self.register_buffer("component_counts", None)


class ManeuverMixtureMeanFlowGenerator(nn.Module):
    """Maneuver-Mixture MeanFlow trajectory generator (M³F)."""

    def __init__(self) -> None:
        super().__init__()


class ManeuverFlowHypothesizer(nn.Module):
    """Generate maneuver-conditioned trajectory hypotheses."""

    def __init__(self) -> None:
        super().__init__()
        self.perception = PlanningContextEncoder()
        self.maneuver_gmp = ManeuverGaussianMixturePrior()
        # This registered attribute is the immutable paper-v1 checkpoint
        # namespace; the module type carries the full M³F paper name.
        self.meanflow_generator = ManeuverMixtureMeanFlowGenerator()
        self.register_parameter("proposal_anchor_bank", None)
        self.register_buffer("maneuver_anchor_bank", None)


class ManeuverFlowImprover(nn.Module):
    """Evidence-coupled trajectory improvement stage."""

    def __init__(self) -> None:
        super().__init__()


__all__ = [
    "ManeuverFlowHypothesizer",
    "ManeuverFlowImprover",
    "ManeuverGaussianMixturePrior",
    "ManeuverMixtureMeanFlowGenerator",
    "PlanningContextEncoder",
]
