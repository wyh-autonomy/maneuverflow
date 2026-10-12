"""Lazy public exports for the ManeuverFlow model and its reusable modules."""

from __future__ import annotations

from importlib import import_module
from typing import Dict, Tuple


_EXPORTS: Dict[str, Tuple[str, str]] = {
    "EvidenceCoupledFrenetRefiner": (".ecfr", "EvidenceCoupledFrenetRefiner"),
    "FutureBEVModel": (".future_bev_model", "FutureBEVModel"),
    "ManeuverFlowAgent": (".agent", "ManeuverFlowAgent"),
    "ManeuverFlowConfig": (".config", "ManeuverFlowConfig"),
    "ManeuverFlowHypothesizer": (".architecture", "ManeuverFlowHypothesizer"),
    "ManeuverFlowImprover": (".architecture", "ManeuverFlowImprover"),
    "ManeuverFlowInferenceOutput": (".inference_api", "ManeuverFlowInferenceOutput"),
    "ManeuverFlowModel": (".model", "ManeuverFlowModel"),
    "ManeuverFlowVerifier": (".verifier", "ManeuverFlowVerifier"),
    "ManeuverGaussianMixturePrior": (
        ".architecture",
        "ManeuverGaussianMixturePrior",
    ),
    "ManeuverMixtureMeanFlowGenerator": (
        ".architecture",
        "ManeuverMixtureMeanFlowGenerator",
    ),
    "ManeuverMixtureMeanFlowField": (".m3f", "ManeuverMixtureMeanFlowField"),
    "MetricPrism": (".metric_prism", "MetricPrism"),
    "PlanningContextEncoder": (".architecture", "PlanningContextEncoder"),
    "TrajectoryGuidedBEVFeatureProbe": (
        ".trajectory_evidence_probe",
        "TrajectoryGuidedBEVFeatureProbe",
    ),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
