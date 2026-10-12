"""Validation and integrity helpers for released ManeuverFlow checkpoints."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Iterable, Literal, Mapping

import torch

from .config import (
    DEFAULT_ECFR_FINAL_POOL_COMPOSITION,
    MANEUVERFLOW_INFERENCE_POLICY,
    MANEUVERFLOW_TRAINING_RECIPE,
)

CHECKPOINT_SCHEMA_NAME = "maneuverflow_inference_state_dict"
CHECKPOINT_FORMAT_VERSION = 3
CHECKPOINT_KEY_SCHEMA = "maneuverflow.paper.v1"
CHECKPOINT_MODEL_NAME = "ManeuverFlow"
CHECKPOINT_ENVELOPE_KEYS = frozenset(
    {
        "format",
        "format_version",
        "key_schema",
        "model",
        "state_dict",
        "state_dict_summary",
        "inference_protocol",
    }
)

PAPER_V1_SCHEMA = CHECKPOINT_KEY_SCHEMA
EMA_TRAINING_SCHEMA = "maneuverflow.ema.training.v1"
EMA_TARGET_PREFIX = "model.ema_target_encoder."
EMA_ENCODER_PREFIX = EMA_TARGET_PREFIX + "encoder."
ONLINE_ENCODER_PREFIX = "model.hypothesizer.perception.transfuser_backbone."
EMA_METADATA_KEYS = frozenset({EMA_TARGET_PREFIX + "decay", EMA_TARGET_PREFIX + "num_updates"})
STAGE1_SOURCE_RECIPES = frozenset({MANEUVERFLOW_TRAINING_RECIPE})
STAGE1_LINEAGE_KEYS = frozenset({"stage1_source_recipe", "stage1_source_tensor_sha256"})
UNKNOWN_SCHEMA = "unknown"
StateDictSchema = Literal["maneuverflow.paper.v1", "unknown"]

_CANONICAL_KEY_PREFIXES = (
    "model.hypothesizer.",
    "model.verifier.",
    "model.improver.",
)


class CheckpointSchemaError(ValueError):
    """Raised when a checkpoint violates the released ManeuverFlow schema."""


class StateDictValidationError(CheckpointSchemaError):
    """Raised when checkpoint tensors do not exactly match a model."""


def validate_stage1_lineage_metadata(metadata: Mapping[str, object]) -> dict[str, str]:
    """Require the paired Stage-1 identity for joint checkpoints."""

    present = STAGE1_LINEAGE_KEYS.intersection(metadata)
    if not present and metadata.get("training_phase") != "joint_finetuning":
        return {}
    recipe = metadata.get("stage1_source_recipe")
    digest = metadata.get("stage1_source_tensor_sha256")
    if (
        present != STAGE1_LINEAGE_KEYS
        or not isinstance(recipe, str)
        or recipe not in STAGE1_SOURCE_RECIPES
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise CheckpointSchemaError("Stage-1 lineage requires an explicit source recipe and canonical tensor SHA256.")
    return {"stage1_source_recipe": recipe, "stage1_source_tensor_sha256": digest}


def validate_optimizer_metadata(metadata: Mapping[str, object]) -> dict[str, object]:
    """Validate the optimizer and scheduler identity recorded during training."""
    controls = {}
    for name in ("learning_rate", "min_lr", "weight_decay"):
        value = metadata.get(name)
        if type(value) not in {int, float} or not math.isfinite(value) or value < 0:
            raise CheckpointSchemaError(f"Training optimizer metadata requires finite non-negative {name}.")
        if name == "learning_rate" and value == 0:
            raise CheckpointSchemaError("Training optimizer metadata requires positive learning_rate.")
        controls[name] = float(value)
    warmup = metadata.get("warmup_epochs")
    duration = metadata.get("max_epochs")
    if type(duration) is not int or duration < 1 or type(warmup) is not int or warmup < 0:
        raise CheckpointSchemaError("Training optimizer metadata requires non-negative warmup_epochs and positive max_epochs.")
    controls["warmup_epochs"] = warmup
    return controls


def detect_state_dict_schema(keys: Iterable[object]) -> StateDictSchema:
    """Identify a non-empty state dict in the released paper namespace.

    There are no alternate or mixed-schema states. Any non-canonical key
    makes the complete state dictionary unknown and therefore unloadable.
    Exact parameter membership is checked separately against the instantiated
    model by :func:`validate_state_dict_against_model`.
    """

    has_key = False
    for key in keys:
        has_key = True
        if not isinstance(key, str) or not any(
            key.startswith(prefix) and len(key) > len(prefix)
            for prefix in _CANONICAL_KEY_PREFIXES
        ):
            return UNKNOWN_SCHEMA
    return PAPER_V1_SCHEMA if has_key else UNKNOWN_SCHEMA


def detect_training_state_dict_schema(keys: Iterable[object]) -> str:
    """Recognize the explicit EMA namespace without broadening release keys."""

    keys = tuple(keys)
    online_keys = []
    teacher_keys = []
    for key in keys:
        if not isinstance(key, str):
            return UNKNOWN_SCHEMA
        if key.startswith(EMA_TARGET_PREFIX):
            if key not in EMA_METADATA_KEYS and not (
                key.startswith(EMA_ENCODER_PREFIX) and len(key) > len(EMA_ENCODER_PREFIX)
            ):
                return UNKNOWN_SCHEMA
            teacher_keys.append(key)
        else:
            online_keys.append(key)
    if detect_state_dict_schema(online_keys) != PAPER_V1_SCHEMA:
        return UNKNOWN_SCHEMA
    if not teacher_keys:
        return PAPER_V1_SCHEMA
    return EMA_TRAINING_SCHEMA


def validate_ema_training_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    *,
    expected_decay: float | None = None,
) -> dict[str, torch.Tensor]:
    """Validate every teacher tensor before returning the online release state.

    Exact backbone parity prevents a misspelled or partial training namespace
    from being silently discarded during export or full-state resume.
    """

    validate_finite_state_dict(state_dict)
    if detect_training_state_dict_schema(state_dict) != EMA_TRAINING_SCHEMA:
        raise StateDictValidationError("EMA state must use only the explicit online/teacher training namespace.")
    online = {key: tensor for key, tensor in state_dict.items() if not key.startswith(EMA_TARGET_PREFIX)}
    online_encoder = {
        key[len(ONLINE_ENCODER_PREFIX):]: tensor
        for key, tensor in online.items() if key.startswith(ONLINE_ENCODER_PREFIX)
    }
    teacher_encoder = {
        key[len(EMA_ENCODER_PREFIX):]: tensor
        for key, tensor in state_dict.items() if key.startswith(EMA_ENCODER_PREFIX)
    }
    if not online_encoder:
        raise StateDictValidationError("EMA training state must include the online backbone.")
    try:
        validate_state_dict_against_model(teacher_encoder, online_encoder)
    except StateDictValidationError as error:
        raise StateDictValidationError(f"EMA teacher backbone mismatch: {error}") from error
    if any(tensor.is_floating_point() and tensor.dtype != torch.float32 for tensor in teacher_encoder.values()):
        raise StateDictValidationError("EMA teacher floating tensors must use FP32.")
    counter = state_dict.get(EMA_TARGET_PREFIX + "num_updates")
    if not torch.is_tensor(counter) or counter.shape != torch.Size([]) or counter.dtype != torch.int64 or counter.item() < 0:
        raise StateDictValidationError("EMA num_updates must be a nonnegative scalar int64 tensor.")
    decay = state_dict.get(EMA_TARGET_PREFIX + "decay")
    if not torch.is_tensor(decay) or decay.shape != torch.Size([]) or decay.dtype != torch.float64 or not 0.0 <= float(decay.item()) < 1.0:
        raise StateDictValidationError("EMA decay must be a scalar float64 tensor in [0, 1).")
    if expected_decay is not None and float(decay.item()) != float(expected_decay):
        raise StateDictValidationError(
            f"EMA decay mismatch: checkpoint={decay.item()}, expected={expected_decay}."
        )
    return online


def validate_stage1_state_dict_against_model(
    state_dict: Mapping[str, torch.Tensor],
    model_state_dict: Mapping[str, torch.Tensor],
) -> None:
    """Allow only missing EMA teacher keys at the explicit Stage-1 boundary."""

    if detect_state_dict_schema(state_dict) != PAPER_V1_SCHEMA:
        raise StateDictValidationError("Stage-1 weights must contain only the online release namespace.")
    if detect_training_state_dict_schema(model_state_dict) not in {PAPER_V1_SCHEMA, EMA_TRAINING_SCHEMA}:
        raise StateDictValidationError("Stage-1 target model has an unknown training namespace.")
    online_model = {
        key: tensor for key, tensor in model_state_dict.items()
        if not key.startswith(EMA_TARGET_PREFIX)
    }
    validate_state_dict_against_model(state_dict, online_model)


def summarize_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, int]:
    """Return the deterministic tensor summary stored in the release envelope."""

    _validate_tensor_mapping(state_dict, label="Checkpoint state_dict")
    return {
        "num_tensors": len(state_dict),
        "num_elements": sum(tensor.numel() for tensor in state_dict.values()),
        "tensor_bytes": sum(
            tensor.numel() * tensor.element_size()
            for tensor in state_dict.values()
        ),
    }


def inference_protocol_from_config(config: object) -> dict[str, object]:
    """Build the canonical paper inference protocol from a runtime config."""

    protocol = {
        "lifecycle": "Hypothesize--Verify--Improve",
        "num_base_hypotheses": int(
            getattr(config, "inference_num_base_hypotheses")
        ),
        "mgmp_top_k": int(getattr(config, "inference_mgmp_top_k")),
        "num_refined_hypotheses": int(
            getattr(config, "inference_num_refined_hypotheses")
        ),
        "num_retained_base_hypotheses": int(
            getattr(config, "inference_num_retained_base_hypotheses")
        ),
        "num_final_hypotheses": int(
            getattr(config, "inference_num_base_hypotheses")
        ),
        "future_bev_feature_dimension": int(
            getattr(config, "future_bev_feature_dimension")
        ),
        "metric_prism_feature_dimension": int(
            getattr(config, "metric_prism_feature_dimension")
        ),
        "ecfr_feature_dimension": int(getattr(config, "ecfr_d_model")),
        "ecfr_bounds": [
            float(getattr(config, "ecfr_max_ds")),
            float(getattr(config, "ecfr_max_dl")),
            float(getattr(config, "ecfr_max_dpsi")),
        ],
        "ecfr_final_pool_composition": str(
            getattr(
                config,
                "ecfr_final_pool_composition",
                DEFAULT_ECFR_FINAL_POOL_COMPOSITION,
            )
        ),
        "score_composition": "fixed log-PDM",
        "refined_score_delta_cost": {
            "xy": float(getattr(
                config,
                "refined_score_delta_cost_xy",
                MANEUVERFLOW_INFERENCE_POLICY.refined_score_delta_cost_xy,
            )),
            "yaw": float(getattr(
                config,
                "refined_score_delta_cost_yaw",
                MANEUVERFLOW_INFERENCE_POLICY.refined_score_delta_cost_yaw,
            )),
            "smooth": float(getattr(
                config,
                "refined_score_delta_cost_smooth",
                MANEUVERFLOW_INFERENCE_POLICY.refined_score_delta_cost_smooth,
            )),
        },
    }
    if getattr(config, "bev_coordinate_convention") != "legacy_descending_x":
        protocol["bev_coordinate_convention"] = "lidar_increasing_x"
    return protocol


def validate_checkpoint_envelope(
    checkpoint: Mapping[str, object],
) -> Mapping[str, torch.Tensor]:
    """Validate release metadata, canonical keys, and the tensor summary."""

    actual_keys = set(checkpoint)
    if actual_keys != CHECKPOINT_ENVELOPE_KEYS:
        raise CheckpointSchemaError(
            "ManeuverFlow checkpoint must contain exactly the released "
            f"envelope fields: expected={sorted(CHECKPOINT_ENVELOPE_KEYS)}, "
            f"actual={sorted(map(str, actual_keys))}."
        )

    expected_metadata = {
        "format": CHECKPOINT_SCHEMA_NAME,
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "key_schema": CHECKPOINT_KEY_SCHEMA,
        "model": CHECKPOINT_MODEL_NAME,
    }
    actual_metadata = {
        key: checkpoint.get(key) for key in expected_metadata
    }
    if actual_metadata != expected_metadata:
        raise CheckpointSchemaError(
            "ManeuverFlow checkpoint envelope mismatch: "
            f"expected={expected_metadata}, actual={actual_metadata}."
        )

    if not isinstance(checkpoint.get("inference_protocol"), Mapping):
        raise CheckpointSchemaError(
            "ManeuverFlow checkpoint inference_protocol must be a mapping."
        )
    protocol = checkpoint["inference_protocol"]
    if (
        protocol.get("ecfr_final_pool_composition") == "source_replacement"
        and protocol.get("bev_coordinate_convention") != "lidar_increasing_x"
    ):
        raise CheckpointSchemaError("EMA inference requires the lidar_increasing_x BEV coordinate convention.")

    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise CheckpointSchemaError(
            "ManeuverFlow checkpoint envelope must contain a state_dict mapping."
        )
    _validate_tensor_mapping(state_dict, label="Checkpoint state_dict")
    validate_finite_state_dict(state_dict)
    schema = detect_state_dict_schema(state_dict.keys())
    if schema != PAPER_V1_SCHEMA:
        raise CheckpointSchemaError(
            "Checkpoint state_dict must use only the released "
            f"{CHECKPOINT_KEY_SCHEMA!r} namespace; detected {schema!r}."
        )

    expected_summary = summarize_state_dict(state_dict)
    actual_summary = checkpoint.get("state_dict_summary")
    if actual_summary != expected_summary:
        raise CheckpointSchemaError(
            "ManeuverFlow checkpoint tensor summary mismatch: "
            f"expected={expected_summary}, actual={actual_summary}."
        )
    return state_dict


def validate_mgmp_inference_buffers(
    state_dict: Mapping[str, torch.Tensor],
) -> None:
    """Fail closed on invalid checkpoint-backed M-GMP statistics."""

    prefix = "model.hypothesizer.maneuver_gmp."
    finite_names = (
        "displacement_mean",
        "displacement_std",
        "heading_mean",
        "heading_std",
    )
    for name in finite_names:
        tensor = state_dict.get(prefix + name)
        if not torch.is_tensor(tensor) or not bool(torch.isfinite(tensor).all()):
            raise StateDictValidationError(
                f"Checkpoint M-GMP buffer {name!r} must be a finite tensor."
            )

    for name in ("displacement_std", "heading_std"):
        tensor = state_dict[prefix + name]
        if not bool((tensor > 0).all()):
            raise StateDictValidationError(
                f"Checkpoint M-GMP buffer {name!r} must be strictly positive."
            )

    counts = state_dict.get(prefix + "component_counts")
    if not torch.is_tensor(counts) or counts.dtype != torch.int64:
        raise StateDictValidationError(
            "Checkpoint M-GMP component_counts must be an int64 tensor."
        )
    if not bool((counts > 0).all()):
        raise StateDictValidationError(
            "Checkpoint M-GMP component_counts must be strictly positive."
        )


def validate_perception_variant_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    perception_variant: str,
) -> None:
    """Require the released physical-LiDAR TF input topology."""

    variant = str(perception_variant)
    if variant != "TF":
        raise StateDictValidationError(
            f"Unsupported perception variant in checkpoint validation: {variant!r}."
        )

    prefix = "model.hypothesizer.perception.transfuser_backbone."
    latent_key = prefix + "lidar_latent"
    stem_key = prefix + "lidar_encoder.conv1.weight"
    stem = state_dict.get(stem_key)
    if not torch.is_tensor(stem):
        raise StateDictValidationError(
            f"Checkpoint is missing perception stem tensor {stem_key!r}."
        )

    expected_stem_shape = (64, 1, 7, 7)
    if tuple(stem.shape) != expected_stem_shape:
        raise StateDictValidationError(
            f"ManeuverFlow-{variant} requires {stem_key} shape "
            f"{expected_stem_shape}, got {tuple(stem.shape)}."
        )

    if latent_key in state_dict:
        raise StateDictValidationError(
            f"ManeuverFlow-TF checkpoint must not contain {latent_key!r}."
        )


def validate_finite_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> None:
    """Require strided layout and finite values throughout a checkpoint state."""

    _validate_tensor_mapping(state_dict, label="Checkpoint state_dict")
    non_strided = [
        key
        for key, tensor in state_dict.items()
        if tensor.layout != torch.strided
    ]
    if non_strided:
        raise StateDictValidationError(
            "Checkpoint state_dict contains non-strided tensor layouts at: "
            f"{_format_keys(non_strided)}"
        )
    invalid = [
        key
        for key, tensor in state_dict.items()
        if (tensor.is_floating_point() or tensor.is_complex())
        and not bool(torch.isfinite(tensor).all())
    ]
    if invalid:
        raise StateDictValidationError(
            "Checkpoint state_dict contains non-finite tensor values at: "
            f"{_format_keys(invalid)}"
        )


def _validate_tensor_mapping(
    state_dict: Mapping[object, object],
    *,
    label: str,
) -> None:
    invalid_keys = [key for key in state_dict if not isinstance(key, str)]
    if invalid_keys:
        raise StateDictValidationError(
            f"{label} contains {len(invalid_keys)} non-string key(s)."
        )

    invalid_values = [
        key for key, value in state_dict.items() if not torch.is_tensor(value)
    ]
    if invalid_values:
        raise StateDictValidationError(
            f"{label} contains non-tensor values at: "
            f"{_format_keys(invalid_values)}"
        )


def _format_keys(keys: Iterable[str]) -> str:
    ordered = sorted(keys)
    preview = ", ".join(repr(key) for key in ordered[:8])
    if len(ordered) > 8:
        preview += f", ... (+{len(ordered) - 8} more)"
    return preview


def validate_state_dict_against_model(
    state_dict: Mapping[str, object],
    model_state_dict: Mapping[str, object],
) -> Mapping[str, object]:
    """Validate exact key, shape, and dtype parity before strict loading."""

    _validate_tensor_mapping(state_dict, label="Checkpoint state_dict")
    _validate_tensor_mapping(model_state_dict, label="Model state_dict")

    checkpoint_keys = set(state_dict)
    model_keys = set(model_state_dict)
    errors = []

    missing = model_keys - checkpoint_keys
    unexpected = checkpoint_keys - model_keys
    if missing:
        errors.append(f"missing keys ({len(missing)}): {_format_keys(missing)}")
    if unexpected:
        errors.append(
            f"unexpected keys ({len(unexpected)}): {_format_keys(unexpected)}"
        )

    shape_mismatches = []
    dtype_mismatches = []
    layout_mismatches = []
    for key in sorted(checkpoint_keys & model_keys):
        checkpoint_tensor = state_dict[key]
        model_tensor = model_state_dict[key]

        checkpoint_shape = tuple(checkpoint_tensor.shape)
        model_shape = tuple(model_tensor.shape)
        if checkpoint_shape != model_shape:
            shape_mismatches.append(
                f"{key!r}: checkpoint={checkpoint_shape}, model={model_shape}"
            )
        if checkpoint_tensor.dtype != model_tensor.dtype:
            dtype_mismatches.append(
                f"{key!r}: checkpoint={checkpoint_tensor.dtype}, "
                f"model={model_tensor.dtype}"
            )
        if checkpoint_tensor.layout != model_tensor.layout:
            layout_mismatches.append(
                f"{key!r}: checkpoint={checkpoint_tensor.layout}, "
                f"model={model_tensor.layout}"
            )

    if shape_mismatches:
        errors.append(
            f"shape mismatches ({len(shape_mismatches)}): "
            + "; ".join(shape_mismatches[:8])
        )
    if dtype_mismatches:
        errors.append(
            f"dtype mismatches ({len(dtype_mismatches)}): "
            + "; ".join(dtype_mismatches[:8])
        )
    if layout_mismatches:
        errors.append(
            f"layout mismatches ({len(layout_mismatches)}): "
            + "; ".join(layout_mismatches[:8])
        )

    if errors:
        raise StateDictValidationError(
            "Checkpoint state_dict is incompatible with the model: "
            + " | ".join(errors)
        )
    return state_dict


def checkpoint_file_sha256(path: Path) -> str:
    """Hash the serialized checkpoint envelope without loading it."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def value_only_tensor_sha256(
    state_dict: Mapping[str, torch.Tensor],
) -> str:
    """Hash tensor dtype, shape, and bytes in checkpoint insertion order."""

    _validate_tensor_mapping(state_dict, label="Checkpoint state_dict")
    digest = hashlib.sha256()
    for tensor in state_dict.values():
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(
            (str(contiguous.dtype) + str(tuple(contiguous.shape))).encode()
        )
        digest.update(
            contiguous.reshape(-1).view(torch.uint8).numpy().tobytes()
        )
    return digest.hexdigest()


def canonical_keyed_tensor_sha256(
    state_dict: Mapping[str, torch.Tensor],
) -> str:
    """Hash sorted canonical names plus each tensor's dtype, shape, and bytes."""

    _validate_tensor_mapping(state_dict, label="Checkpoint state_dict")
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        contiguous = state_dict[key].detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(contiguous.dtype).encode())
        digest.update(str(tuple(contiguous.shape)).encode())
        digest.update(
            contiguous.reshape(-1).view(torch.uint8).numpy().tobytes()
        )
    return digest.hexdigest()
