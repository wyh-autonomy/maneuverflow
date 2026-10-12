"""Export Joint Lightning weights into the strict inference envelope."""

from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path
import tempfile
from typing import Any

import torch

from .checkpoint_schema import (
    CHECKPOINT_ENVELOPE_KEYS,
    CHECKPOINT_FORMAT_VERSION,
    CHECKPOINT_KEY_SCHEMA,
    CHECKPOINT_MODEL_NAME,
    CHECKPOINT_SCHEMA_NAME,
    EMA_TRAINING_SCHEMA,
    canonical_keyed_tensor_sha256,
    checkpoint_file_sha256,
    detect_training_state_dict_schema,
    inference_protocol_from_config,
    summarize_state_dict,
    validate_checkpoint_envelope,
    validate_ema_training_state_dict,
    validate_mgmp_inference_buffers,
    validate_optimizer_metadata,
    validate_stage1_lineage_metadata,
    validate_state_dict_against_model,
)
from .config import (
    MANEUVERFLOW_TRAINING_RECIPE,
    MANEUVERFLOW_EMA_DECAY,
    ManeuverFlowConfig,
)

DEFAULT_EXPORT_POOL_COMPOSITION = "source_replacement"


def joint_checkpoint_metadata(
    *,
    max_epochs: int = 20,
    learning_rate: float = 2e-4,
    warmup_epochs: int = 3,
    min_lr: float = 5e-6,
    weight_decay: float = 0.1,
    ema_target_decay: float = MANEUVERFLOW_EMA_DECAY,
    ema_target_buffer_policy: str = "copy",
    future_bev_loss_weight: float = ManeuverFlowConfig.future_bev_loss_weight,
    future_bev_loss_decay_epochs: int = 0,
) -> dict[str, object]:
    """Return the exact TF Joint checkpoint identity."""

    return {
        "agent_name": "ManeuverFlowAgent",
        "training_phase": "joint_finetuning",
        "training_recipe": MANEUVERFLOW_TRAINING_RECIPE,
        "perception_variant": "TF",
        "max_epochs": max_epochs,
        "learning_rate": learning_rate,
        "warmup_epochs": warmup_epochs,
        "min_lr": min_lr,
        "weight_decay": weight_decay,
        "bev_coordinate_convention": "lidar_increasing_x",
        "ema_target_decay": ema_target_decay,
        "ema_target_buffer_policy": ema_target_buffer_policy,
        "future_bev_loss_weight": future_bev_loss_weight,
        "future_bev_loss_decay_epochs": future_bev_loss_decay_epochs,
    }


RELEASE_ENVELOPE_KEYS = set(CHECKPOINT_ENVELOPE_KEYS)


class CheckpointExportError(ValueError):
    """Raised when a training checkpoint cannot be exported safely."""


def validate_joint_checkpoint_metadata(
    checkpoint: Mapping[str, Any],
) -> None:
    """Require a Joint checkpoint with the released inference protocol."""

    actual = checkpoint.get("navsim_agent_metadata")
    if not isinstance(actual, Mapping):
        raise CheckpointExportError("EMA Joint export requires training recipe metadata.")
    validate_stage1_lineage_metadata(actual)
    max_epochs = actual.get("max_epochs")
    decay = actual.get("ema_target_decay")
    if type(max_epochs) is not int or max_epochs < 1:
        raise CheckpointExportError("EMA Joint training recipe max_epochs must be positive.")
    if type(decay) not in {int, float} or float(decay) != MANEUVERFLOW_EMA_DECAY:
        raise CheckpointExportError("EMA Joint training recipe requires decay=0.999.")
    controls = validate_optimizer_metadata(actual)
    expected_metadata = joint_checkpoint_metadata(
        max_epochs=max_epochs, **controls,
    )
    runtime_metadata = {
        key: value for key, value in actual.items()
        if key not in {"stage1_source_recipe", "stage1_source_tensor_sha256"}
    }
    if runtime_metadata != expected_metadata:
        raise CheckpointExportError(
            "Inference export requires a checkpoint saved by the EMA "
            "Joint fine-tuning recipe: "
            f"expected={expected_metadata}, actual={actual}."
        )
    expected_protocol = inference_protocol_from_config(
        inference_config()
    )
    actual_protocol = checkpoint.get("inference_protocol")
    normalized_protocol = dict(actual_protocol) if isinstance(actual_protocol, Mapping) else actual_protocol
    if normalized_protocol != expected_protocol:
        raise CheckpointExportError(
            "Joint checkpoint inference protocol mismatch. Train with the "
            "released ManeuverFlow cardinalities and ECFR bounds before exporting: "
            f"expected={expected_protocol}, actual={actual_protocol}."
        )


def normalize_joint_lightning_state_dict(
    state_dict: Mapping[object, object],
    *,
    expected_decay: float | None = None,
) -> dict[str, torch.Tensor]:
    """Validate the teacher, then strip its exact training-only namespace."""

    if not isinstance(state_dict, Mapping) or not state_dict:
        raise CheckpointExportError(
            "Joint checkpoint state_dict must be a non-empty mapping."
        )

    normalized: dict[str, torch.Tensor] = {}
    for raw_key, raw_tensor in state_dict.items():
        if not isinstance(raw_key, str):
            raise CheckpointExportError(
                "Joint checkpoint state_dict keys must be strings."
            )
        if not raw_key.startswith("agent."):
            raise CheckpointExportError(
                "Every Joint Lightning state_dict key must start with exactly "
                f"'agent.': {raw_key!r}."
            )
        key = raw_key[len("agent.") :]
        if key.startswith("agent."):
            raise CheckpointExportError(
                f"Joint Lightning state_dict key has a repeated prefix: {raw_key!r}."
            )
        if not torch.is_tensor(raw_tensor):
            raise CheckpointExportError(
                f"Joint checkpoint value is not a tensor: {raw_key!r}."
            )
        if raw_tensor.layout != torch.strided:
            raise CheckpointExportError(
                f"Joint checkpoint tensor must use strided layout: {raw_key!r}."
            )
        if key in normalized:
            raise CheckpointExportError(
                f"Duplicate normalized checkpoint key: {key!r}."
            )
        normalized[key] = raw_tensor.detach().cpu().contiguous()

    schema = detect_training_state_dict_schema(normalized.keys())
    if schema != EMA_TRAINING_SCHEMA:
        raise CheckpointExportError(
            "EMA Joint checkpoint tensors must use the explicit training "
            f"{EMA_TRAINING_SCHEMA!r} namespace after stripping 'agent.'; "
            f"detected {schema!r}."
        )
    return validate_ema_training_state_dict(normalized, expected_decay=expected_decay)


def inference_config() -> ManeuverFlowConfig:
    """Return a strict inference-time protocol configuration."""

    try:
        return ManeuverFlowConfig(
            training_phase="inference",
            mgmp_assets_from_checkpoint=True,
            ecfr_max_ds=1.0,
            ecfr_max_dl=0.8,
            ecfr_max_dpsi=0.35,
            ecfr_final_pool_composition=DEFAULT_EXPORT_POOL_COMPOSITION,
        )
    except ValueError as error:
        raise CheckpointExportError(str(error)) from error


def canonical_model_state_dict_on_meta() -> Mapping[str, torch.Tensor]:
    """Construct the expected state schema without allocating model storage."""

    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from .agent import ManeuverFlowAgent

    config = inference_config()
    trajectory_sampling = TrajectorySampling(
        time_horizon=4.0,
        interval_length=0.5,
    )
    with torch.device("meta"):
        agent = ManeuverFlowAgent(
            config=config,
            trajectory_sampling=trajectory_sampling,
            checkpoint_path=None,
            initialize_from_checkpoint=False,
        )
    return agent.state_dict()


def build_release_envelope(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, object]:
    """Create the minimal seven-field inference checkpoint envelope."""

    config = inference_config()
    return {
        "format": CHECKPOINT_SCHEMA_NAME,
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "key_schema": CHECKPOINT_KEY_SCHEMA,
        "model": CHECKPOINT_MODEL_NAME,
        "state_dict": dict(state_dict),
        "state_dict_summary": summarize_state_dict(state_dict),
        "inference_protocol": inference_protocol_from_config(config),
    }


def validate_release_envelope(
    checkpoint: Mapping[str, Any],
    model_state_dict: Mapping[str, torch.Tensor],
) -> Mapping[str, torch.Tensor]:
    """Validate exact fields, metadata, protocol, and model tensor parity."""

    actual_keys = set(checkpoint)
    if actual_keys != RELEASE_ENVELOPE_KEYS:
        raise CheckpointExportError(
            "Exported checkpoint fields do not match the strict release "
            f"envelope: expected={sorted(RELEASE_ENVELOPE_KEYS)}, "
            f"actual={sorted(actual_keys)}."
        )
    if checkpoint.get("model") != CHECKPOINT_MODEL_NAME:
        raise CheckpointExportError(
            f"Exported checkpoint model must be {CHECKPOINT_MODEL_NAME!r}."
        )
    config = inference_config()
    expected_protocol = inference_protocol_from_config(config)
    if checkpoint.get("inference_protocol") != expected_protocol:
        raise CheckpointExportError(
            "Exported checkpoint inference protocol does not match the "
            f"released configuration: expected={expected_protocol}, "
            f"actual={checkpoint.get('inference_protocol')}."
        )

    state_dict = validate_checkpoint_envelope(checkpoint)
    validate_state_dict_against_model(state_dict, model_state_dict)
    validate_mgmp_inference_buffers(state_dict)
    return state_dict


def _temporary_torch_save(
    checkpoint: Mapping[str, Any],
    output_path: Path,
) -> Path:
    """Serialize beside the destination without publishing the file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        # A binary stream gives PyTorch the neutral internal ZIP root
        # ``archive/`` instead of embedding a destination or temporary stem.
        with temporary_path.open("wb") as stream:
            torch.save(dict(checkpoint), stream)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return temporary_path


def export_joint_checkpoint(
    input_path: Path,
    output_path: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Export and round-trip validate one Joint Lightning checkpoint."""

    input_path = Path(input_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Joint checkpoint does not exist: {input_path}")
    if input_path == output_path:
        raise CheckpointExportError(
            "Input and output checkpoint paths must be different."
        )
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output checkpoint already exists: {output_path}. "
            "Pass --force to replace it."
        )

    source_checkpoint = torch.load(
        input_path,
        map_location=torch.device("cpu"),
        mmap=True,
        weights_only=True,
    )
    if not isinstance(source_checkpoint, Mapping):
        raise CheckpointExportError("Joint checkpoint must be a mapping.")
    validate_joint_checkpoint_metadata(
        source_checkpoint,
    )

    source_state_dict = source_checkpoint.get("state_dict")
    if not isinstance(source_state_dict, Mapping):
        raise CheckpointExportError(
            "Joint checkpoint must contain a state_dict mapping."
        )
    state_dict = normalize_joint_lightning_state_dict(
        source_state_dict,
        expected_decay=float(source_checkpoint["navsim_agent_metadata"]["ema_target_decay"]),
    )
    model_state_dict = canonical_model_state_dict_on_meta()
    validate_state_dict_against_model(state_dict, model_state_dict)

    release_checkpoint = build_release_envelope(
        state_dict,
    )
    validate_release_envelope(
        release_checkpoint,
        model_state_dict,
    )
    temporary_path = _temporary_torch_save(
        release_checkpoint,
        output_path,
    )
    try:
        reloaded = torch.load(
            temporary_path,
            map_location=torch.device("cpu"),
            mmap=True,
            weights_only=True,
        )
        if not isinstance(reloaded, Mapping):
            raise CheckpointExportError(
                "Round-trip validation did not reload a checkpoint mapping."
            )
        reloaded_state_dict = validate_release_envelope(
            reloaded,
            model_state_dict,
        )
        canonical_hash = canonical_keyed_tensor_sha256(
            reloaded_state_dict
        )
        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"Output checkpoint appeared during export: {output_path}."
            )
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    return {
        "output_path": str(output_path),
        "perception_variant": "TF",
        "ecfr_final_pool_composition": DEFAULT_EXPORT_POOL_COMPOSITION,
        "state_dict_summary": dict(reloaded["state_dict_summary"]),
        "canonical_keyed_tensor_sha256": canonical_hash,
        "file_sha256": checkpoint_file_sha256(output_path),
    }
