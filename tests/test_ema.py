"""Training-only EMA target storage, updates, and state restoration."""

import pytest
import torch
from torch import nn

from navsim.agents.maneuverflow import checkpoint_export as export
from navsim.agents.maneuverflow.checkpoint_schema import (
    CheckpointSchemaError,
    canonical_keyed_tensor_sha256,
    inference_protocol_from_config,
    validate_checkpoint_envelope,
)
from navsim.agents.maneuverflow.config import MANEUVERFLOW_TRAINING_RECIPE
from navsim.agents.maneuverflow.ema_target_encoder import EMATargetEncoder


class _Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)
        self.norm = nn.BatchNorm1d(2)
        self.dropout = nn.Dropout(0.8)

    def forward(self, inputs):
        return self.dropout(self.norm(self.linear(inputs)))


def test_teacher_uses_independent_frozen_fp32_storage_and_stays_in_eval():
    online = _Encoder().double().train()
    teacher = EMATargetEncoder(online)
    parent = nn.Sequential(teacher)
    parent.train()

    assert online.training
    assert not teacher.training
    assert all(not module.training for module in teacher.modules())
    for online_parameter, teacher_parameter in zip(online.parameters(), teacher.encoder.parameters()):
        assert teacher_parameter.data_ptr() != online_parameter.data_ptr()
        assert teacher_parameter.dtype == torch.float32
        assert not teacher_parameter.requires_grad
    inputs = torch.ones(3, 2, requires_grad=True)
    outputs = teacher(inputs)
    assert not outputs.requires_grad
    torch.testing.assert_close(outputs, teacher(inputs))
    assert all(parameter.grad is None for parameter in teacher.parameters())


def test_teacher_updates_parameters_exactly_once_and_copies_all_buffers():
    online = _Encoder()
    with torch.no_grad():
        for parameter in online.parameters():
            parameter.fill_(2.0)
    teacher = EMATargetEncoder(online, decay=0.9)
    with torch.no_grad():
        for parameter in online.parameters():
            parameter.fill_(4.0)
        online.norm.running_mean.fill_(9.0)
        online.norm.running_var.fill_(5.0)
        online.norm.num_batches_tracked.fill_(13)

    teacher.update(online)

    for parameter in teacher.encoder.parameters():
        torch.testing.assert_close(parameter, torch.full_like(parameter, 2.2))
        assert parameter.grad is None
    assert teacher.num_updates.item() == 1
    torch.testing.assert_close(teacher.encoder.norm.running_mean, online.norm.running_mean)
    torch.testing.assert_close(teacher.encoder.norm.running_var, online.norm.running_var)
    assert teacher.encoder.norm.num_batches_tracked.item() == 13


def test_reset_hard_copies_loaded_online_weights_and_restarts_counter():
    online = nn.Linear(1, 1, bias=False)
    teacher = EMATargetEncoder(online, decay=0.9)
    teacher.update(online)
    with torch.no_grad():
        online.weight.fill_(7.0)

    teacher.reset_from_online(online)

    assert teacher.encoder.weight.item() == 7.0
    assert teacher.num_updates.item() == 0
    assert teacher.decay.item() == pytest.approx(0.9)
    assert not teacher.encoder.weight.requires_grad
    assert not teacher.encoder.training


def test_full_state_resume_preserves_teacher_decay_and_counter():
    online = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        online.weight.fill_(2.0)
    teacher = EMATargetEncoder(online, decay=0.9)
    with torch.no_grad():
        online.weight.fill_(4.0)
    teacher.update(online)
    saved = {name: tensor.clone() for name, tensor in teacher.state_dict().items()}
    resumed = EMATargetEncoder(online, decay=0.5)

    resumed.load_state_dict(saved, strict=True)

    assert resumed.encoder.weight.item() == pytest.approx(2.2)
    assert resumed.num_updates.item() == 1
    assert resumed.decay.item() == pytest.approx(0.9)
    resumed.update(online)
    assert resumed.encoder.weight.item() == pytest.approx(2.38)
    assert resumed.num_updates.item() == 2


def test_teacher_float_buffers_stay_fp32_when_parent_changes_dtype():
    teacher = EMATargetEncoder(_Encoder(), decay=0.999)

    teacher.half()

    assert all(parameter.dtype == torch.float32 for parameter in teacher.encoder.parameters())
    assert teacher.encoder.norm.running_mean.dtype == torch.float32
    assert teacher.encoder.norm.num_batches_tracked.dtype == torch.int64
    assert teacher.num_updates.dtype == torch.int64
    assert teacher.decay.item() == pytest.approx(0.999)


@pytest.fixture
def joint_checkpoint():
    online_prefix = "model.hypothesizer.perception.transfuser_backbone."
    prior_prefix = "model.hypothesizer.maneuver_gmp."
    online = {online_prefix + "weight": torch.tensor([[2.0]])}
    for name in ("displacement_mean", "displacement_std", "heading_mean", "heading_std"):
        online[prior_prefix + name] = torch.ones(8, 1)
    online[prior_prefix + "component_counts"] = torch.ones(8, dtype=torch.int64)
    state = {"agent." + name: value for name, value in online.items()}
    state.update({
        "agent.model.ema_target_encoder.encoder.weight": torch.tensor([[1.0]]),
        "agent.model.ema_target_encoder.decay": torch.tensor(0.999, dtype=torch.float64),
        "agent.model.ema_target_encoder.num_updates": torch.tensor(7, dtype=torch.int64),
    })
    metadata = export.joint_checkpoint_metadata()
    metadata.update({
        "stage1_source_recipe": MANEUVERFLOW_TRAINING_RECIPE,
        "stage1_source_tensor_sha256": canonical_keyed_tensor_sha256(online),
    })
    return {
        "state_dict": state,
        "navsim_agent_metadata": metadata,
        "inference_protocol": inference_protocol_from_config(export.inference_config()),
        "optimizer_states": [{"state": "training-only fixture"}],
    }, online


def test_joint_export_preserves_online_tensors_and_strips_training_state(joint_checkpoint, tmp_path, monkeypatch):
    checkpoint, online = joint_checkpoint
    source, destination = tmp_path / "joint.ckpt", tmp_path / "inference.pt"
    torch.save(checkpoint, source)
    monkeypatch.setattr(export, "canonical_model_state_dict_on_meta", lambda: online)

    export.export_joint_checkpoint(source, destination)

    released = torch.load(destination, weights_only=True)
    state = export.validate_release_envelope(released, online)
    assert set(released) == export.RELEASE_ENVELOPE_KEYS
    assert canonical_keyed_tensor_sha256(state) == canonical_keyed_tensor_sha256(online)
    assert released["inference_protocol"]["bev_coordinate_convention"] == "lidar_increasing_x"
    assert not any("ema_target_encoder" in name for name in state)
    with pytest.raises(FileExistsError):
        export.export_joint_checkpoint(source, destination)


@pytest.mark.parametrize("mutation", ["missing_lineage", "old_recipe", "wrong_decay", "old_geometry"])
def test_joint_export_rejects_incompatible_training_contracts(joint_checkpoint, mutation):
    checkpoint, _ = joint_checkpoint
    metadata = checkpoint["navsim_agent_metadata"]
    if mutation == "missing_lineage":
        metadata.pop("stage1_source_tensor_sha256")
    elif mutation == "old_recipe":
        metadata["training_recipe"] = "maneuverflow.ema.v2"
    elif mutation == "wrong_decay":
        metadata["ema_target_decay"] = 0.9
    else:
        checkpoint["inference_protocol"].pop("bev_coordinate_convention")
    with pytest.raises(ValueError):
        export.validate_joint_checkpoint_metadata(checkpoint)


def test_ema_inference_envelope_requires_explicit_histogram_geometry(joint_checkpoint):
    _, online = joint_checkpoint
    envelope = export.build_release_envelope(online)
    envelope["inference_protocol"].pop("bev_coordinate_convention")
    with pytest.raises(CheckpointSchemaError, match="coordinate convention"):
        validate_checkpoint_envelope(envelope)


def test_joint_export_rejects_partial_teacher_before_stripping(joint_checkpoint):
    checkpoint, _ = joint_checkpoint
    checkpoint["state_dict"].pop("agent.model.ema_target_encoder.encoder.weight")
    with pytest.raises(CheckpointSchemaError, match="teacher backbone mismatch"):
        export.normalize_joint_lightning_state_dict(checkpoint["state_dict"], expected_decay=0.999)
