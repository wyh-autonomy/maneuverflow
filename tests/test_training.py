"""Optimizer-step EMA, training state, losses, and validation derivatives."""

from pathlib import Path
import json
from types import MethodType, SimpleNamespace

from omegaconf import OmegaConf
import pytorch_lightning as pl
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from navsim.agents.maneuverflow.agent import ManeuverFlowAgent
from navsim.agents.maneuverflow.config import MANEUVERFLOW_TRAINING_RECIPE, ManeuverFlowConfig
from navsim.agents.maneuverflow.checkpoint_schema import canonical_keyed_tensor_sha256
from navsim.agents.maneuverflow.ema_target_encoder import EMATargetEncoder
from navsim.agents.maneuverflow.losses import (
    compute_future_bev_feature_loss,
    compute_joint_finetuning_loss,
    compute_m3f_loss,
    compute_metric_prism_factor_loss,
)
from navsim.agents.maneuverflow.model import ManeuverFlowModel
from navsim.planning.training.agent_lightning_module import AgentLightningModule
from navsim.planning.script import run_training as training_entry
from navsim.planning.training.data_split import training_data_metadata
from navsim.planning.training.dataset import CacheOnlyDataset, Dataset, dump_feature_target_to_pickle


class _Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.lidar_encoder = nn.Module()
        self.lidar_encoder.conv1 = nn.Conv2d(1, 64, 7, bias=False)
        self.norm = nn.BatchNorm1d(2)


class _Prior(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("displacement_mean", torch.zeros(8, 8, 2))
        self.register_buffer("displacement_std", torch.full((8, 8, 2), 0.03))
        self.register_buffer("heading_mean", torch.zeros(8, 8))
        self.register_buffer("heading_std", torch.full((8, 8), 0.05))
        self.register_buffer("component_counts", torch.ones(8, dtype=torch.int64))


class _Model(nn.Module):
    def __init__(self, joint):
        super().__init__()
        self.hypothesizer = nn.Module()
        self.hypothesizer.perception = nn.Module()
        perception = self.hypothesizer.perception
        perception.transfuser_backbone = _Backbone()
        for name in (
            "bev_memory_embedding",
            "query_embedding",
            "bev_projection",
            "state_navigation_encoder",
            "bev_semantic_head",
            "query_decoder",
            "agent_prediction_head",
        ):
            setattr(perception, name, nn.Linear(2, 2))
        self.hypothesizer.maneuver_decoder = nn.Linear(2, 2)
        self.hypothesizer.maneuver_gmp = _Prior()
        self.hypothesizer.meanflow_generator = nn.Module()
        for name in (
            "interval_encoder",
            "trajectory_tokenizer",
            "field",
            "field_decoder",
            "bev_position_embedding",
        ):
            setattr(self.hypothesizer.meanflow_generator, name, nn.Linear(2, 2))
        self.verifier = nn.Linear(2, 2)
        self.improver = nn.Module()
        self.improver.ecfr = nn.Linear(2, 2)
        self.verifier.requires_grad_(joint)
        self.improver.requires_grad_(joint)
        self.ema_target_encoder = EMATargetEncoder(perception.transfuser_backbone, decay=0.999) if joint else None


def _agent(joint=True):
    agent = ManeuverFlowAgent.__new__(ManeuverFlowAgent)
    nn.Module.__init__(agent)
    agent.config = ManeuverFlowConfig(
        training_phase="joint_finetuning" if joint else "generation_pretraining",
        mgmp_assets_from_checkpoint=True,
        metric_cache_path="metric-cache",
        warmup_epochs=0,
    )
    agent.model = _Model(joint)
    agent.max_epochs = agent.config.max_epochs
    agent.min_lr = agent.config.min_lr
    agent._lr = 2e-4 if joint else 4e-4
    agent._checkpoint_initialized = False
    agent._checkpoint_path = None
    agent._stage1_source_recipe = MANEUVERFLOW_TRAINING_RECIPE if joint else None
    agent._initialized_checkpoint_tensor_sha256 = (
        canonical_keyed_tensor_sha256({
            name: value for name, value in agent.state_dict().items()
            if not name.startswith("model.ema_target_encoder.")
        }) if joint else None
    )
    return agent


def _checkpoint(agent):
    return {
        "state_dict": {"agent." + name: tensor.clone() for name, tensor in agent.state_dict().items()},
        "navsim_agent_metadata": agent.training_checkpoint_metadata(),
        "inference_protocol": agent.inference_protocol_metadata(),
    }


def test_ema_updates_only_after_real_optimizer_steps_and_never_enters_groups():
    agent = _agent()
    online = agent.model.hypothesizer.perception.transfuser_backbone
    teacher = agent.model.ema_target_encoder
    optimizer = agent.get_optimizers()["optimizer"]
    assert len(optimizer._optimizer_step_post_hooks) == 1
    teacher_ids = {id(parameter) for parameter in teacher.parameters()}
    assert teacher_ids.isdisjoint({id(parameter) for group in optimizer.param_groups for parameter in group["params"]})
    initial_teacher = teacher.encoder.lidar_encoder.conv1.weight.detach().clone()
    loss = online.lidar_encoder.conv1.weight.sum()
    loss.backward()
    assert teacher.num_updates.item() == 0
    optimizer.zero_grad()
    assert teacher.num_updates.item() == 0
    online.lidar_encoder.conv1.weight.sum().backward()

    optimizer.step()

    assert teacher.num_updates.item() == 1
    torch.testing.assert_close(
        teacher.encoder.lidar_encoder.conv1.weight,
        initial_teacher * 0.999 + online.lidar_encoder.conv1.weight.detach() * 0.001,
    )
    optimizer.step()
    assert teacher.num_updates.item() == 2


def test_stage1_optimizer_has_no_ema_hook():
    agent = _agent(joint=False)
    optimizer = agent.get_optimizers()["optimizer"]
    assert not optimizer._optimizer_step_post_hooks


def test_joint_initialization_requires_stage1_checkpoint_path():
    with pytest.raises(RuntimeError, match="checkpoint_path"):
        _agent().initialize()


def test_training_metadata_records_actual_optimizer_controls():
    agent = _agent(joint=False)
    agent._lr = 7e-4
    agent.config.warmup_epochs = 4
    agent.min_lr = 3e-6
    agent.config.weight_decay = 0.07

    metadata = agent.training_checkpoint_metadata()

    assert {key: metadata.get(key) for key in (
        "learning_rate", "warmup_epochs", "min_lr", "weight_decay", "bev_coordinate_convention",
    )} == {
        "learning_rate": 7e-4,
        "warmup_epochs": 4,
        "min_lr": 3e-6,
        "weight_decay": 0.07,
        "bev_coordinate_convention": "lidar_increasing_x",
    }


@pytest.mark.parametrize("missing", ["recipe", "digest", "both"])
def test_joint_training_metadata_requires_complete_stage1_lineage(missing):
    agent = _agent()
    if missing in {"recipe", "both"}:
        agent._stage1_source_recipe = None
    if missing in {"digest", "both"}:
        agent._initialized_checkpoint_tensor_sha256 = None

    with pytest.raises(ValueError, match="Stage-1 lineage"):
        agent.training_checkpoint_metadata()


@pytest.mark.parametrize("control", ["learning_rate", "warmup_epochs", "min_lr", "weight_decay"])
def test_full_resume_rejects_optimizer_control_drift(control):
    checkpoint = _checkpoint(_agent())
    resumed = _agent()
    if control == "learning_rate":
        resumed._lr *= 2
    elif control == "warmup_epochs":
        resumed.config.warmup_epochs += 1
    elif control == "min_lr":
        resumed.min_lr *= 2
    else:
        resumed.config.weight_decay *= 2

    with pytest.raises(RuntimeError, match="resume checkpoint metadata mismatch"):
        resumed.validate_resume_checkpoint_metadata(checkpoint)


@pytest.mark.parametrize("control", ["learning_rate", "warmup_epochs", "min_lr", "weight_decay"])
def test_stage1_handoff_rejects_invalid_optimizer_identity(control):
    checkpoint = _checkpoint(_agent(joint=False))
    checkpoint["navsim_agent_metadata"][control] = float("nan")

    with pytest.raises(RuntimeError, match="Stage-1 source"):
        _agent()._validate_checkpoint_training_handoff(checkpoint)


def test_stage1_initialize_loads_online_weights_then_hard_syncs_teacher(tmp_path):
    stage1 = _agent(joint=False)
    with torch.no_grad():
        for parameter in stage1.parameters():
            parameter.fill_(2.0)
    path = tmp_path / "stage1.ckpt"
    torch.save(_checkpoint(stage1), path)
    joint = _agent()
    joint._checkpoint_path = str(path)

    joint.initialize()

    teacher = joint.model.ema_target_encoder
    assert joint._checkpoint_initialized
    assert teacher.num_updates.item() == 0
    for name, tensor in joint.model.hypothesizer.perception.transfuser_backbone.state_dict().items():
        torch.testing.assert_close(teacher.encoder.state_dict()[name], tensor)
    assert joint.model.verifier.weight[0, 0].item() == 2.0


def test_full_resume_restores_teacher_without_hard_sync_and_accepts_unchanged_recipe():
    joint = _agent()
    online = joint.model.hypothesizer.perception.transfuser_backbone
    with torch.no_grad():
        online.lidar_encoder.conv1.weight.fill_(4.0)
    joint.model.ema_target_encoder.update(online)
    checkpoint = _checkpoint(joint)
    resumed = _agent()

    resumed.validate_resume_checkpoint_metadata(checkpoint)
    resumed.load_state_dict(
        {key.removeprefix("agent."): value for key, value in checkpoint["state_dict"].items()}, strict=True
    )

    assert resumed.model.ema_target_encoder.num_updates.item() == 1
    assert not torch.equal(
        resumed.model.ema_target_encoder.encoder.lidar_encoder.conv1.weight, online.lidar_encoder.conv1.weight
    )
    for name, value in joint.state_dict().items():
        torch.testing.assert_close(resumed.state_dict()[name], value)


class _CacheBuilder:
    requires_online_target_construction = False

    def __init__(self, name):
        self.name = name

    def get_unique_name(self):
        return self.name


def _cache_sample(cache, log, token):
    sample_path = cache / log / token
    sample_path.mkdir(parents=True)
    for name in ("feature", "target"):
        dump_feature_target_to_pickle(sample_path / f"{name}.gz", {name: torch.tensor([1.0])})


def test_cache_only_dataset_filters_tokens_before_scene_limit(tmp_path):
    for token in ("excluded", "selected_a", "selected_b"):
        _cache_sample(tmp_path, "train", token)

    dataset = CacheOnlyDataset(
        cache_path=str(tmp_path),
        feature_builders=[_CacheBuilder("feature")],
        target_builders=[_CacheBuilder("target")],
        log_names=["train"],
        tokens=["selected_a", "selected_b"],
        max_scenes=1,
    )

    assert len(dataset) == 1
    assert dataset.tokens[0] in {"selected_a", "selected_b"}
    features, targets = dataset[0]
    assert features["token"] == dataset.tokens[0]
    assert targets["target"].item() == 1.0


def test_online_dataset_reads_cache_only_from_selected_scene_log(tmp_path):
    for log, token in (("training", "selected"), ("validation", "selected"), ("unrelated", "other")):
        _cache_sample(tmp_path, log, token)
    loader = SimpleNamespace(
        tokens=["selected"], get_tokens_list_per_log=lambda: {"training": ["selected"]},
    )

    dataset = Dataset(
        scene_loader=loader, feature_builders=[_CacheBuilder("feature")],
        target_builders=[_CacheBuilder("target")], cache_path=str(tmp_path),
    )

    assert dataset._valid_cache_paths == {"selected": tmp_path / "training/selected"}
    assert dataset.tokens == ["selected"]
    features, targets = dataset[0]
    assert features["token"] == "selected"
    assert targets["target"].item() == 1.0


def test_cache_only_entry_intersects_partition_logs_with_scene_filter_without_sensors(tmp_path):
    cfg = _navtrain_config()
    train_logs = list(cfg.train_logs[:2])
    val_log = cfg.val_logs[0]
    train_token, excluded_token, val_token = cfg.train_test_split.scene_filter.tokens[:3]
    for log, token in (
        (train_logs[0], train_token), (train_logs[0], "outside_population"),
        (train_logs[1], excluded_token), (val_log, val_token),
    ):
        _cache_sample(tmp_path, log, token)
    cfg.cache_path = str(tmp_path)
    cfg.train_logs = train_logs
    cfg.val_logs = [val_log]
    cfg.train_test_split.scene_filter.log_names = [train_logs[0], val_log]
    cfg.train_test_split.scene_filter.tokens = [train_token, excluded_token, val_token]
    cfg.train_test_split.scene_filter.max_scenes = 1
    agent = SimpleNamespace(
        get_feature_builders=lambda: [_CacheBuilder("feature")],
        get_target_builders=lambda: [_CacheBuilder("target")],
    )

    train, val = training_entry.build_cache_only_datasets(cfg, agent)

    assert train.tokens == [train_token]
    assert val.tokens == [val_token]


def test_training_partition_overlap_is_rejected_before_cache_loading(tmp_path):
    cfg = OmegaConf.create({
        "cache_path": str(tmp_path),
        "train_logs": ["shared_log"],
        "val_logs": ["shared_log"],
        "train_test_split": {"scene_filter": {
            "_target_": "navsim.common.dataclasses.SceneFilter", "log_names": None,
        }},
    })

    with pytest.raises(ValueError, match="disjoint"):
        training_entry.build_cache_only_datasets(cfg, SimpleNamespace())


def _navtrain_config():
    root = Path(__file__).resolve().parents[1] / "navsim/planning/script/config"
    partitions = OmegaConf.load(root / "training/default_train_val_test_log_split.yaml")
    scene_filter = OmegaConf.load(root / "common/train_test_split/scene_filter/navtrain.yaml")
    return OmegaConf.create({
        "train_logs": partitions.train_logs,
        "val_logs": partitions.val_logs,
        "train_test_split": {"scene_filter": scene_filter},
    })


def test_training_filter_rejects_validation_log_reassigned_to_training():
    cfg = _navtrain_config()
    validation_logs = [name for name in cfg.val_logs if name in cfg.train_test_split.scene_filter.log_names]
    cfg.train_logs = validation_logs[:1]
    cfg.val_logs = validation_logs[1:2]

    with pytest.raises(ValueError, match="training log partition"):
        tuple(training_entry._partition_scene_filters(cfg))


def test_training_filter_rejects_tokens_outside_navtrain():
    cfg = _navtrain_config()
    cfg.train_test_split.scene_filter.tokens = ["not_a_navtrain_token"]

    with pytest.raises(ValueError, match="navtrain tokens"):
        tuple(training_entry._partition_scene_filters(cfg))


@pytest.mark.parametrize("expected_tokens", [
    ["training_scene", "validation_scene", "missing_scene"],
    ["training_scene"],
])
def test_training_rejects_incomplete_or_extra_scene_coverage(expected_tokens):
    with pytest.raises(ValueError, match="scene coverage"):
        training_data_metadata(["training_scene"], ["validation_scene"], expected_tokens=expected_tokens)


def test_training_checkpoint_preserves_population_without_raw_token_list():
    module = AgentLightningModule(_agent(joint=False))
    module.training_data_metadata = {
        "split": "navtrain", "train": {"num_scenes": 8, "tokens_sha256": "a" * 64},
        "validation": {"num_scenes": 2, "tokens_sha256": "b" * 64},
    }
    checkpoint = {}

    module.on_save_checkpoint(checkpoint)

    assert checkpoint.get("training_data") == module.training_data_metadata


def test_resume_rejects_changed_training_population():
    module = AgentLightningModule(_agent(joint=False))
    checkpoint = _checkpoint(module.agent)
    checkpoint["training_data"] = {
        "split": "navtrain", "train": {"num_scenes": 8, "tokens_sha256": "a" * 64},
        "validation": {"num_scenes": 2, "tokens_sha256": "b" * 64},
    }
    module.training_data_metadata = {
        **checkpoint["training_data"],
        "train": {"num_scenes": 8, "tokens_sha256": "c" * 64},
    }

    with pytest.raises(ValueError, match="training population"):
        module.on_load_checkpoint(checkpoint)


def test_training_rejects_prior_fitted_on_larger_population_before_first_step():
    module = AgentLightningModule(_agent(joint=False))
    module.training_data_metadata = {"train": {"num_scenes": 7}}

    with pytest.raises(ValueError, match="M-GMP prior.*training population"):
        module.on_fit_start()


@pytest.mark.parametrize("joint", [False, True])
def test_fit_checks_prior_or_stage1_population_hash(tmp_path, joint):
    module = AgentLightningModule(_agent(joint=joint))
    population = {
        "split": "navtrain", "train": {"num_scenes": 8, "tokens_sha256": "a" * 64},
        "validation": {"num_scenes": 2, "tokens_sha256": "b" * 64},
    }
    module.training_data_metadata = population
    if joint:
        module.agent._source_training_data_metadata = population
    else:
        module.agent.config.mgmp_prior_dir = str(tmp_path)
        metadata_path = tmp_path / "maneuver_mgmp_metadata.json"
        metadata_path.write_text(json.dumps({"training_data": {
            "split": "navtrain", "partition": "train", **population["train"],
        }}), encoding="utf-8")

    module.on_fit_start()
    module.training_data_metadata = {**population, "train": {**population["train"], "tokens_sha256": "c" * 64}}

    with pytest.raises(ValueError, match="training population"):
        module.on_fit_start()


def test_resume_checks_restored_prior_without_external_prior_metadata():
    module = AgentLightningModule(_agent(joint=False))
    module.training_data_metadata = {
        "split": "navtrain", "train": {"num_scenes": 8, "tokens_sha256": "a" * 64},
        "validation": {"num_scenes": 2, "tokens_sha256": "b" * 64},
    }
    checkpoint = _checkpoint(module.agent)
    module.on_save_checkpoint(checkpoint)

    module.on_load_checkpoint(checkpoint)
    module.on_fit_start()
    module.agent.model.hypothesizer.maneuver_gmp.component_counts.add_(1)

    with pytest.raises(ValueError, match="M-GMP prior.*training population"):
        module.on_fit_start()


def test_joint_loss_rank_local_nonfinite_diagnostic_keeps_log_key_parity():
    config = ManeuverFlowConfig(training_phase="joint_finetuning")
    base = torch.zeros(1, 1, 8, 3)
    teacher = base.clone()
    teacher[..., 0] = 0.5
    finite_predictions = {
        "generated_trajectories_base": base,
        "joint_finetuning_teacher_traj": teacher,
        "joint_finetuning_delta_traj_pred": torch.zeros_like(base, requires_grad=True),
        "joint_finetuning_delta_traj_pred_frenet": torch.zeros_like(base, requires_grad=True),
        "joint_finetuning_teacher_delta_frenet": teacher,
        "joint_finetuning_refine_supervision_weight": torch.ones(1, 1),
    }
    nonfinite_predictions = {
        **finite_predictions,
        "joint_finetuning_delta_traj_pred_frenet": torch.full_like(base, float("nan"), requires_grad=True),
    }

    finite_rank = compute_joint_finetuning_loss({}, finite_predictions, config)
    nonfinite_rank = compute_joint_finetuning_loss({}, nonfinite_predictions, config)

    assert finite_rank.keys() == nonfinite_rank.keys()
    key = "joint_finetuning_nonfinite_loss_terms_monitor"
    assert finite_rank[key].ndim == nonfinite_rank[key].ndim == 0
    assert finite_rank[key].item() == 0
    assert nonfinite_rank[key].item() > 0
    assert torch.isfinite(nonfinite_rank[key])


@pytest.mark.parametrize("missing", [
    "joint_finetuning_delta_traj_pred_frenet",
    "joint_finetuning_teacher_delta_frenet",
    "joint_finetuning_refine_supervision_weight",
])
def test_ecfr_rejects_incomplete_main_objective(missing):
    predictions = {
        "joint_finetuning_delta_traj_pred_frenet": torch.zeros(1, 2, 8, 3),
        "joint_finetuning_teacher_delta_frenet": torch.ones(1, 2, 8, 3),
        "joint_finetuning_refine_supervision_weight": torch.ones(1, 2),
    }
    predictions.pop(missing)
    with pytest.raises(KeyError, match=missing):
        compute_joint_finetuning_loss({}, predictions, ManeuverFlowConfig(training_phase="joint_finetuning"))


def test_ecfr_valid_pair_average_detaches_reference_and_masks_gradients():
    prediction = torch.zeros(1, 2, 8, 3, requires_grad=True)
    reference = torch.full_like(prediction, 0.5, requires_grad=True)
    weights = torch.tensor([[1.0, 0.0]], requires_grad=True)
    losses = compute_joint_finetuning_loss({}, {
        "joint_finetuning_delta_traj_pred_frenet": prediction,
        "joint_finetuning_teacher_delta_frenet": reference,
        "joint_finetuning_refine_supervision_weight": weights,
    }, ManeuverFlowConfig(training_phase="joint_finetuning"))
    loss = losses["joint_finetuning_delta_traj_loss"]
    torch.testing.assert_close(loss, torch.tensor(0.625))
    loss.backward()
    torch.testing.assert_close(prediction.grad[:, 0], torch.full((1, 8, 3), -2.5 / 24))
    assert torch.count_nonzero(prediction.grad[:, 1]) == 0
    assert reference.grad is None
    assert weights.grad is None


@pytest.mark.parametrize("shape", [(1, 5, 8, 8), (1, 1, 5, 64)])
def test_factor_objective_rejects_cached_grid_and_time_axis_labels(shape):
    predictions = {
        "nc_logits": torch.zeros(1, 64, 1),
        "binary_pdm_factor_logits": torch.zeros(1, 3, 64),
        "ep_logits": torch.zeros(1, 64),
        "ep_pred": torch.full((1, 64), 0.5),
    }
    with pytest.raises(ValueError, match="shape"):
        compute_metric_prism_factor_loss(torch.ones(shape), predictions)


def test_training_targets_cannot_be_injected_through_features():
    model = ManeuverFlowModel.__new__(ManeuverFlowModel)
    nn.Module.__init__(model)
    model._config = SimpleNamespace(training_phase="generation_pretraining")

    def unexpected_perception(features):
        raise AssertionError("Targets must be checked before perception.")

    model._forward_perception = unexpected_perception
    with pytest.raises(KeyError, match="targets"):
        model.forward({"__targets__": {"gt_trajectory": torch.zeros(1, 8, 3)}})


@pytest.mark.parametrize("proposal_count", [1, 6])
def test_m3f_uses_displacement_l1_and_detaches_its_target(proposal_count):
    prediction = torch.ones(1, proposal_count, 8, 4, requires_grad=True)
    target = torch.zeros_like(prediction, requires_grad=True)
    loss = compute_m3f_loss({}, {
        "m3f_velocity_prediction": prediction,
        "m3f_target_velocity": target,
    }, ManeuverFlowConfig())["m3f_loss"]
    torch.testing.assert_close(loss, torch.tensor(1.0))
    loss.backward()
    torch.testing.assert_close(prediction.grad, torch.full_like(prediction, 1.0 / prediction.numel()))
    assert target.grad is None


class _LossAgent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.config = SimpleNamespace(input_target=True, training_phase="joint_finetuning")
        self.updates = 0
        self.validation_calls = []

    def forward(self, features, targets):
        assert targets["target"].shape == features["input"].shape
        return self.weight * features["input"]

    def compute_loss(self, features, targets, prediction):
        aggregate = (prediction - targets["target"]).square().mean()
        return {"loss": aggregate, "component_loss": 100 * aggregate, "diagnostic_metric": aggregate.detach() + 1000}

    def get_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
        optimizer.register_step_post_hook(lambda *args: setattr(self, "updates", self.updates + 1))
        return optimizer

    def training_checkpoint_metadata(self):
        return {"training_phase": "joint_finetuning", "training_recipe": "test-ema-online64"}

    def inference_protocol_metadata(self):
        return {"ecfr_final_pool_composition": "source_replacement"}

    def validate_resume_checkpoint_metadata(self, checkpoint):
        self.validation_calls.append(checkpoint)
        if checkpoint.get("invalid"):
            raise ValueError("Resume metadata mismatch")


def test_wrapper_uses_authoritative_aggregate_and_delegates_checkpoint_metadata():
    agent = _LossAgent()
    wrapper = AgentLightningModule(agent)
    wrapper.log = lambda *args, **kwargs: None
    batch = ({"input": torch.ones(2, 1)}, {"target": torch.zeros(2, 1)})
    loss = wrapper.training_step(batch, 0)
    loss.backward()
    assert loss.item() == 1.0
    assert agent.weight.grad.item() == 2.0
    checkpoint = {}
    wrapper.on_save_checkpoint(checkpoint)
    assert checkpoint["navsim_agent_metadata"] == agent.training_checkpoint_metadata()
    assert checkpoint["inference_protocol"] == agent.inference_protocol_metadata()
    wrapper.on_load_checkpoint(checkpoint)
    assert agent.validation_calls == [checkpoint]
    with pytest.raises(ValueError, match="Resume metadata mismatch"):
        wrapper.on_load_checkpoint({"invalid": True})


def test_wrapper_does_not_duplicate_agent_optimizer_step_hook_with_gradient_accumulation():
    agent = _LossAgent()
    wrapper = AgentLightningModule(agent)
    batches = [({"input": torch.ones(1)}, {"target": torch.zeros(1)}) for _ in range(4)]
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        accumulate_grad_batches=2,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    trainer.fit(wrapper, train_dataloaders=DataLoader(batches, batch_size=1))
    assert trainer.global_step == 2
    assert agent.updates == 2


def test_entry_passes_full_resume_to_lightning_and_disables_stage1_initialization(monkeypatch, tmp_path):
    module = training_entry
    captured = {}
    agent = _LossAgent()
    agent.get_training_callbacks = lambda: []
    cfg = OmegaConf.create(
        {
            "agent": {
                "config": {"training_phase": "joint_finetuning", "max_epochs": 20, "metric_cache_path": "/metrics"},
                "initialize_from_checkpoint": True,
                "checkpoint_path": "/stage1.ckpt",
            },
            "trainer": {"params": {"max_epochs": 20, "precision": "32-true"}},
            "dataloader": {"params": {"num_workers": 0, "prefetch_factor": 2, "batch_size": 1}},
            "seed": 0,
            "output_dir": str(tmp_path),
            "use_cache_without_dataset": False,
            "resume_ckpt_path": "/resume.ckpt",
            "train_test_split": {"scene_filter": {"tokens": ["training_scene", "validation_scene"]}},
        }
    )

    def instantiate(config):
        captured["initialization"] = (config.initialize_from_checkpoint, config.checkpoint_path)
        return agent

    class _Trainer:
        def __init__(self, **kwargs):
            captured["trainer"] = kwargs

        def fit(self, **kwargs):
            captured["fit"] = kwargs

    class Samples(list):
        def __init__(self, token):
            super().__init__([({"input": torch.ones(1)}, {"target": torch.zeros(1)})])
            self.tokens = [token]

    train_samples, val_samples = Samples("training_scene"), Samples("validation_scene")
    monkeypatch.setattr(module, "instantiate", instantiate)
    monkeypatch.setattr(module, "build_datasets", lambda *args: (train_samples, val_samples))
    monkeypatch.setattr(module, "navtrain_scene_filters", lambda: (
        SimpleNamespace(tokens=["training_scene", "validation_scene"]), None,
    ))
    monkeypatch.setattr(pl, "Trainer", _Trainer)
    module.main.__wrapped__(cfg)
    assert captured["initialization"] == (False, None)
    assert captured["fit"]["ckpt_path"] == "/resume.ckpt"
    assert captured["fit"]["model"].training_data_metadata["train"]["num_scenes"] == 1
    assert any(callback.save_last for callback in captured["trainer"]["callbacks"] if hasattr(callback, "save_last"))


def test_training_entry_rejects_configured_navtrain_subset_before_trainer(monkeypatch, tmp_path):
    cfg = OmegaConf.merge(_navtrain_config(), {
        "agent": {"config": {"training_phase": "generation_pretraining", "max_epochs": 100}},
        "trainer": {"params": {"max_epochs": 100, "precision": "32-true"}},
        "dataloader": {"params": {"num_workers": 0, "batch_size": 1}},
        "seed": 0, "output_dir": str(tmp_path), "use_cache_without_dataset": False,
        "resume_ckpt_path": None,
    })
    selected = list(cfg.train_test_split.scene_filter.tokens[:2])
    cfg.train_test_split.scene_filter.tokens = selected

    class Samples(list):
        def __init__(self, token):
            super().__init__([({}, {})])
            self.tokens = [token]

    agent = _LossAgent()
    agent.get_training_callbacks = lambda: []
    monkeypatch.setattr(training_entry, "instantiate", lambda config: agent)
    monkeypatch.setattr(training_entry, "build_datasets", lambda *args: tuple(Samples(token) for token in selected))

    def unexpected_trainer(**kwargs):
        raise AssertionError("An incomplete main training partition must fail before constructing Trainer.")

    monkeypatch.setattr(pl, "Trainer", unexpected_trainer)
    with pytest.raises(ValueError, match="scene coverage"):
        training_entry.main.__wrapped__(cfg)


def test_future_supervision_has_constant_weight_and_trains_both_predictions():
    config = ManeuverFlowConfig(training_phase="joint_finetuning")
    absolute = torch.ones(1, 4, 2, 2, 2, requires_grad=True)
    residual = torch.ones_like(absolute, requires_grad=True)
    predictions = {
        "future_bev_absolute_feature_prediction": absolute,
        "future_bev_absolute_feature_target": torch.zeros_like(absolute),
        "future_bev_residual_feature_prediction": residual,
        "future_bev_residual_feature_target": torch.zeros_like(residual),
        "future_bev_feature_valid_mask": torch.ones(1, 4, dtype=torch.bool),
    }
    losses = compute_future_bev_feature_loss({}, predictions, config)
    assert losses["future_bev_feature_loss_weight_monitor"].item() == pytest.approx(0.5)
    assert losses["future_bev_feature_loss"].item() == pytest.approx(
        0.5 * losses["future_bev_feature_loss_raw_monitor"].item()
    )
    assert losses["future_bev_feature_loss"].item() > 0
    losses["future_bev_feature_loss"].backward()
    assert absolute.grad is not None and absolute.grad.abs().sum() > 0
    assert residual.grad is not None and residual.grad.abs().sum() > 0


def test_lightning_validation_retains_the_actual_m3f_jvp():
    model = ManeuverFlowModel.__new__(ManeuverFlowModel)
    nn.Module.__init__(model)

    def field(self, z, r, t, **kwargs):
        return z * t.view(-1, 1, 1, 1), None, None

    model._evaluate_m3f = MethodType(field, model)

    class ValidationModule(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.model = model
            self.derivative = None

        def validation_step(self, batch, batch_idx):
            z = torch.ones(1, 1, 1, 1)
            time = torch.full((1,), 0.25)
            _, derivative = self.model._run_m3f_jvp(
                z_t=z,
                tangent_v=torch.full_like(z, 2.0),
                r=torch.zeros_like(time),
                t=time,
                c_state=z,
                c_bev=z,
                c_agent=z,
                maneuver_component_weights=z,
            )
            self.derivative = derivative

    root = Path(__file__).resolve().parents[1]
    config = OmegaConf.load(root / "navsim/planning/script/config/training/default_training.yaml")
    module = ValidationModule()
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        inference_mode=config.trainer.params.get("inference_mode", True),
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.validate(module, dataloaders=DataLoader([torch.ones(1)]))
    # For u=z*t, D u[(v,0,1)] = t*v + z = .25*2 + 1.
    torch.testing.assert_close(module.derivative, torch.full((1, 1, 1, 1), 1.5))
