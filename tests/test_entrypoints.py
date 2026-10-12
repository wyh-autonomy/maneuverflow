"""Public TF training, checkpoint-export, evaluation, and submission entrypoints."""

import importlib.util
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
from types import SimpleNamespace

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
import numpy as np
import pytest

from navsim.planning.script.run_training import prepare_training_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "navsim/planning/script/config/training"


@pytest.fixture
def training_env(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    for suffix in ("navsim_logs/trainval", "sensor_blobs/trainval", "maps"):
        (data_root / suffix).mkdir(parents=True)
    cache = tmp_path / "current_cache"
    metrics = tmp_path / "metric_cache"
    prior = tmp_path / "prior"
    for directory in (cache, metrics, prior):
        directory.mkdir()
    checkpoint = tmp_path / "stage1.ckpt"
    checkpoint.touch()
    for name, shape in (
        ("displacement_mean", (8, 8, 2)),
        ("displacement_std", (8, 8, 2)),
        ("heading_mean", (8, 8)),
        ("heading_std", (8, 8)),
    ):
        np.save(prior / f"maneuver_{name}.npy", np.ones(shape, dtype=np.float32))
    np.save(prior / "maneuver_component_counts.npy", np.ones(8, dtype=np.int64))
    (prior / "maneuver_mgmp_metadata.json").write_text("{}\n", encoding="utf-8")
    values = {
        "OPENSCENE_DATA_ROOT": data_root.as_posix(),
        "NAVSIM_EXP_ROOT": (tmp_path / "output").as_posix(),
        "NUPLAN_MAPS_ROOT": (data_root / "maps").as_posix(),
        "MANEUVERFLOW_TRAIN_CACHE": cache.as_posix(),
        "MANEUVERFLOW_METRIC_CACHE": metrics.as_posix(),
        "MANEUVERFLOW_MGMP_PRIOR_DIR": prior.as_posix(),
        "MANEUVERFLOW_STAGE1_CHECKPOINT": checkpoint.as_posix(),
        "PYTHON_BIN": Path(sys.executable).as_posix(),
        "DRY_RUN": "1",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    for key in ("MANEUVERFLOW_STAGE1_MAX_EPOCHS", "MANEUVERFLOW_JOINT_MAX_EPOCHS", "RESUME_CKPT_PATH"):
        monkeypatch.delenv(key, raising=False)
    return values


def _compose(agent: str, overrides=()):
    assert CONFIG_DIR.is_dir(), "TF training configuration must be published"
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(
            config_name="default_training",
            overrides=["experiment_name=contract", "train_test_split=navtrain", f"agent={agent}", *overrides],
        )


@pytest.mark.parametrize(
    ("agent", "epochs", "lr", "weights"),
    [
        ("maneuverflow_generation_pretraining", 100, 4e-4, (1, 4, 0, 0)),
        ("maneuverflow_joint_finetuning", 20, 2e-4, (1, 4, 2, 2)),
    ],
)
def test_tf_phase_configs_compose_with_recipe_defaults(training_env, agent, epochs, lr, weights):
    cfg = _compose(agent)
    instantiate(cfg.agent.config)
    assert cfg.agent.config.max_epochs == cfg.trainer.params.max_epochs == epochs
    assert cfg.agent.lr == pytest.approx(lr)
    assert cfg.trainer.params.precision == "32-true"
    assert cfg.trainer.params.gradient_clip_val == 1.0
    assert cfg.agent.config.weight_decay == 0.1
    assert (
        tuple(
            cfg.agent.config[key]
            for key in (
                "auxiliary_loss_weight",
                "m3f_loss_weight",
                "verification_loss_weight",
                "refinement_loss_weight",
            )
        )
        == weights
    )


def test_joint_config_has_online64_ema_and_static_metric_cache(training_env):
    cfg = _compose("maneuverflow_joint_finetuning")
    config = instantiate(cfg.agent.config)
    assert config.metric_cache_path == training_env["MANEUVERFLOW_METRIC_CACHE"]
    assert config.ema_target_decay == 0.999
    assert config.ema_target_buffer_policy == "copy"
    assert config.joint_probability_source_count == 64
    assert config.future_bev_loss_decay_epochs == 0
    assert config.future_bev_loss_weight == 0.5
    assert config.joint_generation_lr_multiplier == 0.2


@pytest.mark.parametrize("overrides, error", [
    (["agent.checkpoint_path=null"], "checkpoint_path"),
    (["agent.checkpoint_path=/missing-stage1.ckpt"], "Stage-1 checkpoint"),
    (["agent.initialize_from_checkpoint=false"], "initialize_from_checkpoint"),
])
def test_direct_fresh_joint_entry_requires_enabled_real_stage1_handoff(training_env, overrides, error):
    cfg = _compose("maneuverflow_joint_finetuning", overrides)

    with pytest.raises((ValueError, FileNotFoundError), match=error):
        prepare_training_config(cfg)


def test_direct_joint_full_resume_bypasses_stage1_handoff(training_env):
    cfg = _compose("maneuverflow_joint_finetuning", [
        "agent.checkpoint_path=null", "resume_ckpt_path=/resume.ckpt",
    ])

    prepare_training_config(cfg)

    assert cfg.agent.initialize_from_checkpoint is False
    assert cfg.agent.checkpoint_path is None


def _run_script(name: str, bash_executable, env=None):
    script = ROOT / "scripts/training" / name
    assert script.is_file(), f"Training entrypoint must be published: {name}"
    return subprocess.run(
        [bash_executable, script.as_posix()], cwd=ROOT, env=env, capture_output=True, text=True, check=False
    )


@pytest.mark.parametrize(
    ("name", "epochs", "agent", "cache_only"),
    [
        ("train_stage1.sh", 100, "maneuverflow_generation_pretraining", "true"),
        ("train_stage2.sh", 20, "maneuverflow_joint_finetuning", "false"),
    ],
)
def test_training_shell_dry_run_builds_tf_only_command(training_env, name, epochs, agent, cache_only, bash_executable):
    result = _run_script(name, bash_executable)
    assert result.returncode == 0, result.stderr
    assert f"agent={agent}" in result.stdout
    assert f"trainer.params.max_epochs={epochs}" in result.stdout
    assert f"use_cache_without_dataset={cache_only}" in result.stdout
    assert "trainer.params.precision=32-true" in result.stdout
    assert "trainer.params.gradient_clip_val=1.0" in result.stdout


def test_joint_shell_requires_metric_cache_and_validates_global_batch(training_env, bash_executable):
    env = dict(os.environ)
    env.pop("MANEUVERFLOW_METRIC_CACHE")
    missing = _run_script("train_stage2.sh", bash_executable, env)
    assert missing.returncode != 0
    assert "MANEUVERFLOW_METRIC_CACHE" in missing.stderr
    env = dict(os.environ, NUM_GPUS="1", PER_DEVICE_BATCH_SIZE="8")
    wrong_batch = _run_script("train_stage2.sh", bash_executable, env)
    assert wrong_batch.returncode != 0
    assert "global batch size 32" in wrong_batch.stderr


def test_joint_shell_resume_uses_full_state_without_stage1_handoff(
    training_env, tmp_path, monkeypatch, bash_executable
):
    resume = tmp_path / "resume.ckpt"
    resume.touch()
    monkeypatch.setenv("RESUME_CKPT_PATH", resume.as_posix())
    monkeypatch.delenv("MANEUVERFLOW_STAGE1_CHECKPOINT")
    result = _run_script("train_stage2.sh", bash_executable)
    assert result.returncode == 0, result.stderr
    assert "resume_ckpt_path=" in result.stdout
    assert "agent.initialize_from_checkpoint=false" in result.stdout
    assert "agent.checkpoint_path=null" in result.stdout


def test_current_cache_and_prior_entrypoints_are_tf_only(training_env, bash_executable):
    result = _run_script("cache_dataset.sh", bash_executable)
    assert result.returncode == 0, result.stderr
    assert "agent=maneuverflow_generation_pretraining" in result.stdout
    assert "run_dataset_caching.py" in result.stdout
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/training/build_mgmp_prior.py"), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--navsim-log-path" in result.stdout
    assert "--input-trajectories" not in result.stdout


def test_prior_builds_from_training_logs_and_matches_navsim_trajectory(tmp_path):
    from pyquaternion import Quaternion
    from navsim.agents.maneuverflow.mgmp_prior import load_maneuver_gmp_prior
    from navsim.common.dataclasses import Scene
    from navsim.planning.training.data_split import navtrain_scene_filters, population_metadata

    path = ROOT / "scripts/training/build_mgmp_prior.py"
    spec = importlib.util.spec_from_file_location("prior_cli_training_split", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    train_filter, val_filter = navtrain_scene_filters()
    tokens = train_filter.tokens[:9]

    def raw_frames(token, log_name, family):
        steps = np.full(8, 2.0)
        if family == 0:
            steps[:] = 0.2
        elif family == 2:
            steps = np.linspace(1.0, 4.0, 8)
        elif family == 3:
            steps = np.linspace(4.0, 1.0, 8)
        yaw = np.deg2rad([0, 0, 0, 0, 8, 20, -8, -20][family])
        xy = np.cumsum(steps)
        poses = [(10.0, -5.0, np.pi / 2)] * 4
        poses += [(10.0, -5.0 + float(x), np.pi / 2 + yaw) for x in xy]
        poses += poses[-1:] * 2
        return [{
            "token": token if i == 3 else f"unused_{token}_{i}",
            "log_name": log_name, "roadblock_ids": ["road"],
            "ego2global_translation": [x, y, 0.0],
            "ego2global_rotation": Quaternion(axis=[0, 0, 1], angle=heading).elements,
            "ego_dynamic_state": np.zeros(4), "driving_command": np.array([0, 1, 0, 0]),
        } for i, (x, y, heading) in enumerate(poses)]

    train_log = train_filter.log_names[0]
    training_frames = [raw_frames(tokens[i], train_log, i) for i in range(8)]
    with (tmp_path / f"{train_log}.pkl").open("wb") as stream:
        pickle.dump([frame for scene in training_frames for frame in scene], stream)
    validation_log = val_filter.log_names[0]
    with (tmp_path / f"{validation_log}.pkl").open("wb") as stream:
        pickle.dump(raw_frames(tokens[8], validation_log, 0), stream)
    with (tmp_path / "outside_navtrain.pkl").open("wb") as stream:
        pickle.dump(raw_frames(tokens[8], "outside_navtrain", 0), stream)

    trajectories, population = module.load_training_trajectories(tmp_path)

    assert trajectories.shape == (8, 8, 3)
    assert population == population_metadata(tokens[:8])
    expected = {}
    for token, frames in zip(tokens[:8], training_frames):
        scene = Scene(
            scene_metadata=SimpleNamespace(num_history_frames=4), map_api=None,
            frames=[SimpleNamespace(ego_status=Scene._build_ego_status(frame)) for frame in frames],
        )
        expected[token] = scene.get_future_trajectory(num_trajectory_frames=8).poses.astype(np.float32)
    np.testing.assert_array_equal(trajectories, np.stack([expected[token] for token in sorted(expected)]))
    output = tmp_path.parent / (tmp_path.name + "_prior")
    assert module.main(["--navsim-log-path", str(tmp_path), "--output-dir", str(output)]) == 0
    prior = load_maneuver_gmp_prior(output)
    np.testing.assert_array_equal(prior["component_counts"], np.ones(8, dtype=np.int64))
    metadata = json.loads((output / "maneuver_mgmp_metadata.json").read_text(encoding="utf-8"))
    assert metadata["training_data"] == {"split": "navtrain", "partition": "train", **population}
    assert "tokens" not in metadata["training_data"]
    assert str(tmp_path) not in json.dumps(metadata)


def test_export_cli_accepts_checkpoint_paths_without_model_variants():
    path = ROOT / "scripts/evaluation/export_checkpoint.py"
    assert path.is_file(), "Checkpoint export CLI must be published"
    spec = importlib.util.spec_from_file_location("tf_export_cli_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = module.parse_args(["--input", "joint.ckpt", "--output", "inference.pt"])
    assert args.input == Path("joint.ckpt")
    assert args.output == Path("inference.pt")
    assert not hasattr(args, "ecfr_final_pool_composition")
    assert not hasattr(args, "perception_variant")


@pytest.mark.parametrize(
    "config_name",
    ["default_run_pdm_score", "default_run_create_submission_pickle"],
)
def test_default_evaluation_config_uses_released_tf_agent(config_name, monkeypatch, tmp_path):
    monkeypatch.setenv("NAVSIM_DEVKIT_ROOT", ROOT.as_posix())
    monkeypatch.setenv("NAVSIM_EXP_ROOT", tmp_path.as_posix())
    with initialize_config_dir(
        config_dir=str(ROOT / "navsim/planning/script/config/pdm_scoring"),
        version_base=None,
    ):
        config = compose(config_name=config_name, overrides=["experiment_name=release"])
    assert config.agent._target_ == "navsim.agents.maneuverflow.agent.ManeuverFlowAgent"
    agent_config = instantiate(config.agent.config)
    assert agent_config.training_phase == "inference"
    assert agent_config.perception_variant == "TF"
    assert agent_config.mgmp_assets_from_checkpoint is True
    assert agent_config.ecfr_final_pool_composition == "independent_ranked_retention"


@pytest.fixture
def evaluation_env(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    for suffix in ("navsim_logs/test", "sensor_blobs/test", "maps", "metric_cache"):
        (data_root / suffix).mkdir(parents=True)
    for name, value in {
        "OPENSCENE_DATA_ROOT": data_root.as_posix(),
        "NAVSIM_EXP_ROOT": (tmp_path / "experiments").as_posix(),
        "NUPLAN_MAPS_ROOT": (data_root / "maps").as_posix(),
        "METRIC_CACHE_PATH": (data_root / "metric_cache").as_posix(),
        "OUTPUT_DIR": (tmp_path / "submission").as_posix(),
        "TEAM_NAME": "test team",
        "AUTHORS": "test authors",
        "EMAIL": "test@example.org",
        "INSTITUTION": "test institution",
        "COUNTRY": "test country",
        "PYTHON_BIN": Path(sys.executable).as_posix(),
        "DRY_RUN": "1",
        "MAX_SCENES": "1",
    }.items():
        monkeypatch.setenv(name, value)
    for name in ("CHECKPOINT", "AGENT_CONFIG"):
        monkeypatch.delenv(name, raising=False)
    checkpoint = tmp_path / "ema.pt"
    checkpoint.touch()
    return checkpoint


def _evaluation_command(bash_executable, name):
    script = ROOT / "scripts/evaluation" / name
    assert script.is_file(), f"Evaluation entrypoint must be published: {name}"
    return subprocess.run(
        [bash_executable, script.as_posix()],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("name", ["evaluate.sh", "create_submission.sh"])
def test_evaluation_shell_defaults_to_bundled_checkpoint(name, bash_executable, evaluation_env):
    result = _evaluation_command(bash_executable, name)
    assert result.returncode == 0, result.stderr
    assert "agent=maneuverflow_agent" in result.stdout
    assert "/checkpoints/maneuverflow_tf_0.906.pt" in result.stdout
    assert "max_scenes=1" in result.stdout


@pytest.mark.parametrize("name", ["evaluate.sh", "create_submission.sh"])
def test_evaluation_shell_supports_explicit_ema_checkpoint(name, bash_executable, evaluation_env, monkeypatch):
    monkeypatch.setenv("AGENT_CONFIG", "maneuverflow_ema_agent")
    monkeypatch.setenv("CHECKPOINT", evaluation_env.as_posix())
    result = _evaluation_command(bash_executable, name)
    assert result.returncode == 0, result.stderr
    assert "agent=maneuverflow_ema_agent" in result.stdout
    assert "ema.pt" in result.stdout


@pytest.mark.parametrize("name", ["evaluate.sh", "create_submission.sh"])
def test_ema_evaluation_requires_explicit_checkpoint(name, bash_executable, evaluation_env, monkeypatch):
    monkeypatch.setenv("AGENT_CONFIG", "maneuverflow_ema_agent")
    result = _evaluation_command(bash_executable, name)
    assert result.returncode != 0
    assert "Set CHECKPOINT" in result.stderr
