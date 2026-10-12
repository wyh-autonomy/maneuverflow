"""Train a NAVSIM agent with explicit Stage-1 handoff and full-state resume."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Tuple

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, open_dict

from navsim.planning.training.data_split import navtrain_scene_filters, training_data_metadata

if TYPE_CHECKING:
    from navsim.agents.abstract_agent import AbstractAgent
    from navsim.planning.training.dataset import Dataset


logger = logging.getLogger(__name__)
CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"


def prepare_training_config(cfg: DictConfig) -> None:
    """Validate the TF recipe before constructing its model or loading weights."""
    phase = cfg.agent.config.get("training_phase")
    if phase not in {"generation_pretraining", "joint_finetuning"}:
        return
    model_epochs = int(cfg.agent.config.max_epochs)
    trainer_epochs = int(cfg.trainer.params.max_epochs)
    if model_epochs <= 0 or model_epochs != trainer_epochs:
        raise ValueError(
            "agent.config.max_epochs and trainer.params.max_epochs must be "
            "the same positive value so the scheduler matches training. "
            "Override agent.config.max_epochs; the trainer default follows it."
        )
    if str(cfg.trainer.params.precision) not in {"32", "32-true"}:
        raise ValueError("ManeuverFlow core training requires FP32 (precision=32-true).")
    if cfg.trainer.params.get("inference_mode", False) is not False:
        raise ValueError(
            "ManeuverFlow validation requires trainer.params.inference_mode=false "
            "so the M3F directional derivative remains active."
        )
    with open_dict(cfg.trainer.params):
        cfg.trainer.params.inference_mode = False
    strategy = cfg.trainer.params.get("strategy", "auto")
    if (
        isinstance(strategy, str)
        and strategy.startswith("ddp")
        and not strategy.endswith("_find_unused_parameters_true")
    ):
        raise ValueError(
            "Sparse maneuver experts require a DDP strategy with unused-parameter "
            "detection; use ddp_find_unused_parameters_true."
        )
    if phase == "joint_finetuning":
        if cfg.use_cache_without_dataset:
            raise ValueError(
                "Joint training requires SceneLoader for future raw observations; "
                "set use_cache_without_dataset=false."
            )
        if not cfg.agent.config.get("metric_cache_path"):
            raise ValueError("Joint training requires metric_cache_path for static scene metrics.")
    if cfg.get("resume_ckpt_path"):
        # Lightning restores the whole online/teacher state, optimizers and
        # scheduler. A separate weight handoff would initialize from the wrong
        # lineage before that strict restore.
        with open_dict(cfg.agent):
            cfg.agent.initialize_from_checkpoint = False
            cfg.agent.checkpoint_path = None
    elif phase == "joint_finetuning":
        if not cfg.agent.get("initialize_from_checkpoint", False):
            raise ValueError("Fresh joint training requires initialize_from_checkpoint=true for the Stage-1 handoff.")
        checkpoint_path = cfg.agent.get("checkpoint_path")
        if not checkpoint_path:
            raise ValueError("Fresh joint training requires a Stage-1 checkpoint_path.")
        if not Path(checkpoint_path).expanduser().is_file():
            raise FileNotFoundError(f"Stage-1 checkpoint not found: {checkpoint_path}")


def _partition_scene_filters(cfg: DictConfig):
    return navtrain_scene_filters(cfg.train_test_split.scene_filter, cfg.train_logs, cfg.val_logs)


def build_cache_only_datasets(cfg: DictConfig, agent: AbstractAgent):
    """Apply the selected scene population to Stage-1 caches without sensors."""
    from navsim.planning.training.dataset import CacheOnlyDataset

    datasets = []
    for scene_filter in _partition_scene_filters(cfg):
        datasets.append(CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=scene_filter.log_names,
            tokens=scene_filter.tokens,
            max_scenes=scene_filter.max_scenes,
        ))
    return tuple(datasets)


def build_datasets(cfg: DictConfig, agent: AbstractAgent) -> Tuple[Dataset, Dataset]:
    """Combine the current cache with SceneLoader's future raw observations."""
    from navsim.common.dataloader import SceneLoader
    from navsim.planning.training.dataset import Dataset

    datasets = []
    for scene_filter in _partition_scene_filters(cfg):
        scene_loader = SceneLoader(
            sensor_blobs_path=Path(cfg.sensor_blobs_path),
            data_path=Path(cfg.navsim_log_path),
            scene_filter=scene_filter,
            sensor_config=agent.get_sensor_config(),
        )
        datasets.append(Dataset(
            scene_loader=scene_loader,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            cache_path=cfg.cache_path,
            force_cache_computation=cfg.force_cache_computation,
        ))
    return tuple(datasets)


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    prepare_training_config(cfg)

    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint
    from torch.utils.data import DataLoader

    from navsim.planning.training.agent_lightning_module import AgentLightningModule

    pl.seed_everything(cfg.seed, workers=True)
    logger.info("Training outputs: %s", cfg.output_dir)
    agent = instantiate(cfg.agent)
    lightning_module = AgentLightningModule(agent)
    if cfg.use_cache_without_dataset:
        if cfg.force_cache_computation or cfg.cache_path is None:
            raise ValueError("Cache-only training requires cache_path and force_cache_computation=false.")
        train_data, val_data = build_cache_only_datasets(cfg, agent)
    else:
        train_data, val_data = build_datasets(cfg, agent)
    if len(train_data) == 0 or len(val_data) == 0:
        raise ValueError("Training and validation partitions must both contain samples; check data/cache coverage.")
    lightning_module.training_data_metadata = training_data_metadata(
        train_data.tokens, val_data.tokens, expected_tokens=navtrain_scene_filters()[0].tokens,
    )
    logger.info("navtrain: %d training scenes; %d validation scenes.", len(train_data), len(val_data))
    loader_params = dict(cfg.dataloader.params)
    if int(loader_params.get("num_workers", 0)) == 0:
        loader_params.pop("prefetch_factor", None)
    train_loader = DataLoader(train_data, **loader_params, shuffle=True)
    val_loader = DataLoader(val_data, **loader_params, shuffle=False)
    callbacks = list(agent.get_training_callbacks())
    callbacks.append(ModelCheckpoint(monitor="val/loss", mode="min", save_top_k=1, save_last=True))
    trainer = pl.Trainer(**cfg.trainer.params, callbacks=callbacks)
    fit_kwargs = dict(model=lightning_module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    if cfg.resume_ckpt_path:
        fit_kwargs["ckpt_path"] = cfg.resume_ckpt_path
    trainer.fit(**fit_kwargs)


if __name__ == "__main__":
    main()
