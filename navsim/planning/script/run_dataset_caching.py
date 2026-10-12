"""Cache current observations and supervision without constructing future targets."""

from pathlib import Path
from typing import Dict, List
import logging

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from nuplan.planning.utils.multithreading.worker_utils import worker_map
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SensorConfig
from navsim.planning.script.builders.worker_pool_builder import build_worker
from navsim.planning.training.dataset import Dataset


logger = logging.getLogger(__name__)


def cache_features(args: List[Dict]) -> List:
    cfg = args[0]["cfg"]
    agent = instantiate(cfg.agent)
    scene_filter = instantiate(cfg.train_test_split.scene_filter)
    scene_filter.log_names = [entry["log_file"] for entry in args]
    scene_filter.tokens = [token for entry in args for token in entry["tokens"]]
    loader = SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
    )
    Dataset(
        scene_loader=loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )
    return []


@hydra.main(config_path="config/training", config_name="default_training", version_base=None)
def main(cfg: DictConfig) -> None:
    if not cfg.cache_path:
        raise ValueError("cache_path is required for dataset caching.")
    worker = build_worker(cfg)
    loader = SceneLoader(
        sensor_blobs_path=None,
        data_path=Path(cfg.navsim_log_path),
        scene_filter=instantiate(cfg.train_test_split.scene_filter),
        sensor_config=SensorConfig.build_no_sensors(),
    )
    items = [dict(cfg=cfg, log_file=log, tokens=tokens) for log, tokens in loader.get_tokens_list_per_log().items()]
    worker_map(worker, cache_features, items)
    logger.info("Cached current observations and supervision for %d scenes", len(loader))


if __name__ == "__main__":
    main()
