"""Shared navtrain selection for main-model training and prior construction."""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
import hashlib
from pathlib import Path
from typing import Optional, Sequence

from hydra.utils import instantiate
from omegaconf import OmegaConf


CONFIG_ROOT = Path(__file__).resolve().parents[1] / "script/config"


@lru_cache(maxsize=1)
def _navtrain_defaults():
    scene_filter = OmegaConf.load(CONFIG_ROOT / "common/train_test_split/scene_filter/navtrain.yaml")
    partitions = OmegaConf.load(CONFIG_ROOT / "training/default_train_val_test_log_split.yaml")
    return scene_filter, partitions


def navtrain_scene_filters(scene_filter_cfg=None, train_logs=None, val_logs=None):
    """Select disjoint subsets of the official navtrain train/validation logs."""
    canonical, partitions = _navtrain_defaults()
    train_logs = list(partitions.train_logs if train_logs is None else train_logs)
    val_logs = list(partitions.val_logs if val_logs is None else val_logs)
    if set(train_logs).intersection(val_logs):
        raise ValueError("Training and validation log partitions must be disjoint.")
    if not set(train_logs).issubset(partitions.train_logs):
        raise ValueError("Training logs must stay within the navtrain training log partition.")
    if not set(val_logs).issubset(partitions.val_logs):
        raise ValueError("Validation logs must stay within the navtrain validation log partition.")

    scene_filter = instantiate(canonical if scene_filter_cfg is None else scene_filter_cfg)
    if scene_filter.tokens is None or not set(scene_filter.tokens).issubset(canonical.tokens):
        raise ValueError("Training requires the selected navtrain tokens; other scene populations are unsupported.")
    if scene_filter.log_names is None or not set(scene_filter.log_names).issubset(canonical.log_names):
        raise ValueError("Training requires navtrain log names.")
    for field in ("num_history_frames", "num_future_frames", "frame_interval", "has_route"):
        if getattr(scene_filter, field) != canonical[field]:
            raise ValueError(f"Training requires the navtrain scene filter's {field}={canonical[field]}.")
    result = []
    for logs in (train_logs, val_logs):
        allowed_logs = set(logs)
        result.append(replace(scene_filter, log_names=[name for name in scene_filter.log_names if name in allowed_logs]))
    return tuple(result)


def population_metadata(tokens: Sequence[str]) -> dict[str, object]:
    """Identify selected scenes without saving their raw token list."""
    unique_tokens = sorted(set(tokens))
    if not unique_tokens or len(unique_tokens) != len(tokens):
        raise ValueError("A training population must contain nonempty, unique scene tokens.")
    return {
        "num_scenes": len(unique_tokens),
        "tokens_sha256": hashlib.sha256("\n".join(unique_tokens).encode("utf-8")).hexdigest(),
    }


def training_data_metadata(
    train_tokens: Sequence[str], val_tokens: Sequence[str], expected_tokens: Optional[Sequence[str]] = None,
) -> dict[str, object]:
    if set(train_tokens).intersection(val_tokens):
        raise ValueError("Training and validation scene tokens must be disjoint.")
    if expected_tokens is not None and set(train_tokens).union(val_tokens) != set(expected_tokens):
        raise ValueError("Training/validation scene coverage differs from the selected navtrain tokens; check logs and caches.")
    return {
        "split": "navtrain",
        "train": population_metadata(train_tokens),
        "validation": population_metadata(val_tokens),
    }
