from __future__ import annotations

import json
from pathlib import Path

import pytorch_lightning as pl

from torch import Tensor
from typing import TYPE_CHECKING, Dict, Tuple

if TYPE_CHECKING:
    from navsim.agents.abstract_agent import AbstractAgent


class AgentLightningModule(pl.LightningModule):
    """Pytorch lightning wrapper for learnable agent."""

    def __init__(self, agent: AbstractAgent):
        """
        Initialise the lightning module wrapper.
        :param agent: agent interface in NAVSIM
        """
        super().__init__()
        self.agent = agent
        self.training_data_metadata = None
        self._resumed_training_population = False

    @staticmethod
    def _is_optimization_loss_key(loss_key: str) -> bool:
        """Return whether a loss-dict entry is an optimization objective."""
        return loss_key == "loss" or loss_key.endswith("_loss")

    def _step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], logging_prefix: str) -> Tensor:
        """
        Propagates the model forward and backwards and computes/logs losses and metrics.
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param logging_prefix: prefix where to log step
        :return: scalar loss
        """
        features, targets = batch
        config = getattr(self.agent, "config", None)
        if bool(getattr(config, "input_target", False)):
            prediction = self.agent.forward(features, targets)
        else:
            prediction = self.agent.forward(features)
        loss = self.agent.compute_loss(features, targets, prediction)

        if isinstance(loss, Tensor):
            total_loss = loss
        elif isinstance(loss, dict):
            # A provided aggregate is authoritative. Otherwise, sum the named
            # optimization terms while leaving metrics/monitors log-only.
            has_aggregate_loss = "loss" in loss
            optimization_losses = []
            for loss_key, loss_value in loss.items():
                is_optimization_loss = (
                    loss_key == "loss"
                    if has_aggregate_loss
                    else self._is_optimization_loss_key(loss_key)
                )
                if is_optimization_loss:
                    optimization_losses.append(loss_value)
                if loss_key != "loss":
                    self.log(
                        f"{logging_prefix}/{loss_key}",
                        loss_value,
                        on_step=True,
                        on_epoch=True,
                        prog_bar=is_optimization_loss,
                        sync_dist=True,
                    )

            if not optimization_losses:
                raise ValueError("Loss dictionary does not contain any optimization loss keys.")
            total_loss = sum(optimization_losses[1:], optimization_losses[0])
        else:
            raise TypeError(f"Agent compute_loss must return a Tensor or dict, got {type(loss).__name__}.")

        self.log(
            f"{logging_prefix}/loss",
            total_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return total_loss

    def training_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int) -> Tensor:
        """
        Step called on training samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        return self._step(batch, "train")

    def validation_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int):
        """
        Step called on validation samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        return self._step(batch, "val")

    def configure_optimizers(self):
        """Inherited, see superclass."""
        return self.agent.get_optimizers()

    def on_fit_start(self) -> None:
        """Check the restored prior and data selection before any optimizer step."""
        if self.training_data_metadata is None:
            return
        population = self.training_data_metadata["train"]
        prior = self.agent.model.hypothesizer.maneuver_gmp
        if int(prior.component_counts.sum().item()) != population["num_scenes"]:
            raise ValueError("M-GMP prior does not match the training population; rebuild it from training logs.")
        if self._resumed_training_population:
            return
        if self.agent.config.training_phase == "generation_pretraining":
            from navsim.agents.maneuverflow.mgmp_prior import MGMP_PRIOR_METADATA_FILENAME

            path = Path(self.agent.config.mgmp_prior_dir) / MGMP_PRIOR_METADATA_FILENAME
            if not path.is_file():
                raise ValueError("M-GMP prior has no training population metadata; rebuild it from training logs.")
            metadata = json.loads(path.read_text(encoding="utf-8"))
            expected = {"split": "navtrain", "partition": "train", **population}
            if metadata.get("training_data") != expected:
                raise ValueError("M-GMP prior and the actual training population differ; check logs and cache coverage.")
        elif getattr(self.agent, "_source_training_data_metadata", None) != self.training_data_metadata:
            raise ValueError("Stage-1 checkpoint and Stage-2 training population differ; use the same navtrain partition.")

    def on_save_checkpoint(self, checkpoint: Dict) -> None:
        """Persist the agent/phase identity needed for strict stage handoffs."""

        config = getattr(self.agent, "config", None)
        training_phase = getattr(config, "training_phase", None)
        if training_phase is None:
            return
        if self.training_data_metadata is not None:
            checkpoint["training_data"] = self.training_data_metadata
        metadata_builder = getattr(
            self.agent,
            "training_checkpoint_metadata",
            None,
        )
        if callable(metadata_builder):
            checkpoint["navsim_agent_metadata"] = metadata_builder()
        else:
            checkpoint["navsim_agent_metadata"] = {
                "agent_name": self.agent.name(),
                "training_phase": str(training_phase),
            }
        protocol_builder = getattr(
            self.agent,
            "inference_protocol_metadata",
            None,
        )
        if callable(protocol_builder):
            checkpoint["inference_protocol"] = protocol_builder()

    def on_load_checkpoint(self, checkpoint: Dict) -> None:
        """Validate optional agent-specific full-state resume lineage."""

        if self.training_data_metadata is not None:
            if checkpoint.get("training_data") != self.training_data_metadata:
                raise ValueError("Resume checkpoint and current training population differ.")
            self._resumed_training_population = True
        validator = getattr(
            self.agent,
            "validate_resume_checkpoint_metadata",
            None,
        )
        if callable(validator):
            validator(checkpoint)
