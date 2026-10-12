from collections.abc import Mapping
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.lr_scheduler import _LRScheduler
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, SensorConfig, Trajectory
from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
)

from .model import ManeuverFlowModel
from .config import (
    MANEUVERFLOW_INFERENCE_POLICY,
    MANEUVERFLOW_TRAINING_RECIPE,
    ManeuverFlowConfig,
)
from .losses import compute_maneuverflow_loss
from .inference_api import ManeuverFlowInferenceOutput

from .targets import (
    ManeuverFlowTargetBuilder,
)
from .features import ManeuverFlowFeatureBuilder
from .checkpoint_schema import (
    CHECKPOINT_KEY_SCHEMA,
    EMA_TARGET_PREFIX,
    PAPER_V1_SCHEMA,
    STAGE1_SOURCE_RECIPES,
    canonical_keyed_tensor_sha256,
    detect_state_dict_schema,
    inference_protocol_from_config,
    validate_checkpoint_envelope,
    validate_finite_state_dict,
    validate_ema_training_state_dict,
    validate_mgmp_inference_buffers,
    validate_perception_variant_state_dict,
    validate_stage1_lineage_metadata,
    validate_optimizer_metadata,
    validate_stage1_state_dict_against_model,
    validate_state_dict_against_model,
)


def _normalize_checkpoint_state_dict(state_dict: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """Remove one Lightning prefix while rejecting aliases and non-tensors."""

    normalized = {}
    for key, value in state_dict.items():
        if not isinstance(key, str) or not torch.is_tensor(value):
            raise TypeError("Checkpoint state_dict must map string keys to tensors.")
        if key.startswith("agent."):
            key = key[len("agent.") :]
        if key in normalized:
            raise ValueError(f"Duplicate normalized checkpoint key: {key!r}")
        normalized[key] = value
    return normalized


class ManeuverFlowAgent(AbstractAgent):
    """NAVSIM agent for the paper's Hypothesize--Verify--Improve policy.

    ManeuverFlow first samples maneuver-conditioned hypotheses with M-GMP and
    M³F, verifies them against one shared future-BEV evidence field through
    Metric Prism, and improves the selected subset with ECFR before a joint
    final re-score.
    """
    def __init__(
        self,
        config: ManeuverFlowConfig,
        trajectory_sampling: TrajectorySampling,
        lr: Optional[float] = None,
        checkpoint_path: Optional[str] = None,
        slice_indices: Optional[List[int]] = None,
        initialize_from_checkpoint: bool = False,
    ):
        super().__init__()
        self.config = config
        self._trajectory_sampling = trajectory_sampling
        self._lr = float(lr) if lr is not None else (
            4e-4 if config.training_phase == "generation_pretraining" else 2e-4
        )
        if not math.isfinite(self._lr) or self._lr <= 0.0:
            raise ValueError("lr must be finite and positive.")
        self._checkpoint_path = checkpoint_path
        self.slice_indices = [3] if slice_indices is None else list(slice_indices)

        self.max_epochs = int(config.max_epochs)
        self.min_lr = float(config.min_lr)

        self.model = ManeuverFlowModel(config)

        self._checkpoint_initialized = False
        self._initialized_checkpoint_tensor_sha256: Optional[str] = None
        self._stage1_source_recipe: Optional[str] = None

        if initialize_from_checkpoint:
            self.initialize()

    def name(self) -> str:
        return self.__class__.__name__

    def inference_protocol_metadata(self) -> dict[str, object]:
        """Return protocol metadata for strict checkpoint export."""

        protocol = inference_protocol_from_config(self.config)
        if self.config.training_phase != "inference":
            protocol["ecfr_bounds"] = [
                MANEUVERFLOW_INFERENCE_POLICY.ecfr_max_ds,
                MANEUVERFLOW_INFERENCE_POLICY.ecfr_max_dl,
                MANEUVERFLOW_INFERENCE_POLICY.ecfr_max_dpsi,
            ]
        return protocol

    def training_checkpoint_metadata(self) -> dict[str, object]:
        """Record the actual EMA recipe and full-resume duration contract."""

        metadata: dict[str, object] = {
            "agent_name": self.name(),
            "training_phase": self.config.training_phase,
            "training_recipe": MANEUVERFLOW_TRAINING_RECIPE,
            "perception_variant": self.config.perception_variant,
            "max_epochs": int(self.max_epochs),
            "learning_rate": float(self._lr),
            "warmup_epochs": int(self.config.warmup_epochs),
            "min_lr": float(self.min_lr),
            "weight_decay": float(self.config.weight_decay),
            "bev_coordinate_convention": self.config.bev_coordinate_convention,
        }
        if self.config.training_phase == "joint_finetuning":
            metadata.update({
                "ema_target_decay": float(self.config.ema_target_decay),
                "ema_target_buffer_policy": self.config.ema_target_buffer_policy,
                "future_bev_loss_weight": float(self.config.future_bev_loss_weight),
                "future_bev_loss_decay_epochs": int(self.config.future_bev_loss_decay_epochs),
            })
            source_recipe = getattr(self, "_stage1_source_recipe", None)
            metadata.update({
                "stage1_source_recipe": source_recipe,
                "stage1_source_tensor_sha256": self._initialized_checkpoint_tensor_sha256,
            })
            validate_stage1_lineage_metadata(metadata)
        return metadata

    def validate_resume_checkpoint_metadata(
        self,
        checkpoint: Mapping[str, Any],
    ) -> None:
        """Reject full-state resumes from another phase or ManeuverFlow protocol."""

        if self.config.training_phase not in {
            "generation_pretraining",
            "joint_finetuning",
        }:
            raise RuntimeError(
                "ManeuverFlow full-state resume is available only for a "
                "released training phase."
            )

        actual_metadata = checkpoint.get("navsim_agent_metadata")
        if isinstance(actual_metadata, Mapping):
            lineage = validate_stage1_lineage_metadata(actual_metadata)
            # Lineage is inherited state; the recipe/EMA/duration are runtime
            # contracts and must still match exactly below.
            self._stage1_source_recipe = lineage.get("stage1_source_recipe")
            self._initialized_checkpoint_tensor_sha256 = lineage.get("stage1_source_tensor_sha256")
        expected_metadata = self.training_checkpoint_metadata()
        if actual_metadata != expected_metadata:
            raise RuntimeError(
                "ManeuverFlow resume checkpoint metadata mismatch: "
                f"expected={expected_metadata}, actual={actual_metadata}."
            )

        self._require_exact_checkpoint_inference_protocol(
            checkpoint,
            context="resume checkpoint",
        )
        state_dict = checkpoint.get("state_dict")
        if not isinstance(state_dict, Mapping):
            raise TypeError("Full-state resume requires a state_dict mapping.")
        normalized = _normalize_checkpoint_state_dict(state_dict)
        if self.config.training_phase == "joint_finetuning":
            validate_ema_training_state_dict(
                normalized, expected_decay=float(self.config.ema_target_decay)
            )
        else:
            if detect_state_dict_schema(normalized) != PAPER_V1_SCHEMA:
                raise RuntimeError("Stage-1 resume requires the online checkpoint namespace.")
            validate_finite_state_dict(normalized)
        validate_state_dict_against_model(normalized, self.state_dict())
        validate_perception_variant_state_dict(normalized, self.config.perception_variant)
        validate_mgmp_inference_buffers(normalized)

    def initialize(self) -> None:
        """Load model weights with exact key/shape/dtype parity.

        Inference additionally requires the released paper-schema envelope.
        Joint fine-tuning accepts a metadata-identified Stage-1 Lightning
        checkpoint and discards its optimizer/loop state at this explicit
        weight boundary.
        """
        if self._checkpoint_initialized:
            return
        if self._checkpoint_path is None:
            if self.config.training_phase in {"inference", "joint_finetuning"}:
                raise RuntimeError(
                    f"{self.config.training_phase} requires a checkpoint_path."
                )
            return

        checkpoint_path = Path(self._checkpoint_path).expanduser()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Inference checkpoint not found: {checkpoint_path}")
        print(
            "[ManeuverFlowAgent] Loading checkpoint: "
            f"{checkpoint_path.name}"
        )

        # Loading on CPU avoids transient GPU spikes. weights_only=True also
        # rejects arbitrary pickle payloads.
        ckpt = torch.load(
            checkpoint_path,
            map_location=torch.device("cpu"),
            mmap=True,
            weights_only=True,
        )
        if not isinstance(ckpt, Mapping):
            raise TypeError("Checkpoint must be a mapping or a state_dict mapping.")

        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        if not isinstance(state_dict, Mapping):
            raise TypeError("Checkpoint state_dict must be a mapping.")

        normalized_state_dict = _normalize_checkpoint_state_dict(state_dict)

        schema = detect_state_dict_schema(normalized_state_dict.keys())
        if schema != PAPER_V1_SCHEMA:
            raise RuntimeError(
                "Checkpoint keys must use the released "
                f"{CHECKPOINT_KEY_SCHEMA!r} namespace; detected {schema!r}."
            )

        self._validate_checkpoint_training_handoff(ckpt)
        if self.config.training_phase == "inference":
            validate_checkpoint_envelope(ckpt)
        self._validate_checkpoint_inference_protocol(ckpt)

        validate_perception_variant_state_dict(
            normalized_state_dict,
            self.config.perception_variant,
        )
        if self.config.training_phase == "joint_finetuning":
            validate_stage1_state_dict_against_model(normalized_state_dict, self.state_dict())
        else:
            validate_state_dict_against_model(normalized_state_dict, self.state_dict())
        if self.config.training_phase != "inference":
            validate_finite_state_dict(normalized_state_dict)
        validate_mgmp_inference_buffers(normalized_state_dict)
        if self.config.training_phase != "inference":
            self._initialized_checkpoint_tensor_sha256 = (
                canonical_keyed_tensor_sha256(normalized_state_dict)
            )
        is_stage1_handoff = self.config.training_phase == "joint_finetuning"
        incompatible = self.load_state_dict(
            normalized_state_dict,
            strict=not is_stage1_handoff,
        )
        if is_stage1_handoff:
            expected_missing = {key for key in self.state_dict() if key.startswith(EMA_TARGET_PREFIX)}
            # Older BatchNorm load metadata may synthesize its integer
            # counter. Online parity was checked above; every reported
            # omission must still belong to this exact teacher whitelist.
            if not set(incompatible.missing_keys).issubset(expected_missing) or incompatible.unexpected_keys:
                raise RuntimeError("Stage-1 handoff may omit only the exact EMA teacher namespace.")
            teacher = self.model.ema_target_encoder
            if teacher is None:
                raise RuntimeError("Joint fine-tuning requires an EMA target encoder.")
            teacher.reset_from_online(self.model.hypothesizer.perception.transfuser_backbone)
            self._stage1_source_recipe = ckpt["navsim_agent_metadata"]["training_recipe"]
            self._source_training_data_metadata = ckpt.get("training_data")
        self._checkpoint_initialized = True
        self._move_to_inference_device()
        print(
            "[ManeuverFlowAgent] Checkpoint loaded with exact online parity. "
            f"Missing: {len(incompatible.missing_keys)}, "
            f"Unexpected: {len(incompatible.unexpected_keys)}"
        )

    def _validate_checkpoint_training_handoff(
        self,
        checkpoint: Mapping[str, Any],
    ) -> None:
        """Accept only explicitly identified Stage-1 weight source recipes."""

        if self.config.training_phase == "inference":
            return
        metadata = checkpoint.get("navsim_agent_metadata")
        if not isinstance(metadata, Mapping):
            raise RuntimeError("ManeuverFlow Stage-1 weight initialization requires training metadata.")
        recipe = metadata.get("training_recipe")
        expected: dict[str, object] = {
            "agent_name": self.name(),
            "training_phase": "generation_pretraining",
            "training_recipe": recipe,
            "perception_variant": self.config.perception_variant,
        }
        if recipe in STAGE1_SOURCE_RECIPES:
            max_epochs = metadata.get("max_epochs")
            if type(max_epochs) is not int or max_epochs < 1:
                raise RuntimeError("EMA Stage-1 source must record positive max_epochs.")
            expected["max_epochs"] = max_epochs
            try:
                expected.update(validate_optimizer_metadata(metadata))
            except ValueError as error:
                raise RuntimeError(f"Stage-1 source optimizer identity is invalid: {error}") from error
            expected["bev_coordinate_convention"] = "lidar_increasing_x"
        if recipe not in STAGE1_SOURCE_RECIPES or metadata != expected:
            raise RuntimeError(
                "ManeuverFlow weight initialization requires a checkpoint "
                "saved by an explicit generation_pretraining source recipe: "
                f"expected={expected}, actual={metadata}."
            )
        self._require_exact_checkpoint_inference_protocol(checkpoint, context="Stage-1 source")

    def _validate_checkpoint_inference_protocol(
        self,
        checkpoint: Mapping[str, Any],
    ) -> None:
        """Match a strict release config to the checkpoint's paper protocol."""

        if self.config.training_phase != "inference":
            return
        self._require_exact_checkpoint_inference_protocol(
            checkpoint,
            context="release checkpoint",
        )

    def _require_exact_checkpoint_inference_protocol(
        self,
        checkpoint: Mapping[str, Any],
        *,
        context: str,
    ) -> None:
        """Require exact protocol keys and values for load-time lineage."""

        protocol = checkpoint.get("inference_protocol")
        if not isinstance(protocol, Mapping):
            raise RuntimeError(
                f"Strict ManeuverFlow {context} loading requires "
                "checkpoint['inference_protocol'] metadata."
            )

        expected = self.inference_protocol_metadata()
        actual = dict(protocol)
        if actual != expected:
            raise RuntimeError(
                f"ManeuverFlow {context} inference protocol mismatch: "
                f"expected={expected}, actual={actual}."
            )

    def _move_to_inference_device(self) -> None:
        """Optionally place the initialized agent on the evaluation device."""
        if not bool(
            getattr(self.config, "move_to_inference_device_on_initialize", False)
        ):
            return
        requested = str(getattr(self.config, "inference_device", "auto")).strip().lower()
        if requested == "auto":
            requested = "cuda:0" if torch.cuda.is_available() else "cpu"
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"inference_device={requested!r} requested but CUDA is unavailable."
            )
        device = torch.device(requested)
        self.to(device)
        self.eval()
        print(f"[ManeuverFlowAgent] Inference device: {device}")

    def get_sensor_config(self) -> SensorConfig:
        """Load TF sensors at the current frame and Stage-2 teacher frames."""
        sensor_indices = list(self.slice_indices)
        needs_future_sensor_teacher = (
            self.config.training_phase == "joint_finetuning"
        )
        if needs_future_sensor_teacher:
            cur_idx = int(self.slice_indices[0])
            sensor_indices.extend([cur_idx + 2, cur_idx + 4, cur_idx + 6, cur_idx + 8])
        sensor_indices = sorted(set(sensor_indices))
        return SensorConfig(
            cam_f0=sensor_indices,
            cam_l0=sensor_indices,
            cam_l1=False,
            cam_l2=False,
            cam_r0=sensor_indices,
            cam_r1=False,
            cam_r2=False,
            cam_b0=False,
            lidar_pc=sensor_indices,
        )

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        return [
            ManeuverFlowTargetBuilder(
                trajectory_sampling=self._trajectory_sampling,
                slice_indices=self.slice_indices,
                config=self.config,
            ),
        ]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        return [ManeuverFlowFeatureBuilder(config=self.config)]

    def get_training_callbacks(self) -> List[pl.Callback]:
        """Record the optimizer learning rates."""
        return [LearningRateMonitor(logging_interval="step")]

    def forward(self, features: Dict[str, torch.Tensor], targets: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """Run training with separate sensor features and supervision."""
        if targets is None:
            raise ValueError(
                "ManeuverFlow training requires targets in both released phases."
            )
        return self.model(features, targets)

    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute the objective for the configured training phase."""
        return compute_maneuverflow_loss(
            targets,
            predictions,
            self.config,
        )

    def get_optimizers(self) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        """Build the optimizer over the released parameter groups."""
        param_groups = self._get_param_groups()

        parameter_tensors = sum(len(group["params"]) for group in param_groups)
        if parameter_tensors == 0:
            raise RuntimeError(
                "ManeuverFlow has no trainable parameters in the configured phase."
            )
        return self._build_optimizer(param_groups)

    def _get_param_groups(self):
        """Build the released optimizer groups.

        Generation pretraining uses the base learning rate for perception and
        M³F.  Joint fine-tuning keeps four inference-time modules active and
        applies ``joint_generation_lr_multiplier`` only to perception/M³F.
        """
        phase = self.config.training_phase
        if phase not in {"generation_pretraining", "joint_finetuning"}:
            raise RuntimeError("Optimizers are unavailable in inference phase.")

        perception_modules = [
            self.model.hypothesizer.perception.transfuser_backbone,
            self.model.hypothesizer.perception.bev_memory_embedding,
            self.model.hypothesizer.perception.query_embedding,
            self.model.hypothesizer.perception.bev_projection,
            self.model.hypothesizer.perception.state_navigation_encoder,
            self.model.hypothesizer.perception.bev_semantic_head,
            self.model.hypothesizer.perception.query_decoder,
            self.model.hypothesizer.perception.agent_prediction_head,
            self.model.hypothesizer.maneuver_decoder,
        ]
        m3f_modules = [
            self.model.hypothesizer.meanflow_generator.interval_encoder,
            self.model.hypothesizer.meanflow_generator.trajectory_tokenizer,
            self.model.hypothesizer.meanflow_generator.field,
            self.model.hypothesizer.meanflow_generator.field_decoder,
            self.model.hypothesizer.meanflow_generator.bev_position_embedding,
        ]

        processed: set[int] = set()

        def collect(modules: List[torch.nn.Module]) -> List[torch.nn.Parameter]:
            params: List[torch.nn.Parameter] = []
            for module in modules:
                for parameter in module.parameters():
                    if parameter.requires_grad and id(parameter) not in processed:
                        params.append(parameter)
                        processed.add(id(parameter))
            return params

        perception_params = collect(perception_modules)
        m3f_params = collect(m3f_modules)
        verifier_params = (
            collect([self.model.verifier])
            if phase == "joint_finetuning"
            else []
        )
        ecfr_params = (
            collect([self.model.improver.ecfr])
            if phase == "joint_finetuning"
            else []
        )

        unassigned = [
            name
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad and id(parameter) not in processed
        ]
        if unassigned:
            raise RuntimeError(
                "Trainable parameters are outside the released optimizer groups: "
                + ", ".join(unassigned[:12])
            )

        generation_multiplier = (
            1.0
            if phase == "generation_pretraining"
            else float(self.config.joint_generation_lr_multiplier)
        )
        group_specs = [
            ("perception", perception_params, generation_multiplier),
            ("m3f", m3f_params, generation_multiplier),
            ("verifier", verifier_params, 1.0),
            ("ecfr", ecfr_params, 1.0),
        ]
        groups = [
            {
                "name": name,
                "params": params,
                "lr": self._lr * multiplier,
            }
            for name, params, multiplier in group_specs
            if params
        ]
        for group in groups:
            print(
                f"[Optimizer] {group['name']}: lr={group['lr']:.2e}, "
                f"tensors={len(group['params'])}"
            )
        return groups

    def _build_optimizer(self, parameter_groups):
        """Build AdamW with the released warm-up/cosine schedule."""
        optimizer = optim.AdamW(
            parameter_groups,
            lr=self._lr,
            weight_decay=float(self.config.weight_decay),
        )
        if self.config.training_phase == "joint_finetuning":
            if self.model.ema_target_encoder is None:
                raise RuntimeError("Joint fine-tuning requires an EMA target encoder.")

            def update_target(optimizer, args, kwargs):
                self.model.ema_target_encoder.update(
                    self.model.hypothesizer.perception.transfuser_backbone
                )

            # GradScaler skips optimizer.step on overflow, and accumulation
            # invokes it only when gradients are actually applied.
            optimizer.register_step_post_hook(update_target)
        scheduler = WarmupCosLR(
            optimizer=optimizer,
            min_lr=self.min_lr,
            epochs=self.max_epochs,
            warmup_epochs=int(self.config.warmup_epochs),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }


    def compute_trajectory(self, agent_input: AgentInput) -> Trajectory:
        """
        Computes the ego vehicle trajectory.
        Run ManeuverFlow and select the highest-scoring final hypothesis.
        """
        if (
            self.config.training_phase == "inference"
            and not self._checkpoint_initialized
        ):
            raise RuntimeError(
                "Inference checkpoint is not initialized. Call initialize() "
                "before compute_trajectory()."
            )
        self.eval()
        features: Dict[str, torch.Tensor] = {}
        for builder in self.get_feature_builders():
            features.update(builder.compute_features(agent_input))

        device = next(self.parameters()).device
        features = {k: v.unsqueeze(0).to(device) for k, v in features.items()}

        with torch.no_grad():
            output: ManeuverFlowInferenceOutput = self.model.inference(features)
            final_hypotheses = output["final_hypotheses"]
            final_scores = output["final_hypothesis_scores"]
            best_index = torch.argmax(final_scores[0])
            poses = final_hypotheses[0, best_index].cpu().numpy()
        return Trajectory(poses, self._trajectory_sampling)

class WarmupCosLR(_LRScheduler):
    """Epoch-wise linear warm-up followed by cosine annealing."""

    def __init__(
        self,
        optimizer,
        min_lr,
        warmup_epochs,
        epochs,
        last_epoch=-1,
    ) -> None:
        self.min_lr = min_lr
        self.epochs = max(1, int(epochs))
        self.warmup_epochs = max(0, int(warmup_epochs))
        super(WarmupCosLR, self).__init__(optimizer, last_epoch)

    def state_dict(self):
        return {
            key: value for key, value in self.__dict__.items() if key != "optimizer"
        }

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def get_lr(self):
        lrs = []
        effective_epoch = self.last_epoch
        for base_lr in self.base_lrs:
            if self.warmup_epochs > 0 and effective_epoch < self.warmup_epochs:
                lr = base_lr * (effective_epoch + 1) / (self.warmup_epochs + 1e-6)
            else:
                target_min = min(self.min_lr, base_lr)
                decay_denominator = self.epochs - self.warmup_epochs + 1e-6
                decay_progress = min(
                    1.0,
                    max(0.0, float(effective_epoch - self.warmup_epochs) / decay_denominator),
                )
                lr = target_min + 0.5 * (base_lr - target_min) * (
                    1
                    + math.cos(math.pi * decay_progress)
                )
            lrs.append(lr)

        return lrs
