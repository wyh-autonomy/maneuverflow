"""Release configuration for ManeuverFlow.

Only lifecycle, training-recipe, asset, and runtime controls are
constructor-visible. Released topology and protocol constants are ClassVars.
"""

from dataclasses import dataclass
import math
from typing import ClassVar, Dict, Literal, Optional, Tuple

from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.maps.abstract_map import SemanticMapLayer

MANEUVERFLOW_TRAINING_PHASES: Tuple[str, ...] = (
    "generation_pretraining",
    "joint_finetuning",
    "inference",
)
MANEUVERFLOW_TRAINING_RECIPE = "maneuverflow.ema.v3"
MANEUVERFLOW_EMA_DECAY = 0.999

ManeuverFlowPerceptionVariant = Literal["TF"]
MANEUVERFLOW_PERCEPTION_VARIANTS: Tuple[ManeuverFlowPerceptionVariant, ...] = (
    "TF",
)

ECFRFinalPoolComposition = Literal[
    "source_replacement",
    "independent_ranked_retention",
]
ECFR_FINAL_POOL_COMPOSITIONS: Tuple[ECFRFinalPoolComposition, ...] = (
    "source_replacement",
    "independent_ranked_retention",
)
DEFAULT_ECFR_FINAL_POOL_COMPOSITION: ECFRFinalPoolComposition = (
    "independent_ranked_retention"
)


@dataclass(frozen=True)
class ManeuverFlowInferencePolicy:
    """Code-side contract for fixed-budget ManeuverFlow inference."""

    num_base_hypotheses: int = 64
    maneuver_top_k: int = 3
    minimum_per_selected_component: int = 0
    num_refined_hypotheses: int = 16
    num_retained_base_hypotheses: int = 48
    ecfr_max_ds: float = 1.0
    ecfr_max_dl: float = 0.8
    ecfr_max_dpsi: float = 0.35
    ecfr_source_selection: str = "score_top16_within_top1_routed_component"
    ecfr_final_pool_composition: ECFRFinalPoolComposition = (
        DEFAULT_ECFR_FINAL_POOL_COMPOSITION
    )
    refined_score_delta_cost_xy: float = 0.02
    refined_score_delta_cost_yaw: float = 0.01
    refined_score_delta_cost_smooth: float = 0.01


MANEUVERFLOW_INFERENCE_POLICY = ManeuverFlowInferencePolicy()


@dataclass
class ManeuverFlowConfig:
    """Configuration for two-phase training and fixed-budget inference."""

    # ------------------------------------------------------------------
    # Lifecycle and optimization recipe.
    # ------------------------------------------------------------------
    training_phase: str = "generation_pretraining"
    # The release uses three cameras and physical LiDAR.
    perception_variant: ManeuverFlowPerceptionVariant = "TF"
    joint_generation_lr_multiplier: float = 0.2
    weight_decay: float = 0.1
    warmup_epochs: int = 3
    max_epochs: Optional[int] = None
    min_lr: float = 5e-6
    ema_target_decay: float = MANEUVERFLOW_EMA_DECAY
    ema_target_buffer_policy: str = "copy"

    # ------------------------------------------------------------------
    # Maneuver Gaussian Mixture Prior (M-GMP).
    # ------------------------------------------------------------------
    num_maneuver_components: ClassVar[int] = 8
    mgmp_prior_dir: str = ""
    m3f_training_proposal_count: Optional[int] = None
    maneuver_navigation_mask_probability: ClassVar[float] = 0.1
    maneuver_head_dropout: ClassVar[float] = 0.1

    # ------------------------------------------------------------------
    # Maneuver-Mixture MeanFlow Field (M3F).
    # ------------------------------------------------------------------
    m3f_hidden_dim: ClassVar[int] = 256
    m3f_num_attention_heads: ClassVar[int] = 4
    m3f_dropout: ClassVar[float] = 0.1
    m3f_attention_dropout: ClassVar[float] = 0.0
    m3f_num_layers: ClassVar[int] = 6
    m3f_time_sampler: ClassVar[str] = "logit_normal"
    m3f_time_mu: ClassVar[float] = -0.4
    m3f_time_sigma: ClassVar[float] = 1.0
    # Fraction of non-complete intervals assigned to the general regime.
    m3f_unequal_interval_ratio: ClassVar[float] = 0.5
    m3f_full_interval_ratio: ClassVar[float] = 0.25

    # ------------------------------------------------------------------
    # Future BEV Model.
    # ------------------------------------------------------------------
    future_bev_feature_dimension: ClassVar[int] = 512
    future_bev_transition_layer_count: ClassVar[int] = 4
    future_bev_attention_heads: ClassVar[int] = 8
    future_bev_horizon_steps: ClassVar[int] = 4
    planning_context_dimension: ClassVar[int] = 256
    verifier_dropout: ClassVar[float] = 0.1
    verifier_feed_forward_dimension: ClassVar[int] = 1024

    # ------------------------------------------------------------------
    # Trajectory Evidence Memory.
    # ------------------------------------------------------------------
    current_evidence_tokens_per_step: ClassVar[int] = 4
    future_evidence_tokens_per_step: ClassVar[int] = 4
    current_evidence_sampling_points: ClassVar[int] = 6
    future_evidence_sampling_points: ClassVar[int] = 4
    vehicle_length: ClassVar[float] = 4.0
    vehicle_width: ClassVar[float] = 2.0
    future_residual_loss_weight: ClassVar[float] = 0.5
    future_cosine_loss_weight: ClassVar[float] = 0.02
    future_bev_loss_weight: ClassVar[float] = 0.5
    future_bev_loss_decay_epochs: ClassVar[int] = 0
    current_bev_feature_channels: ClassVar[int] = 64
    bev_minimum_x: ClassVar[float] = -32.0
    bev_maximum_x: ClassVar[float] = 32.0
    bev_minimum_y: ClassVar[float] = -32.0
    bev_maximum_y: ClassVar[float] = 32.0

    # ------------------------------------------------------------------
    # Metric Prism.
    # ------------------------------------------------------------------
    metric_prism_feature_dimension: ClassVar[int] = 512
    metric_prism_attention_heads: ClassVar[int] = 8
    metric_prism_layer_count: ClassVar[int] = 4
    metric_prism_set_attention_heads: ClassVar[int] = 1
    metric_prism_projection_dropout: ClassVar[float] = 0.1
    metric_prism_drop_path_rate: ClassVar[float] = 0.2
    metric_prism_layer_scale_initial_value: ClassVar[float] = 0.0
    factor_prior_clamp_minimum: ClassVar[float] = 1e-4
    factor_prior_clamp_maximum: ClassVar[float] = 0.99
    non_collision_binary_safe_prior: ClassVar[float] = 0.7312
    factor_prior_means: ClassVar[Dict[str, float]] = {
        "dac": 0.6592,
        "ep": 0.5615,
        "ttc": 0.6561,
        "comfort": 0.8214,
    }

    # ------------------------------------------------------------------
    # Evidence-Coupled Frenet Refiner (ECFR).
    # ------------------------------------------------------------------
    ecfr_d_model: ClassVar[int] = 512
    ecfr_num_heads: ClassVar[int] = 8
    ecfr_num_layers: ClassVar[int] = 2
    ecfr_dropout: ClassVar[float] = 0.1
    ecfr_max_ds: Optional[float] = None
    ecfr_max_dl: Optional[float] = None
    ecfr_max_dpsi: Optional[float] = None

    # ------------------------------------------------------------------
    # Fixed Hypothesize--Verify--Improve inference.
    # ------------------------------------------------------------------
    ecfr_final_pool_composition: ECFRFinalPoolComposition = "source_replacement"
    refined_score_delta_cost_xy: float = 0.0
    refined_score_delta_cost_yaw: float = 0.0
    refined_score_delta_cost_smooth: float = 0.0
    inference_num_base_hypotheses: ClassVar[int] = 64
    inference_num_refined_hypotheses: ClassVar[int] = 16
    inference_num_retained_base_hypotheses: ClassVar[int] = 48
    inference_mgmp_top_k: ClassVar[int] = 3

    # ------------------------------------------------------------------
    # Trajectory normalization.
    # ------------------------------------------------------------------
    trajectory_displacement_x_mean: ClassVar[float] = 2.950225591659546
    trajectory_displacement_x_scale: ClassVar[float] = 4.52533745765686
    trajectory_displacement_y_mean: ClassVar[float] = 0.0607292577624321
    trajectory_displacement_y_scale: ClassVar[float] = 5.0728104040026665

    # ------------------------------------------------------------------
    # Released TransFuser topology and data geometry.
    # ------------------------------------------------------------------
    image_architecture: ClassVar[str] = "resnet34"
    lidar_architecture: ClassVar[str] = "resnet34"
    block_exp: ClassVar[int] = 4
    n_layer: ClassVar[int] = 4
    n_head: ClassVar[int] = 8
    embd_pdrop: ClassVar[float] = 0.1
    resid_pdrop: ClassVar[float] = 0.1
    attn_pdrop: ClassVar[float] = 0.1
    gpt_linear_layer_init_mean: ClassVar[float] = 0.0
    gpt_linear_layer_init_std: ClassVar[float] = 0.02
    gpt_layer_norm_init_weight: ClassVar[float] = 1.0
    max_height_lidar: ClassVar[float] = 100.0
    pixels_per_meter: ClassVar[float] = 4.0
    hist_max_per_pixel: ClassVar[int] = 5
    lidar_split_height: ClassVar[float] = 0.2
    lidar_seq_len: ClassVar[int] = 1
    lidar_resolution_width: ClassVar[int] = 256
    lidar_resolution_height: ClassVar[int] = 256
    img_vert_anchors: ClassVar[int] = 8
    img_horz_anchors: ClassVar[int] = 32
    lidar_vert_anchors: ClassVar[int] = 8
    lidar_horz_anchors: ClassVar[int] = 8
    num_bounding_boxes: ClassVar[int] = 30
    perception_query_feed_forward_dimension: ClassVar[int] = 1024
    perception_query_num_layers: ClassVar[int] = 2
    perception_query_num_heads: ClassVar[int] = 8
    perception_query_dropout: ClassVar[float] = 0.0

    # ------------------------------------------------------------------
    # BEV and checkpoint tensor contracts.
    # ------------------------------------------------------------------
    proposal_anchor_count: ClassVar[int] = 512
    trajectory_num_poses: ClassVar[int] = 8
    bev_memory_token_count: ClassVar[int] = 64
    ego_box_map_idx: ClassVar[int] = 7
    num_bev_classes: ClassVar[int] = 8
    bev_down_sample_factor: ClassVar[int] = 4
    bev_upsample_factor: ClassVar[int] = 2
    bev_pixel_width: ClassVar[int] = 256
    bev_pixel_height: ClassVar[int] = 128
    bev_pixel_size: ClassVar[float] = 0.25

    # ------------------------------------------------------------------
    # Training data assets.  Empty paths are valid for checkpoint-backed
    # inference; training recipes must provide the assets they consume.
    # ------------------------------------------------------------------
    metric_cache_path: Optional[str] = None
    joint_probability_source_count: ClassVar[int] = 64

    # ------------------------------------------------------------------
    # Objective group coefficients.
    # ------------------------------------------------------------------
    auxiliary_loss_weight: Optional[float] = None
    m3f_loss_weight: Optional[float] = None
    verification_loss_weight: Optional[float] = None
    refinement_loss_weight: Optional[float] = None
    agent_class_weight: ClassVar[float] = 1.0
    agent_box_weight: ClassVar[float] = 0.1
    bev_semantic_weight: ClassVar[float] = 0.25
    focal_loss_alpha: ClassVar[float] = 0.5
    focal_loss_gamma: ClassVar[float] = 2.0
    refinement_delta_loss_weight: ClassVar[float] = 5.0
    metric_prism_loss_weight: ClassVar[float] = 10.0

    # ------------------------------------------------------------------
    # Self-contained release initialization.
    # ------------------------------------------------------------------
    mgmp_assets_from_checkpoint: bool = False
    move_to_inference_device_on_initialize: bool = False
    inference_device: str = "auto"
    bev_semantic_classes = {
        1: ("polygon", [SemanticMapLayer.LANE, SemanticMapLayer.INTERSECTION]),
        2: ("polygon", [SemanticMapLayer.WALKWAYS]),
        3: ("linestring", [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]),
        4: (
            "box",
            [
                TrackedObjectType.CZONE_SIGN,
                TrackedObjectType.BARRIER,
                TrackedObjectType.TRAFFIC_CONE,
                TrackedObjectType.GENERIC_OBJECT,
            ],
        ),
        5: ("box", [TrackedObjectType.VEHICLE]),
        6: ("box", [TrackedObjectType.PEDESTRIAN]),
        7: ("ego_box", [TrackedObjectType.VEHICLE]),
    }

    @property
    def bev_coordinate_convention(self) -> str:
        """Preserve released weights; train against the LiDAR histogram axes."""
        if (
            self.training_phase == "inference"
            and self.ecfr_final_pool_composition == "independent_ranked_retention"
        ):
            return "legacy_descending_x"
        return "lidar_increasing_x"

    def __post_init__(self) -> None:
        if self.max_epochs is None:
            self.max_epochs = 100 if self.training_phase == "generation_pretraining" else 20
        if (
            isinstance(self.max_epochs, bool)
            or not isinstance(self.max_epochs, int)
            or self.max_epochs <= 0
        ):
            raise ValueError("max_epochs must be a positive integer.")
        if type(self.warmup_epochs) is not int or self.warmup_epochs < 0:
            raise ValueError("warmup_epochs must be a non-negative integer.")
        if not math.isfinite(float(self.min_lr)) or float(self.min_lr) < 0.0:
            raise ValueError("min_lr must be finite and non-negative.")
        if (
            not math.isfinite(float(self.ema_target_decay))
            or not 0.0 <= float(self.ema_target_decay) < 1.0
        ):
            raise ValueError("ema_target_decay must be finite and in [0, 1).")
        if self.ema_target_buffer_policy != "copy":
            raise ValueError("ema_target_buffer_policy must be 'copy'.")
        if (
            self.training_phase == "joint_finetuning"
            and float(self.ema_target_decay) != MANEUVERFLOW_EMA_DECAY
        ):
            raise ValueError("joint_finetuning requires ema_target_decay=0.999.")
        for name in (
            "refined_score_delta_cost_xy",
            "refined_score_delta_cost_yaw",
            "refined_score_delta_cost_smooth",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
        if self.perception_variant not in MANEUVERFLOW_PERCEPTION_VARIANTS:
            choices = ", ".join(MANEUVERFLOW_PERCEPTION_VARIANTS)
            raise ValueError(
                f"perception_variant must be one of {{{choices}}}, got "
                f"{self.perception_variant!r}"
            )
        if self.training_phase not in MANEUVERFLOW_TRAINING_PHASES:
            choices = ", ".join(MANEUVERFLOW_TRAINING_PHASES)
            raise ValueError(
                f"training_phase must be one of {{{choices}}}, got "
                f"{self.training_phase!r}"
            )

        if (
            self.training_phase
            in {"generation_pretraining", "joint_finetuning"}
            and float(self.weight_decay) != 0.1
        ):
            raise ValueError(
                "Paper-aligned training requires AdamW weight_decay=0.1."
            )
        if (
            self.training_phase == "joint_finetuning"
            and float(self.joint_generation_lr_multiplier) != 0.2
        ):
            raise ValueError(
                "Paper-aligned joint_finetuning requires "
                "joint_generation_lr_multiplier=0.2."
            )

        objective_names = (
            "auxiliary_loss_weight",
            "m3f_loss_weight",
            "verification_loss_weight",
            "refinement_loss_weight",
        )
        objective_profile = (
            (1.0, 4.0, 0.0, 0.0)
            if self.training_phase == "generation_pretraining"
            else (1.0, 4.0, 2.0, 2.0)
        )
        for name, expected in zip(objective_names, objective_profile):
            value = getattr(self, name)
            if value is None:
                setattr(self, name, expected)
                continue
            value = float(value)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
            if (
                self.training_phase
                in {"generation_pretraining", "joint_finetuning"}
                and value != expected
            ):
                raise ValueError(
                    f"{name}={value} conflicts with the released "
                    f"{self.training_phase} coefficient {expected}."
                )

        expected_proposal_count = 6
        proposal_count = self.m3f_training_proposal_count
        if proposal_count is None:
            proposal_count = expected_proposal_count
            self.m3f_training_proposal_count = proposal_count
        if (
            isinstance(proposal_count, bool)
            or not isinstance(proposal_count, int)
            or proposal_count != expected_proposal_count
        ):
            raise ValueError(
                f"{self.training_phase} requires "
                f"{expected_proposal_count} M3F proposals per scene."
            )
        if self.ecfr_max_ds is None:
            self.ecfr_max_ds = (
                1.4 if self.training_phase == "joint_finetuning" else 1.0
            )
        if self.ecfr_max_dl is None:
            self.ecfr_max_dl = (
                0.9 if self.training_phase == "joint_finetuning" else 0.8
            )
        if self.ecfr_max_dpsi is None:
            self.ecfr_max_dpsi = 0.35
        if self.training_phase == "joint_finetuning":
            training_bounds = (
                float(self.ecfr_max_ds),
                float(self.ecfr_max_dl),
                float(self.ecfr_max_dpsi),
            )
            expected_training_bounds = (1.4, 0.9, 0.35)
            if training_bounds != expected_training_bounds:
                raise ValueError(
                    "joint_finetuning requires ECFR training bounds "
                    f"{expected_training_bounds}, got {training_bounds}."
                )

        self.validate_maneuverflow_inference_contract()

    def validate_maneuverflow_inference_contract(
        self,
        *,
        require_deployment_bounds: Optional[bool] = None,
    ) -> None:
        """Fail closed when runtime settings drift from the paper contract."""

        policy = MANEUVERFLOW_INFERENCE_POLICY
        if self.ecfr_final_pool_composition not in ECFR_FINAL_POOL_COMPOSITIONS:
            choices = ", ".join(ECFR_FINAL_POOL_COMPOSITIONS)
            raise ValueError(
                "ecfr_final_pool_composition must be one of "
                f"{{{choices}}}, got {self.ecfr_final_pool_composition!r}"
            )
        base_count = int(self.inference_num_base_hypotheses)
        refined_count = int(self.inference_num_refined_hypotheses)
        retained_count = int(self.inference_num_retained_base_hypotheses)
        if base_count <= 0:
            raise ValueError("inference_num_base_hypotheses must be positive")
        if refined_count < 0 or retained_count < 0:
            raise ValueError(
                "inference refined/retained hypothesis counts must be non-negative"
            )
        if retained_count + refined_count != base_count:
            raise ValueError(
                "ManeuverFlow cardinality mismatch: retained base + refined must equal base"
            )
        if (
            base_count,
            refined_count,
            retained_count,
        ) != (
            policy.num_base_hypotheses,
            policy.num_refined_hypotheses,
            policy.num_retained_base_hypotheses,
        ):
            raise ValueError(
                "The released ManeuverFlow protocol requires "
                "64 generated bases, 16 ECFR refinements, and 48 retained bases."
            )
        if int(self.inference_mgmp_top_k) != policy.maneuver_top_k:
            raise ValueError(
                "The released ManeuverFlow protocol requires Top-3 M-GMP routing."
            )

        if require_deployment_bounds is None:
            require_deployment_bounds = self.training_phase == "inference"
        if require_deployment_bounds:
            actual_bounds = (
                float(self.ecfr_max_ds),
                float(self.ecfr_max_dl),
                float(self.ecfr_max_dpsi),
            )
            expected_bounds = (
                policy.ecfr_max_ds,
                policy.ecfr_max_dl,
                policy.ecfr_max_dpsi,
            )
            if actual_bounds != expected_bounds:
                raise ValueError(
                    "The released ManeuverFlow inference protocol requires "
                    f"ECFR bounds {expected_bounds}, got {actual_bounds}."
                )

    @property
    def bev_semantic_frame(self) -> Tuple[int, int]:
        return (self.bev_pixel_height, self.bev_pixel_width)

    @property
    def bev_radius(self) -> float:
        values = [
            self.bev_minimum_x,
            self.bev_maximum_x,
            self.bev_minimum_y,
            self.bev_maximum_y,
        ]
        return max(abs(value) for value in values)

    @property
    def input_target(self) -> bool:
        return self.training_phase != "inference"
