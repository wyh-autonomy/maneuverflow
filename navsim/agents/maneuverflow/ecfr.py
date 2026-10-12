"""Evidence-Coupled Frenet Refinement (ECFR).

ECFR is the paper-facing late-refinement module in ManeuverFlow.  It encodes
each generated trajectory hypothesis into four motion tokens, couples those
tokens to shared world-evidence memory through cross-attention, and predicts a
bounded local residual in ``(ds, dl, dpsi)`` Frenet coordinates.
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .frenet_geometry import frenet_residual_to_trajectory
from .trajectory_codec import flow_state_to_trajectory, trajectory_to_flow_state


class ECFRTrajectoryEncoder(nn.Module):
    """Encode each flow-state trajectory into four ECFR motion tokens.

    Every motion token represents two consecutive waypoints.  Separate
    projections encode the two waypoint states and their within-pair
    displacement before motion-level fusion.  This is the trajectory-query
    side of the paper's Evidence-Coupled Frenet Refiner.
    """

    def __init__(self, config: Any):
        super().__init__()
        self.d_model = int(config.ecfr_d_model)
        self.num_motion_tokens = 4
        self.waypoints_per_token = 2

        self.waypoint_projection = nn.Sequential(
            nn.LayerNorm(4),
            nn.Linear(4, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.displacement_projection = nn.Sequential(
            nn.LayerNorm(4),
            nn.Linear(4, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.motion_projection = nn.Sequential(
            nn.LayerNorm(self.d_model * 3),
            nn.Linear(self.d_model * 3, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.token_normalization = nn.LayerNorm(self.d_model)
        self.waypoint_position_embedding = nn.Parameter(
            torch.randn(
                1,
                1,
                self.num_motion_tokens,
                self.waypoints_per_token,
                self.d_model,
            )
            * 0.02
        )
        self.motion_position_embedding = nn.Parameter(
            torch.randn(1, 1, self.num_motion_tokens, self.d_model) * 0.02
        )

    def forward(self, hypotheses_flow_state: torch.Tensor) -> torch.Tensor:
        """Return four motion tokens for every trajectory hypothesis."""
        batch_size, num_hypotheses, num_waypoints, state_dimension = (
            hypotheses_flow_state.shape
        )
        if num_waypoints != 8 or state_dimension < 4:
            raise ValueError(
                "ECFRTrajectoryEncoder expects [B, N, 8, 4], "
                f"got {tuple(hypotheses_flow_state.shape)}"
            )

        paired_waypoint_states = hypotheses_flow_state[..., :4].reshape(
            batch_size,
            num_hypotheses,
            self.num_motion_tokens,
            self.waypoints_per_token,
            4,
        )
        waypoint_tokens = (
            self.waypoint_projection(paired_waypoint_states)
            + self.waypoint_position_embedding
        )
        first_waypoint_token = waypoint_tokens[..., 0, :]
        second_waypoint_token = waypoint_tokens[..., 1, :]
        within_pair_displacement = (
            paired_waypoint_states[..., 1, :]
            - paired_waypoint_states[..., 0, :]
        )
        displacement_token = self.displacement_projection(
            within_pair_displacement
        )
        motion_features = torch.cat(
            [
                first_waypoint_token,
                second_waypoint_token,
                displacement_token,
            ],
            dim=-1,
        )
        motion_tokens = self.motion_projection(motion_features)
        return self.token_normalization(
            motion_tokens + self.motion_position_embedding
        )


class ECFRBlock(nn.Module):
    """Couple trajectory tokens to shared world-evidence memory.

    The layer order is trajectory self-attention, trajectory FFN, evidence
    cross-attention, and cross-attention FFN, each with pre-normalization and a
    residual connection.
    """

    def __init__(self, config: Any):
        super().__init__()
        self.d_model = int(config.ecfr_d_model)
        self.dropout = float(config.ecfr_dropout)
        self.num_heads = int(config.ecfr_num_heads)

        self.self_attention_normalization = nn.LayerNorm(self.d_model)
        self.self_ffn_normalization = nn.LayerNorm(self.d_model)
        self.evidence_memory_normalization = nn.LayerNorm(self.d_model)
        self.cross_attention_normalization = nn.LayerNorm(self.d_model)
        self.cross_ffn_normalization = nn.LayerNorm(self.d_model)

        self.self_attention = nn.MultiheadAttention(
            self.d_model,
            self.num_heads,
            dropout=self.dropout,
            batch_first=True,
        )
        self.cross_attention = nn.MultiheadAttention(
            self.d_model,
            self.num_heads,
            dropout=self.dropout,
            batch_first=True,
        )

        self.self_feed_forward = nn.Sequential(
            nn.Linear(self.d_model, 2 * self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(2 * self.d_model, self.d_model),
        )
        self.cross_feed_forward = nn.Sequential(
            nn.Linear(self.d_model, 2 * self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(2 * self.d_model, self.d_model),
        )
        self.residual_dropout = nn.Dropout(self.dropout)

    def forward(
        self,
        trajectory_tokens: torch.Tensor,
        evidence_memory_tokens: torch.Tensor,
        evidence_memory_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Refine hypothesis tokens against their corresponding evidence memory."""
        batch_size, num_hypotheses, num_tokens, _ = trajectory_tokens.shape
        flat_trajectory_tokens = trajectory_tokens.reshape(
            batch_size * num_hypotheses,
            num_tokens,
            self.d_model,
        )
        flat_evidence_memory = evidence_memory_tokens.reshape(
            batch_size * num_hypotheses,
            -1,
            self.d_model,
        )
        flat_evidence_memory = self.evidence_memory_normalization(
            flat_evidence_memory
        )

        flat_evidence_memory_padding_mask = None
        if evidence_memory_padding_mask is not None:
            expected_shape = (
                batch_size,
                num_hypotheses,
                evidence_memory_tokens.shape[2],
            )
            if tuple(evidence_memory_padding_mask.shape) != expected_shape:
                raise ValueError(
                    "evidence_memory_padding_mask must match [B, N, T], "
                    f"got {tuple(evidence_memory_padding_mask.shape)} "
                    f"versus {expected_shape}"
                )
            flat_evidence_memory_padding_mask = (
                evidence_memory_padding_mask.reshape(
                    batch_size * num_hypotheses,
                    evidence_memory_tokens.shape[2],
                ).to(device=evidence_memory_tokens.device, dtype=torch.bool)
            )

        self_attention_output, _ = self.self_attention(
            self.self_attention_normalization(flat_trajectory_tokens),
            self.self_attention_normalization(flat_trajectory_tokens),
            self.self_attention_normalization(flat_trajectory_tokens),
        )
        flat_trajectory_tokens = (
            flat_trajectory_tokens
            + self.residual_dropout(self_attention_output)
        )
        flat_trajectory_tokens = (
            flat_trajectory_tokens
            + self.residual_dropout(
                self.self_feed_forward(
                    self.self_ffn_normalization(flat_trajectory_tokens)
                )
            )
        )

        cross_attention_output, _ = self.cross_attention(
            self.cross_attention_normalization(flat_trajectory_tokens),
            flat_evidence_memory,
            flat_evidence_memory,
            key_padding_mask=flat_evidence_memory_padding_mask,
        )
        flat_trajectory_tokens = (
            flat_trajectory_tokens
            + self.residual_dropout(cross_attention_output)
        )
        flat_trajectory_tokens = (
            flat_trajectory_tokens
            + self.residual_dropout(
                self.cross_feed_forward(
                    self.cross_ffn_normalization(flat_trajectory_tokens)
                )
            )
        )

        return flat_trajectory_tokens.view(
            batch_size,
            num_hypotheses,
            num_tokens,
            self.d_model,
        )


class EvidenceCoupledFrenetRefiner(nn.Module):
    """Refine generated hypotheses with shared world evidence in Frenet space.

    ECFR reuses the evidence memory produced for trajectory evaluation.  For
    each base hypothesis, it predicts a bounded eight-step residual
    ``(ds, dl, dpsi)`` and applies that residual in the hypothesis-local Frenet
    frame.  Zero initialization keeps the branch an exact no-op before its
    refinement objective is learned.
    """

    def __init__(self, config: Any):
        super().__init__()
        self.config = config
        self.d_model = int(config.ecfr_d_model)
        self.num_layers = int(config.ecfr_num_layers)
        self.max_ds = float(config.ecfr_max_ds)
        self.max_dl = float(config.ecfr_max_dl)
        self.max_dpsi = float(config.ecfr_max_dpsi)

        self.trajectory_encoder = ECFRTrajectoryEncoder(config)
        self.refinement_blocks = nn.ModuleList(
            [ECFRBlock(config) for _ in range(self.num_layers)]
        )
        self.frenet_residual_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, 6),
        )

        nn.init.zeros_(self.frenet_residual_head[-1].weight)
        nn.init.zeros_(self.frenet_residual_head[-1].bias)

    def forward(
        self,
        base_hypotheses_flow_state: torch.Tensor,
        evidence_memory_outputs: Dict[str, torch.Tensor],
        base_hypotheses: Optional[torch.Tensor] = None,
        residual_scale: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """Apply bounded evidence-conditioned residuals to base hypotheses."""
        if base_hypotheses is None:
            base_hypotheses = flow_state_to_trajectory(
                base_hypotheses_flow_state,
                self.config,
            )
        if "memory_tokens" not in evidence_memory_outputs:
            raise KeyError(
                "EvidenceCoupledFrenetRefiner expects evidence-reader output "
                "containing 'memory_tokens'."
            )

        trajectory_tokens = self.trajectory_encoder(
            base_hypotheses_flow_state
        )
        evidence_memory_tokens = evidence_memory_outputs["memory_tokens"]
        if (
            evidence_memory_tokens.shape[:2]
            != base_hypotheses_flow_state.shape[:2]
        ):
            raise ValueError(
                "memory_tokens must share [B, N] with the base hypotheses, "
                f"got memory={tuple(evidence_memory_tokens.shape)} "
                f"base={tuple(base_hypotheses_flow_state.shape)}"
            )
        if evidence_memory_tokens.shape[-1] != self.d_model:
            raise ValueError(
                "memory_tokens last dim must be "
                f"{self.d_model}, got {evidence_memory_tokens.shape[-1]}"
            )

        evidence_memory_padding_mask = evidence_memory_outputs.get(
            "memory_key_padding_mask",
            None,
        )
        for refinement_block in self.refinement_blocks:
            trajectory_tokens = refinement_block(
                trajectory_tokens=trajectory_tokens,
                evidence_memory_tokens=evidence_memory_tokens,
                evidence_memory_padding_mask=evidence_memory_padding_mask,
            )

        raw_frenet_residual_pairs = self.frenet_residual_head(
            trajectory_tokens
        ).view(
            base_hypotheses_flow_state.shape[0],
            base_hypotheses_flow_state.shape[1],
            4,
            2,
            3,
        )
        raw_frenet_residuals = raw_frenet_residual_pairs.reshape(
            base_hypotheses_flow_state.shape[0],
            base_hypotheses_flow_state.shape[1],
            8,
            3,
        )
        residual_bounds = raw_frenet_residuals.new_tensor(
            [self.max_ds, self.max_dl, self.max_dpsi]
        ).view(1, 1, 1, 3)
        frenet_residuals = (
            torch.tanh(raw_frenet_residuals)
            * residual_bounds
            * float(residual_scale)
        )

        (
            refined_hypotheses,
            trajectory_residuals,
            _,
        ) = frenet_residual_to_trajectory(
            base_hypotheses,
            frenet_residuals,
        )
        refined_hypotheses_flow_state = trajectory_to_flow_state(
            refined_hypotheses,
            self.config,
        ).to(
            device=base_hypotheses_flow_state.device,
            dtype=base_hypotheses_flow_state.dtype,
        )
        flow_residuals = (
            refined_hypotheses_flow_state - base_hypotheses_flow_state
        )

        return {
            "refined_hypotheses": refined_hypotheses,
            "refined_hypotheses_flow_space": refined_hypotheses_flow_state,
            "trajectory_residuals": trajectory_residuals,
            "flow_residuals": flow_residuals,
            "frenet_residuals": frenet_residuals,
            "trajectory_tokens": trajectory_tokens,
        }
