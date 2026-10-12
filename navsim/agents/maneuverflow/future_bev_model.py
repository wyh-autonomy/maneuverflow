"""Shared future-BEV prediction for ManeuverFlow verification."""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FutureBEVTransitionLayer(nn.Module):
    """One recurrent transition layer for the shared future-BEV field."""

    def __init__(
        self,
        model_dimension: int,
        attention_heads: int,
        feed_forward_dimension: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.spatial_attention = nn.MultiheadAttention(
            model_dimension,
            attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.current_context_attention = nn.MultiheadAttention(
            model_dimension,
            attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.history_attention = nn.MultiheadAttention(
            model_dimension,
            attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(model_dimension),
            nn.Linear(model_dimension, feed_forward_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feed_forward_dimension, model_dimension),
        )
        self.spatial_normalization = nn.LayerNorm(model_dimension)
        self.current_context_normalization = nn.LayerNorm(model_dimension)
        self.history_normalization = nn.LayerNorm(model_dimension)
        self.transition_normalization = nn.LayerNorm(model_dimension)
        self.transition_projection = nn.Linear(model_dimension, model_dimension)
        self.transition_gate = nn.Sequential(
            nn.LayerNorm(model_dimension * 3),
            nn.Linear(model_dimension * 3, model_dimension),
        )
        self.residual_dropout = nn.Dropout(dropout)

    def forward(
        self,
        state: torch.Tensor,
        previous_state: torch.Tensor,
        current_context: torch.Tensor,
        history_context: torch.Tensor,
        state_navigation_token: torch.Tensor,
    ) -> torch.Tensor:
        spatial_update, _ = self.spatial_attention(state, state, state)
        state = self.spatial_normalization(
            state + self.residual_dropout(spatial_update)
        )

        current_update, _ = self.current_context_attention(
            state,
            current_context,
            current_context,
        )
        state = self.current_context_normalization(
            state + self.residual_dropout(current_update)
        )

        history_update, _ = self.history_attention(
            state,
            history_context,
            history_context,
        )
        state = self.history_normalization(
            state + self.residual_dropout(history_update)
        )

        state = state + self.residual_dropout(self.feed_forward(state))
        expanded_state_navigation = state_navigation_token.expand(
            -1,
            state.shape[1],
            -1,
        )
        gate = torch.sigmoid(
            self.transition_gate(
                torch.cat(
                    [previous_state, state, expanded_state_navigation],
                    dim=-1,
                )
            )
        )
        transition = self.transition_projection(state)
        return self.transition_normalization(
            previous_state + gate * self.residual_dropout(transition)
        )


class CausalFutureBEVRefiner(nn.Module):
    """Axial spatial and causal temporal refinement of future-BEV states."""

    def __init__(
        self,
        model_dimension: int,
        attention_heads: int,
        feed_forward_dimension: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.spatial_attention = nn.MultiheadAttention(
            model_dimension,
            attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.causal_temporal_attention = nn.MultiheadAttention(
            model_dimension,
            attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(model_dimension),
            nn.Linear(model_dimension, feed_forward_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feed_forward_dimension, model_dimension),
        )
        self.spatial_normalization = nn.LayerNorm(model_dimension)
        self.temporal_normalization = nn.LayerNorm(model_dimension)
        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, future_states: torch.Tensor) -> torch.Tensor:
        batch_size, future_steps, spatial_tokens, channels = future_states.shape

        spatial_states = future_states.reshape(
            batch_size * future_steps,
            spatial_tokens,
            channels,
        )
        spatial_update, _ = self.spatial_attention(
            spatial_states,
            spatial_states,
            spatial_states,
        )
        spatial_states = self.spatial_normalization(
            spatial_states + self.residual_dropout(spatial_update)
        )
        future_states = spatial_states.view(
            batch_size,
            future_steps,
            spatial_tokens,
            channels,
        )

        temporal_states = future_states.permute(0, 2, 1, 3).reshape(
            batch_size * spatial_tokens,
            future_steps,
            channels,
        )
        causal_mask = torch.triu(
            torch.ones(
                future_steps,
                future_steps,
                device=future_states.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )
        temporal_update, _ = self.causal_temporal_attention(
            temporal_states,
            temporal_states,
            temporal_states,
            attn_mask=causal_mask,
        )
        temporal_states = self.temporal_normalization(
            temporal_states + self.residual_dropout(temporal_update)
        )
        future_states = temporal_states.view(
            batch_size,
            spatial_tokens,
            future_steps,
            channels,
        ).permute(0, 2, 1, 3)
        return future_states + self.residual_dropout(
            self.feed_forward(future_states)
        )


class FutureBEVModel(nn.Module):
    """Predict one shared future-BEV field for all trajectory hypotheses."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.future_steps = int(config.future_bev_horizon_steps)
        self.feature_dimension = int(config.future_bev_feature_dimension)
        planning_context_dimension = int(config.planning_context_dimension)
        dropout = float(config.verifier_dropout)
        transition_layer_count = int(
            config.future_bev_transition_layer_count
        )
        feed_forward_dimension = int(
            config.verifier_feed_forward_dimension
        )
        attention_heads = int(config.future_bev_attention_heads)

        self.state_navigation_conditioner = nn.Sequential(
            nn.Linear(planning_context_dimension, self.feature_dimension),
            nn.GELU(),
            nn.Linear(self.feature_dimension, self.feature_dimension),
        )
        self.state_navigation_token_encoder = nn.Sequential(
            nn.Linear(planning_context_dimension, self.feature_dimension),
            nn.GELU(),
            nn.Linear(self.feature_dimension, self.feature_dimension),
        )
        self.temporal_embedding = nn.Embedding(
            self.future_steps,
            self.feature_dimension,
        )
        self.spatial_embedding = nn.Embedding(
            8 * 8,
            self.feature_dimension,
        )
        self.transition_layers = nn.ModuleList(
            [
                FutureBEVTransitionLayer(
                    self.feature_dimension,
                    attention_heads=attention_heads,
                    feed_forward_dimension=feed_forward_dimension,
                    dropout=dropout,
                )
                for _ in range(transition_layer_count)
            ]
        )
        self.causal_temporal_refiner = CausalFutureBEVRefiner(
            self.feature_dimension,
            attention_heads=attention_heads,
            feed_forward_dimension=feed_forward_dimension,
            dropout=dropout,
        )

    def encode_future_bev_targets(
        self,
        future_bev_high: Optional[torch.Tensor],
        future_bev_low: torch.Tensor,
        current_bev_high: Optional[torch.Tensor] = None,
        current_bev_low: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Build stop-gradient absolute and residual future-BEV targets."""

        del future_bev_high, current_bev_high
        if future_bev_low.dim() != 5:
            raise ValueError(
                "future_bev_low must be [B, T, C, 8, 8], "
                f"got {tuple(future_bev_low.shape)}"
            )
        if future_bev_low.shape[2] != self.feature_dimension:
            raise ValueError(
                "future_bev_low channels must match future-BEV dimension "
                f"{self.feature_dimension}, got {future_bev_low.shape[2]}."
            )

        absolute_targets = future_bev_low.detach()
        residual_targets = torch.zeros_like(absolute_targets)
        if current_bev_low is not None:
            if current_bev_low.shape[1] != self.feature_dimension:
                raise ValueError(
                    "current_bev_low channels must match future-BEV dimension "
                    f"{self.feature_dimension}, got {current_bev_low.shape[1]}."
                )
            residual_targets[:, :1] = (
                absolute_targets[:, :1]
                - current_bev_low.unsqueeze(1).to(dtype=absolute_targets.dtype)
            )
        if absolute_targets.shape[1] > 1:
            residual_targets[:, 1:] = (
                absolute_targets[:, 1:] - absolute_targets[:, :-1]
            )
        return {
            "future_bev_absolute_feature_target": absolute_targets,
            "future_bev_residual_feature_target": residual_targets.detach(),
        }

    def select_current_bev(
        self,
        planning_context: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Select the perception grid matching the future-BEV feature width."""

        for key in ("bev_world", "bev_low"):
            value = planning_context.get(key)
            if torch.is_tensor(value) and int(value.shape[1]) == self.feature_dimension:
                return value
        available = {
            key: tuple(value.shape)
            for key, value in planning_context.items()
            if key in {"bev_world", "bev_low"} and torch.is_tensor(value)
        }
        raise ValueError(
            "No current BEV grid matches future-BEV dimension "
            f"{self.feature_dimension}: {available}"
        )

    def rollout_world(
        self,
        planning_context: Dict[str, torch.Tensor],
        future_bev_targets: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
    ]:
        """Roll out the shared future-BEV field once for a planning step."""

        batch_size = planning_context["status_feat"].shape[0]
        device = planning_context["status_feat"].device
        current_high_resolution_bev = planning_context.get(
            "bev_high",
            torch.zeros(batch_size, 64, 8, 8, device=device),
        )
        current_world_bev = self.select_current_bev(planning_context)
        if current_high_resolution_bev.shape[-2:] != current_world_bev.shape[-2:]:
            current_high_resolution_bev = F.adaptive_avg_pool2d(
                current_high_resolution_bev,
                output_size=current_world_bev.shape[-2:],
            )

        state_navigation_bias = self.state_navigation_conditioner(
            planning_context["status_feat"]
        ).view(batch_size, 1, self.feature_dimension)
        state_navigation_token = self.state_navigation_token_encoder(
            planning_context["status_feat"]
        ).view(batch_size, 1, self.feature_dimension)

        spatial_state = current_world_bev.flatten(2).transpose(1, 2)
        temporal_ids = torch.arange(
            self.future_steps,
            device=device,
            dtype=torch.long,
        )
        spatial_ids = torch.arange(8 * 8, device=device, dtype=torch.long)
        spatial_encoding = self.spatial_embedding(spatial_ids).view(
            1,
            8 * 8,
            self.feature_dimension,
        )
        current_state = (
            spatial_state + state_navigation_bias + spatial_encoding
        )
        current_context = torch.cat(
            [current_state, state_navigation_token],
            dim=1,
        )

        history_states = [current_state]
        predicted_states = []
        state = current_state
        for step_index in range(self.future_steps):
            step_state = state + self.temporal_embedding(
                temporal_ids[step_index]
            ).view(1, 1, self.feature_dimension)
            for transition_layer in self.transition_layers:
                history_context = torch.cat(history_states, dim=1)
                step_state = transition_layer(
                    step_state,
                    previous_state=state,
                    current_context=current_context,
                    history_context=history_context,
                    state_navigation_token=state_navigation_token,
                )
            state = step_state
            history_states.append(state)
            predicted_states.append(state)

        future_states = torch.stack(predicted_states, dim=1)
        future_states = self.causal_temporal_refiner(future_states)
        future_absolute_bev = future_states.view(
            batch_size,
            self.future_steps,
            8,
            8,
            self.feature_dimension,
        ).permute(0, 1, 4, 2, 3).contiguous()

        reconstruction_outputs = {
            "future_bev_absolute_feature_prediction": future_absolute_bev,
        }
        if future_bev_targets is not None:
            reconstruction_outputs.update(
                {
                    key: value
                    for key, value in future_bev_targets.items()
                    if torch.is_tensor(value)
                }
            )

        trajectory_evidence_field = {
            "cur_grid": current_high_resolution_bev,
            "fut_abs": future_absolute_bev,
        }
        future_residual_bev = torch.zeros_like(future_absolute_bev)
        future_residual_bev[:, :1] = (
            future_absolute_bev[:, :1] - current_world_bev.unsqueeze(1)
        )
        if self.future_steps > 1:
            future_residual_bev[:, 1:] = (
                future_absolute_bev[:, 1:]
                - future_absolute_bev[:, :-1]
            )
        reconstruction_outputs["future_bev_residual_feature_prediction"] = (
            future_residual_bev
        )
        trajectory_evidence_field["fut_res"] = future_residual_bev

        return (
            future_states,
            future_states,
            reconstruction_outputs,
            trajectory_evidence_field,
        )
