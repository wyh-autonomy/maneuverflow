"""Maneuver-Mixture MeanFlow Field (M³F).

Each block queries BEV--ego context followed by agent memory, then applies a
maneuver-routed feed-forward expert.  Interval-conditioned AdaLN parameterizes
the one-step interval-averaged MeanFlow field described in the paper.
"""

from typing import Any

import torch
import torch.nn as nn


class ManeuverExpertMLP(nn.Module):
    """Feed-forward expert used by the shared and routed M3F sublayers."""

    def __init__(self, config: Any):
        super().__init__()
        model_dim = int(config.m3f_hidden_dim)
        dropout = float(config.m3f_dropout)
        hidden_dim = model_dim * 4
        self.layers = nn.Sequential(
            nn.Linear(model_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, model_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.layers(x)


def sparse_maneuver_expert_forward(
    x: torch.Tensor,
    weights: torch.Tensor,
    experts: nn.ModuleList,
) -> torch.Tensor:
    """Apply one-hot maneuver routing with the checkpoint's sparse topology.

    Released M3F call sites provide an exact one-hot M-GMP route. The fixed
    two-slot implementation is retained for numerical checkpoint compatibility;
    its second selected slot therefore always carries zero weight.
    """
    if weights.ndim < 2 or weights.shape[-1] != len(experts):
        raise ValueError("M3F maneuver weights must end with one value per expert.")
    one_hot = (weights == 0) | (weights == 1)
    if not bool(one_hot.all()) or not bool((weights.sum(dim=-1) == 1).all()):
        raise ValueError("M3F maneuver expert routing requires one-hot weights.")
    top_k = 2
    topk_weights, topk_indices = torch.topk(weights, top_k, dim=-1)
    topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-6)
    original_shape = x.shape[:-1]
    feature_dimension = x.shape[-1]
    x_flat = x.reshape(-1, feature_dimension)
    token_count = x_flat.shape[0]
    routed_x = (
        x_flat.unsqueeze(1)
        .expand(-1, top_k, -1)
        .reshape(-1, feature_dimension)
    )
    routed_expert_indices = topk_indices.reshape(-1)
    routed_weights = topk_weights.reshape(-1, 1)
    routed_token_indices = (
        torch.arange(token_count, device=x.device)
        .unsqueeze(1)
        .expand(-1, top_k)
        .reshape(-1)
    )
    output_flat = torch.zeros_like(x_flat)

    for expert_idx, expert in enumerate(experts):
        mask = routed_expert_indices == expert_idx
        if mask.any():
            selected_output = expert(routed_x[mask])
            output_flat.index_add_(
                0,
                routed_token_indices[mask],
                routed_weights[mask] * selected_output,
            )
    return output_flat.reshape(*original_shape, feature_dimension)


class M3FBlock(nn.Module):
    """One maneuver-routed M³F block with AdaLN-Zero conditioning."""

    def __init__(self, config: Any):
        super().__init__()
        self.config = config
        d_model = int(config.m3f_hidden_dim)

        residual_sublayers = 4
        self.interval_adaln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                d_model,
                residual_sublayers * 3 * d_model,
                bias=True,
            ),
        )
        nn.init.constant_(self.interval_adaln[-1].weight, 0)
        nn.init.constant_(self.interval_adaln[-1].bias, 0)

        # Layer 1: BEV perception context plus ego-state conditioning.
        self.bev_state_normalization = nn.LayerNorm(
            d_model, elementwise_affine=False, eps=1e-6
        )
        attn_dropout = float(config.m3f_attention_dropout)
        num_heads = int(config.m3f_num_attention_heads)
        self.bev_state_attention = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.bev_state_type_embedding = nn.Parameter(torch.zeros(2, d_model))
        nn.init.normal_(self.bev_state_type_embedding, std=0.02)

        # Layer 2: time-conditioned FFN on the single trajectory token.
        self.field_normalization = nn.LayerNorm(
            d_model, elementwise_affine=False, eps=1e-6
        )
        self.shared_feed_forward = ManeuverExpertMLP(config)

        # Layer 3: dynamic-agent context.
        self.agent_normalization = nn.LayerNorm(
            d_model, elementwise_affine=False, eps=1e-6
        )
        self.agent_attention = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )

        # Layer 4: one-hot M-GMP routing over the eight maneuver experts. The
        # two-slot sparse implementation preserves the released checkpoint's
        # numerical path while the second slot has zero weight.
        self.expert_normalization = nn.LayerNorm(
            d_model, elementwise_affine=False, eps=1e-6
        )
        self.maneuver_experts = nn.ModuleList(
            [ManeuverExpertMLP(config) for _ in range(8)]
        )

    @staticmethod
    def _expand_cond(param: torch.Tensor) -> torch.Tensor:
        return param.unsqueeze(1).unsqueeze(1)

    def modulate(self, x, shift, scale):
        return x * (1 + self._expand_cond(scale)) + self._expand_cond(shift)

    def forward(
        self,
        x,
        t_emb,
        c_state,
        c_bev,
        c_agent,
        maneuver_component_weights,
    ):
        """
        Shapes:
            x: [batch, hypotheses, tokens, channels]
            c_agent: [batch, agents, channels]
        """
        batch_size, hypothesis_count, token_count, channels = x.shape
        params = self.interval_adaln(t_emb).view(
            batch_size,
            4,
            3,
            channels,
        )

        # Layer 1: BEV and state context.
        shift, scale, gate = params[:, 0, 0], params[:, 0, 1], params[:, 0, 2]
        residual = x
        x_flat = x.view(
            batch_size * hypothesis_count,
            token_count,
            channels,
        )
        x_norm = self.modulate(
            self.bev_state_normalization(x_flat).view(
                batch_size,
                hypothesis_count,
                token_count,
                channels,
            ),
            shift,
            scale,
        )
        x_norm_flat = x_norm.view(
            batch_size * hypothesis_count,
            token_count,
            channels,
        )

        state_mem = c_state + self.bev_state_type_embedding[0].view(
            1,
            1,
            channels,
        )
        bev_mem = c_bev + self.bev_state_type_embedding[1].view(
            1,
            1,
            channels,
        )
        bev_state_mem = torch.cat([state_mem, bev_mem], dim=1)
        bev_state_ex = (
            bev_state_mem.unsqueeze(1)
            .repeat(1, hypothesis_count, 1, 1)
            .view(batch_size * hypothesis_count, -1, channels)
        )
        out, _ = self.bev_state_attention(x_norm_flat, bev_state_ex, bev_state_ex)
        x = residual + self._expand_cond(gate) * out.view(
            batch_size,
            hypothesis_count,
            token_count,
            channels,
        )

        # Layer 2: interval-conditioned shared feed-forward network.
        shift, scale, gate = params[:, 1, 0], params[:, 1, 1], params[:, 1, 2]
        residual = x
        x_norm = self.modulate(self.field_normalization(x), shift, scale)
        ffn_out = self.shared_feed_forward(x_norm)
        x = residual + self._expand_cond(gate) * ffn_out

        # Layer 3: dynamic-agent interaction.
        shift, scale, gate = params[:, 2, 0], params[:, 2, 1], params[:, 2, 2]
        residual = x
        x_flat = x.view(
            batch_size * hypothesis_count,
            token_count,
            channels,
        )
        x_norm = self.modulate(
            self.agent_normalization(x_flat).view(
                batch_size,
                hypothesis_count,
                token_count,
                channels,
            ),
            shift,
            scale,
        ).view(
            batch_size * hypothesis_count,
            token_count,
            channels,
        )
        c_agent_ex = (
            c_agent.unsqueeze(1)
            .repeat(1, hypothesis_count, 1, 1)
            .view(batch_size * hypothesis_count, -1, channels)
        )
        out, _ = self.agent_attention(x_norm, c_agent_ex, c_agent_ex)
        out = out.view(
            batch_size,
            hypothesis_count,
            token_count,
            channels,
        )
        x = residual + self._expand_cond(gate) * out

        # Layer 4: M-GMP-conditioned maneuver expert routing.
        shift, scale, gate = params[:, 3, 0], params[:, 3, 1], params[:, 3, 2]
        residual = x
        x_norm = self.modulate(self.expert_normalization(x), shift, scale)

        maneuver_weights = maneuver_component_weights.unsqueeze(2).expand(
            -1,
            -1,
            token_count,
            -1,
        )
        maneuver_output = sparse_maneuver_expert_forward(
            x_norm,
            maneuver_weights,
            self.maneuver_experts,
        )
        x = residual + self._expand_cond(gate) * maneuver_output

        return x

class ManeuverMixtureMeanFlowField(nn.Module):
    """Stack maneuver-routed M³F blocks into the MeanFlow vector field."""

    def __init__(self, config: Any):
        super().__init__()
        self.config = config

        num_layers = int(config.m3f_num_layers)
        hidden_dim = int(config.m3f_hidden_dim)
        self.field_blocks = nn.ModuleList(
            [M3FBlock(config) for _ in range(num_layers)]
        )

        self.output_normalization = nn.LayerNorm(
            hidden_dim, elementwise_affine=False, eps=1e-6
        )
        self.output_adaln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim)
        )
        nn.init.constant_(self.output_adaln[-1].weight, 0)
        nn.init.constant_(self.output_adaln[-1].bias, 0)

    def forward(
        self,
        x,
        t_emb,
        c_state,
        c_bev,
        c_agent,
        maneuver_component_weights,
    ):
        """
        Shapes:
            x: [batch, hypotheses, tokens, channels]
            c_agent: [batch, agents, channels]
        """
        for block in self.field_blocks:
            x = block(
                x,
                t_emb,
                c_state,
                c_bev,
                c_agent,
                maneuver_component_weights,
            )

        shift, scale = self.output_adaln(t_emb).chunk(2, dim=1)
        x = self.output_normalization(x)
        x = x * (1 + scale.unsqueeze(1).unsqueeze(1)) + shift.unsqueeze(1).unsqueeze(1)

        return x
