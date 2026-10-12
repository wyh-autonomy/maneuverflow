"""Metric Prism: set-level comparison over trajectory evidence memories."""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


def non_collision_score(logits: torch.Tensor) -> torch.Tensor:
    if logits.shape[-1] != 1:
        raise ValueError(
            "Metric Prism requires one non-collision logit, "
            f"got {tuple(logits.shape)}"
        )
    return torch.sigmoid(logits.squeeze(-1))


def _drop_path(
    value: torch.Tensor,
    drop_probability: float = 0.0,
    training: bool = False,
    scale_by_keep: bool = True,
) -> torch.Tensor:
    if drop_probability == 0.0 or not training:
        return value
    keep_probability = 1.0 - drop_probability
    shape = (value.shape[0],) + (1,) * (value.ndim - 1)
    random_tensor = value.new_empty(shape).bernoulli_(keep_probability)
    if keep_probability > 0.0 and scale_by_keep:
        random_tensor.div_(keep_probability)
    return value * random_tensor


class DropPath(nn.Module):
    def __init__(
        self,
        drop_probability: float = 0.0,
        scale_by_keep: bool = True,
    ) -> None:
        super().__init__()
        self.drop_probability = float(drop_probability)
        self.scale_by_keep = bool(scale_by_keep)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return _drop_path(
            value,
            self.drop_probability,
            self.training,
            self.scale_by_keep,
        )


class LayerScale(nn.Module):
    def __init__(
        self,
        dimension: int,
        initial_value: float = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = bool(inplace)
        self.scale = nn.Parameter(
            float(initial_value) * torch.ones(dimension)
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if self.inplace:
            return value.mul_(self.scale)
        return value * self.scale


class MetricPrismAttention(nn.Module):
    """Attention wrapper matching the released single-head scorer path."""

    def __init__(
        self,
        dimension: int,
        attention_heads: int = 1,
        projection_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.attention_heads = int(attention_heads)
        self.projection_dropout = nn.Dropout(float(projection_dropout))
        self.attention = nn.MultiheadAttention(
            dimension,
            self.attention_heads,
            dropout=0.0,
            bias=True,
            add_bias_kv=False,
            add_zero_attn=False,
            batch_first=True,
        )

    def forward(
        self,
        query: torch.Tensor,
        key_value: Optional[torch.Tensor] = None,
        *,
        need_weights: bool = False,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if key_value is None:
            output, weights = self.attention(
                query=query,
                key=query,
                value=query,
                need_weights=need_weights,
                average_attn_weights=True,
            )
        else:
            output, weights = self.attention(
                query=query,
                key=key_value,
                value=key_value,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                average_attn_weights=True,
            )
        return self.projection_dropout(output), weights


class MetricPrismFeedForward(nn.Module):
    def __init__(
        self,
        dimension: int,
        hidden_dimension: int,
        projection_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(dimension, hidden_dimension)
        self.activation = nn.GELU()
        self.input_dropout = nn.Dropout(float(projection_dropout))
        self.output_projection = nn.Linear(hidden_dimension, dimension)
        self.output_dropout = nn.Dropout(float(projection_dropout))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.input_projection(value)
        value = self.activation(value)
        value = self.input_dropout(value)
        value = self.output_projection(value)
        return self.output_dropout(value)


class MetricPrismLayer(nn.Module):
    """Set comparison, evidence attention, and feed-forward update."""

    def __init__(
        self,
        model_dimension: int,
        attention_heads: int,
        hidden_dimension: int,
        projection_dropout: float,
        drop_path_rate: float,
        layer_scale_initial_value: float,
    ) -> None:
        super().__init__()
        self.set_attention_input_normalization = nn.LayerNorm(
            model_dimension
        )
        self.set_attention = MetricPrismAttention(
            model_dimension,
            attention_heads=attention_heads,
            projection_dropout=projection_dropout,
        )
        self.set_attention_layer_scale = (
            LayerScale(
                model_dimension,
                initial_value=layer_scale_initial_value,
            )
            if layer_scale_initial_value > 0.0
            else nn.Identity()
        )
        self.set_attention_drop_path = (
            DropPath(drop_path_rate)
            if drop_path_rate > 0.0
            else nn.Identity()
        )

        self.evidence_key_value_normalization = nn.LayerNorm(
            model_dimension
        )
        self.evidence_query_normalization = nn.LayerNorm(model_dimension)
        self.evidence_attention = MetricPrismAttention(
            model_dimension,
            attention_heads=attention_heads,
            projection_dropout=projection_dropout,
        )
        self.evidence_attention_layer_scale = (
            LayerScale(
                model_dimension,
                initial_value=layer_scale_initial_value,
            )
            if layer_scale_initial_value > 0.0
            else nn.Identity()
        )
        self.evidence_attention_drop_path = (
            DropPath(drop_path_rate)
            if drop_path_rate > 0.0
            else nn.Identity()
        )

        self.feed_forward_input_normalization = nn.LayerNorm(
            model_dimension
        )
        self.feed_forward = MetricPrismFeedForward(
            model_dimension,
            hidden_dimension,
            projection_dropout=projection_dropout,
        )
        self.feed_forward_layer_scale = (
            LayerScale(
                model_dimension,
                initial_value=layer_scale_initial_value,
            )
            if layer_scale_initial_value > 0.0
            else nn.Identity()
        )
        self.feed_forward_drop_path = (
            DropPath(drop_path_rate)
            if drop_path_rate > 0.0
            else nn.Identity()
        )

    def forward(
        self,
        hypothesis_tokens: torch.Tensor,
        evidence_memory: torch.Tensor,
        evidence_memory_padding_mask: Optional[
            torch.Tensor
        ] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        set_update, set_attention_weights = self.set_attention(
            self.set_attention_input_normalization(hypothesis_tokens),
            need_weights=True,
        )
        hypothesis_tokens = (
            hypothesis_tokens
            + self.set_attention_drop_path(
                self.set_attention_layer_scale(set_update)
            )
        )

        batch_size, hypothesis_count, model_dimension = (
            hypothesis_tokens.shape
        )
        evidence_tokens = evidence_memory.shape[2]
        query = self.evidence_query_normalization(
            hypothesis_tokens
        ).reshape(
            batch_size * hypothesis_count,
            1,
            model_dimension,
        )
        key_value = self.evidence_key_value_normalization(
            evidence_memory
        ).reshape(
            batch_size * hypothesis_count,
            evidence_tokens,
            model_dimension,
        )

        flattened_padding_mask = None
        if evidence_memory_padding_mask is not None:
            expected_shape = (
                batch_size,
                hypothesis_count,
                evidence_tokens,
            )
            if tuple(evidence_memory_padding_mask.shape) != expected_shape:
                raise ValueError(
                    "evidence_memory_padding_mask must match [B, N, T], "
                    f"got {tuple(evidence_memory_padding_mask.shape)} "
                    f"versus {expected_shape}"
                )
            flattened_padding_mask = (
                evidence_memory_padding_mask.reshape(
                    batch_size * hypothesis_count,
                    evidence_tokens,
                ).to(
                    device=evidence_memory.device,
                    dtype=torch.bool,
                )
            )

        evidence_update, evidence_attention_weights = (
            self.evidence_attention(
                query,
                key_value,
                need_weights=True,
                key_padding_mask=flattened_padding_mask,
            )
        )
        evidence_update = evidence_update.reshape(
            batch_size,
            hypothesis_count,
            model_dimension,
        )
        hypothesis_tokens = (
            hypothesis_tokens
            + self.evidence_attention_drop_path(
                self.evidence_attention_layer_scale(evidence_update)
            )
        )
        hypothesis_tokens = (
            hypothesis_tokens
            + self.feed_forward_drop_path(
                self.feed_forward_layer_scale(
                    self.feed_forward(
                        self.feed_forward_input_normalization(
                            hypothesis_tokens
                        )
                    )
                )
            )
        )
        if (
            set_attention_weights is None
            or evidence_attention_weights is None
        ):
            raise RuntimeError(
                "MetricPrismLayer expected attention weights"
            )
        return (
            hypothesis_tokens,
            set_attention_weights,
            evidence_attention_weights.reshape(
                batch_size,
                hypothesis_count,
                evidence_tokens,
            ),
        )


def _factor_head(
    model_dimension: int,
    hidden_dimension: int,
    output_dimension: int,
    dropout: float,
    activation: type[nn.Module],
) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(model_dimension),
        nn.Linear(model_dimension, hidden_dimension),
        activation(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dimension, output_dimension),
    )


def _final_linear(head: nn.Sequential) -> nn.Linear:
    final = head[-1]
    if not isinstance(final, nn.Linear):
        raise TypeError(
            "Expected factor head to end with nn.Linear, "
            f"got {type(final).__name__}"
        )
    return final


def _set_binary_prior_bias(
    head: nn.Sequential,
    prior: float,
    clamp_minimum: float,
    clamp_maximum: float,
) -> None:
    final = _final_linear(head)
    prior_tensor = torch.tensor(
        float(prior),
        device=final.bias.device,
        dtype=final.bias.dtype,
    ).clamp(
        min=float(clamp_minimum),
        max=float(clamp_maximum),
    )
    logit = torch.log(prior_tensor / (1.0 - prior_tensor))
    final.bias.copy_(logit.expand_as(final.bias))


def _initialize_factor_prior_biases(
    prism: nn.Module,
    config: Any,
) -> None:
    clamp_minimum = float(config.factor_prior_clamp_minimum)
    clamp_maximum = float(config.factor_prior_clamp_maximum)
    factor_means = config.factor_prior_means

    with torch.no_grad():
        _set_binary_prior_bias(
            prism.non_collision_head,
            float(config.non_collision_binary_safe_prior),
            clamp_minimum,
            clamp_maximum,
        )
        _set_binary_prior_bias(
            prism.drivable_area_compliance_head,
            float(factor_means.get("dac", 0.7189)),
            clamp_minimum,
            clamp_maximum,
        )
        _set_binary_prior_bias(
            prism.ego_progress_head,
            float(factor_means.get("ep", 0.5615)),
            clamp_minimum,
            clamp_maximum,
        )
        _set_binary_prior_bias(
            prism.time_to_collision_head,
            float(factor_means.get("ttc", 0.7710)),
            clamp_minimum,
            clamp_maximum,
        )
        _set_binary_prior_bias(
            prism.comfort_head,
            float(factor_means.get("comfort", 0.9978)),
            clamp_minimum,
            clamp_maximum,
        )

class MetricPrism(nn.Module):
    """Released scorer: one token per hypothesis with set comparison."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.model_dimension = int(config.metric_prism_feature_dimension)
        self.factor_count = 5
        attention_heads = int(config.metric_prism_set_attention_heads)
        if attention_heads != 1:
            raise ValueError(
                "metric_prism_set_attention_heads must be 1 to match the "
                f"released scorer path, got {attention_heads}."
            )
        hidden_dimension = int(config.verifier_feed_forward_dimension)
        dropout = float(config.verifier_dropout)
        projection_dropout = float(
            config.metric_prism_projection_dropout
        )
        drop_path_rate = float(config.metric_prism_drop_path_rate)
        layer_scale_initial_value = float(
            config.metric_prism_layer_scale_initial_value
        )
        layer_count = int(config.metric_prism_layer_count)

        self.evidence_normalization = nn.LayerNorm(self.model_dimension)
        self.hypothesis_position_embedding = nn.Sequential(
            nn.Linear(8 * 3, hidden_dimension),
            nn.ReLU(),
            nn.Linear(hidden_dimension, self.model_dimension),
        )
        self.comparison_layers = nn.ModuleList(
            [
                MetricPrismLayer(
                    self.model_dimension,
                    attention_heads,
                    hidden_dimension,
                    projection_dropout,
                    drop_path_rate,
                    layer_scale_initial_value,
                )
                for _ in range(layer_count)
            ]
        )
        self.non_collision_head = _factor_head(
            self.model_dimension,
            hidden_dimension,
            1,
            dropout,
            nn.ReLU,
        )
        self.drivable_area_compliance_head = _factor_head(
            self.model_dimension,
            hidden_dimension,
            1,
            dropout,
            nn.ReLU,
        )
        self.ego_progress_head = _factor_head(
            self.model_dimension,
            hidden_dimension,
            1,
            dropout,
            nn.ReLU,
        )
        self.time_to_collision_head = _factor_head(
            self.model_dimension,
            hidden_dimension,
            1,
            dropout,
            nn.ReLU,
        )
        self.comfort_head = _factor_head(
            self.model_dimension,
            hidden_dimension,
            1,
            dropout,
            nn.ReLU,
        )
        _initialize_factor_prior_biases(self, config)

    def forward(
        self,
        evidence_outputs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        evidence_memory = self.evidence_normalization(
            evidence_outputs["memory_tokens"]
        )
        batch_size, hypothesis_count, evidence_tokens, _ = (
            evidence_memory.shape
        )

        trajectory_hypotheses = evidence_outputs.get("trajectory_phys")
        if not torch.is_tensor(trajectory_hypotheses):
            raise ValueError(
                "MetricPrism requires evidence_outputs['trajectory_phys']"
            )
        if (
            trajectory_hypotheses.shape[:2]
            != (batch_size, hypothesis_count)
            or trajectory_hypotheses.shape[-2:] != (8, 3)
        ):
            raise ValueError(
                "MetricPrism requires trajectory_phys [B, N, 8, 3], "
                f"got {tuple(trajectory_hypotheses.shape)}"
            )

        flattened_trajectories = trajectory_hypotheses.detach().to(
            device=evidence_memory.device,
            dtype=self.hypothesis_position_embedding[0].weight.dtype,
        ).reshape(batch_size, hypothesis_count, 8 * 3)
        hypothesis_tokens = self.hypothesis_position_embedding(
            flattened_trajectories
        ).to(dtype=evidence_memory.dtype)

        evidence_padding_mask = evidence_outputs.get(
            "memory_key_padding_mask"
        )
        for comparison_layer in self.comparison_layers:
            hypothesis_tokens, _, _ = comparison_layer(
                hypothesis_tokens,
                evidence_memory,
                evidence_memory_padding_mask=evidence_padding_mask,
            )
        hypothesis_tokens = (
            hypothesis_tokens
            + evidence_outputs["ego_token"].to(
                device=hypothesis_tokens.device,
                dtype=hypothesis_tokens.dtype,
            )
        )

        factor_logits = torch.cat(
            [
                self.non_collision_head(hypothesis_tokens),
                self.drivable_area_compliance_head(hypothesis_tokens),
                self.ego_progress_head(hypothesis_tokens),
                self.time_to_collision_head(hypothesis_tokens),
                self.comfort_head(hypothesis_tokens),
            ],
            dim=-1,
        )
        return factor_logits
