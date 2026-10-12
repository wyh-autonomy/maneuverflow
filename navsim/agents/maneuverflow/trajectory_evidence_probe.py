"""Trajectory-guided current/future BEV feature probing."""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def _normalize_grid_coordinates(
    xy: torch.Tensor,
    config: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    x = xy[..., 0]
    y = xy[..., 1]
    minimum_x = float(config.bev_minimum_x)
    maximum_x = float(config.bev_maximum_x)
    minimum_y = float(config.bev_minimum_y)
    maximum_y = float(config.bev_maximum_y)
    grid_x = (
        2.0 * (y - minimum_y) / max(maximum_y - minimum_y, 1e-6) - 1.0
    )
    grid_y = (
        1.0 - 2.0 * (x - minimum_x) / max(maximum_x - minimum_x, 1e-6)
    )
    if config.bev_coordinate_convention == "lidar_increasing_x":
        grid_y = -grid_y
    grid = torch.stack([grid_x, grid_y], dim=-1)
    valid = (
        (grid[..., 0] >= -1.0)
        & (grid[..., 0] <= 1.0)
        & (grid[..., 1] >= -1.0)
        & (grid[..., 1] <= 1.0)
    )
    return grid.clamp(min=-1.0, max=1.0), valid


def _point_frenet_frame(
    trajectory: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xy = trajectory[..., :2]
    heading = trajectory[..., 2]
    previous_xy = torch.cat([xy[..., :1, :], xy[..., :-1, :]], dim=-2)
    next_xy = torch.cat([xy[..., 1:, :], xy[..., -1:, :]], dim=-2)
    tangent_raw = next_xy - previous_xy
    tangent_norm = tangent_raw.norm(dim=-1, keepdim=True)
    tangent = tangent_raw / tangent_norm.clamp_min(1e-6)
    heading_vector = torch.stack(
        [torch.cos(heading), torch.sin(heading)],
        dim=-1,
    )
    tangent = torch.where(
        (tangent_norm < 1e-3).expand_as(tangent),
        heading_vector,
        tangent,
    )
    tangent = tangent / tangent.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    normal = torch.stack([-tangent[..., 1], tangent[..., 0]], dim=-1)
    return tangent, normal


def _arc_length(xy: torch.Tensor) -> torch.Tensor:
    displacements = xy[..., 1:, :] - xy[..., :-1, :]
    step_length = displacements.norm(dim=-1)
    zero = torch.zeros(
        *xy.shape[:-2],
        1,
        device=xy.device,
        dtype=xy.dtype,
    )
    return torch.cat([zero, step_length.cumsum(dim=-1)], dim=-1)


def _curvature(trajectory: torch.Tensor) -> torch.Tensor:
    xy = trajectory[..., :2]
    heading = trajectory[..., 2]
    displacements = xy[..., 1:, :] - xy[..., :-1, :]
    distance = displacements.norm(dim=-1).clamp_min(1e-3)
    heading_change = _wrap_angle(heading[..., 1:] - heading[..., :-1])
    curvature = heading_change / distance
    return torch.cat([curvature[..., :1], curvature], dim=-1)


def _trajectory_sampling_coordinates(
    centers: torch.Tensor,
    headings: torch.Tensor,
    config: Any,
    sampling_points: int,
) -> torch.Tensor:
    tangent = torch.stack(
        [torch.cos(headings), torch.sin(headings)],
        dim=-1,
    )
    normal = torch.stack([-tangent[..., 1], tangent[..., 0]], dim=-1)
    half_length = 0.5 * float(config.vehicle_length)
    half_width = 0.5 * float(config.vehicle_width)
    corners = torch.stack(
        [
            centers + half_length * tangent + half_width * normal,
            centers + half_length * tangent - half_width * normal,
            centers - half_length * tangent + half_width * normal,
            centers - half_length * tangent - half_width * normal,
        ],
        dim=-2,
    )
    if int(sampling_points) <= 4:
        return corners[..., : int(sampling_points), :]

    extra_pattern = (
        (0.0, 0.0),
        (0.5, 0.0),
        (-0.5, 0.0),
        (0.0, 0.5),
        (0.0, -0.5),
    )
    extra_count = int(sampling_points) - 4
    extras = [
        centers
        + float(longitudinal_scale) * tangent
        + float(lateral_scale) * normal
        for longitudinal_scale, lateral_scale in extra_pattern[:extra_count]
    ]
    if extras:
        return torch.cat([corners, torch.stack(extras, dim=-2)], dim=-2)
    return corners


def _sample_bev_along_trajectory(
    bev_features: torch.Tensor,
    coordinates: torch.Tensor,
    config: Any,
    return_valid: bool = False,
) -> Any:
    if bev_features.dim() == 4:
        bev_features = bev_features.unsqueeze(1).expand(
            -1,
            coordinates.shape[2],
            -1,
            -1,
            -1,
        ).contiguous()

    batch_size, time_steps, channels, height, width = bev_features.shape
    hypothesis_count = coordinates.shape[1]
    sampling_points = coordinates.shape[3]
    flattened_bev = bev_features.reshape(
        batch_size * time_steps,
        channels,
        height,
        width,
    )
    normalized_coordinates, valid = _normalize_grid_coordinates(
        coordinates,
        config,
    )
    flattened_coordinates = normalized_coordinates.permute(
        0,
        2,
        1,
        3,
        4,
    ).reshape(
        batch_size * time_steps,
        hypothesis_count * sampling_points,
        1,
        2,
    )
    sampled = F.grid_sample(
        flattened_bev,
        flattened_coordinates,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=config.bev_coordinate_convention == "legacy_descending_x",
    )
    sampled = sampled.view(
        batch_size,
        time_steps,
        channels,
        hypothesis_count,
        sampling_points,
    )
    sampled = sampled.permute(0, 3, 1, 4, 2).contiguous()
    sampled = sampled * valid.unsqueeze(-1).to(dtype=sampled.dtype)
    if return_valid:
        return sampled, valid.to(device=sampled.device)
    return sampled


class EvidenceSlotPool(nn.Module):
    """Pool spatial probe samples into a fixed number of evidence tokens."""

    def __init__(
        self,
        model_dimension: int,
        output_tokens: int,
        attention_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.output_tokens = int(output_tokens)
        self.slot_query_embedding = nn.Parameter(
            torch.randn(1, self.output_tokens, model_dimension) * 0.02
        )
        self.evidence_context_projection = nn.Sequential(
            nn.LayerNorm(model_dimension),
            nn.Linear(model_dimension, model_dimension),
            nn.GELU(),
            nn.Linear(model_dimension, model_dimension),
        )
        self.slot_attention = nn.MultiheadAttention(
            model_dimension,
            attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.output_normalization = nn.LayerNorm(model_dimension)

    def forward(
        self,
        sampled_features: torch.Tensor,
        sample_validity: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, hypothesis_count, steps, sampling_points, channels = (
            sampled_features.shape
        )
        flattened_features = sampled_features.reshape(
            batch_size * hypothesis_count * steps,
            sampling_points,
            channels,
        )
        context = self.evidence_context_projection(
            flattened_features.mean(dim=1)
        )
        query = self.slot_query_embedding.expand(
            flattened_features.shape[0],
            -1,
            -1,
        ) + context.unsqueeze(1)

        key_padding_mask = None
        if sample_validity is not None:
            flattened_validity = sample_validity.reshape(
                flattened_features.shape[0],
                sampling_points,
            ).to(dtype=torch.bool)
            all_invalid = ~flattened_validity.any(dim=1)
            flattened_validity = flattened_validity.clone()
            flattened_validity[:, 0] = (
                flattened_validity[:, 0] | all_invalid
            )
            first_sample = torch.arange(
                sampling_points,
                device=flattened_features.device,
            ).eq(0)
            invalid_fallback = (
                all_invalid.unsqueeze(1) & first_sample.unsqueeze(0)
            )
            flattened_features = flattened_features.masked_fill(
                invalid_fallback.unsqueeze(-1),
                0.0,
            )
            key_padding_mask = ~flattened_validity

        pooled, _ = self.slot_attention(
            query,
            flattened_features,
            flattened_features,
            key_padding_mask=key_padding_mask,
        )
        pooled = self.output_normalization(pooled + context.unsqueeze(1))
        return pooled.view(
            batch_size,
            hypothesis_count,
            steps,
            self.output_tokens,
            channels,
        )


class TrajectoryGeometryEncoder(nn.Module):
    """Encode waypoint geometry into one hypothesis-level evidence token."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.model_dimension = int(config.metric_prism_feature_dimension)
        attention_heads = int(config.metric_prism_attention_heads)
        dropout = float(config.verifier_dropout)
        feed_forward_dimension = int(
            config.verifier_feed_forward_dimension
        )

        self.waypoint_feature_projection = nn.Linear(8, self.model_dimension)
        self.trajectory_summary_projection = nn.Sequential(
            nn.LayerNorm(6),
            nn.Linear(6, self.model_dimension),
            nn.GELU(),
            nn.Linear(self.model_dimension, self.model_dimension),
        )
        self.trajectory_summary_token = nn.Parameter(
            torch.randn(1, 1, self.model_dimension) * 0.02
        )
        self.waypoint_position_embedding = nn.Embedding(
            8,
            self.model_dimension,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.model_dimension,
            nhead=attention_heads,
            dim_feedforward=max(
                self.model_dimension * 2,
                feed_forward_dimension,
            ),
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.trajectory_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=1,
        )
        self.output_normalization = nn.LayerNorm(self.model_dimension)

    def forward(self, trajectory: torch.Tensor) -> torch.Tensor:
        batch_size, hypothesis_count, steps, _ = trajectory.shape
        xy = trajectory[..., :2]
        heading = trajectory[..., 2]
        tangent, _ = _point_frenet_frame(trajectory)
        arc_length = _arc_length(xy)
        curvature = _curvature(trajectory)
        waypoint_features = torch.cat(
            [
                xy,
                torch.sin(heading).unsqueeze(-1),
                torch.cos(heading).unsqueeze(-1),
                tangent,
                arc_length.unsqueeze(-1),
                curvature.unsqueeze(-1),
            ],
            dim=-1,
        )
        waypoint_embeddings = self.waypoint_feature_projection(
            waypoint_features
        )
        waypoint_ids = torch.arange(
            steps,
            device=trajectory.device,
        ).clamp(max=7)
        waypoint_embeddings = (
            waypoint_embeddings
            + self.waypoint_position_embedding(waypoint_ids).view(
                1,
                1,
                steps,
                self.model_dimension,
            )
        )
        flattened_waypoints = waypoint_embeddings.reshape(
            batch_size * hypothesis_count,
            steps,
            self.model_dimension,
        )
        summary_token = self.trajectory_summary_token.expand(
            batch_size * hypothesis_count,
            -1,
            -1,
        )
        encoded = self.trajectory_encoder(
            torch.cat([summary_token, flattened_waypoints], dim=1)
        )[:, 0]

        final_xy = xy[..., -1, :]
        total_arc_length = arc_length[..., -1:]
        net_heading = _wrap_angle(
            heading[..., -1] - heading[..., 0]
        ).unsqueeze(-1)
        maximum_absolute_curvature = curvature.abs().amax(
            dim=-1,
            keepdim=True,
        )
        maximum_absolute_lateral_offset = xy[..., 1].abs().amax(
            dim=-1,
            keepdim=True,
        )
        trajectory_summary = torch.cat(
            [
                final_xy,
                total_arc_length,
                net_heading,
                maximum_absolute_curvature,
                maximum_absolute_lateral_offset,
            ],
            dim=-1,
        )
        return self.output_normalization(
            encoded.view(
                batch_size,
                hypothesis_count,
                self.model_dimension,
            )
            + self.trajectory_summary_projection(trajectory_summary)
        )


class TrajectoryKinematicsEncoder(nn.Module):
    """Encode trajectory kinematics into one hypothesis-level token."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.model_dimension = int(config.metric_prism_feature_dimension)
        self.time_interval = 0.5
        self.motion_feature_projection = nn.Linear(5, self.model_dimension)
        self.temporal_encoder = nn.GRU(
            self.model_dimension,
            self.model_dimension,
            batch_first=True,
        )
        self.extreme_motion_projection = nn.Sequential(
            nn.LayerNorm(4),
            nn.Linear(4, self.model_dimension),
            nn.GELU(),
            nn.Linear(self.model_dimension, self.model_dimension),
        )
        self.output_normalization = nn.LayerNorm(self.model_dimension)

    def forward(
        self,
        trajectory: torch.Tensor,
        state_navigation_feature: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, hypothesis_count, steps, _ = trajectory.shape
        xy = trajectory[..., :2]
        heading = trajectory[..., 2]
        displacements = xy[..., 1:, :] - xy[..., :-1, :]
        future_speed = displacements.norm(dim=-1) / max(
            self.time_interval,
            1e-3,
        )
        current_speed = state_navigation_feature[:, 4:6].float().norm(
            dim=-1
        ).view(batch_size, 1, 1)
        speed = torch.cat(
            [
                current_speed.expand(-1, hypothesis_count, -1),
                future_speed,
            ],
            dim=-1,
        )
        acceleration = torch.zeros_like(speed)
        acceleration[..., 0] = state_navigation_feature[
            :, 6:8
        ].float().norm(dim=-1).view(batch_size, 1).expand(
            -1,
            hypothesis_count,
        )
        acceleration[..., 1:] = (
            speed[..., 1:] - speed[..., :-1]
        ) / max(self.time_interval, 1e-3)
        curvature = _curvature(trajectory)
        lateral_acceleration = speed.square() * curvature
        jerk = torch.zeros_like(acceleration)
        jerk[..., 1:] = (
            acceleration[..., 1:] - acceleration[..., :-1]
        ) / max(self.time_interval, 1e-3)
        yaw_rate = torch.zeros_like(speed)
        yaw_rate[..., 1:] = _wrap_angle(
            heading[..., 1:] - heading[..., :-1]
        ) / max(self.time_interval, 1e-3)
        motion_features = torch.stack(
            [
                speed,
                acceleration,
                lateral_acceleration,
                jerk,
                yaw_rate,
            ],
            dim=-1,
        )
        flattened_motion = self.motion_feature_projection(
            motion_features
        ).reshape(
            batch_size * hypothesis_count,
            steps,
            self.model_dimension,
        )
        _, hidden = self.temporal_encoder(flattened_motion)
        sequence_token = hidden[-1].view(
            batch_size,
            hypothesis_count,
            self.model_dimension,
        )
        extreme_motion = torch.stack(
            [
                acceleration.abs().amax(dim=-1),
                lateral_acceleration.abs().amax(dim=-1),
                jerk.abs().amax(dim=-1),
                yaw_rate.abs().amax(dim=-1),
            ],
            dim=-1,
        )
        return self.output_normalization(
            sequence_token
            + self.extreme_motion_projection(extreme_motion)
        )


class StateNavigationEncoder(nn.Module):
    """Encode ego state and navigation into a hypothesis-broadcast token."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.model_dimension = int(config.metric_prism_feature_dimension)
        self.state_navigation_projection = nn.Sequential(
            nn.LayerNorm(8),
            nn.Linear(8, self.model_dimension),
            nn.GELU(),
            nn.Linear(self.model_dimension, self.model_dimension),
        )

    def forward(
        self,
        state_navigation_feature: torch.Tensor,
        hypothesis_count: int,
    ) -> torch.Tensor:
        token = self.state_navigation_projection(
            state_navigation_feature.float()
        )
        return token.unsqueeze(1).expand(-1, hypothesis_count, -1)


class TrajectoryGuidedBEVFeatureProbe(nn.Module):
    """Build trajectory-specific evidence memory from shared BEV fields."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config
        self.model_dimension = int(config.metric_prism_feature_dimension)
        self.future_bev_dimension = int(
            config.future_bev_feature_dimension
        )
        self.future_steps = int(config.future_bev_horizon_steps)
        attention_heads = int(config.metric_prism_attention_heads)
        dropout = float(config.verifier_dropout)
        self.current_tokens_per_step = int(
            config.current_evidence_tokens_per_step
        )
        self.future_tokens_per_step = int(
            config.future_evidence_tokens_per_step
        )
        self.current_sampling_points = int(
            config.current_evidence_sampling_points
        )
        self.future_sampling_points = int(
            config.future_evidence_sampling_points
        )

        self.maneuver_component_count = int(
            config.num_maneuver_components
        )
        current_bev_channels = int(config.current_bev_feature_channels)

        self.current_bev_projection = nn.Conv2d(
            current_bev_channels,
            self.model_dimension,
            kernel_size=1,
        )
        self.future_absolute_projection = nn.Conv2d(
            self.future_bev_dimension,
            self.model_dimension,
            kernel_size=1,
        )
        self.current_evidence_pool = EvidenceSlotPool(
            self.model_dimension,
            self.current_tokens_per_step,
            attention_heads,
            dropout,
        )
        self.future_absolute_evidence_pool = EvidenceSlotPool(
            self.model_dimension,
            self.future_tokens_per_step,
            attention_heads,
            dropout,
        )
        self.future_residual_projection = nn.Conv2d(
            self.future_bev_dimension,
            self.model_dimension,
            kernel_size=1,
        )
        self.future_residual_evidence_pool = EvidenceSlotPool(
            self.model_dimension,
            self.future_tokens_per_step,
            attention_heads,
            dropout,
        )

        self.trajectory_geometry_encoder = TrajectoryGeometryEncoder(config)
        self.trajectory_kinematics_encoder = TrajectoryKinematicsEncoder(
            config
        )
        self.state_navigation_encoder = StateNavigationEncoder(config)

        self.evidence_type_embedding = nn.Embedding(6, self.model_dimension)
        self.current_step_embedding = nn.Embedding(8, self.model_dimension)
        self.future_step_embedding = nn.Embedding(
            self.future_steps,
            self.model_dimension,
        )
        self.maneuver_route_projection = nn.Sequential(
            nn.LayerNorm(self.maneuver_component_count),
            nn.Linear(
                self.maneuver_component_count,
                self.model_dimension,
            ),
            nn.GELU(),
            nn.Linear(self.model_dimension, self.model_dimension),
        )
        self.maneuver_route_type_embedding = nn.Parameter(
            torch.randn(1, 1, self.model_dimension) * 0.02
        )
        self.metadata_normalization = nn.LayerNorm(self.model_dimension)
        self.evidence_memory_normalization = nn.LayerNorm(
            self.model_dimension
        )

    @staticmethod
    def _future_step_indices(
        device: torch.device,
        trajectory_steps: int,
    ) -> torch.Tensor:
        if trajectory_steps >= 8:
            return torch.tensor(
                [1, 3, 5, 7],
                device=device,
                dtype=torch.long,
            )
        return torch.linspace(
            0,
            max(trajectory_steps - 1, 0),
            steps=4,
            device=device,
        ).round().long()

    def _add_step_and_type_embeddings(
        self,
        tokens: torch.Tensor,
        evidence_type: int,
        step_embedding: nn.Embedding,
    ) -> torch.Tensor:
        steps = tokens.shape[2]
        step_ids = torch.arange(
            steps,
            device=tokens.device,
            dtype=torch.long,
        )
        return (
            tokens
            + self.evidence_type_embedding.weight[evidence_type].view(
                1,
                1,
                1,
                1,
                -1,
            )
            + step_embedding(step_ids).view(
                1,
                1,
                steps,
                1,
                -1,
            )
        )

    def prepare_evidence_field(
        self,
        evidence_field: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Project shared BEV fields once for repeated trajectory queries."""

        current_bev = evidence_field["cur_grid"]
        future_absolute_bev = evidence_field["fut_abs"]
        if future_absolute_bev.shape[2] != self.future_bev_dimension:
            raise ValueError(
                "TrajectoryGuidedBEVFeatureProbe future-absolute channel "
                f"mismatch: expected {self.future_bev_dimension}, "
                f"got {future_absolute_bev.shape[2]}."
            )
        future_residual_bev = evidence_field["fut_res"]
        if future_residual_bev.shape[2] != self.future_bev_dimension:
            raise ValueError(
                "TrajectoryGuidedBEVFeatureProbe future-residual channel "
                f"mismatch: expected {self.future_bev_dimension}, "
                f"got {future_residual_bev.shape[2]}."
            )
        if current_bev.shape[-2:] != future_absolute_bev.shape[-2:]:
            current_bev = F.adaptive_avg_pool2d(
                current_bev,
                output_size=future_absolute_bev.shape[-2:],
            )

        prepared = {
            "current_bev_projection": self.current_bev_projection(
                current_bev
            )
        }
        batch_size = int(future_absolute_bev.shape[0])
        flattened_absolute_bev = future_absolute_bev.reshape(
            batch_size * self.future_steps,
            self.future_bev_dimension,
            future_absolute_bev.shape[-2],
            future_absolute_bev.shape[-1],
        )
        prepared["future_absolute_projection"] = (
            self.future_absolute_projection(flattened_absolute_bev).view(
                batch_size,
                self.future_steps,
                self.model_dimension,
                *future_absolute_bev.shape[-2:],
            )
        )
        flattened_residual_bev = future_residual_bev.reshape(
            batch_size * self.future_steps,
            self.future_bev_dimension,
            future_residual_bev.shape[-2],
            future_residual_bev.shape[-1],
        )
        prepared["future_residual_projection"] = (
            self.future_residual_projection(
                flattened_residual_bev
            ).view(
                batch_size,
                self.future_steps,
                self.model_dimension,
                *future_residual_bev.shape[-2:],
            )
        )
        return prepared

    def forward(
        self,
        trajectory_hypotheses: torch.Tensor,
        state_navigation_feature: torch.Tensor,
        evidence_field: Dict[str, torch.Tensor],
        maneuver_component_weights: Optional[torch.Tensor] = None,
        prepared_evidence_field: Optional[
            Dict[str, torch.Tensor]
        ] = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size, hypothesis_count, trajectory_steps, _ = (
            trajectory_hypotheses.shape
        )
        if prepared_evidence_field is None:
            prepared_evidence_field = self.prepare_evidence_field(
                evidence_field
            )

        current_projection = prepared_evidence_field[
            "current_bev_projection"
        ]
        current_coordinates = _trajectory_sampling_coordinates(
            trajectory_hypotheses[..., :2],
            trajectory_hypotheses[..., 2],
            self.config,
            self.current_sampling_points,
        )
        current_samples, current_validity = _sample_bev_along_trajectory(
            current_projection,
            current_coordinates,
            self.config,
            return_valid=True,
        )
        current_tokens = self.current_evidence_pool(
            current_samples,
            current_validity,
        )
        current_tokens = self._add_step_and_type_embeddings(
            current_tokens,
            evidence_type=0,
            step_embedding=self.current_step_embedding,
        )
        current_tokens = current_tokens.reshape(
            batch_size,
            hypothesis_count,
            8 * self.current_tokens_per_step,
            self.model_dimension,
        )

        future_indices = self._future_step_indices(
            trajectory_hypotheses.device,
            trajectory_steps,
        )
        future_trajectory = trajectory_hypotheses.index_select(
            2,
            future_indices,
        )
        future_coordinates = _trajectory_sampling_coordinates(
            future_trajectory[..., :2],
            future_trajectory[..., 2],
            self.config,
            self.future_sampling_points,
        )
        future_absolute_projection = prepared_evidence_field[
            "future_absolute_projection"
        ]
        absolute_samples, absolute_validity = (
            _sample_bev_along_trajectory(
                future_absolute_projection,
                future_coordinates,
                self.config,
                return_valid=True,
            )
        )
        absolute_tokens = self.future_absolute_evidence_pool(
            absolute_samples,
            absolute_validity,
        )
        absolute_tokens = self._add_step_and_type_embeddings(
            absolute_tokens,
            evidence_type=1,
            step_embedding=self.future_step_embedding,
        )
        future_absolute_tokens = absolute_tokens.reshape(
            batch_size,
            hypothesis_count,
            self.future_steps * self.future_tokens_per_step,
            self.model_dimension,
        )
        future_residual_projection = prepared_evidence_field[
            "future_residual_projection"
        ]
        residual_samples, residual_validity = (
            _sample_bev_along_trajectory(
                future_residual_projection,
                future_coordinates,
                self.config,
                return_valid=True,
            )
        )
        residual_tokens = self.future_residual_evidence_pool(
            residual_samples,
            residual_validity,
        )
        residual_tokens = self._add_step_and_type_embeddings(
            residual_tokens,
            evidence_type=2,
            step_embedding=self.future_step_embedding,
        )
        future_residual_tokens = residual_tokens.reshape(
            batch_size,
            hypothesis_count,
            self.future_steps * self.future_tokens_per_step,
            self.model_dimension,
        )

        geometry_token = self.trajectory_geometry_encoder(
            trajectory_hypotheses
        ) + self.evidence_type_embedding.weight[3]
        kinematics_token = self.trajectory_kinematics_encoder(
            trajectory_hypotheses,
            state_navigation_feature,
        ) + self.evidence_type_embedding.weight[4]
        state_navigation_token = self.state_navigation_encoder(
            state_navigation_feature,
            hypothesis_count,
        ) + self.evidence_type_embedding.weight[5]
        metadata_tokens = self.metadata_normalization(
            torch.stack(
                [geometry_token, kinematics_token, state_navigation_token],
                dim=2,
            )
        )

        if maneuver_component_weights is None:
            maneuver_input = trajectory_hypotheses.new_zeros(
                batch_size,
                hypothesis_count,
                self.maneuver_component_count,
            )
        else:
            if maneuver_component_weights.dim() != 3:
                raise ValueError(
                    "maneuver_component_weights must be [B, N, M], "
                    f"got {tuple(maneuver_component_weights.shape)}"
                )
            if maneuver_component_weights.shape[:2] != (
                batch_size,
                hypothesis_count,
            ):
                raise ValueError(
                    "maneuver_component_weights batch/hypothesis mismatch: "
                    f"expected {(batch_size, hypothesis_count)}, "
                    f"got {tuple(maneuver_component_weights.shape[:2])}"
                )
            if (
                maneuver_component_weights.shape[2]
                != self.maneuver_component_count
            ):
                raise ValueError(
                    "maneuver_component_weights component mismatch: "
                    f"expected {self.maneuver_component_count}, "
                    f"got {maneuver_component_weights.shape[2]}"
                )
            maneuver_input = maneuver_component_weights.to(
                device=trajectory_hypotheses.device,
                dtype=trajectory_hypotheses.dtype,
            )
        maneuver_route_token = self.maneuver_route_projection(
            maneuver_input
        ).to(dtype=metadata_tokens.dtype)
        maneuver_route_token = (
            maneuver_route_token
            + self.maneuver_route_type_embedding.to(
                dtype=metadata_tokens.dtype
            )
        )
        metadata_tokens = torch.cat(
            [metadata_tokens, maneuver_route_token.unsqueeze(2)],
            dim=2,
        )

        memory_tokens = self.evidence_memory_normalization(
            torch.cat(
                [
                    current_tokens,
                    future_absolute_tokens,
                    future_residual_tokens,
                    metadata_tokens,
                ],
                dim=2,
            )
        )
        memory_type_ids = torch.tensor(
            [0] * current_tokens.shape[2]
            + [1] * future_absolute_tokens.shape[2]
            + [2] * future_residual_tokens.shape[2]
            + [3, 4, 5, 6],
            device=memory_tokens.device,
            dtype=torch.long,
        )
        future_step_ids = torch.arange(
            self.future_steps,
            device=memory_tokens.device,
        ).repeat_interleave(self.future_tokens_per_step)
        memory_step_ids = torch.cat(
            [
                torch.arange(
                    8,
                    device=memory_tokens.device,
                ).repeat_interleave(self.current_tokens_per_step),
                future_step_ids,
                future_step_ids,
                torch.full(
                    (metadata_tokens.shape[2],),
                    -1,
                    device=memory_tokens.device,
                    dtype=torch.long,
                ),
            ],
            dim=0,
        )
        memory_is_future = (
            (memory_type_ids == 1) | (memory_type_ids == 2)
        )
        memory_is_meta = memory_type_ids >= 3
        memory_key_padding_mask = None

        alpha_time_shared = torch.full(
            (batch_size, hypothesis_count, self.future_steps),
            1.0 / float(self.future_steps),
            device=memory_tokens.device,
            dtype=memory_tokens.dtype,
        )
        return {
            "memory_tokens": memory_tokens,
            "memory_type_ids": memory_type_ids,
            "memory_step_ids": memory_step_ids,
            "memory_is_future": memory_is_future,
            "memory_is_meta": memory_is_meta,
            "memory_key_padding_mask": memory_key_padding_mask,
            "current_tokens": current_tokens,
            "future_abs_tokens": future_absolute_tokens,
            "future_res_tokens": future_residual_tokens,
            "geom_token": geometry_token,
            "kinematics_token": kinematics_token,
            "ego_token": state_navigation_token,
            "maneuver_route_token": maneuver_route_token,
            "trajectory_phys": trajectory_hypotheses,
            "alpha_time_shared": alpha_time_shared,
            "m_cur_tokens": current_tokens,
            "m_fut_tokens": torch.cat(
                [future_absolute_tokens, future_residual_tokens],
                dim=2,
            ),
        }
