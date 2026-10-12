"""Trajectory representation used by the Maneuver-Mixture MeanFlow Field."""

from typing import Any, Protocol, Union

import torch
import torch.nn as nn


class TrajectoryCodecConfig(Protocol):
    """Configuration fields consumed by this module."""

    trajectory_displacement_x_mean: float
    trajectory_displacement_x_scale: float
    trajectory_displacement_y_mean: float
    trajectory_displacement_y_scale: float
    m3f_hidden_dim: int


def trajectory_to_flow_state(
    trajectory: Union[torch.Tensor, Any], config: TrajectoryCodecConfig
) -> torch.Tensor:
    """
    Convert absolute trajectory [..., L, 3] to M-GMP displacement state [..., L, 4]:
    [dx_norm, dy_norm, sin(heading), cos(heading)].
    """
    if not isinstance(trajectory, torch.Tensor):
        trajectory = torch.tensor(trajectory, dtype=torch.float32)

    traj = trajectory.float()
    if traj.shape[-1] < 3:
        raise ValueError(f"trajectory_to_flow_state expects last dim >=3, got {traj.shape}")

    x = traj[..., 0:1]
    y = traj[..., 1:2]
    heading = traj[..., 2:3]

    x_prepend = torch.zeros_like(x[..., :1, :])
    y_prepend = torch.zeros_like(y[..., :1, :])
    x_diff = torch.diff(x, dim=-2, prepend=x_prepend)
    y_diff = torch.diff(y, dim=-2, prepend=y_prepend)

    x_mean = float(config.trajectory_displacement_x_mean)
    x_range = float(config.trajectory_displacement_x_scale)
    y_mean = float(config.trajectory_displacement_y_mean)
    y_range = float(config.trajectory_displacement_y_scale)
    x_range = x_range if abs(x_range) > 1e-6 else 1.0
    y_range = y_range if abs(y_range) > 1e-6 else 1.0

    x_diff_norm = (x_diff - x_mean) / x_range
    y_diff_norm = (y_diff - y_mean) / y_range
    sin_heading = torch.sin(heading)
    cos_heading = torch.cos(heading)

    return torch.cat([x_diff_norm, y_diff_norm, sin_heading, cos_heading], dim=-1)


def flow_state_to_trajectory(
    flow_state: Union[torch.Tensor, Any], config: TrajectoryCodecConfig
) -> torch.Tensor:
    """
    Convert M-GMP displacement state [..., L, 4] to absolute trajectory [..., L, 3]:
    [x, y, heading].
    """
    if not isinstance(flow_state, torch.Tensor):
        flow_state = torch.tensor(flow_state, dtype=torch.float32)

    feat = flow_state.float()
    if feat.shape[-1] < 4:
        raise ValueError(f"flow_state_to_trajectory expects last dim >=4, got {feat.shape}")

    x_mean = float(config.trajectory_displacement_x_mean)
    x_range = float(config.trajectory_displacement_x_scale)
    y_mean = float(config.trajectory_displacement_y_mean)
    y_range = float(config.trajectory_displacement_y_scale)
    x_range = x_range if abs(x_range) > 1e-6 else 1.0
    y_range = y_range if abs(y_range) > 1e-6 else 1.0

    dx = feat[..., 0:1] * x_range + x_mean
    dy = feat[..., 1:2] * y_range + y_mean
    x = torch.cumsum(dx, dim=-2)
    y = torch.cumsum(dy, dim=-2)

    heading_vec = feat[..., 2:4]
    heading_vec = heading_vec / heading_vec.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    sin_heading = heading_vec[..., 0:1]
    cos_heading = heading_vec[..., 1:2]
    heading = torch.atan2(sin_heading, cos_heading)

    return torch.cat([x, y, heading], dim=-1)


class M3FTrajectoryTokenizer(nn.Module):
    """Tokenize one complete trajectory state for the M³F field."""

    def __init__(self, config: TrajectoryCodecConfig, input_dim: int = 4):
        """Build a whole-trajectory tokenizer: `[8, 4] -> [1, D]`."""
        super().__init__()
        self.hidden_dim = int(config.m3f_hidden_dim)
        self.input_dim = input_dim
        self.horizon = 8

        self.state_projection = nn.Linear(self.horizon * input_dim, self.hidden_dim)
        self.position_embedding = nn.Parameter(
            torch.randn(1, 1, self.hidden_dim) * 0.02
        )
        self.token_normalization = nn.LayerNorm(self.hidden_dim)

    def forward(self, trajectory_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            trajectory_state: `[batch, hypotheses, 8, 4]` flow state.

        Returns:
            `[batch, hypotheses, 1, model_dimension]` tokens.
        """
        batch_size, num_hypotheses, horizon, channels = trajectory_state.shape
        if horizon != self.horizon or channels != self.input_dim:
            raise ValueError(
                "M3FTrajectoryTokenizer expects "
                f"[*, *, {self.horizon}, {self.input_dim}], got {tuple(trajectory_state.shape)}"
            )

        tokens = trajectory_state.reshape(
            batch_size * num_hypotheses, horizon * channels
        )
        tokens = tokens.to(dtype=self.state_projection.weight.dtype)
        tokens = self.state_projection(tokens).unsqueeze(1)
        tokens = self.token_normalization(tokens + self.position_embedding)
        return tokens.view(batch_size, num_hypotheses, 1, self.hidden_dim)


class M3FFieldDecoder(nn.Module):
    """Decode an M³F token into the interval-averaged trajectory field."""

    def __init__(self, config: TrajectoryCodecConfig, output_dim: int = 4):
        """Build a token decoder: `[1, D] -> [8, 4]` flow field."""
        super().__init__()
        self.hidden_dim = int(config.m3f_hidden_dim)
        self.output_dim = output_dim
        self.horizon = 8

        self.field_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.horizon * output_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: [Batch, Na, 1, d_model]

        Returns:
            flow_v: [Batch, Na, 8, 4] internal flow feature
        """
        batch_size, num_hypotheses, num_tokens, hidden_dim = tokens.shape
        if num_tokens != 1:
            raise ValueError(
                f"M3FFieldDecoder expects a single token, got {tuple(tokens.shape)}"
            )

        field = self.field_head(
            tokens.reshape(batch_size * num_hypotheses, hidden_dim)
        )
        return field.view(
            batch_size, num_hypotheses, self.horizon, self.output_dim
        )
