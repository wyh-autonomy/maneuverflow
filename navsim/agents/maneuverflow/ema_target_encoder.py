"""Frozen EMA encoder used only to produce future-BEV training targets."""

from __future__ import annotations

from copy import deepcopy
import math

import torch
from torch import nn


class EMATargetEncoder(nn.Module):
    """Keep an independent FP32 teacher updated after online optimizer steps.

    Parameters use EMA; every encoder buffer is copied from the online model.
    Both the decay and update counter belong to the training checkpoint.
    """

    def __init__(self, online_encoder: nn.Module, decay: float = 0.999):
        super().__init__()
        if not math.isfinite(float(decay)) or not 0.0 <= float(decay) < 1.0:
            raise ValueError("EMA decay must be finite and in [0, 1).")
        self.encoder = deepcopy(online_encoder).float()
        self.encoder.requires_grad_(False)
        reference = next(self.encoder.parameters(), None)
        if reference is None:
            reference = next(self.encoder.buffers(), None)
        device = None if reference is None else reference.device
        self.register_buffer("decay", torch.tensor(float(decay), dtype=torch.float64, device=device))
        self.register_buffer("num_updates", torch.zeros((), dtype=torch.int64, device=device))
        self.train(False)

    def train(self, mode: bool = True):
        """A parent entering train mode must never enable teacher BN/dropout."""

        super().train(False)
        return self

    def _apply(self, fn, recurse=True):
        """Move devices with the parent while preserving the teacher's dtype."""

        def preserve_dtype(tensor):
            result = fn(tensor)
            if result.dtype != tensor.dtype:
                return tensor.to(device=result.device, dtype=tensor.dtype)
            return result

        return super()._apply(preserve_dtype, recurse=recurse)

    @torch.no_grad()
    def reset_from_online(self, online_encoder: nn.Module) -> None:
        """Hard-sync only at the explicit Stage-1 weight handoff boundary."""

        self.encoder.load_state_dict(online_encoder.state_dict(), strict=True)
        self.encoder.requires_grad_(False)
        self.num_updates.zero_()
        self.train(False)

    @torch.no_grad()
    def update(self, online_encoder: nn.Module) -> None:
        """Perform one parameter EMA and copy/synchronize all teacher buffers."""

        online_parameters = dict(online_encoder.named_parameters())
        teacher_parameters = dict(self.encoder.named_parameters())
        online_buffers = dict(online_encoder.named_buffers())
        teacher_buffers = dict(self.encoder.named_buffers())
        for teacher_tensors, online_tensors, kind in (
            (teacher_parameters, online_parameters, "parameters"),
            (teacher_buffers, online_buffers, "buffers"),
        ):
            if teacher_tensors.keys() != online_tensors.keys() or any(
                teacher_tensors[name].shape != online_tensors[name].shape
                for name in teacher_tensors
            ):
                raise ValueError(f"EMA online and teacher {kind} must match exactly.")

        decay = float(self.decay.item())
        for name, parameter in teacher_parameters.items():
            source = online_parameters[name].detach().to(parameter)
            parameter.mul_(decay).add_(source, alpha=1.0 - decay)
        for name, buffer in teacher_buffers.items():
            buffer.copy_(online_buffers[name].detach())
        self.num_updates.add_(1)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            for buffer in self.buffers():
                torch.distributed.broadcast(buffer, src=0)
        self.train(False)

    @torch.no_grad()
    def forward(self, *args, **kwargs):
        self.train(False)
        with torch.autocast(device_type=self.decay.device.type, enabled=False):
            return self.encoder(*args, **kwargs)
