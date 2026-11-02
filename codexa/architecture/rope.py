import torch
from torch import Tensor, nn


class RoPE(nn.Module):
    inv_freq: Tensor

    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        self.register_buffer(
            "inv_freq",
            1.0 / base ** (torch.arange(0, head_dim, 2).float() / head_dim),
            persistent=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        # [batch, heads, sequence, head_dim], adjacent coordinate pairs.
        with torch.autocast(device_type=x.device.type, enabled=False):
            angles = torch.outer(
                torch.arange(x.shape[-2], device=x.device).float(), self.inv_freq.float()
            )
            c, s = angles.cos(), angles.sin()
            a, b = x.float()[..., 0::2], x.float()[..., 1::2]
            return torch.stack((a * c - b * s, a * s + b * c), dim=-1).flatten(-2).to(x.dtype)
