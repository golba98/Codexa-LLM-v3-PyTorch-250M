from contextlib import nullcontext

from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.bias import causal_lower_right

from codexa.config import ModelConfig

from .rope import RoPE


class CausalAttention(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.heads, self.head_dim, self.dropout = c.heads, c.dim // c.heads, c.dropout
        self.qkv = nn.Linear(c.dim, 3 * c.dim, bias=False)
        self.out = nn.Linear(c.dim, c.dim, bias=False)
        self.rope = RoPE(self.head_dim, c.rope_base, c.rope_factor)
        self.backend = c.attention_backend

    def forward(
        self, x: Tensor, cache: tuple[Tensor, Tensor] | None = None, offset: int = 0
    ) -> Tensor:
        b, t, d = x.shape
        q, k, v = (
            self.qkv(x).reshape(b, t, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        )
        q, k = self.rope(q, offset), self.rope(k, offset)
        if cache is not None:
            if cache[0].dtype != k.dtype:
                raise ValueError("KV cache precision changed")
            cache[0][:, :, offset : offset + t].copy_(k)
            cache[1][:, :, offset : offset + t].copy_(v)
            k, v = cache[0][:, :, : offset + t], cache[1][:, :, : offset + t]
        backends = {
            "flash": [SDPBackend.FLASH_ATTENTION],
            "efficient": [SDPBackend.EFFICIENT_ATTENTION],
            "math": [SDPBackend.MATH],
        }
        choices = backends.get(self.backend)
        if choices is None and x.is_cuda and k.shape[-2] > 2048:
            choices = [
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.CUDNN_ATTENTION,
            ]
        with sdpa_kernel(choices) if choices else nullcontext():
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=causal_lower_right(t, offset + t) if offset else None,
                is_causal=not bool(offset),
                dropout_p=self.dropout if self.training else 0.0,
            )
        return self.out(y.transpose(1, 2).contiguous().reshape(b, t, d))
