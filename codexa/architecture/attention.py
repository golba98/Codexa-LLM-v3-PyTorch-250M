from torch import Tensor, nn
from torch.nn import functional as F

from codexa.config import ModelConfig

from .rope import RoPE


class CausalAttention(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.heads, self.head_dim, self.dropout = c.heads, c.dim // c.heads, c.dropout
        self.qkv = nn.Linear(c.dim, 3 * c.dim, bias=False)
        self.out = nn.Linear(c.dim, c.dim, bias=False)
        self.rope = RoPE(self.head_dim, c.rope_base)

    def forward(self, x: Tensor) -> Tensor:
        b, t, d = x.shape
        q, k, v = (
            self.qkv(x).reshape(b, t, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        )
        y = F.scaled_dot_product_attention(
            self.rope(q),
            self.rope(k),
            v,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.out(y.transpose(1, 2).contiguous().reshape(b, t, d))
