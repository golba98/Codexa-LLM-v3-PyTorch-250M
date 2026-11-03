from torch import Tensor, nn
from torch.nn import functional as F

from codexa.config import ModelConfig

from .attention import CausalAttention
from .normalization import RMSNorm


class SwiGLU(nn.Module):
    def __init__(self, dim: int, intermediate: int):
        super().__init__()
        self.gate = nn.Linear(dim, intermediate, bias=False)
        self.up = nn.Linear(dim, intermediate, bias=False)
        self.down = nn.Linear(intermediate, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.attn_norm, self.ff_norm = RMSNorm(c.dim, c.norm_eps), RMSNorm(c.dim, c.norm_eps)
        self.attention = CausalAttention(c)
        self.ff = SwiGLU(c.dim, c.intermediate)

    def forward(
        self, x: Tensor, cache: tuple[Tensor, Tensor] | None = None, offset: int = 0
    ) -> Tensor:
        x = x + self.attention(self.attn_norm(x), cache, offset)
        return x + self.ff(self.ff_norm(x))
