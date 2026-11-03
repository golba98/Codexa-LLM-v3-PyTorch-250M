import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from codexa.config import ModelConfig

from .normalization import RMSNorm
from .transformer import Block


@dataclass
class KVCache:
    layers: list[tuple[Tensor, Tensor]]
    capacity: int
    length: int = 0
    valid: bool = True
    owner: int = 0


class Codexa(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = False
        self.embedding = nn.Embedding(config.vocab_size, config.dim)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.layers))
        self.norm = RMSNorm(config.dim, config.norm_eps)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
        # Initialize the shared matrix once.
        nn.init.normal_(self.embedding.weight, std=0.02)
        for block in self.blocks:
            assert isinstance(block, Block)
            for module in block.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, std=0.02)
            nn.init.normal_(block.attention.out.weight, std=0.02 / math.sqrt(2 * config.layers))
            nn.init.normal_(block.ff.down.weight, std=0.02 / math.sqrt(2 * config.layers))

    def forward(self, input_ids: Tensor, last_logits_only: bool = False) -> Tensor:
        if input_ids.ndim != 2 or not 0 < input_ids.shape[1] <= self.config.context:
            raise ValueError("Expected nonempty [batch, sequence] within context")
        x = self.embedding(input_ids)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                from torch.utils.checkpoint import checkpoint

                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return self.lm_head(self.norm(x[:, -1:] if last_logits_only else x))

    @torch.inference_mode()
    def forward_cached(
        self, input_ids: Tensor, cache: KVCache | None = None, capacity: int | None = None
    ) -> tuple[Tensor, KVCache]:
        if self.training:
            raise ValueError("KV caching requires eval mode")
        if input_ids.ndim != 2 or input_ids.shape[1] < 1:
            raise ValueError("Expected nonempty [batch, sequence]")
        b, t = input_ids.shape
        if cache is None:
            cap = capacity if capacity is not None else self.config.context
            if not t <= cap <= self.config.context:
                raise ValueError("Cache capacity exceeds context or is smaller than prompt")
            dtype = (
                torch.get_autocast_dtype(input_ids.device.type)
                if torch.is_autocast_enabled(input_ids.device.type)
                else self.embedding.weight.dtype
            )
            shape = (b, self.config.heads, cap, self.config.dim // self.config.heads)
            cache = KVCache(
                [
                    (
                        torch.empty(shape, device=input_ids.device, dtype=dtype),
                        torch.empty(shape, device=input_ids.device, dtype=dtype),
                    )
                    for _ in self.blocks
                ],
                cap,
                owner=id(self),
            )
        if (
            cache.owner != id(self)
            or cache.length + t > self.config.context
            or not cache.valid
            or cache.length + t > cache.capacity
            or len(cache.layers) != len(self.blocks)
        ):
            raise ValueError("Invalid or exhausted KV cache")
        if any(k.shape[0] != b or k.device != input_ids.device for k, v in cache.layers):
            raise ValueError("Cache batch/device mismatch")
        cache.valid = False
        x = self.embedding(input_ids)
        for block, past in zip(self.blocks, cache.layers):
            x = block(x, past, cache.length)
        logits = self.lm_head(self.norm(x[:, -1:]))
        cache.length += t
        cache.valid = True
        return logits, cache

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @staticmethod
    def expected_parameters(c: ModelConfig) -> int:
        return (
            c.vocab_size * c.dim
            + c.layers * (4 * c.dim * c.dim + 3 * c.dim * c.intermediate + 2 * c.dim)
            + c.dim
        )

    @classmethod
    def count_without_allocation(cls, c: ModelConfig) -> int:
        with torch.device("meta"):
            return cls(c).parameter_count()
