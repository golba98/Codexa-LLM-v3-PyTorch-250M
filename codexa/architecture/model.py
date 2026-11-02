import math

import torch
from torch import Tensor, nn

from codexa.config import ModelConfig

from .normalization import RMSNorm
from .transformer import Block


class Codexa(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
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

    def forward(self, input_ids: Tensor) -> Tensor:
        if input_ids.ndim != 2 or not 0 < input_ids.shape[1] <= self.config.context:
            raise ValueError("Expected nonempty [batch, sequence] within context")
        x = self.embedding(input_ids)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.norm(x))

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
