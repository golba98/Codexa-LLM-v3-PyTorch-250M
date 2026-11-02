import torch

from codexa.architecture.model import Codexa
from codexa.config import TrainConfig


def make_optimizer(model: Codexa, c: TrainConfig) -> torch.optim.AdamW:
    params = list(model.parameters())
    return torch.optim.AdamW(
        [
            {"params": [p for p in params if p.ndim >= 2], "weight_decay": c.weight_decay},
            {"params": [p for p in params if p.ndim < 2], "weight_decay": 0.0},
        ],
        lr=c.lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        foreach=False,
    )
