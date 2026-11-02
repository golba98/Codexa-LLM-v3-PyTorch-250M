import math

import torch

from codexa.config import TrainConfig


def make_scheduler(
    optimizer: torch.optim.Optimizer, c: TrainConfig
) -> torch.optim.lr_scheduler.LambdaLR:
    def factor(step: int) -> float:
        if step < c.warmup:
            return (step + 1) / c.warmup
        progress = min(1.0, (step - c.warmup) / max(1, c.steps - c.warmup - 1))
        return c.min_lr / c.lr + (1 - c.min_lr / c.lr) * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
