import math
from pathlib import Path
from typing import Any

import torch

from codexa.architecture.model import Codexa
from codexa.data.packing import PackedData, fingerprint
from codexa.training.trainer import causal_loss


@torch.inference_mode()
def evaluate(model: Codexa, data: str, max_batches: int | None = None) -> dict[str, Any]:
    if max_batches is not None and max_batches <= 0:
        raise ValueError("max_batches must be positive")
    ds = PackedData(Path(data) / "val.bin", model.config.context)
    count = min(ds.size, max_batches) if max_batches else ds.size
    total = 0.0
    training = model.training
    model.eval()
    try:
        for i in range(count):
            x, y = ds.batch([i], next(model.parameters()).device)
            total += float(causal_loss(model(x), y)) * y.numel()
    finally:
        model.train(training)
    tokens = count * ds.sequence
    loss = total / tokens
    return {
        "loss": loss,
        "perplexity": math.exp(loss) if loss < 700 else None,
        "tokens": tokens,
        "sequence": ds.sequence,
        "parameters": model.parameter_count(),
        "data_sha256": fingerprint(Path(data) / "val.bin"),
    }
