import math
from pathlib import Path
from typing import Any

import torch

from codexa.architecture.model import Codexa
from codexa.data.packing import PackedData, fingerprint
from codexa.training.trainer import causal_loss


@torch.inference_mode()
def evaluate(
    model: Codexa, data: str, max_batches: int | None = None, split: str = "val"
) -> dict[str, Any]:
    if max_batches is not None and max_batches <= 0:
        raise ValueError("max_batches must be positive")
    import json

    if split not in ("val", "test"):
        raise ValueError("Evaluation split must be val or test")
    manifest = json.loads((Path(data) / "manifest.json").read_text())
    if manifest.get("version") == 2:
        from codexa.data.production import ShardedData

        ds: Any = ShardedData(data, split, model.config.context)
        data_hash = fingerprint(Path(data) / "manifest.json")
    else:
        ds = PackedData(Path(data) / f"{split}.bin", model.config.context)
        data_hash = fingerprint(Path(data) / f"{split}.bin")
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
        "data_sha256": data_hash,
    }
