import time
from typing import Any

import torch

from codexa.architecture.model import Codexa
from codexa.config import Config
from codexa.training.optimizer import make_optimizer
from codexa.training.trainer import causal_loss, choose_device, seed_all


def benchmark(
    config: Config,
    device: str,
    sequence: int,
    steps: int = 5,
    warmup: int = 2,
    accumulation: int = 1,
) -> dict[str, Any]:
    if not 0 < sequence <= config.model.context or min(steps, accumulation) <= 0 or warmup < 0:
        raise ValueError("Invalid benchmark dimensions")
    dev = choose_device(device)
    seed_all(42)
    model = Codexa(config.model).to(dev)
    opt = make_optimizer(model, config.training)
    bf16 = dev.type == "cuda" and torch.cuda.is_bf16_supported()
    x = torch.randint(config.model.vocab_size, (1, sequence + 1), device=dev)

    def step() -> None:
        opt.zero_grad(set_to_none=True)
        for _ in range(accumulation):
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=bf16):
                loss = causal_loss(model(x[:, :-1]), x[:, 1:]) / accumulation
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        opt.step()

    for _ in range(warmup):
        step()
    if dev.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(steps):
        step()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    n = model.parameter_count()
    tps = sequence * accumulation * steps / elapsed
    return {
        "measured": True,
        "device": str(dev),
        "gpu": torch.cuda.get_device_name() if dev.type == "cuda" else None,
        "torch": torch.__version__,
        "precision": "bf16_autocast" if bf16 else "fp32",
        "parameters": n,
        "weight_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
        "gradient_bytes_estimate": n * 4,
        "adam_moments_bytes_estimate": n * 8,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated()
        if dev.type == "cuda"
        else None,
        "sequence": sequence,
        "micro_batch": 1,
        "accumulation": accumulation,
        "effective_global_batch_sequences": accumulation,
        "tokens_per_step": sequence * accumulation,
        "warmup_steps": warmup,
        "measured_steps": steps,
        "step_seconds": elapsed / steps,
        "tokens_per_second": tps,
        "five_billion_tokens_seconds_extrapolated": 5e9 / tps,
        "five_billion_tokens_flops_estimate": 6 * n * 5_000_000_000,
    }
