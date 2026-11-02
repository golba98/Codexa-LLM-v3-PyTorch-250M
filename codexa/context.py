"""Explicit stage preparation and isolated, bounded context benchmarks."""

import argparse
import json
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any


def main(argv=None):
    result: dict[str, Any]
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    bench = sub.add_parser("benchmark")
    bench.add_argument("--lengths", default="2048,4096,8192,16384,32768")
    bench.add_argument("--mode", choices=["inference", "training", "both"], default="inference")
    bench.add_argument("--output", default="outputs/benchmarks/context.json")
    bench.add_argument("--config", default="configs/250m.yaml")
    worker = sub.add_parser("_case")
    worker.add_argument("--length", type=int)
    worker.add_argument("--mode")
    worker.add_argument("--config")
    worker.add_argument("--checkpointing", action="store_true")
    transition = sub.add_parser("transition")
    transition.add_argument("--checkpoint", required=True)
    transition.add_argument("--config", required=True)
    transition.add_argument("--output", required=True)
    transition.add_argument("--reset-data-order", action="store_true", required=True)
    a = p.parse_args(argv)
    if a.action == "transition":
        import torch

        from codexa.config import Config
        from codexa.training.checkpoint import load, save_atomic

        state = load(a.checkpoint)
        old = Config.from_dict(state["config"])
        new = Config.load(a.config)
        if state["step"] >= new.training.steps:
            raise ValueError(
                "Schedule exhausted; a separately reviewed schedule initialization is required"
            )
        permitted = {"context", "rope_scaling", "rope_factor", "attention_backend"}
        if any(
            v != new.model.__dict__[k] for k, v in old.model.__dict__.items() if k not in permitted
        ):
            raise ValueError("Architecture changed")
        allowed = {"sequence", "gradient_checkpointing", "batch_size", "accumulation"}
        if any(
            v != new.training.__dict__[k]
            for k, v in old.training.__dict__.items()
            if k not in allowed
        ):
            raise ValueError(
                "Schedule/optimizer changes require separately reviewed initialization"
            )
        target = Path(a.output)
        if target.exists():
            raise FileExistsError(target)
        # Sampler size depends on the prepared array: require explicit dataset in a later train call.
        state["config"] = new.to_dict()
        state["context_transition"] = {
            "parent_step": state["step"],
            "parent_checkpoint": str(a.checkpoint),
            "reset_data_order": True,
        }
        state.pop("sampler", None)
        save_atomic(state, target)
        print(target)
        return
    if a.action == "benchmark":
        results = []
        for n in map(int, a.lengths.split(",")):
            for mode in ["inference", "training"] if a.mode == "both" else [a.mode]:
                cmd = [
                    sys.executable,
                    "-m",
                    "codexa.context",
                    "_case",
                    "--length",
                    str(n),
                    "--mode",
                    mode,
                    "--config",
                    a.config,
                ]
                try:
                    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)
                    result = (
                        json.loads(r.stdout.strip().splitlines()[-1])
                        if r.returncode == 0
                        else {"status": "error", "detail": r.stderr[-1000:]}
                    )
                except subprocess.TimeoutExpired:
                    result = {"status": "timeout", "limit_seconds": 60}
                result.update(length=n, mode=mode)
                results.append(result)
                if result.get("status") == "oom" and mode == "training":
                    try:
                        r = subprocess.run(
                            cmd + ["--checkpointing"],
                            capture_output=True,
                            text=True,
                            timeout=60,
                            check=False,
                        )
                        retry = (
                            json.loads(r.stdout.strip().splitlines()[-1])
                            if r.returncode == 0
                            else {"status": "error", "detail": r.stderr[-500:]}
                        )
                        retry.update(length=n, mode=mode, checkpointing=True)
                    except subprocess.TimeoutExpired:
                        retry = {
                            "status": "timeout",
                            "limit_seconds": 60,
                            "length": n,
                            "mode": mode,
                            "checkpointing": True,
                        }
                    results.append(retry)
        path = Path(a.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(results, indent=2))
        print(path)
        return
    import torch

    from codexa.architecture.model import Codexa
    from codexa.config import Config
    from codexa.training.optimizer import make_optimizer
    from codexa.training.trainer import causal_loss

    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        print(json.dumps({"status": "unavailable"}))
        return
    torch.manual_seed(42)
    start = time.perf_counter()
    cfg = Config.load(a.config)
    cfg.model = replace(cfg.model, context=a.length)
    # Reserve desktop headroom. OOM results describe this explicit allocator budget.
    free, total = torch.cuda.mem_get_info()
    fraction = min(0.9, max(0.1, (free - 1073741824) / total))
    torch.cuda.set_per_process_memory_fraction(fraction)
    result = {
        "memory_budget_fraction": fraction,
        "seed": 42,
        "precision": "bf16",
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "checkpointing": a.checkpointing,
        "micro_batch": 1,
        "accumulation": 1,
    }
    try:
        model = Codexa(cfg.model).cuda()
        model.gradient_checkpointing = a.checkpointing
        torch.cuda.synchronize()
        result["initialization_seconds"] = time.perf_counter() - start
        torch.cuda.reset_peak_memory_stats()
        x = torch.randint(cfg.model.vocab_size, (1, a.length), device="cuda")
        if a.mode == "training":
            model.train()
            opt = make_optimizer(model, cfg.training)
            result["completed_updates"] = 0
            result["warmup_updates"] = 1
            result["measured_updates"] = 1
            for update in range(2):
                opt.zero_grad(set_to_none=True)
                start = time.perf_counter()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = causal_loss(model(x), torch.roll(x, -1, 1))
                loss.backward()
                opt.step()
                torch.cuda.synchronize()
                result["completed_updates"] += 1
                result["cold_step_seconds" if update == 0 else "training_step_seconds"] = (
                    time.perf_counter() - start
                )
        else:
            model.eval()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                start = time.perf_counter()
                _, cache = model.forward_cached(x, capacity=a.length)
                torch.cuda.synchronize()
                result["prefill_seconds"] = time.perf_counter() - start
                result["kv_cache_bytes"] = sum(
                    k.numel() * k.element_size() + v.numel() * v.element_size()
                    for k, v in cache.layers
                )
                del cache
                _, cache = model.forward_cached(x[:, :-8], capacity=a.length)
                torch.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(8):
                    model.forward_cached(x[:, -1:], cache)
                torch.cuda.synchronize()
                result["decode_seconds_per_token"] = (time.perf_counter() - start) / 8
        if a.mode == "training":
            result["training_tokens_per_second"] = a.length / result["training_step_seconds"]
        else:
            result["prefill_tokens_per_second"] = a.length / result["prefill_seconds"]
            result["decode_tokens_per_second"] = 1 / result["decode_seconds_per_token"]
        result.update(
            status="ok",
            parameters=model.parameter_count(),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        )
    except torch.OutOfMemoryError:
        result.update(
            status="oom",
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
