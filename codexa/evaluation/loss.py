"""Forward-only held-out loss and precision controls within the shared GPU budget."""

import argparse
import fcntl
import json
import math
import time
from pathlib import Path
from typing import Any

from codexa.data.production import ShardedData, atomic, digest, verify
from codexa.data.v2_storage import Storage
from codexa.evaluation.benchmark import evaluation_directory, load_frozen, model_hash


def evaluate(
    root: Path,
    checkpoint: Path,
    datasets: list[Path],
    name: str,
    maximum_seconds: int = 600,
    output_path: Path | None = None,
) -> dict[str, Any]:
    import torch

    from codexa.safety import ATTEMPTS

    storage = Storage.load(root)
    storage.check(64 * 1024**2)
    sha = digest(checkpoint)
    output = evaluation_directory(root, sha, output_path)
    report_path = output / (name + ".json")
    if report_path.exists() or Path(name).name != name:
        raise ValueError("Use a fresh bounded evaluation name")
    with (root / "tmp/gpu-evaluation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        budget_path = output / "budget.json"
        budget = json.loads(budget_path.read_text())
        if budget.get("active_since"):
            budget["used_seconds"] += max(0, time.time() - budget.pop("active_since"))
        before_seconds = budget["used_seconds"]
        allowance = min(maximum_seconds, budget["limit_seconds"] - before_seconds)
        if allowance <= 0:
            raise RuntimeError("Shared 90-minute GPU budget exhausted")
        start = time.monotonic()
        atomic(budget_path, budget | {"active_since": time.time()})
        results = []
        frozen = None
        before_hash = None
        complete = True
        try:
            frozen, _tok, _state = load_frozen(checkpoint)
            before_hash = model_hash(frozen)
            with torch.inference_mode():
                for dataset in datasets:
                    manifest = verify(dataset)
                    data = ShardedData(dataset, "val", 2048)
                    summed = 0.0
                    count = 0
                    for i in range(data.size):
                        if time.monotonic() - start >= allowance:
                            complete = False
                            break
                        x, y = data.batch([i], torch.device("cuda"))
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            logits = frozen(x)
                        summed += float(
                            torch.nn.functional.cross_entropy(
                                logits.float().flatten(0, 1), y.flatten(), reduction="sum"
                            )
                        )
                        count += y.numel()
                        if i % 32 == 0:
                            atomic(
                                budget_path,
                                budget
                                | {
                                    "used_seconds": before_seconds + time.monotonic() - start,
                                    "active_since": time.time(),
                                },
                            )
                    results.append(
                        {
                            "dataset": str(dataset),
                            "manifest_sha256": digest(dataset / "manifest.json"),
                            "vocab_size": manifest["vocab_size"],
                            "evaluated_tokens": count,
                            "windows": count // 2048,
                            "available_windows": data.size,
                            "loss": summed / count if count else None,
                            "perplexity": math.exp(summed / count) if count else None,
                            "precision": "bf16 forward, fp32 logits CE",
                            "complete": count // 2048 == data.size,
                            "tail_policy": "full 2048-target windows; fewer than 2048 trailing targets omitted",
                        }
                    )
                    if not complete:
                        break
        finally:
            budget["used_seconds"] = before_seconds + time.monotonic() - start
            atomic(budget_path, budget)
            unchanged = frozen is not None and before_hash == model_hash(frozen)
            result = {
                "complete": complete and len(results) == len(datasets),
                "datasets": results,
                "checkpoint_sha256": sha,
                "weights_unchanged": unchanged,
                "forbidden_attempts": ATTEMPTS,
                "backward_calls": 0,
                "optimizer_updates": 0,
                "elapsed_seconds": time.monotonic() - start,
                "shared_budget": budget,
                "gpu_peak_allocated": torch.cuda.max_memory_allocated(),
                "gpu_peak_reserved": torch.cuda.max_memory_reserved(),
            }
            atomic(report_path, result)
            if not unchanged or ATTEMPTS:
                raise RuntimeError("Forward-only integrity verification failed")
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, action="append", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--seconds", type=int, default=600)
    a = parser.parse_args()
    print(json.dumps(evaluate(a.root, a.checkpoint, a.data, a.name, a.seconds), indent=2))


if __name__ == "__main__":
    main()
