"""Frozen, resumable forward-only baseline generation and transparent scoring."""

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from codexa.data.production import atomic, digest
from codexa.data.v2_storage import Storage


def repetition(ids: list[int]) -> dict[str, Any]:
    grams = [tuple(ids[i : i + 4]) for i in range(max(0, len(ids) - 3))]
    longest = 0
    for period in range(1, min(32, len(ids) // 3) + 1):
        run = best = 0
        for a, b in zip(ids, ids[period:]):
            run = run + 1 if a == b else 0
            best = max(best, run)
        if best >= period * 2:
            longest = max(longest, best + period)
    return {
        "repeated_token_4gram_fraction": 1 - len(set(grams)) / len(grams) if grams else 0,
        "longest_tandem_loop": longest,
        "severe_loop": longest >= 24,
        "eos": bool(ids and ids[-1] == 3),
        "tokens": len(ids),
    }


def model_hash(model: Any) -> str:
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def load_frozen(checkpoint: Path, device: str = "cuda") -> tuple[Any, Any, dict[str, Any]]:
    import torch
    from tokenizers import Tokenizer

    from codexa.architecture.model import Codexa
    from codexa.config import Config
    from codexa.safety import forward_only
    from codexa.tokenizer.bpe import CodexaTokenizer

    forward_only()
    torch.set_num_threads(4)
    state = torch.load(checkpoint, weights_only=True, map_location="cpu", mmap=True)
    if state["version"] != 1:
        raise ValueError("Expected native v1 checkpoint")
    model = Codexa(Config.from_dict(state["config"]).model)
    model.load_state_dict(state["model"], strict=True)
    model.requires_grad_(False)
    model.eval()
    return model.to(device), CodexaTokenizer(Tokenizer.from_str(state["tokenizer_json"])), state


def evaluate(
    root: Path, checkpoint: Path, seconds: int = 5400, output_path: Path | None = None
) -> dict[str, Any]:
    import fcntl

    with (root / "tmp/gpu-evaluation.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another evaluation owns the GPU lease") from None
        return _evaluate(root, checkpoint, seconds, output_path)


def evaluation_directory(root: Path, checkpoint_sha: str, output_path: Path | None = None) -> Path:
    output = output_path or root / "benchmarks" / ("baseline-" + checkpoint_sha[:16])
    if not output.resolve().is_relative_to((root / "benchmarks").resolve()):
        raise ValueError("Evaluation artifacts must stay under the preparation benchmarks root")
    output.mkdir(parents=True, exist_ok=True)
    return output


def _evaluate(
    root: Path,
    checkpoint: Path,
    seconds: int = 5400,
    output_path: Path | None = None,
    kinds: set[str] | None = None,
) -> dict[str, Any]:
    import torch

    from codexa.inference.generate import generate
    from codexa.safety import ATTEMPTS

    storage = Storage.load(root)
    storage.check(64 * 1024**2)
    fixtures = root / "benchmarks/v3"
    manifest = json.loads((fixtures / "manifest.json").read_text())
    for name, sha in manifest["files"].items():
        if digest(fixtures / name) != sha:
            raise ValueError("Evaluation fixture fingerprint changed")
    checkpoint_sha = digest(checkpoint)
    output = evaluation_directory(root, checkpoint_sha, output_path)
    if kinds is not None:
        if not kinds or not kinds <= {
            "paragraph",
            "sentence",
            "fact",
            "context",
            "grammar",
            "vocabulary",
            "math",
        }:
            raise ValueError("Unknown or empty benchmark task selection")
        selection = output / "selection.json"
        requested = {
            "kinds": sorted(kinds),
            "fixture_manifest_sha256": digest(fixtures / "manifest.json"),
        }
        if selection.exists():
            if json.loads(selection.read_text()) != requested:
                raise ValueError("Benchmark task selection changed on resume")
        elif (output / "outputs.jsonl").exists():
            raise ValueError("Selected evaluation cannot resume an unfiltered output")
        else:
            atomic(selection, requested)
    elif (output / "selection.json").exists():
        raise ValueError("Selected benchmark output requires its original task selection")
    budget_path = output / "budget.json"
    budget = (
        json.loads(budget_path.read_text())
        if budget_path.exists()
        else {"limit_seconds": seconds, "used_seconds": 0.0}
    )
    if budget.get("active_since"):
        # Conservatively charge unrecorded time after an interrupted process.
        budget["used_seconds"] += max(0, time.time() - budget.pop("active_since"))
    limit = min(seconds, budget["limit_seconds"]) - budget["used_seconds"]
    if limit <= 0:
        raise RuntimeError("Baseline GPU budget exhausted")
    started = time.monotonic()
    used_before = budget["used_seconds"]
    atomic(budget_path, budget | {"active_since": time.time()})
    model, tok, _state = load_frozen(checkpoint)
    before = model_hash(model)
    items = [
        json.loads(line)
        for split in ("dev", "test")
        for line in (fixtures / f"{split}.jsonl").read_text().splitlines()
    ]
    if kinds is not None:
        items = [item for item in items if item["kind"] in kinds]
    records_path = output / "outputs.jsonl"
    seen: set[str] = set()
    if records_path.exists():
        for line in records_path.read_text().splitlines():
            seen.add(json.loads(line)["key"])
    complete = True
    failed = False
    done = 0
    try:
        with records_path.open("a") as f:
            for item in items:
                kind = item["kind"]
                jobs: list[tuple[str, bool, int, int]] = []
                if "candidates" in item:
                    jobs.append(("candidate", True, 42, 0))
                if kind == "paragraph":
                    jobs.extend(
                        [("greedy", True, 42, 256)]
                        + [("sample", False, s, 256) for s in (42, 123, 2026)]
                    )
                elif kind in ("sentence", "fact", "context", "math"):
                    jobs.extend(
                        [
                            ("greedy", True, 42, 48 if kind == "math" else 64),
                            ("sample", False, 42, 48 if kind == "math" else 64),
                        ]
                    )
                for label, greedy, seed, cap in jobs:
                    key = item["id"] + f":{label}:{seed}"
                    if key in seen:
                        continue
                    if time.monotonic() - started >= limit:
                        complete = False
                        break
                    row: dict[str, Any] = {
                        "key": key,
                        "item_id": item["id"],
                        "split": item["split"],
                        "kind": kind,
                        "family": item["family"],
                        "setting": label,
                        "seed": seed,
                        "prompt": item["prompt"],
                    }
                    prefix = [2, *tok.encode(item["prompt"])]
                    if label == "candidate":
                        values = []
                        for candidate in item["candidates"]:
                            continuation = candidate if not item["prompt"] else " " + candidate
                            ids = prefix + tok.encode(continuation)
                            # Shared terminal punctuation is excluded from candidate scoring.
                            if [2, *tok.encode(item["prompt"] + continuation)][
                                : len(prefix)
                            ] != prefix:
                                raise ValueError("Candidate prefix changed")
                            logits = model(torch.tensor([ids[:-1]], device="cuda"))[0].float()
                            target = torch.tensor(ids[1:], device="cuda")
                            nll = torch.nn.functional.cross_entropy(
                                logits, target, reduction="none"
                            )[len(prefix) - 1 :]
                            if continuation.endswith(".") and tok.decode([ids[-1]]) == ".":
                                nll = nll[:-1]
                            values.append(
                                {
                                    "candidate": candidate,
                                    "tokens": len(nll),
                                    "sum_nll": float(nll.sum()),
                                    "mean_nll": float(nll.mean()),
                                }
                            )
                        row.update(
                            precision="fp32",
                            scores=values,
                            correct_by_sum=min(
                                range(len(values)), key=lambda i: values[i]["sum_nll"]
                            )
                            == 0,
                            correct_by_mean=min(
                                range(len(values)), key=lambda i: values[i]["mean_nll"]
                            )
                            == 0,
                            limitation="Restricted candidates, not successful free generation",
                        )
                    else:
                        ids = generate(
                            model,
                            prefix,
                            max_tokens=cap,
                            greedy=greedy,
                            seed=seed,
                            precision="bf16",
                            temperature=0.7,
                            top_k=40,
                            top_p=0.9,
                        )[len(prefix) :]
                        row.update(
                            ids=ids,
                            text=tok.decode(ids),
                            precision="bf16",
                            max_tokens=cap,
                            temperature=0.7,
                            top_k=40,
                            top_p=0.9,
                            **repetition(ids),
                            prefixes={
                                str(n): repetition(ids[:n]) for n in (64, 128, 256) if n <= cap
                            },
                            rubric_status="unreviewed",
                            grammar=None,
                            coherence=None,
                            contradiction=None,
                        )
                    f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    f.flush()
                    seen.add(key)
                    done += 1
                    if done % 20 == 0:
                        atomic(
                            budget_path,
                            budget
                            | {
                                "used_seconds": used_before + time.monotonic() - started,
                                "active_since": time.time(),
                            },
                        )
                        print(
                            json.dumps(
                                {
                                    "split": item["split"],
                                    "records_this_session": done,
                                    "total_records": len(seen),
                                    "elapsed_seconds": time.monotonic() - started,
                                }
                            ),
                            flush=True,
                        )
                if not complete:
                    break
    except BaseException:
        complete = False
        failed = True
        raise
    finally:
        budget["used_seconds"] += time.monotonic() - started
        atomic(budget_path, budget)
        unchanged = before == model_hash(model)
        assert unchanged and not ATTEMPTS
        result = {
            "complete": complete,
            "requested_kinds": sorted(kinds) if kinds is not None else None,
            "failed": failed,
            "checkpoint_sha256": checkpoint_sha,
            "fixture_manifest_sha256": digest(fixtures / "manifest.json"),
            "records": len(seen),
            "new_records": done,
            "weights_unchanged": unchanged,
            "backward_calls": 0,
            "optimizer_updates": 0,
            "forbidden_attempts": ATTEMPTS,
            "gpu_peak_allocated": torch.cuda.max_memory_allocated(),
            "gpu_peak_reserved": torch.cuda.max_memory_reserved(),
            "budget": budget,
            "human_readiness": "unreviewed, not established",
        }
        atomic(output / "verification.json", result)
    summarize(output)
    return result


def summarize(output: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in (output / "outputs.jsonl").read_text().splitlines()]
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = f"{row['split']}:{row['kind']}:{row['setting']}:{row['seed']}"
        groups.setdefault(key, []).append(row)
    result = {}
    for key, group in groups.items():
        if "scores" in group[0]:
            result[key] = {
                "n": len(group),
                "correct_by_sum": sum(r["correct_by_sum"] for r in group),
                "correct_by_mean": sum(r["correct_by_mean"] for r in group),
            }
        else:
            result[key] = {
                "n": len(group),
                "mean_repeated_4gram": sum(r["repeated_token_4gram_fraction"] for r in group)
                / len(group),
                "severe_loops": sum(r["severe_loop"] for r in group),
                "eos": sum(r["eos"] for r in group),
                "rubric_reviewed": sum(r["rubric_status"] != "unreviewed" for r in group),
            }
    atomic(output / "summary.json", result)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "evaluate", "summarize"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--seconds", type=int, default=5400)
    args = parser.parse_args(argv)
    if args.action == "freeze":
        from codexa.evaluation.fixtures import freeze

        print(json.dumps(freeze(args.root), indent=2))
    elif args.action == "evaluate":
        if args.checkpoint is None:
            parser.error("--checkpoint required")
        print(json.dumps(evaluate(args.root, args.checkpoint, args.seconds), indent=2))
    else:
        print(json.dumps(summarize(args.root), indent=2))


if __name__ == "__main__":
    main()
