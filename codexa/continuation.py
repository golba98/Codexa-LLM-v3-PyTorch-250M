"""Explicit native-v1 phase preparation. This module has no training action."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from codexa.data.production import PermutationSampler, atomic, digest, verify
from codexa.data.v2_storage import Storage


def phase_state(
    parent: dict[str, Any],
    config: Any,
    dataset_sha: str,
    tokenizer_sha: str,
    windows: int | None,
    policy: str,
    parent_sha: str,
    additional_tokens: int | None = None,
    *,
    stream_sampler: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import torch

    from codexa.architecture.model import Codexa
    from codexa.config import Config
    from codexa.safety import forward_only
    from codexa.training.optimizer import make_optimizer
    from codexa.training.scheduler import make_scheduler

    forward_only()
    if parent["version"] != 1 or policy not in ("preserve", "reset"):
        raise ValueError("Expected native v1 and an explicit optimizer policy")
    old = Config.from_dict(parent["config"])
    if old.model != config.model:
        raise ValueError("Architecture/context compatibility changed")
    if hashlib.sha256(parent["tokenizer_json"].encode()).hexdigest() != tokenizer_sha:
        raise ValueError("Tokenizer compatibility changed")
    model = Codexa(config.model)
    model.load_state_dict(parent["model"], strict=True)
    model.requires_grad_(False)
    model.eval()
    if not torch.equal(parent["model"]["embedding.weight"], parent["model"]["lm_head.weight"]):
        raise ValueError("Untied parent tensor state")
    if not all(bool(torch.isfinite(t).all()) for t in model.state_dict().values()):
        raise ValueError("Nonfinite model state")
    optimizer = make_optimizer(model, config.training)
    if policy == "preserve":
        # load_state_dict preserves moments and step values; only the explicitly new LR changes.
        optimizer.load_state_dict(parent["optimizer"])
        parent_phase = parent.get("phase", {})
        parent_cumulative = parent_phase.get("parent_updates", 0) + parent["step"]
        optimizer_origin = parent_phase.get(
            "optimizer_origin_updates",
            parent_phase.get("parent_updates", 0)
            if parent_phase.get("optimizer_policy") == "reset"
            else 0,
        )
        expected_updates = parent_cumulative - optimizer_origin
        if optimizer_origin < 0 or expected_updates < 0:
            raise ValueError("Invalid optimizer-origin accounting")
        for group in optimizer.param_groups:
            if tuple(group["betas"]) != (0.9, 0.95) or group["eps"] != 1e-8:
                raise ValueError("Preserved optimizer algorithm is incompatible")
            for parameter in group["params"]:
                state = optimizer.state.get(parameter, {})
                if expected_updates:
                    if not all(k in state for k in ("step", "exp_avg", "exp_avg_sq")):
                        raise ValueError("Trained checkpoint has incomplete optimizer state")
                    step = state["step"]
                    if (
                        not isinstance(step, torch.Tensor)
                        or step.numel() != 1
                        or float(step) != expected_updates
                    ):
                        raise ValueError(
                            "Optimizer update counter differs from checkpoint accounting"
                        )
                for name in ("exp_avg", "exp_avg_sq"):
                    if name in state and (
                        state[name].shape != parameter.shape
                        or not torch.isfinite(state[name]).all()
                    ):
                        raise ValueError("Optimizer tensor mapping invalid")
                if "exp_avg_sq" in state and bool((state["exp_avg_sq"] < 0).any()):
                    raise ValueError("Negative optimizer second moment")
    for group in optimizer.param_groups:
        group["lr"] = config.training.lr
        group["initial_lr"] = config.training.lr
    scheduler = make_scheduler(optimizer, config.training)
    prior_phase = parent.get("phase", {})
    prior_updates = prior_phase.get("parent_updates", 0) + parent["step"]
    same_data = parent["fingerprints"].get("dataset") == dataset_sha
    if stream_sampler is not None:
        if same_data:
            raise ValueError("A streaming phase must have its own source identity")
        sampler = copy.deepcopy(stream_sampler)
    else:
        if windows is None or windows < 1:
            raise ValueError("Native token arrays require a positive window count")
        sampler = (
            copy.deepcopy(parent["sampler"])
            if same_data
            else PermutationSampler(windows, config.training.seed).state_dict()
        )
        if sampler["size"] != windows:
            raise ValueError("Sampler size incompatible with prepared data")
    phase = {
        "parent_sha256": parent_sha,
        "parent_tokens": parent["tokens"],
        "parent_updates": prior_updates,
        "parent_microsteps": parent.get("microsteps"),
        "parent_fingerprints": parent["fingerprints"],
        "parent_sampler": parent["sampler"],
        "optimizer_policy": policy,
        "optimizer_origin_updates": prior_updates
        if policy == "reset"
        else prior_phase.get(
            "optimizer_origin_updates",
            prior_phase.get("parent_updates", 0)
            if prior_phase.get("optimizer_policy") == "reset"
            else 0,
        ),
        "rng_policy": "preserve_parent",
        "sampler_policy": "preserve parent data order"
        if same_data
        else "fresh seed-42 order for proven-disjoint new dataset",
        "sampler_origin": copy.deepcopy(sampler),
        "schedule_policy": "fresh warmup/cosine; completed parent schedule retained only in parent",
        "phase_updates": 0,
        "phase_tokens": 0,
        "cumulative_updates": prior_updates,
        "minimum_free_bytes": 40 * 1024**3,
        "volume_uuid": "279352d0-0011-4733-8348-6815a4308196",
        "parent_scheduler": copy.deepcopy(parent["scheduler"]),
    }
    if additional_tokens is not None:
        full_update = (
            config.training.batch_size * config.training.accumulation * config.training.sequence
        )
        if (
            additional_tokens <= 0
            or (additional_tokens + full_update - 1) // full_update != config.training.steps
        ):
            raise ValueError("Exact token budget and phase update count disagree")
        phase["additional_target_tokens"] = additional_tokens
    if stream_sampler is not None:
        phase["sampler_policy"] = (
            "fresh deterministic published stream; seeded shard order and sequential records"
        )
    output = {
        "version": 1,
        "microsteps": 0,
        "best_validation_loss": None,
        "config": config.to_dict(),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": 0,
        "tokens": parent["tokens"],
        "rng": copy.deepcopy(parent["rng"]),
        "sampler": sampler,
        "fingerprints": {"dataset": dataset_sha, "tokenizer": tokenizer_sha},
        "tokenizer_json": parent["tokenizer_json"],
        "device_type": parent["device_type"],
        "bf16": parent["bf16"],
        "phase": phase,
    }
    return output


def prepare(
    root: Path,
    parent_path: Path,
    dataset: Path,
    tokenizer: Path,
    expected_sha: str,
    policy: str = "preserve",
    output_path: Path | None = None,
    additional_tokens: int = 50003968,
) -> dict[str, Any]:
    import torch
    import yaml

    from codexa.config import Config, TrainConfig
    from codexa.training.checkpoint import save_atomic

    storage = Storage.load(root)
    storage.check(4 * 1024**3)
    if digest(parent_path) != expected_sha:
        raise ValueError("Original checkpoint integrity mismatch")
    parent_stat = parent_path.stat()
    manifest = verify(dataset)
    audit_path = dataset / "audit.json"
    if not audit_path.exists():
        raise ValueError("Continuation requires an immutable audited dataset-v2 snapshot")
    audit = json.loads(audit_path.read_text())
    if audit.get("manifest_sha256") != digest(dataset / "manifest.json") or not audit.get(
        "all_accepted_documents_screened_against_final_policy"
    ):
        raise ValueError("Snapshot audit is missing or incompatible")
    if manifest["tokenizer_hash"] != digest(tokenizer):
        raise ValueError("Dataset/tokenizer mismatch")
    parent = torch.load(parent_path, map_location="cpu", weights_only=True, mmap=True)
    config = Config.from_dict(parent["config"])
    if additional_tokens <= 0:
        raise ValueError("Positive additional-token budget required")
    config.training = TrainConfig(
        steps=(additional_tokens + 32767) // 32768,
        batch_size=1,
        accumulation=16,
        sequence=2048,
        lr=1e-4,
        min_lr=1e-5,
        warmup=100,
        weight_decay=0.1,
        clip=1.0,
        seed=42,
        validate_every=100,
        checkpoint_every=250,
        validation_batches=32,
        precision="bf16",
        gradient_checkpointing=False,
    )
    windows = (sum(x["tokens"] for x in manifest["splits"]["train"]) - 1) // 2048
    if windows < config.training.steps * 16:
        raise ValueError("New data cannot support a 50M-token phase without replay")
    new_sha = digest(dataset / "manifest.json")
    old_sha = parent["fingerprints"].get("dataset")
    excluded = manifest.get("identity", {}).get("excluded_dataset_fingerprints", [])
    if new_sha != old_sha and old_sha not in excluded:
        raise ValueError("Changed dataset lacks a verified exclusion record for prior exposure")
    if new_sha == old_sha and windows - parent["sampler"]["position"] < config.training.steps * 16:
        raise ValueError("Insufficient unseen windows for a phase without replay")
    Config.from_dict(config.to_dict())
    output = output_path or root / "continuation" / ("healthcheck-" + policy)
    if not output.resolve().is_relative_to((root / "continuation").resolve()):
        raise ValueError("Phase bundle must stay under the verified continuation root")
    output.mkdir(parents=True, exist_ok=False)
    state = phase_state(
        parent,
        config,
        digest(dataset / "manifest.json"),
        digest(tokenizer),
        windows,
        policy,
        expected_sha,
        additional_tokens,
    )
    (output / "config.yaml").write_text(yaml.safe_dump(config.to_dict(), sort_keys=True))
    save_atomic(state, output / "initial.pt")
    report = validate(output / "initial.pt", parent_path, dataset, tokenizer, expected_sha)
    assert parent_path.stat().st_mtime_ns == parent_stat.st_mtime_ns
    report.update(
        phase_updates=config.training.steps,
        additional_tokens=additional_tokens,
        cumulative_target_tokens=parent["tokens"] + additional_tokens,
        training_started=False,
    )
    atomic(output / "verification.json", report)
    return report


def validate(
    path: Path, parent: Path, dataset: Path, tokenizer: Path, parent_sha: str
) -> dict[str, Any]:
    import torch

    from codexa.architecture.model import Codexa
    from codexa.config import Config
    from codexa.safety import ATTEMPTS, forward_only

    forward_only()
    if digest(parent) != parent_sha:
        raise ValueError("Parent integrity changed")
    checkpoint_sha = digest(path)
    record = path.parent / "verification.json"
    if (
        record.exists()
        and json.loads(record.read_text()).get("checkpoint_sha256") != checkpoint_sha
    ):
        raise ValueError("Converted checkpoint integrity record mismatch")
    original = torch.load(parent, weights_only=True, map_location="cpu", mmap=True)
    state = torch.load(path, weights_only=True, map_location="cpu", mmap=True)
    cfg = Config.from_dict(state["config"])
    config_path = path.parent / "config.yaml"
    if config_path.exists():
        import yaml

        if yaml.safe_load(config_path.read_text()) != state["config"]:
            raise ValueError("Phase configuration differs from its native checkpoint")
    if cfg.model != Config.from_dict(original["config"]).model:
        raise ValueError("Architecture changed")
    if state["fingerprints"] != {
        "dataset": digest(dataset / "manifest.json"),
        "tokenizer": digest(tokenizer),
    }:
        raise ValueError("Exact-resume fingerprints invalid")
    if (
        state["step"] != 0
        or state["tokens"] != original["tokens"]
        or state["sampler"] != state["phase"]["sampler_origin"]
    ):
        raise ValueError("Phase token/update accounting invalid")
    if not all(torch.equal(v, original["model"][k]) for k, v in state["model"].items()):
        raise ValueError("Converted model weights changed")
    for name, value in original["rng"].items():
        converted = state["rng"][name]
        if isinstance(value, torch.Tensor):
            equal = torch.equal(value, converted)
        elif name == "cuda":
            equal = len(value) == len(converted) and all(
                torch.equal(a, b) for a, b in zip(value, converted)
            )
        else:
            equal = value == converted
        if not equal:
            raise ValueError("Preserved RNG state changed")
    model = Codexa(cfg.model)
    model.load_state_dict(state["model"], strict=True)
    model.requires_grad_(False)
    model.eval()
    with torch.inference_mode():
        logits = model(torch.tensor([[2, 7, 11, 3]]), last_logits_only=True)
    if not torch.isfinite(logits).all():
        raise ValueError("Dry-run forward nonfinite")
    if state["phase"]["optimizer_policy"] == "preserve":
        if set(state["optimizer"]["state"]) != set(original["optimizer"]["state"]):
            raise ValueError("Preserved optimizer parameter-state identities changed")
        for key, old in original["optimizer"]["state"].items():
            new = state["optimizer"]["state"][key]
            for name, value in old.items():
                if isinstance(value, torch.Tensor) and not torch.equal(value, new[name]):
                    raise ValueError("Preserved optimizer tensors changed")
    elif state["optimizer"]["state"]:
        raise ValueError("Reset policy retained optimizer tensors")
    assert not ATTEMPTS
    return {
        "status": "passed",
        "checkpoint_sha256": checkpoint_sha,
        "parent_sha256": parent_sha,
        "weights_identical": True,
        "optimizer_policy": state["phase"]["optimizer_policy"],
        "optimizer_updates": 0,
        "optimizer_parameter_states": len(state["optimizer"]["state"]),
        "backward_calls": 0,
        "forbidden_attempts": ATTEMPTS,
        "forward_only_cpu_logits_finite": True,
        "phase": state["phase"],
    }


def prepare_stream(
    root: Path,
    parent_path: Path,
    stream_config: Path,
    tokenizer: Path,
    config_path: Path,
    expected_sha: str,
    additional_tokens: int,
) -> dict[str, Any]:
    from codexa.config import Config
    from codexa.data.stream_policy import configured_streams
    from codexa.training.checkpoint import load, save_atomic

    if additional_tokens != 1_000_000_000:
        raise ValueError("Streaming preparation requires the explicit 1B budget")
    if root.exists():
        raise FileExistsError("Use a fresh streaming continuation directory")
    if digest(parent_path) != expected_sha:
        raise ValueError("Preserved parent checkpoint fingerprint changed")
    source_config = json.loads(stream_config.read_text())
    if source_config.get("parent_checkpoint_sha256") != expected_sha:
        raise ValueError("Streaming authorization targets another parent checkpoint")
    config = Config.load(config_path)
    if hashlib.sha256(
        json.dumps(config.to_dict(), sort_keys=True).encode()
    ).hexdigest() != source_config.get("training_config_identity"):
        raise ValueError(
            "Continuation configuration differs from its authorization and compute evidence"
        )
    train, val = configured_streams(
        str(stream_config), str(tokenizer), config.training.sequence, config.training.seed
    )
    parent = load(parent_path)
    if parent["tokens"] + additional_tokens != source_config.get("cumulative_target_tokens"):
        raise ValueError("Expanded cumulative token budget differs from the parent")
    state = phase_state(
        parent,
        config,
        train.identity,
        digest(tokenizer),
        None,
        "preserve",
        expected_sha,
        additional_tokens,
        stream_sampler=train.state_dict(),
    )
    state["fingerprints"]["validation"] = val.identity
    Storage(root, state["phase"]["volume_uuid"], state["phase"]["minimum_free_bytes"]).check(
        4 * 1024**3
    )
    root.mkdir(parents=True)
    path = root / "initial-stream.pt"
    save_atomic(state, path)
    return {
        "checkpoint": str(path),
        "sha256": digest(path),
        "additional_target_tokens": additional_tokens,
        "training_started": False,
        "optimizer_updates": 0,
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["prepare", "verify", "stream-prepare"])
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--parent-sha256", required=True)
    p.add_argument("--optimizer-policy", choices=["preserve", "reset"], default="preserve")
    p.add_argument("--config", type=Path)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--additional-tokens", type=int, default=50003968)
    a = p.parse_args(argv)
    if a.action == "stream-prepare":
        if a.optimizer_policy != "preserve":
            p.error("Streaming continuation preserves the parent optimizer state")
        if a.config is None:
            p.error("--config required for streaming continuation")
        result = prepare_stream(
            a.root, a.parent, a.data, a.tokenizer, a.config, a.parent_sha256, a.additional_tokens
        )
    elif a.action == "prepare":
        result = prepare(
            a.root,
            a.parent,
            a.data,
            a.tokenizer,
            a.parent_sha256,
            a.optimizer_policy,
            a.output,
            a.additional_tokens,
        )
    else:
        if a.checkpoint is None:
            p.error("--checkpoint required")
        result = validate(a.checkpoint, a.parent, a.data, a.tokenizer, a.parent_sha256)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
