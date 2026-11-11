import json
import math
import random
import signal
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from codexa.architecture.model import Codexa
from codexa.config import Config
from codexa.data.packing import PackedData, Sampler, fingerprint
from codexa.tokenizer.bpe import CodexaTokenizer

from .callbacks.telemetry import Callback, EventBus
from .checkpoint import check_volume, load, restore_rng, rng_state, save_atomic
from .optimizer import make_optimizer
from .rolling import RollingCheckpoints
from .scheduler import make_scheduler


def causal_loss(logits: Tensor, targets: Tensor) -> Tensor:
    # Targets are shifted exactly once by PackedData.batch.
    return F.cross_entropy(logits.float().flatten(0, 1), targets.flatten())


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(name: str) -> torch.device:
    return (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if name == "auto"
        else torch.device(name)
    )


class Trainer:
    def __init__(
        self,
        config: Config,
        data: str,
        tokenizer: str,
        output: str,
        device: str = "auto",
        callbacks: tuple[Callback, ...] = (),
        checkpoint_volume_uuid: str | None = None,
        stream_data: tuple[Any, Any] | None = None,
    ):
        self.config, self.output = config, Path(output)
        self.checkpoint_volume_uuid = checkpoint_volume_uuid
        check_volume(self.output, checkpoint_volume_uuid)
        self.output.mkdir(parents=True, exist_ok=True)
        self.device = choose_device(device)
        c = config.training
        seed_all(c.seed)
        self.tokenizer_json = Path(tokenizer).read_text()
        tok = CodexaTokenizer.load(tokenizer)
        if tok.backend.get_vocab_size() != config.model.vocab_size:
            raise ValueError("Tokenizer vocabulary differs from model")
        manifest = (
            {"dataset_name": "unapproved-stream"}
            if stream_data is not None
            else json.loads((Path(data) / "manifest.json").read_text())
        )
        self.train_data: Any
        self.val_data: Any
        production = manifest.get("version") == 2
        self.streaming = stream_data is not None
        if stream_data is not None:
            self.train_data, self.val_data = stream_data
            if self.train_data.split != "train" or self.val_data.split != "val":
                raise ValueError("Streaming train and validation splits required")
            import hashlib

            tokenizer_identity = hashlib.sha256(tok.backend.to_str().encode()).hexdigest()
            config_identity = hashlib.sha256(
                json.dumps(config.to_dict(), sort_keys=True).encode()
            ).hexdigest()
            if any(
                stream.tokenizer_identity != tokenizer_identity
                or stream.sequence != c.sequence
                or (
                    stream.required_config_identity is not None
                    and stream.required_config_identity != config_identity
                )
                for stream in stream_data
            ):
                raise ValueError("Streaming tokenizer or sequence differs from trainer")
            self.fingerprints = {
                "dataset": self.train_data.identity,
                "validation": self.val_data.identity,
                "tokenizer": fingerprint(tokenizer),
            }
            self.validation_stream_start = self.val_data.state_dict()
        elif production:
            if manifest["vocab_size"] != config.model.vocab_size:
                raise ValueError("Dataset vocabulary mismatch")
            from codexa.data.production import ShardedData, digest

            self.train_data = ShardedData(data, "train", c.sequence)
            self.val_data = ShardedData(data, "val", c.sequence)
            self.fingerprints = {
                "dataset": digest(Path(data) / "manifest.json"),
                "tokenizer": fingerprint(tokenizer),
            }
            if manifest["tokenizer_hash"] != self.fingerprints["tokenizer"]:
                raise ValueError("Tokenizer fingerprint mismatch")
        else:
            self.train_data = PackedData(Path(data) / "train.bin", c.sequence)
            self.val_data = PackedData(Path(data) / "val.bin", c.sequence)
            manifest = json.loads((Path(data) / "manifest.json").read_text())
            self.fingerprints = {
                name: fingerprint(Path(data) / f"{name}.bin") for name in ("train", "val")
            }
            self.fingerprints["tokenizer"] = fingerprint(tokenizer)
            if manifest["tokenizer_hash"] != self.fingerprints["tokenizer"] or any(
                manifest["splits"][name]["sha256"] != self.fingerprints[name]
                for name in ("train", "val")
            ):
                raise ValueError("Prepared data fingerprint mismatch")
            for split in (self.train_data, self.val_data):
                if int(split.tokens.max()) >= config.model.vocab_size:
                    raise ValueError("Token ID outside vocabulary")
        self.model = Codexa(config.model).to(self.device)
        self.optimizer = make_optimizer(self.model, c)
        self.scheduler = make_scheduler(self.optimizer, c)
        from codexa.data.production import PermutationSampler

        self.sampler = (
            self.train_data
            if self.streaming
            else (PermutationSampler if production else Sampler)(self.train_data.size, c.seed)
        )
        supported = self.device.type == "cuda" and torch.cuda.is_bf16_supported()
        self.bf16 = supported and c.precision != "fp32"
        self.step, self.tokens, self.stop = 0, 0, False
        self.resumed = False
        self.callbacks = callbacks
        self.microsteps = 0
        self.events: EventBus | None = None
        self.dataset_name = manifest.get("dataset_name", Path(data).name)
        self.model.gradient_checkpointing = c.gradient_checkpointing
        self.best_validation_loss: float | None = None
        self.phase: dict[str, Any] | None = None
        self.recovery = (
            RollingCheckpoints(self.output, c.checkpoint_keep, checkpoint_volume_uuid)
            if c.checkpoint_every_tokens is not None
            else None
        )

    def amp(self) -> torch.autocast:
        return torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.bf16)

    def emit(self, kind: str, **data: Any) -> None:
        if self.events:
            if self.device.type != "cuda" and "peak_cuda_bytes" in data:
                data["peak_cuda_bytes"] = None
            self.events.emit(
                kind,
                self.step,
                self.tokens,
                **{k: v for k, v in data.items() if k not in ("step", "tokens")},
            )

    def checkpoint(self, name: str = "latest.pt") -> None:
        check_volume(self.output, self.checkpoint_volume_uuid)
        if self.phase:
            from codexa.data.v2_storage import Storage

            Storage(self.output, self.phase["volume_uuid"], self.phase["minimum_free_bytes"]).check(
                3 * 1024**3
            )
        started = time.perf_counter()
        self.emit("checkpoint_started", identifier=name)
        state = {
            "version": 1,
            "microsteps": self.microsteps,
            "best_validation_loss": self.best_validation_loss,
            "config": self.config.to_dict(),
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "step": self.step,
            "tokens": self.tokens,
            "rng": rng_state(),
            "sampler": self.sampler.state_dict(),
            "fingerprints": self.fingerprints,
            "tokenizer_json": self.tokenizer_json,
            "device_type": self.device.type,
            "bf16": self.bf16,
        } | (
            {
                "phase": self.phase
                | {
                    "phase_updates": self.step,
                    "phase_tokens": self.tokens - self.phase["parent_tokens"],
                    "cumulative_updates": self.phase["parent_updates"] + self.step,
                }
            }
            if self.phase
            else {}
        )
        extra = {}
        if self.recovery is not None:
            entry = self.recovery.save(state, final=self.step == self.config.training.steps)
            name = entry["file"]
            extra = self.recovery.status()
        else:
            save_atomic(state, self.output / name)
        self.emit(
            "checkpoint_saved",
            **(
                extra
                | {
                    "identifier": name,
                    "size_bytes": (self.output / name).stat().st_size,
                    "duration_seconds": time.perf_counter() - started,
                    "verification_status": "readback_verified"
                    if self.recovery
                    else "atomic_save_only",
                }
            ),
        )

    def resume(self, path: str) -> None:
        source = Path(path)
        if source.is_dir():
            source = RollingCheckpoints(source).latest()
        state = load(source)
        if (
            state["version"] != 1
            or Config.from_dict(state["config"]).to_dict() != self.config.to_dict()
            or state["fingerprints"] != self.fingerprints
        ):
            raise ValueError("Resume configuration or data mismatch")
        if state["device_type"] != self.device.type or state["bf16"] != self.bf16:
            raise ValueError("Exact training resume requires the same device type and precision")
        if state.get("phase"):
            from codexa.data.v2_storage import Storage

            phase = state["phase"]
            Storage(self.output, phase["volume_uuid"], phase["minimum_free_bytes"]).check(
                3 * 1024**3
            )
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        if "context_transition" in state and "sampler" not in state:
            self.sampler = type(self.sampler)(self.train_data.size, self.config.training.seed)
        else:
            self.sampler.load_state_dict(state["sampler"])
        self.step, self.tokens = state["step"], state["tokens"]
        self.microsteps = state.get(
            "microsteps", self.step * Config.from_dict(state["config"]).training.accumulation
        )
        self.transition = state.get("context_transition")
        self.phase = state.get("phase")
        if self.phase and "additional_target_tokens" in self.phase:
            budget = self.phase["additional_target_tokens"]
            c = self.config.training
            update = c.batch_size * c.accumulation * c.sequence
            if (
                not isinstance(budget, int)
                or budget <= 0
                or (budget + update - 1) // update != c.steps
                or self.tokens - self.phase["parent_tokens"] != min(self.step * update, budget)
                or self.microsteps
                != (min(self.step * update, budget) + c.batch_size * c.sequence - 1)
                // (c.batch_size * c.sequence)
            ):
                raise ValueError("Exact phase-token accounting mismatch")
        self.resumed = True
        self.best_validation_loss = state.get("best_validation_loss")
        restore_rng(state["rng"])

    @torch.no_grad()
    def validate(self) -> float:
        started = time.perf_counter()
        batches = (
            self.config.training.validation_batches
            if self.streaming
            else min(self.config.training.validation_batches, self.val_data.size)
        )
        self.emit("validation_started", total_batches=batches)
        state, training = rng_state(), self.model.training
        self.model.eval()
        losses = []
        try:
            if self.streaming:
                self.val_data.load_state_dict(self.validation_stream_start)
            progress_at = started
            for i in range(batches):
                indices = self.val_data.next(1) if self.streaming else [i]
                x, y = self.val_data.batch(indices, self.device)
                with self.amp():
                    losses.append(float(causal_loss(self.model(x), y)))
                if time.perf_counter() - progress_at >= 0.25 or i + 1 == batches:
                    self.emit("validation_progress", completed_batches=i + 1, total_batches=batches)
                    progress_at = time.perf_counter()
            value = sum(losses) / len(losses)
            if not math.isfinite(value):
                raise FloatingPointError("Nonfinite validation loss")
            improved = self.best_validation_loss is None or value < self.best_validation_loss
            self.best_validation_loss = (
                value
                if self.best_validation_loss is None
                else min(self.best_validation_loss, value)
            )
            if improved and self.recovery is not None:
                self.recovery.save_best(
                    {
                        "artifact_kind": "codexa-private-weights-1",
                        "model": self.model.state_dict(),
                        "config": self.config.to_dict(),
                        "tokenizer_json": self.tokenizer_json,
                        "fingerprints": self.fingerprints,
                        "step": self.step,
                        "tokens": self.tokens,
                        "validation_loss": value,
                    }
                )
            self.emit(
                "validation_completed",
                validation_loss=value,
                perplexity=math.exp(value) if value < 700 else None,
                evaluated_tokens=len(losses) * self.config.training.sequence,
                duration_seconds=time.perf_counter() - started,
            )
            return value
        finally:
            self.model.train(training)
            restore_rng(state)

    def train(
        self, until: int | None = None, handle_signals: bool = True, ui: str = "auto"
    ) -> list[dict[str, Any]]:
        from codexa.safety import require_training_allowed

        require_training_allowed()
        if self.streaming:
            if self.train_data.launch_check is None:
                raise RuntimeError(
                    "Streaming training is blocked pending source approval and gate integration"
                )
            self.train_data.launch_check()
            if not self.phase or self.phase.get("additional_target_tokens") != 500_000_000:
                raise RuntimeError(
                    "Streaming requires an explicitly prepared 500M continuation phase"
                )
        from .terminal import choose_terminal

        c = self.config.training
        end = min(until if until is not None else c.steps, c.steps)
        if not self.resumed and (self.output / "latest.pt").exists():
            raise FileExistsError("Existing checkpoint: use explicit resume or a fresh output")
        terminal, ui_warning = choose_terminal(ui)
        self.events = EventBus(
            self.output,
            {
                "model": self.config.model.__dict__,
                "training": c.__dict__,
                "parameters": self.model.parameter_count(),
                "model_name": "Codexa v3",
                "torch_version": str(torch.__version__),
                "target_tokens": self.phase["parent_tokens"]
                + self.phase["additional_target_tokens"]
                if self.phase and "additional_target_tokens" in self.phase
                else self.tokens
                + max(0, c.steps - self.step) * c.batch_size * c.accumulation * c.sequence,
                "architecture": "codexa-decoder-v1",
                "device": str(self.device),
                "precision": "bf16" if self.bf16 else "fp32",
                "gpu_name": torch.cuda.get_device_name(self.device)
                if self.device.type == "cuda"
                else None,
                "optimizer": "AdamW · warmup/cosine",
                "parent_tokens": self.phase["parent_tokens"] if self.phase else 0,
                "parent_updates": self.phase["parent_updates"] if self.phase else 0,
                "fingerprints": self.fingerprints,
                "context_target": 32768,
                "dataset": self.dataset_name,
                "phase": "English Wikipedia continuation" if self.phase else "Base pretraining",
                "additional_target_tokens": self.phase.get("additional_target_tokens")
                if self.phase
                else None,
            },
            self.callbacks + ((terminal,) if terminal else ()),
        )
        self.emit(
            "started",
            resumed=self.resumed,
            resumed_from_step=self.step,
            best_validation_loss=self.best_validation_loss,
            metadata=self.events.summary["metadata"],
        )
        if ui_warning:
            import warnings

            self.emit("warning", message=ui_warning)
            warnings.warn(ui_warning, stacklevel=2)
        if getattr(self, "transition", None):
            self.emit("context_changed", transition=self.transition)
        records = []
        old = {}

        def stop(signum: int, frame: Any) -> None:
            self.stop = True

        if handle_signals:
            for sig in (signal.SIGINT, signal.SIGTERM):
                old[sig] = signal.signal(sig, stop)
        try:
            self.model.train()
            while self.step < end and not self.stop:
                start = time.perf_counter()
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                self.optimizer.zero_grad(set_to_none=True)
                loss_value = 0.0
                loading_seconds = 0.0
                full_update = c.batch_size * c.sequence * c.accumulation
                update_targets = full_update
                if self.phase and "additional_target_tokens" in self.phase:
                    update_targets = min(
                        full_update,
                        self.phase["parent_tokens"]
                        + self.phase["additional_target_tokens"]
                        - self.tokens,
                    )
                if update_targets <= 0:
                    raise ValueError("Phase-token budget exhausted before configured final step")
                used_targets = used_microsteps = unused_targets = 0
                for _ in range(c.accumulation):
                    if used_targets == update_targets:
                        break
                    load_start = time.perf_counter()
                    x, y = self.train_data.batch(self.sampler.next(c.batch_size), self.device)
                    loading_seconds += time.perf_counter() - load_start
                    valid_targets = min(y.numel(), update_targets - used_targets)
                    with self.amp():
                        if update_targets == full_update:
                            loss = causal_loss(self.model(x), y) / c.accumulation
                        else:
                            if c.batch_size == 1:
                                x, y = x[:, :valid_targets], y[:, :valid_targets]
                            logits = self.model(x)
                            loss = (
                                F.cross_entropy(
                                    logits.float().flatten(0, 1)[:valid_targets],
                                    y.flatten()[:valid_targets],
                                    reduction="sum",
                                )
                                / update_targets
                            )
                    if not torch.isfinite(loss):
                        raise FloatingPointError(
                            "Nonfinite loss; last completed checkpoint retained"
                        )
                    loss.backward()
                    self.microsteps += 1
                    used_microsteps += 1
                    used_targets += valid_targets
                    unused_targets += c.batch_size * c.sequence - valid_targets
                    loss_value += float(loss.detach())
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), c.clip, error_if_nonfinite=True
                )
                lr = self.optimizer.param_groups[0]["lr"]
                self.optimizer.step()
                self.scheduler.step()
                self.step += 1
                tokens = used_targets
                self.tokens += tokens
                if self.phase and unused_targets:
                    self.phase["unused_reserved_window_targets"] = (
                        self.phase.get("unused_reserved_window_targets", 0) + unused_targets
                    )
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                seconds = time.perf_counter() - start
                record = {
                    "step": self.step,
                    "tokens": self.tokens,
                    "loss": loss_value,
                    "lr": lr,
                    "seconds": seconds,
                    "tokens_per_second": tokens / seconds,
                    "peak_cuda_bytes": torch.cuda.max_memory_allocated()
                    if self.device.type == "cuda"
                    else 0,
                }
                self.emit(
                    "step_completed",
                    **record,
                    gradient_norm=float(gradient_norm),
                    clipped=bool(float(gradient_norm) + 1e-6 > c.clip),
                    sequence_length=c.sequence,
                    accumulation=used_microsteps,
                    effective_batch=c.batch_size * c.accumulation,
                    microstep=self.microsteps,
                    data_loading_seconds=loading_seconds,
                    data_loading_tokens_per_second=tokens / loading_seconds
                    if loading_seconds
                    else None,
                    dataset=self.dataset_name,
                    checkpoint_next_tokens=(
                        (
                            (self.tokens - (self.phase["parent_tokens"] if self.phase else 0))
                            // c.checkpoint_every_tokens
                            + 1
                        )
                        * c.checkpoint_every_tokens
                        if c.checkpoint_every_tokens
                        else None
                    ),
                    recovery_status=self.recovery.status() if self.recovery else None,
                    cuda_allocated_bytes=torch.cuda.memory_allocated()
                    if self.device.type == "cuda"
                    else None,
                    cuda_reserved_bytes=torch.cuda.memory_reserved()
                    if self.device.type == "cuda"
                    else None,
                )
                if self.step % c.validate_every == 0 or self.step == end:
                    record["validation_loss"] = self.validate()
                    if terminal is None:
                        print(json.dumps(record), flush=True)
                records.append(record)
                with (self.output / "metrics.jsonl").open("a") as f:
                    f.write(json.dumps(record) + "\n")
                phase_tokens = self.tokens - (self.phase["parent_tokens"] if self.phase else 0)
                if c.checkpoint_every_tokens is not None:
                    due = (
                        phase_tokens // c.checkpoint_every_tokens
                        > (phase_tokens - tokens) // c.checkpoint_every_tokens
                    )
                else:
                    due = self.step % c.checkpoint_every == 0
                if due:
                    self.checkpoint()
            self.checkpoint()
            self.emit(
                "interrupted" if self.stop else "completed" if self.step == c.steps else "paused"
            )
            return records
        except Exception as exc:
            self.emit("failed", error_type=type(exc).__name__, message=str(exc)[:500])
            raise
        finally:
            try:
                if self.events:
                    self.events.close()
                    self.events = None
            finally:
                if terminal:
                    terminal.close()
                for sig, previous in old.items():
                    signal.signal(sig, previous)
