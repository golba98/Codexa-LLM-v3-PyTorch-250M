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
from .checkpoint import load, restore_rng, rng_state, save_atomic
from .optimizer import make_optimizer
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
    ):
        self.config, self.output = config, Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.device = choose_device(device)
        c = config.training
        seed_all(c.seed)
        self.tokenizer_json = Path(tokenizer).read_text()
        tok = CodexaTokenizer.load(tokenizer)
        if tok.backend.get_vocab_size() != config.model.vocab_size:
            raise ValueError("Tokenizer vocabulary differs from model")
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
        self.sampler = Sampler(self.train_data.size, c.seed)
        supported = self.device.type == "cuda" and torch.cuda.is_bf16_supported()
        self.bf16 = supported and c.precision != "fp32"
        self.step, self.tokens, self.stop = 0, 0, False
        self.resumed = False
        self.callbacks = callbacks
        self.microsteps = 0
        self.events: EventBus | None = None
        self.model.gradient_checkpointing = c.gradient_checkpointing
        self.best_validation_loss: float | None = None

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
        started = time.perf_counter()
        self.emit("checkpoint_started", identifier=name)
        save_atomic(
            {
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
            },
            self.output / name,
        )

        self.emit(
            "checkpoint_saved",
            identifier=name,
            size_bytes=(self.output / name).stat().st_size,
            duration_seconds=time.perf_counter() - started,
            verification_status="atomic_save_only",
        )

    def resume(self, path: str) -> None:
        state = load(path)
        if (
            state["version"] != 1
            or Config.from_dict(state["config"]).to_dict() != self.config.to_dict()
            or state["fingerprints"] != self.fingerprints
        ):
            raise ValueError("Resume configuration or data mismatch")
        if state["device_type"] != self.device.type or state["bf16"] != self.bf16:
            raise ValueError("Exact training resume requires the same device type and precision")
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        if "context_transition" in state and "sampler" not in state:
            self.sampler = Sampler(self.train_data.size, self.config.training.seed)
        else:
            self.sampler.load_state_dict(state["sampler"])
        self.step, self.tokens = state["step"], state["tokens"]
        self.microsteps = state.get(
            "microsteps", self.step * Config.from_dict(state["config"]).training.accumulation
        )
        self.transition = state.get("context_transition")
        self.resumed = True
        self.best_validation_loss = state.get("best_validation_loss")
        restore_rng(state["rng"])

    @torch.no_grad()
    def validate(self) -> float:
        started = time.perf_counter()
        batches = min(self.config.training.validation_batches, self.val_data.size)
        self.emit("validation_started", total_batches=batches)
        state, training = rng_state(), self.model.training
        self.model.eval()
        losses = []
        try:
            progress_at = started
            for i in range(batches):
                x, y = self.val_data.batch([i], self.device)
                with self.amp():
                    losses.append(float(causal_loss(self.model(x), y)))
                if time.perf_counter() - progress_at >= 0.25 or i + 1 == batches:
                    self.emit("validation_progress", completed_batches=i + 1, total_batches=batches)
                    progress_at = time.perf_counter()
            value = sum(losses) / len(losses)
            self.best_validation_loss = (
                value
                if self.best_validation_loss is None
                else min(self.best_validation_loss, value)
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
                "target_tokens": self.tokens
                + max(0, c.steps - self.step) * c.batch_size * c.accumulation * c.sequence,
                "architecture": "codexa-decoder-v1",
                "device": str(self.device),
                "precision": "bf16" if self.bf16 else "fp32",
                "gpu_name": torch.cuda.get_device_name(self.device)
                if self.device.type == "cuda"
                else None,
                "optimizer": "AdamW · warmup/cosine",
                "fingerprints": self.fingerprints,
                "context_target": 32768,
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
                for _ in range(c.accumulation):
                    x, y = self.train_data.batch(self.sampler.next(c.batch_size), self.device)
                    with self.amp():
                        loss = causal_loss(self.model(x), y) / c.accumulation
                    if not torch.isfinite(loss):
                        raise FloatingPointError(
                            "Nonfinite loss; last completed checkpoint retained"
                        )
                    loss.backward()
                    self.microsteps += 1
                    loss_value += float(loss.detach())
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), c.clip, error_if_nonfinite=True
                )
                lr = self.optimizer.param_groups[0]["lr"]
                self.optimizer.step()
                self.scheduler.step()
                self.step += 1
                tokens = c.batch_size * c.sequence * c.accumulation
                self.tokens += tokens
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
                    accumulation=c.accumulation,
                    effective_batch=c.batch_size * c.accumulation,
                    microstep=self.microsteps,
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
                if self.step % c.checkpoint_every == 0:
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
