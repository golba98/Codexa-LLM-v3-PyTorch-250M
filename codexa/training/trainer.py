import json
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
        self, config: Config, data: str, tokenizer: str, output: str, device: str = "auto"
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

    def amp(self) -> torch.autocast:
        return torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.bf16)

    def checkpoint(self, name: str = "latest.pt") -> None:
        save_atomic(
            {
                "version": 1,
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

    def resume(self, path: str) -> None:
        state = load(path)
        if (
            state["version"] != 1
            or state["config"] != self.config.to_dict()
            or state["fingerprints"] != self.fingerprints
        ):
            raise ValueError("Resume configuration or data mismatch")
        if state["device_type"] != self.device.type or state["bf16"] != self.bf16:
            raise ValueError("Exact training resume requires the same device type and precision")
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.sampler.load_state_dict(state["sampler"])
        self.step, self.tokens = state["step"], state["tokens"]
        restore_rng(state["rng"])

    @torch.no_grad()
    def validate(self) -> float:
        state, training = rng_state(), self.model.training
        self.model.eval()
        losses = []
        try:
            for i in range(min(self.config.training.validation_batches, self.val_data.size)):
                x, y = self.val_data.batch([i], self.device)
                with self.amp():
                    losses.append(float(causal_loss(self.model(x), y)))
            return sum(losses) / len(losses)
        finally:
            self.model.train(training)
            restore_rng(state)

    def train(self, until: int | None = None, handle_signals: bool = True) -> list[dict[str, Any]]:
        c = self.config.training
        end = min(until if until is not None else c.steps, c.steps)
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
                    loss_value += float(loss.detach())
                torch.nn.utils.clip_grad_norm_(
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
                if self.step % c.validate_every == 0 or self.step == end:
                    record["validation_loss"] = self.validate()
                    print(json.dumps(record), flush=True)
                records.append(record)
                with (self.output / "metrics.jsonl").open("a") as f:
                    f.write(json.dumps(record) + "\n")
                if self.step % c.checkpoint_every == 0:
                    self.checkpoint()
            self.checkpoint()
            return records
        finally:
            for sig, previous in old.items():
                signal.signal(sig, previous)
