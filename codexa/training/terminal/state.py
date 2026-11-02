"""Bounded projection of the existing event schema; no model or terminal dependencies."""

from collections import deque
from typing import Any

from codexa.training.callbacks.telemetry import TrainingEvent


class TerminalState:
    def __init__(self) -> None:
        self.metadata: dict[str, Any] = {}
        self.metrics: dict[str, Any] = {}
        self.losses: deque[float] = deque(maxlen=60)
        self.speeds: deque[float] = deque(maxlen=60)
        self.timings: deque[tuple[int, float]] = deque(maxlen=30)
        self.recent: deque[str] = deque(maxlen=6)
        self.step = self.tokens = 0
        self.elapsed = 0.0
        self.run_id = ""
        self.status = "starting"
        self.phase = "Starting"
        self.validation: dict[str, Any] = {}
        self.checkpoint: dict[str, Any] = {}
        self.best: float | None = None
        self.resumed = False
        self.peak_vram: int | None = None

    def update(self, e: TrainingEvent) -> None:
        self.step, self.tokens, self.elapsed = e.step, e.tokens, e.elapsed_seconds
        self.run_id = e.run_id
        d = e.data
        if e.event == "started":
            self.metadata = d.get("metadata", {})
            self.resumed = bool(d.get("resumed"))
            self.best = d.get("best_validation_loss")
            self.status, self.phase = "running", "Base pretraining"
        elif e.event == "step_completed":
            self.metrics = d.copy()
            peak = d.get("peak_cuda_bytes")
            if peak is not None:
                self.peak_vram = max(self.peak_vram or 0, peak)
            if isinstance(d.get("loss"), (float, int)):
                self.losses.append(d["loss"])
            if isinstance(d.get("tokens_per_second"), (float, int)):
                self.speeds.append(d["tokens_per_second"])
            self.timings.append((e.step, e.elapsed_seconds))
            self.status, self.phase = "running", "Base pretraining"
        elif e.event == "validation_started":
            self.status, self.phase = "validating", "Validation"
            self.validation.update(completed_batches=0, total_batches=d.get("total_batches"))
        elif e.event == "validation_progress":
            self.validation.update(d)
        elif e.event == "validation_completed":
            self.validation.update(d)
            value = d.get("validation_loss")
            if value is not None:
                self.best = value if self.best is None else min(self.best, value)
            self.status, self.phase = "running", "Base pretraining"
        elif e.event == "checkpoint_started":
            self.status, self.phase = "checkpointing", "Saving checkpoint"
        elif e.event == "checkpoint_saved":
            self.checkpoint = {**d, "step": e.step, "timestamp": e.timestamp}
            self.status, self.phase = "running", "Base pretraining"
        elif e.event in ("paused", "completed", "interrupted", "failed"):
            self.status, self.phase = e.event, "Run finished"
        if e.event not in ("step_completed", "validation_progress"):
            detail = d.get("message") or d.get("identifier") or d.get("error_type") or ""
            self.recent.append(
                f"{e.timestamp[11:19]}  {e.event.replace('_', ' ')}  {str(detail)[:180]}"
            )

    def snapshot(self) -> dict[str, Any]:
        total = self.metadata.get("training", {}).get("steps")
        remaining = None
        if total and len(self.timings) >= 3 and self.status == "running":
            first, last = self.timings[0], self.timings[-1]
            if last[0] > first[0]:
                remaining = max(0, total - self.step) * (last[1] - first[1]) / (last[0] - first[0])
        return {
            "metadata": self.metadata,
            "metrics": self.metrics,
            "losses": list(self.losses),
            "speeds": list(self.speeds),
            "smoothed_loss": sum(self.losses) / len(self.losses) if self.losses else None,
            "events": list(self.recent),
            "step": self.step,
            "tokens": self.tokens,
            "elapsed": self.elapsed,
            "eta": remaining,
            "run_id": self.run_id,
            "status": self.status,
            "phase": self.phase,
            "validation": self.validation,
            "checkpoint": self.checkpoint,
            "best": self.best,
            "peak_vram": self.peak_vram,
            "resumed": self.resumed,
        }
