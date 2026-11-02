import fcntl
import json
import os
import subprocess
import threading
import time
import uuid
import warnings
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class TrainingEvent:
    schema_version: int
    run_id: str
    session_id: str
    seq: int
    event: str
    step: int
    tokens: int
    timestamp: str
    elapsed_seconds: float
    data: dict[str, Any]


class Callback(Protocol):
    def on_event(self, event: TrainingEvent) -> None: ...


class EventBus:
    def __init__(self, root: Path, metadata: dict[str, Any], callbacks: tuple[Callback, ...] = ()):
        root.mkdir(parents=True, exist_ok=True)
        self.lock = (root / "telemetry.lock").open("a")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = root / "run.json"
        self.run_id = (
            json.loads(manifest.read_text())["run_id"] if manifest.exists() else uuid.uuid4().hex
        )
        self.session = uuid.uuid4().hex
        self.root = root
        self.started = time.monotonic()
        self.seq = 0
        self.callbacks = list(callbacks)
        self.stream = (root / "events.jsonl").open("ab")
        if (root / "events.jsonl").stat().st_size:
            with (root / "events.jsonl").open("rb") as f:
                f.seek(-1, 2)
                if f.read() != b"\n":
                    self.stream.write(b"\n")
        self.summary = {
            "run_id": self.run_id,
            "metadata": metadata,
            "pid": os.getpid(),
            "session_id": self.session,
        }
        tmp = root / ".run.tmp"
        tmp.write_text(json.dumps(self.summary))
        tmp.replace(manifest)
        self.seen: set[tuple[str, int]] = set()
        self.failed = False
        self.gpu_snapshot: dict[str, Any] | None = None
        self.host_snapshot: dict[str, Any] | None = None
        self.cpu_sample: tuple[int, int] | None = None
        self.stop_probe = threading.Event()
        self.cuda_probe = str(metadata.get("device", "")).startswith("cuda")
        threading.Thread(target=self.probe_system, daemon=True).start()

    def probe_host(self) -> dict[str, Any] | None:
        try:
            memory = {
                line.split()[0].rstrip(":"): int(line.split()[1]) * 1024
                for line in Path("/proc/meminfo").read_text().splitlines()
            }
            rss = next(
                int(line.split()[1]) * 1024
                for line in Path("/proc/self/status").read_text().splitlines()
                if line.startswith("VmRSS:")
            )
            ticks = list(map(int, Path("/proc/stat").read_text().splitlines()[0].split()[1:9]))
            total, idle = sum(ticks), ticks[3] + ticks[4]
            utilization = None
            if self.cpu_sample and total > self.cpu_sample[0]:
                utilization = 100 * (1 - (idle - self.cpu_sample[1]) / (total - self.cpu_sample[0]))
            self.cpu_sample = total, idle
            return {
                "available_ram_bytes": memory.get("MemAvailable"),
                "total_ram_bytes": memory.get("MemTotal"),
                "process_rss_bytes": rss,
                "cpu_utilization_percent": utilization,
                "load_1m": os.getloadavg()[0],
                "sampled_at": datetime.now(UTC).isoformat(),
                "scope": "host /proc",
            }
        except (OSError, ValueError, IndexError, StopIteration):
            return None

    def probe_system(self) -> None:
        while not self.stop_probe.is_set():
            self.host_snapshot = self.probe_host()
            if not self.cuda_probe:
                self.stop_probe.wait(2)
                continue
            self.gpu_snapshot = None
            try:
                r = subprocess.run(
                    [
                        "nvidia-smi",
                        "--id=0",
                        "--query-gpu=utilization.gpu,temperature.gpu,power.draw",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=1,
                    check=False,
                )
                values = r.stdout.strip().split(",")
                if r.returncode == 0 and len(values) == 3:
                    self.gpu_snapshot = {
                        "utilization_percent": float(values[0]),
                        "temperature_celsius": float(values[1]),
                        "power_watts": float(values[2]),
                        "sampled_at": datetime.now(UTC).isoformat(),
                        "scope": "physical GPU 0",
                    }
            except (OSError, ValueError, subprocess.TimeoutExpired):
                self.gpu_snapshot = None
            self.stop_probe.wait(2)

    def emit(self, kind: str, step: int, tokens: int, **data: Any) -> None:
        if (kind, step) in self.seen and kind == "step_completed":
            return
        if kind == "step_completed":
            data["gpu_system"] = self.gpu_snapshot
            data["host_system"] = self.host_snapshot
        self.seen.add((kind, step))
        self.seq += 1
        event = TrainingEvent(
            1,
            self.run_id,
            self.session,
            self.seq,
            kind,
            step,
            tokens,
            datetime.now(UTC).isoformat(),
            time.monotonic() - self.started,
            data,
        )
        failures = []
        disabled = []
        for cb in list(self.callbacks):
            try:
                cb.on_event(replace(event, data=deepcopy(event.data)))
            except Exception as exc:  # noqa: BLE001 - observer isolation
                self.callbacks.remove(cb)
                failures.append(type(exc).__name__)
                disabled.append(type(cb).__name__)
                if not getattr(cb, "silent_failure", False):
                    warnings.warn(f"Callback disabled: {type(exc).__name__}", stacklevel=2)
        if self.failed:
            return
        try:
            record = asdict(event)
            record["event_id"] = f"{self.run_id}:{self.session}:{self.seq}"
            self.stream.write(json.dumps(record, allow_nan=False).encode() + b"\n")
            self.stream.flush()
            if self.seq % 20 == 0 or kind in (
                "checkpoint_saved",
                "completed",
                "interrupted",
                "failed",
                "paused",
            ):
                os.fsync(self.stream.fileno())
            self.summary.update({"latest": record})
            tmp = self.root / ".summary.tmp"
            tmp.write_text(json.dumps(self.summary))
            tmp.replace(self.root / "summary.json")
        except (OSError, ValueError):
            self.failed = True
            warnings.warn("Telemetry persistence degraded; training continues", stacklevel=2)

        if failures and not self.failed:
            self.emit(
                "warning",
                step,
                tokens,
                observer_errors=failures,
                message="Observer disabled: " + ", ".join(disabled),
            )

    def close(self) -> None:
        self.stop_probe.set()
        self.stream.close()
        self.lock.close()
