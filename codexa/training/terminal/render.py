"""V1 Rich monitoring layout adapted to v3 event snapshots. See V1_LICENSE."""

import math
import re
from typing import Any

from rich import box
from rich.console import Group
from rich.layout import Layout
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

ACCENT, BORDER, DIM = "deep_sky_blue1", "grey37", "grey62"
SPARK = "▁▂▃▄▅▆▇█"
BADGES = {
    "running": "black on green3",
    "validating": "black on cyan",
    "checkpointing": "white on dodger_blue2",
    "completed": "black on bright_green",
    "paused": "black on gold1",
    "interrupted": "white on magenta",
    "failed": "white on red3",
}


def clean(value: Any) -> str:
    return re.sub(r"[\x00-\x1f\x7f-\x9f]", "?", str(value))


def number(value: Any, digits: int = 3, suffix: str = "") -> str:
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        return "—"
    return f"{value:,.{digits}f}{suffix}"


def duration(value: float | None) -> str:
    if value is None:
        return "—"
    seconds = max(0, int(value))
    return f"{seconds // 3600:02}:{seconds // 60 % 60:02}:{seconds % 60:02}"


def spark(values: list[float], width: int = 24) -> str:
    values = values[-width:]
    if len(values) < 2:
        return "—"
    low, high = min(values), max(values)
    return "".join(SPARK[int((v - low) / (high - low) * 7) if high > low else 3] for v in values)


def panel(rows: list[tuple[str, Any]], title: str) -> Panel:
    table = Table.grid(expand=True, padding=(0, 1))
    table.add_column(style=DIM, no_wrap=True, overflow="ellipsis")
    table.add_column(
        style="bold white", justify="right", no_wrap=True, overflow="ellipsis", ratio=1
    )
    for label, value in rows:
        table.add_row(Text(clean(label)), Text(clean(value)))
    return Panel(
        table,
        title=Text(title, style=f"bold {ACCENT}"),
        title_align="left",
        border_style=BORDER,
        box=box.ROUNDED,
        padding=(0, 1),
    )


def frame(s: dict[str, Any], width: int, height: int) -> Any:
    meta, m, val, ck = s["metadata"], s["metrics"], s["validation"], s["checkpoint"]
    cfg, model = meta.get("training", {}), meta.get("model", {})
    status = s["status"]
    total, step = cfg.get("steps"), s["step"]
    progress = f"{step:,} / {total:,}  {100 * step / total:.1f}%" if total else f"{step:,} / —"
    target = meta.get("target_tokens")
    tokens = number(s["tokens"], 0) + " / " + number(target, 0)
    header = Table.grid(expand=True)
    header.add_column(ratio=1, no_wrap=True, overflow="ellipsis")
    header.add_column(justify="right", no_wrap=True)
    header.add_row(
        Text(
            f"CODEXA v3  ·  {number(meta.get('parameters'), 0)} parameters", style=f"bold {ACCENT}"
        ),
        Text(f" {status.upper()} ", style=BADGES.get(status, "black on grey70")),
    )
    header.add_row(Text(clean(s["run_id"][:16]) + " · " + s["phase"], style=DIM), Text(progress))
    header.add_row(
        ProgressBar(
            total=total or 1,
            completed=step if total else 0,
            pulse=False,
            complete_style=ACCENT,
            finished_style="green3",
        )
        if total
        else Text("Total work unavailable", style=DIM),
        Text(f"  ETA ≈ {duration(s['eta'])}"),
    )
    header.add_row(Text("Tokens " + tokens, style=DIM), Text("Session " + duration(s["elapsed"])))
    if meta.get("parent_tokens"):
        header.add_row(
            Text(
                "Continuation tokens "
                + number(s["tokens"] - meta["parent_tokens"], 0)
                + " / "
                + number(
                    meta.get("additional_target_tokens", (target or 0) - meta["parent_tokens"]), 0
                ),
                style=DIM,
            ),
            Text("Lifetime step " + number(meta.get("parent_updates", 0) + step, 0)),
        )
    top = Panel(header, border_style=BORDER, box=box.ROUNDED, padding=(0, 1))
    mean, loss = number(s["smoothed_loss"]), number(m.get("loss"))
    lr = f"{m['lr']:.2e}" if m.get("lr") is not None else "—"
    grad = number(m.get("gradient_norm")) + (" · clipped" if m.get("clipped") else "")
    speed = number(m.get("tokens_per_second"), 0, " tok/s")
    seconds = m.get("seconds")
    batch = m.get("effective_batch", cfg.get("batch_size", 0) * cfg.get("accumulation", 0)) or None
    sequence = m.get("sequence_length", cfg.get("sequence"))
    optimization = [
        ("Train loss", loss),
        ("Mean loss (≤60)", mean),
        ("Validation loss", number(val.get("validation_loss"))),
        ("Best validation", number(s["best"])),
        ("Val perplexity", number(val.get("perplexity"))),
        ("Learning rate", lr),
        ("Gradient norm", grad),
        ("Optimizer", meta.get("optimizer", "—")),
        ("Loss trend", spark(s["losses"])),
        ("Speed trend", spark(s["speeds"])),
    ]
    performance = [
        ("Update throughput", speed),
        (
            "Effective throughput",
            number(
                (s["tokens"] - s.get("session_start_tokens", s["tokens"])) / s["elapsed"]
                if s["elapsed"] > 0
                else None,
                0,
                " tok/s",
            ),
        ),
        ("Step duration", number(seconds, 4, "s")),
        ("Effective batch", number(batch, 0)),
        ("Accumulation", number(cfg.get("accumulation"), 0)),
        ("Sequence / context", f"{number(sequence, 0)} / {number(model.get('context'), 0)}"),
        ("Dataset", meta.get("dataset", "—")),
        ("Data loading", number(m.get("data_loading_seconds"), 4, "s")),
        ("Load rate", number(m.get("data_loading_tokens_per_second"), 0, " tok/s")),
    ]
    gpu, host = m.get("gpu_system") or {}, m.get("host_system") or {}

    def gib(value):
        return number(value / 1073741824 if value is not None else None, 2, " GiB")

    hardware = [
        ("GPU", meta.get("gpu_name") or "Unavailable"),
        ("VRAM allocated", gib(m.get("cuda_allocated_bytes"))),
        ("VRAM reserved", gib(m.get("cuda_reserved_bytes"))),
        ("Peak (session)", gib(s.get("peak_vram"))),
        ("GPU utilization", number(gpu.get("utilization_percent"), 0, "%")),
        (
            "Temp / power",
            number(gpu.get("temperature_celsius"), 0, "°C")
            + " / "
            + number(gpu.get("power_watts"), 1, " W"),
        ),
        ("Host CPU", number(host.get("cpu_utilization_percent"), 1, "%")),
        ("Host RAM available", gib(host.get("available_ram_bytes"))),
        ("Process RAM", gib(host.get("process_rss_bytes"))),
        (
            "Precision / attention",
            f"{meta.get('precision', '—')} / {model.get('attention_backend', '—')}",
        ),
    ]
    interval = cfg.get("checkpoint_every")
    next_save = (step // interval + 1) * interval if interval else None
    if next_save and total and next_save > total:
        next_save = None
    validation = (
        "active ("
        + number(val.get("completed_batches"), 0)
        + "/"
        + number(val.get("total_batches"), 0)
        + ")"
        if status == "validating"
        else "complete"
        if val.get("validation_loss") is not None
        else "not evaluated"
    )
    recovery = [
        ("Latest checkpoint", ck.get("identifier", "—")),
        ("Last successful save", f"step {ck.get('step', '—')} · {ck.get('timestamp', '—')[11:19]}"),
        ("Interval / next step", number(interval, 0) + " / " + number(next_save, 0)),
        ("Recovery", "Resumed" if s["resumed"] else "New random initialization"),
        (
            "Save state",
            "writing · progress unknown"
            if status == "checkpointing"
            else ck.get("verification_status", "no save recorded"),
        ),
        (
            "Save size / time",
            number(ck.get("size_bytes", 0) / 1048576, 2, " MiB")
            + " / "
            + number(ck.get("duration_seconds"), 3, "s")
            if ck
            else "—",
        ),
        ("Validation", validation),
    ]
    if cfg.get("checkpoint_every_tokens"):
        next_tokens = min(
            m.get("checkpoint_next_tokens") or cfg["checkpoint_every_tokens"],
            meta.get("additional_target_tokens") or float("inf"),
        )
        recovery[2] = ("Next save (phase tokens)", number(next_tokens, 0))
        retained = ck.get("retained", [])
        recovery[5] = (
            "Retained recovery steps",
            ", ".join(str(e["step"]) for e in retained) or "Preparing first save",
        )
        recovery[6] = (
            "Disk free / recovery",
            gib(ck.get("disk_free_bytes")) + " / " + gib(ck.get("recovery_bytes")),
        )
    events = s["events"] or ["Waiting for training events"]
    event_lines = 3 if width >= 86 and height >= 34 else 2 if width >= 86 and height >= 28 else 1
    event_panel = Panel(
        Text("\n".join(clean(v) for v in events[-event_lines:]), style="grey78"),
        title=Text("Recent events", style=f"bold {ACCENT}"),
        title_align="left",
        border_style=BORDER,
        box=box.ROUNDED,
        padding=(0, 1),
    )
    footer = Text(
        "Ctrl+C: close viewer · training continues independently"
        if meta.get("viewer_only")
        else "Ctrl+C: safe checkpoint & stop · viewer has no training controls",
        style=DIM,
        overflow="ellipsis",
        no_wrap=True,
    )
    if height < 16 or width < 38:
        return Group(
            panel(
                [
                    ("State", status),
                    ("Steps", progress),
                    ("Loss / val", loss + " / " + number(val.get("validation_loss"))),
                    ("Speed", speed),
                    ("Checkpoint", ck.get("identifier", "—")),
                ],
                "CODEXA v3 · compact",
            ),
            footer,
        )
    root = Layout()
    if width >= 86 and height >= 28:
        full = height >= 34
        if not full:
            optimization = optimization[:7]
            hardware = hardware[:6] + hardware[-1:]
            performance = performance[:6]
            recovery = recovery[:6]
        left, right = Layout(), Layout()
        left.split_column(
            Layout(panel(optimization, "Training & optimization"), ratio=6),
            Layout(panel(performance, "Performance"), ratio=5),
        )
        right.split_column(
            Layout(panel(hardware, "Resources"), ratio=6),
            Layout(panel(recovery, "Checkpoints & recovery"), ratio=5),
        )
        middle = Layout()
        middle.split_row(left, right)
        root.split_column(
            Layout(top, size=7 if meta.get("parent_tokens") else 6),
            middle,
            Layout(event_panel, size=5 if full else 4),
            Layout(footer, size=1),
        )
    else:
        compact = [
            ("Loss / mean", loss + " / " + mean),
            ("Val / best", number(val.get("validation_loss")) + " / " + number(s["best"])),
            ("LR / gradient", lr + " / " + number(m.get("gradient_norm"))),
            ("Throughput", speed),
            ("Seq / batch", number(sequence, 0) + " / " + number(batch, 0)),
        ]
        middle = Layout()
        middle.split_column(
            Layout(panel(compact, "Training · compact"), ratio=7),
            Layout(
                panel([recovery[0], recovery[3], ("Validation", validation)], "Recovery"), ratio=5
            ),
        )
        root.split_column(
            Layout(top, size=6), middle, Layout(event_panel, size=3), Layout(footer, size=1)
        )
    return root
