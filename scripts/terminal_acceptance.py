"""Run bounded REAL CLI training in a PTY and capture the rendered cells (dev extra)."""

import argparse
import codecs
import fcntl
import html
import json
import os
import pty
import re
import select
import signal
import struct
import subprocess
import sys
import termios
import time
from pathlib import Path

import pyte


class FrameCapture(pyte.Screen):
    """Capture complete Rich frames at VT boundaries, independent of OS read chunks."""

    def __init__(self, columns, lines):
        self.frames = []
        super().__init__(columns, lines)

    def capture_frame(self):
        if not hasattr(self, "buffer"):
            return
        display = "\n".join(self.display)
        if "CODEXA" not in display or not ("Training" in display or "compact" in display):
            return
        if self.frames and self.frames[-1]["text"] == display:
            return
        self.frames.append(
            {
                "text": display,
                "width": self.columns,
                "height": self.lines,
                "cells": [
                    [self.buffer[y][x] for x in range(self.columns)] for y in range(self.lines)
                ],
            }
        )
        if len(self.frames) > 40:
            self.frames.pop(1)

    def cursor_position(self, line=None, column=None):
        if line in (None, 0, 1) and column in (None, 0, 1):
            self.capture_frame()
        super().cursor_position(line, column)

    def reset_mode(self, *modes, **kwargs):
        if kwargs.get("private") and 1049 in modes:
            self.capture_frame()
        super().reset_mode(*modes, **kwargs)


def capture(a):
    if not 1 <= a.until <= 200:
        raise ValueError("Acceptance is bounded to 1–200 optimizer steps")
    a.output.mkdir(parents=True, exist_ok=False)
    master, slave = pty.openpty()
    width, height = a.width, a.height
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", height, width, 0, 0))
    before = termios.tcgetattr(slave)
    command = [
        sys.executable,
        "-m",
        "codexa.cli",
        "train",
        "--config",
        a.config,
        "--data",
        a.data,
        "--tokenizer",
        a.tokenizer,
        "--output",
        str(a.output / "run"),
        "--device",
        a.device,
        "--until",
        str(a.until),
        "--ui",
        a.ui,
    ]
    if a.resume:
        command += ["--resume", a.resume]
    env = os.environ.copy()
    env["TERM"] = "xterm-256color"
    env.pop("NO_COLOR", None)  # Exercise colors in the isolated acceptance terminal.
    process = subprocess.Popen(
        command, stdin=slave, stdout=slave, stderr=slave, start_new_session=True, env=env
    )
    screen = FrameCapture(width, height)
    stream = pyte.Stream(screen)
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    raw = bytearray()
    frames = screen.frames
    metrics_offset = 0
    interrupt_sent = False
    resized = False
    started = time.monotonic()
    try:
        while True:
            if time.monotonic() - started > 60:
                raise TimeoutError("Bounded terminal acceptance timed out")
            ready, _, _ = select.select([master], [], [], 0.1)
            if ready:
                chunk = os.read(master, 65536)
                if not chunk:
                    break
                raw.extend(chunk)
                stream.feed(decoder.decode(chunk))
            metric_path = a.output / "run/events.jsonl"
            current_step = 0
            if metric_path.exists():
                with metric_path.open("rb") as f:
                    f.seek(metrics_offset)
                    lines = f.read().splitlines(keepends=True)
                for line in lines:
                    if not line.endswith(b"\n"):
                        break
                    metrics_offset += len(line)
                    r = json.loads(line)
                    if r["event"] == "step_completed":
                        current_step = max(current_step, r["step"])
            if a.interrupt_after and current_step >= a.interrupt_after and not interrupt_sent:
                os.kill(process.pid, signal.SIGINT)
                interrupt_sent = True
            if a.resize_to and current_step >= 30 and not resized:
                width, height = map(int, a.resize_to.split("x"))
                fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", height, width, 0, 0))
                screen.resize(lines=height, columns=width)
                resized = True
            if process.poll() is not None and not ready:
                break
        code = process.wait(timeout=5)
        restored = termios.tcgetattr(slave) == before
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        os.close(master)
        os.close(slave)
    screen.capture_frame()
    (a.output / "terminal.ansi").write_bytes(raw)
    events = [json.loads(line) for line in (a.output / "run/events.jsonl").read_text().splitlines()]
    steps = [r for r in events if r["event"] == "step_completed"]
    if code or not frames:
        raise AssertionError(
            f"CLI failed or no rendered dashboard: code={code}; tail={raw[-2000:]!r}"
        )
    latest = frames[-1]
    (a.output / "terminal.txt").write_text(latest["text"])
    export_svg(latest, a.output / "terminal.svg")
    rendered_steps = sorted(
        {
            int(m.replace(",", ""))
            for f in frames
            for m in re.findall(r"(\d[\d,]*) / [\d,]+  [\d.]+%", f["text"])
        }
    )
    report = {
        "command": command,
        "returncode": code,
        "frames_captured": len(frames),
        "rendered_steps": rendered_steps,
        "optimizer_steps": len(steps),
        "last_step": events[-1]["step"],
        "last_event": events[-1]["event"],
        "last_loss": steps[-1]["data"]["loss"] if steps else None,
        "gpu_sample_recorded": any(r["data"].get("gpu_system") for r in steps),
        "host_sample_recorded": any(r["data"].get("host_system") for r in steps),
        "checkpoint_saved": any(r["event"] == "checkpoint_saved" for r in events),
        "validation_recorded": any(r["event"] == "validation_completed" for r in events),
        "color_sgr_recorded": bool(re.search(rb"\x1b\[(?:[0-9;]*;)?(?:38|48);", raw)),
        "alternate_screen_entered": b"\x1b[?1049h" in raw,
        "alternate_screen_restored": b"\x1b[?1049l" in raw,
        "cursor_restored": b"\x1b[?25h" in raw,
        "termios_unchanged": restored,
        "resized": resized,
        "interrupt_sent": interrupt_sent,
        "scope": "Actual local smoke training, not simulated metrics",
    }
    (a.output / "acceptance.json").write_text(json.dumps(report, indent=2) + "\n")
    assert report["checkpoint_saved"] and report["alternate_screen_restored"] and restored
    assert len(rendered_steps) >= 2, report
    print(json.dumps(report, indent=2))


def export_svg(frame, path):
    """Faithful color/text export of PTY cells, not a separately constructed UI mockup."""
    colors = {
        "default": "#d9e1ee",
        "black": "#101319",
        "red": "#d75f5f",
        "green": "#5faf5f",
        "brown": "#d7af5f",
        "blue": "#5f87d7",
        "magenta": "#af87d7",
        "cyan": "#5fafaf",
        "white": "#d9e1ee",
        "brightblack": "#555555",
        "brightred": "#ff5555",
        "brightgreen": "#55ff55",
        "brightbrown": "#ffff55",
        "brightblue": "#5555ff",
        "brightmagenta": "#ff55ff",
        "brightcyan": "#55ffff",
        "brightwhite": "#ffffff",
    }

    def color(value, background=False):
        if value == "default":
            return "#101319" if background else "#d9e1ee"
        return (
            "#" + value if re.fullmatch("[0-9a-fA-F]{6}", value) else colors.get(value, "#d9e1ee")
        )

    content = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{frame["width"] * 10 + 24}" height="{frame["height"] * 20 + 24}" viewBox="0 0 {frame["width"] * 10 + 24} {frame["height"] * 20 + 24}">',
        '<rect width="100%" height="100%" fill="#101319"/>',
    ]
    for y, row in enumerate(frame["cells"]):
        for x, cell in enumerate(row):
            if cell.bg != "default":
                content.append(
                    f'<rect x="{x * 10 + 12}" y="{y * 20 + 6}" width="10" height="20" fill="{color(cell.bg, True)}"/>'
                )
            if cell.data.strip():
                content.append(
                    f'<text x="{x * 10 + 12}" y="{y * 20 + 22}" fill="{color(cell.fg)}" font-family="DejaVu Sans Mono,monospace" font-size="16.6" font-weight="{"bold" if cell.bold else "normal"}">{html.escape(cell.data)}</text>'
                )
    path.write_text("\n".join(content) + "\n</svg>\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/smoke.yaml")
    p.add_argument("--data", default="outputs/datasets/smoke")
    p.add_argument("--tokenizer", default="outputs/tokenizers/smoke/tokenizer.json")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--until", type=int, default=200)
    p.add_argument("--ui", choices=["auto", "terminal"], default="terminal")
    p.add_argument("--resume")
    p.add_argument("--width", type=int, default=120)
    p.add_argument("--height", type=int, default=36)
    p.add_argument("--resize-to")
    p.add_argument("--interrupt-after", type=int)
    capture(p.parse_args())


if __name__ == "__main__":
    main()
