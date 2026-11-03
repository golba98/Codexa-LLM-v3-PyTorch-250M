import io
import json
import os
import socket
import time
from pathlib import Path

import pytest
import torch

from codexa.config import Config
from codexa.training.callbacks.telemetry import TrainingEvent
from codexa.training.terminal import TerminalDashboard, choose_terminal
from codexa.training.terminal.state import TerminalState
from codexa.training.trainer import Trainer


def event(kind, step=0, **data):
    return TrainingEvent(
        1,
        "test-run",
        "session",
        step,
        kind,
        step,
        step * 128,
        "2026-11-03T10:00:00+00:00",
        float(step),
        data,
    )


def test_state_metrics_and_lifecycle():
    s = TerminalState()
    s.update(
        event(
            "started", metadata={"training": {"steps": 100}}, resumed=True, best_validation_loss=2.5
        )
    )
    for step in range(1, 70):
        s.update(event("step_completed", step, loss=1 / step, seconds=0.1, lr=0.001))
    s.update(event("validation_started", 69, total_batches=4))
    assert s.status == "validating"
    s.update(event("validation_progress", 69, completed_batches=3, total_batches=4))
    assert s.validation["completed_batches"] == 3
    s.update(event("validation_completed", 69, validation_loss=2.0, perplexity=7.389))
    s.update(event("checkpoint_started", 69))
    assert s.status == "checkpointing"
    s.update(
        event(
            "checkpoint_saved",
            69,
            identifier="latest.pt",
            size_bytes=42,
            verification_status="atomic_save_only",
        )
    )
    assert s.checkpoint["step"] == 69 and s.best == 2.0
    assert len(s.losses) == 60 and len(s.recent) <= 6
    assert s.snapshot()["smoothed_loss"] == pytest.approx(sum(s.losses) / 60)
    assert s.snapshot()["eta"] == pytest.approx(31)
    s.update(event("interrupted", 69))
    assert s.snapshot()["eta"] is None and s.resumed
    assert json.loads(json.dumps(s.snapshot()))["status"] == "interrupted"


def test_unknown_work_and_headless(monkeypatch):
    from codexa.training import terminal

    s = TerminalState().snapshot()
    assert s["eta"] is None and not s["metadata"]
    assert choose_terminal("none", io.StringIO()) == (None, None)
    assert choose_terminal("auto", io.StringIO()) == (None, None)
    assert "TTY" in choose_terminal("terminal", io.StringIO())[1]

    class TTY(io.StringIO):
        def isatty(self):
            return True

    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setattr(terminal.importlib.util, "find_spec", lambda _: None)
    assert "Rich" in choose_terminal("terminal", TTY())[1]
    with pytest.raises(ValueError):
        choose_terminal("invalid")


@pytest.mark.parametrize("size", [(120, 36), (96, 28), (60, 24), (40, 20), (30, 12)])
def test_render_sizes(size):
    pytest.importorskip("rich")
    from rich.console import Console

    from codexa.training.terminal.render import frame

    s = TerminalState()
    s.update(
        event(
            "started",
            metadata={"parameters": 133440, "training": {"steps": 200}, "model": {"context": 64}},
        )
    )
    s.update(event("step_completed", 1, loss=3.25, lr=0.001, sequence_length=64))
    width, height = size
    output = io.StringIO()
    c = Console(file=output, width=width, height=height, color_system=None)
    c.print(frame(s.snapshot(), width, height))
    lines = output.getvalue().splitlines()
    assert "3.250" in output.getvalue() and "╭" in output.getvalue()
    assert len(lines) <= height and max(map(len, lines)) <= width
    assert "Unavailable" in output.getvalue() if width >= 86 else "compact" in output.getvalue()
    assert "0.0%" not in output.getvalue()  # no invented GPU utilization


def test_nonblocking_transport_and_rate_limit(monkeypatch):
    view = TerminalDashboard(io.StringIO())
    a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
    a.setblocking(False)
    a.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)
    view.channel = a
    try:
        start = time.monotonic()
        for _ in range(1000):
            view.send()
        assert time.monotonic() - start < 1
    finally:
        a.close()
        b.close()

    class Process:
        def poll(self):
            return None

    view.process = Process()
    calls = []
    monkeypatch.setattr(view, "send", lambda: calls.append(True) or True)
    for _ in range(100):
        view.on_event(event("step_completed", 1, loss=2.0))
    assert len(calls) == 1


def test_trainer_callback_resume_and_failure(prepared, tmp_path, monkeypatch):
    from codexa.training import terminal

    data, tok, out = prepared
    plain = Trainer(Config(), data, tok, out, "cpu")
    plain_records = plain.train(until=4, handle_signals=False, ui="none")

    class Capture:
        active = True

        def __init__(self):
            self.state = TerminalState()

        def on_event(self, e):
            self.state.update(e)

        def close(self):
            self.active = False

    capture = Capture()
    monkeypatch.setattr(terminal, "choose_terminal", lambda _: (capture, None))
    shown = Trainer(Config(), data, tok, str(tmp_path / "shown"), "cpu")
    records = shown.train(until=4, handle_signals=False, ui="terminal")
    assert [r["loss"] for r in records] == [r["loss"] for r in plain_records]
    for key, value in shown.model.state_dict().items():
        assert torch.equal(value, plain.model.state_dict()[key])
    assert capture.state.metrics["loss"] == records[-1]["loss"]
    assert capture.state.validation["validation_loss"] == records[-1]["validation_loss"]
    assert capture.state.checkpoint["identifier"] == "latest.pt" and not capture.active
    resumed = Trainer(Config(), data, tok, str(tmp_path / "resumed"), "cpu")
    resumed.resume(str(tmp_path / "shown/latest.pt"))
    assert resumed.best_validation_loss == shown.best_validation_loss
    resumed.train(until=5, handle_signals=False, ui="terminal")
    assert capture.state.resumed and resumed.step == 5

    class Failed(Capture):
        active = False

        def on_event(self, e):
            raise RuntimeError("broken renderer")

    broken = Failed()
    monkeypatch.setattr(terminal, "choose_terminal", lambda _: (broken, None))
    failed = Trainer(Config(), data, tok, str(tmp_path / "failed"), "cpu")
    with pytest.warns(UserWarning, match="Callback disabled"):
        failed.train(until=3, handle_signals=False, ui="terminal")
    assert failed.step == 3 and (tmp_path / "failed/latest.pt").is_file()
    assert '"event": "warning"' in (tmp_path / "failed/events.jsonl").read_text()


def test_signal_with_terminal_callback(prepared, monkeypatch):
    import signal

    from codexa.training import terminal

    data, tok, out = prepared
    s = TerminalState()

    class Capture:
        active = True

        def on_event(self, e):
            s.update(e)

        def close(self):
            pass

    monkeypatch.setattr(terminal, "choose_terminal", lambda _: (Capture(), None))
    t = Trainer(Config(), data, tok, out, "cpu")

    def stop(gradient):
        os.kill(os.getpid(), signal.SIGINT)
        return gradient

    t.model.embedding.weight.register_hook(stop)
    t.train(until=4, ui="terminal")
    assert t.step == 1 and s.status == "interrupted" and s.checkpoint


def test_host_probe_and_missing_gpu(tmp_path):
    from codexa.training.callbacks.telemetry import EventBus

    bus = EventBus(tmp_path, {"device": "cpu"})
    try:
        host = bus.probe_host()
        bus.emit("step_completed", 1, 128, loss=1.0)
    finally:
        bus.close()
    row = json.loads((tmp_path / "events.jsonl").read_text())
    assert row["data"]["gpu_system"] is None
    if Path("/proc/meminfo").exists():
        assert host["available_ram_bytes"] > 0 and host["process_rss_bytes"] > 0


@pytest.mark.parametrize("size,mode", [(("120", "36"), "auto"), (("60", "24"), "terminal")])
def test_cli_pty_acceptance(prepared, tmp_path, size, mode, monkeypatch):
    import subprocess
    import sys

    pytest.importorskip("rich")
    pytest.importorskip("pyte")
    data, tok, _ = prepared
    monkeypatch.setenv("COLUMNS", "80")
    monkeypatch.setenv("LINES", "24")
    output = tmp_path / "pty"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/terminal_acceptance.py",
            "--data",
            data,
            "--tokenizer",
            tok,
            "--output",
            str(output),
            "--until",
            "80",
            "--width",
            size[0],
            "--height",
            size[1],
            "--ui",
            mode,
        ],
        check=False,
        timeout=40,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((output / "acceptance.json").read_text())
    assert report["last_step"] == 80 and report["last_event"] == "paused"
    assert report["cursor_restored"] and report["termios_unchanged"]
    assert report["validation_recorded"] and report["checkpoint_saved"]
    assert len(report["rendered_steps"]) >= 2


def test_dead_renderer_training_continues(prepared, tmp_path, monkeypatch):
    import pty

    from codexa.training import terminal

    pytest.importorskip("rich")
    data, tok, out = prepared
    master, slave = pty.openpty()
    stream = os.fdopen(os.dup(slave), "w")
    view = TerminalDashboard(stream)

    class Kill:
        def on_event(self, e):
            if e.event == "step_completed" and view.process and view.process.poll() is None:
                view.process.kill()
                view.process.wait(timeout=3)

    monkeypatch.setattr(terminal, "choose_terminal", lambda _: (view, None))
    import codexa.training.trainer as trainer_module

    def blocked_terminal(*args, **kwargs):
        raise OSError("Terminal output must not block or fail training")

    monkeypatch.setattr(trainer_module, "print", blocked_terminal, raising=False)
    try:
        t = Trainer(Config(), data, tok, out, "cpu", callbacks=(Kill(),))
        t.train(until=4, ui="terminal", handle_signals=False)
        assert t.step == 4 and not view.active and (Path(out) / "latest.pt").is_file()
        assert '"event": "warning"' in (Path(out) / "events.jsonl").read_text()
        assert b"\x1b[?25h\x1b[?1049l" in os.read(master, 4096)
    finally:
        stream.close()
        os.close(master)
        os.close(slave)


def test_frame_capture_chunk_independence():
    import runpy

    pytest.importorskip("pyte")
    import pyte

    capture = runpy.run_path("scripts/terminal_acceptance.py")["FrameCapture"]
    raw = "\x1b[?1049h\x1b[HCODEXA 0 / 200  0.0%\r\nTraining\x1b[HCODEXA 80 / 200  40.0%\r\nTraining\x1b[?1049l"
    for chunks in ([raw], list(raw)):
        screen = capture(60, 24)
        stream = pyte.Stream(screen)
        for chunk in chunks:
            stream.feed(chunk)
        assert len(screen.frames) == 2
        assert "0 / 200" in screen.frames[0]["text"]
        assert "80 / 200" in screen.frames[1]["text"]


def test_worker_preserves_queued_startup(monkeypatch):
    import fcntl
    import pty
    import runpy
    import select
    import struct
    import subprocess
    import sys
    import termios

    pytest.importorskip("rich")
    pyte = pytest.importorskip("pyte")
    capture = runpy.run_path("scripts/terminal_acceptance.py")["FrameCapture"]
    parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 24, 60, 0, 0))
    state = TerminalState()
    state.update(event("started", metadata={"training": {"steps": 200}}))
    parent.send(json.dumps({"snapshot": state.snapshot(), "close": False}).encode())
    state.update(event("step_completed", 80, loss=1.25))
    state.update(event("paused", 80))
    parent.send(json.dumps({"snapshot": state.snapshot(), "close": True}).encode())
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setenv("COLUMNS", "80")
    monkeypatch.setenv("LINES", "24")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "codexa.training.terminal.worker",
            str(child.fileno()),
            str(os.getpid()),
        ],
        pass_fds=(child.fileno(),),
        stdout=slave,
        stderr=slave,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    child.close()
    raw = bytearray()
    deadline = time.monotonic() + 10
    try:
        while time.monotonic() < deadline:
            ready, _, _ = select.select([master], [], [], 0.1)
            if ready:
                raw.extend(os.read(master, 65536))
            elif process.poll() is not None:
                break
        assert process.wait(timeout=1) == 0
        screen = capture(60, 24)
        pyte.Stream(screen).feed(raw.decode())
        assert any("0 / 200" in f["text"] for f in screen.frames)
        assert any("80 / 200" in f["text"] for f in screen.frames)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=2)
        parent.close()
        os.close(master)
        os.close(slave)
