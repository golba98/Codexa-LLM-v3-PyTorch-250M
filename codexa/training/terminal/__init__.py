"""Optional Rich viewer in an isolated process; transport never waits for rendering."""

import importlib.util
import json
import os
import socket
import subprocess
import sys
import time
from typing import TextIO

from codexa.training.callbacks.telemetry import TrainingEvent

from .state import TerminalState


class TerminalDashboard:
    # Persist failures through EventBus, avoiding synchronous writes to a possibly broken TTY.
    silent_failure = True

    def __init__(self, stream: TextIO):
        self.stream = stream
        self.state = TerminalState()
        self.process: subprocess.Popen | None = None
        self.channel: socket.socket | None = None
        self.last_send = -1.0
        self.failed = False
        self.needs_restore = False

    @property
    def active(self) -> bool:
        return not self.failed and self.process is not None and self.process.poll() is None

    def start(self) -> None:
        parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
        parent.setblocking(False)
        parent.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 32768)
        try:
            self.process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "codexa.training.terminal.worker",
                    str(child.fileno()),
                    str(os.getpid()),
                ],
                pass_fds=(child.fileno(),),
                stdin=subprocess.DEVNULL,
                stdout=self.stream,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            self.channel = parent
        except Exception:
            parent.close()
            raise
        finally:
            child.close()

    def send(self, closing: bool = False) -> bool:
        assert self.channel is not None
        try:
            packet = json.dumps(
                {"snapshot": self.state.snapshot(), "close": closing}, allow_nan=False
            ).encode()
            if len(packet) > 60000:
                raise ValueError("Terminal snapshot exceeds bounded transport")
            self.channel.send(packet)
            return True
        except BlockingIOError:
            return False

    def on_event(self, event: TrainingEvent) -> None:
        try:
            self.state.update(event)
            if self.process is None:
                self.start()
            if not self.active:
                raise RuntimeError("Terminal renderer exited; continuing headless")
            now = time.monotonic()
            terminal = event.event in ("completed", "paused", "interrupted", "failed")
            if (terminal or now - self.last_send >= 0.25) and self.send():
                self.last_send = now
        except Exception:
            self.failed = True
            self.needs_restore = True
            if self.process and self.process.poll() is None:
                try:
                    self.process.terminate()
                except OSError:
                    pass
            # EventBus isolates this observer and persists a warning.
            raise

    def close(self) -> None:
        try:
            if self.process is not None and self.process.poll() is None:
                deadline = time.monotonic() + 1.5
                while (
                    not self.failed and not self.send(closing=True) and time.monotonic() < deadline
                ):
                    time.sleep(0.01)
                try:
                    self.process.wait(timeout=max(0.01, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    self.needs_restore = True
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=0.5)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                        self.process.wait(timeout=0.5)
        except (OSError, ValueError, subprocess.TimeoutExpired):
            pass
        finally:
            if self.process is not None and self.process.poll() is None:
                self.needs_restore = True
                try:
                    self.process.kill()
                    self.process.wait(timeout=0.5)
                except (OSError, subprocess.TimeoutExpired):
                    pass
            if self.channel:
                self.channel.close()
            if self.needs_restore:
                self.restore_terminal()

    def restore_terminal(self) -> None:
        """Best-effort recovery after renderer failure, using a separate nonblocking TTY fd."""
        try:
            fd = os.open(
                os.ttyname(self.stream.fileno()), os.O_WRONLY | os.O_NONBLOCK | os.O_NOCTTY
            )
            try:
                os.write(fd, b"\x1b[?25h\x1b[?1049l")
            finally:
                os.close(fd)
        except (OSError, ValueError, AttributeError, TypeError):
            pass


def choose_terminal(
    mode: str = "auto", stream: TextIO | None = None
) -> tuple[TerminalDashboard | None, str | None]:
    if mode not in ("auto", "terminal", "none"):
        raise ValueError("UI must be auto, terminal or none")
    stream = stream if stream is not None else sys.stdout
    if mode == "none":
        return None, None
    reason = None
    if not stream.isatty() or os.environ.get("TERM") == "dumb":
        reason = "Terminal UI requires a capable TTY; continuing headless"
    elif importlib.util.find_spec("rich") is None:
        reason = "Terminal UI requires optional Rich: install codexa[terminal]; continuing headless"
    elif os.name != "posix":
        reason = "Terminal UI transport currently requires POSIX; continuing headless"
    if reason:
        return None, reason if mode == "terminal" or stream.isatty() else None
    return TerminalDashboard(stream), None
