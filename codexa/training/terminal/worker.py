"""Renderer child. No PyTorch imports, network listeners, input grabs or checkpoint reads."""

import json
import os
import select
import signal
import socket
import sys
import time

from rich.console import Console
from rich.live import Live

from .render import frame


def main() -> None:
    channel = socket.socket(fileno=int(sys.argv[1]))
    channel.setblocking(False)
    parent = int(sys.argv[2])
    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    console = Console()
    snapshot = None
    received = time.monotonic()
    with (
        channel,
        Live(
            console=console,
            screen=True,
            auto_refresh=False,
            redirect_stdout=False,
            redirect_stderr=False,
            vertical_overflow="crop",
        ) as live,
    ):
        while not stopping and os.getppid() == parent:
            select.select([channel], [], [], 0.25)
            for _ in range(64):
                try:
                    packet = json.loads(channel.recv(65536))
                except BlockingIOError:
                    break
                first = snapshot is None
                snapshot = packet["snapshot"]
                received = time.monotonic()
                stopping = stopping or packet.get("close", False)
                if first:
                    # Startup is a real lifecycle state; don't coalesce it away
                    # when a very short run finishes before this child starts.
                    live.update(frame(snapshot, console.width, console.height), refresh=True)
            if snapshot is not None:
                view = snapshot.copy()
                if view["status"] not in ("completed", "paused", "interrupted", "failed"):
                    view["elapsed"] += time.monotonic() - received
                live.update(frame(view, console.width, console.height), refresh=True)
    if snapshot is not None:
        # A single durable terminal summary after leaving the alternate screen.
        console.print(
            f"Codexa {snapshot['status']} · step {snapshot['step']:,} · "
            f"{snapshot['tokens']:,} tokens · metrics and checkpoints retained",
            markup=False,
        )


if __name__ == "__main__":
    main()
