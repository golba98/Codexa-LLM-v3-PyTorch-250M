"""Reconnectable read-only follower using the original terminal dashboard."""

import argparse
import json
import time
from pathlib import Path
from typing import Any, BinaryIO

from codexa.training.callbacks.telemetry import TrainingEvent

from . import choose_terminal


def read_events(stream: BinaryIO) -> list[TrainingEvent]:
    events: list[TrainingEvent] = []
    while True:
        offset = stream.tell()
        line = stream.readline()
        if not line or not line.endswith(b"\n"):
            stream.seek(offset)
            return events
        try:
            row: dict[str, Any] = json.loads(line)
            row.pop("event_id", None)
            events.append(TrainingEvent(**row))
        except (ValueError, TypeError):
            continue


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    viewer, reason = choose_terminal("terminal")
    if viewer is None:
        raise RuntimeError(reason)
    receipt = args.run / "viewer.json"
    import os

    try:
        while not (args.run / "events.jsonl").exists():
            time.sleep(0.2)
        with (args.run / "events.jsonl").open("rb") as stream:
            while True:
                events = read_events(stream)
                for event in events:
                    if event.event == "started":
                        event.data["metadata"] = event.data.get("metadata", {}) | {
                            "viewer_only": True
                        }
                for event in events[:-1]:
                    viewer.state.update(event)
                if events:
                    # One original renderer snapshot per batch, including replay on reconnect.
                    viewer.on_event(events[-1])
                    receipt.write_text(
                        json.dumps(
                            {
                                "pid": os.getpid(),
                                "renderer_pid": viewer.process.pid if viewer.process else None,
                                "active": viewer.active,
                                "run_id": events[-1].run_id,
                                "session_id": events[-1].session_id,
                                "step": events[-1].step,
                                "event": events[-1].event,
                                "tokens": events[-1].tokens,
                            }
                        )
                    )
                if viewer.failed:
                    raise RuntimeError("Original terminal renderer failed")
                time.sleep(0.25)
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
