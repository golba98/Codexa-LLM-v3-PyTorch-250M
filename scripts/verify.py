"""Bounded local CI equivalent; writes only ignored verification output."""

import json
import platform
import subprocess
import sys
from pathlib import Path

from codexa.repository import check_repository, source_identity


def main() -> None:
    check_repository()
    commands = [
        [sys.executable, "-m", "ruff", "check", "."],
        [sys.executable, "-m", "ruff", "format", "--check", "."],
        [sys.executable, "-m", "mypy", "codexa"],
        [sys.executable, "-m", "pytest", "-q"],
    ]
    for cmd in commands:
        subprocess.run(cmd, check=True)
    report = {
        "status": "passed",
        "source": source_identity(),
        "python": platform.python_version(),
        "commands": commands,
    }
    path = Path("outputs/logs/verification.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
