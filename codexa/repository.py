"""Source identity and limited, redacted repository-policy checks."""

import hashlib
import subprocess
from pathlib import Path
from typing import Any

from codexa.release.artifacts import secret_findings


def source_identity() -> dict[str, Any]:
    paths = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"]
    ).split(b"\0")
    h = hashlib.sha256()
    for raw in sorted(set(paths)):
        if not raw:
            continue
        p = Path(raw.decode())
        if p.is_file():
            h.update(raw)
            h.update(p.read_bytes())
    return {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "digest": h.hexdigest(),
        "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"])),
    }


def check_repository() -> None:
    paths = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"], text=True
    ).splitlines()
    for name in paths:
        p = Path(name)
        if p.suffix in (
            ".pt",
            ".pth",
            ".ckpt",
            ".safetensors",
            ".bin",
            ".gguf",
            ".onnx",
        ) or name.startswith(("outputs/", ".venv/", ".agents/scratch/")):
            raise ValueError("Forbidden public artifact: " + name)
        if p.is_file() and secret_findings(p.read_text(errors="replace")):
            raise ValueError("Potential secret in " + name + " (redacted)")
    for path, ignored in [
        ("README.md", False),
        ("CHANGELOG.md", False),
        ("codexa/data/production.py", False),
        ("docs/DATASETS.md", False),
        (".agent-reports/report.md", True),
        ("docs/_drafts/plan.md", True),
        ("nested/CLAUDE.md", True),
        ("outputs/datasets/train.bin", True),
        ("outputs/tokenizers/tokenizer.json", True),
        ("AGENTS.md", False),
        ("nested/AGENTS.md", True),
        ("AGENTS.override.md", True),
        (".agents/scratch/task.md", True),
        (".agents/skills/dataset-audit/SKILL.md", False),
        ("docs/ARCHITECTURE.md", False),
        ("requirements-lock.txt", False),
        ("outputs/model.pt", True),
        (".env", True),
        (".env.example", False),
    ]:
        result = subprocess.run(["git", "check-ignore", "--no-index", "-q", path], check=False)
        if (result.returncode == 0) != ignored:
            raise ValueError("Ignore policy mismatch: " + path)

    # Public metadata and local documentation references must be usable.
    import json
    import re

    import yaml

    proposal = json.loads(Path("release/github-metadata.json").read_text())
    topics = proposal["topics"]
    if len(topics) > 20 or any(not re.fullmatch(r"[a-z0-9-]{1,50}", t) for t in topics):
        raise ValueError("Invalid GitHub topics")
    for workflow in Path(".github/workflows").glob("*.yml"):
        if not yaml.safe_load(workflow.read_text()).get("jobs"):
            raise ValueError("Workflow lacks jobs")
    for doc in [Path("README.md"), *Path("docs").rglob("*.md")]:
        for link in re.findall(r"\[[^\]]*\]\(([^)]+)\)", doc.read_text()):
            if (
                "://" not in link
                and not link.startswith("#")
                and not (doc.parent / link.split("#")[0]).exists()
            ):
                raise ValueError("Broken local link in " + str(doc))
