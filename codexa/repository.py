"""Source identity and limited, redacted repository-policy checks."""

import hashlib
import json
import re
import subprocess
import tarfile
from pathlib import Path
from typing import Any
from urllib.parse import unquote

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
        ("docs/datasets.md", False),
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
        ("docs/architecture.md", False),
        ("requirements-lock.txt", False),
        ("outputs/model.pt", True),
        ("reports/language/evidence/continuation-readiness-report.json", False),
        ("outputs/cache.sqlite", True),
        ("outputs/download.pdf", True),
        (".env", True),
        (".env.example", False),
    ]:
        result = subprocess.run(["git", "check-ignore", "--no-index", "-q", path], check=False)
        if (result.returncode == 0) != ignored:
            raise ValueError("Ignore policy mismatch: " + path)

    # Public metadata and local documentation references must be usable.
    import yaml

    proposal = json.loads(Path("release/github-metadata.json").read_text())
    topics = proposal["topics"]
    if len(topics) > 20 or any(not re.fullmatch(r"[a-z0-9-]{1,50}", t) for t in topics):
        raise ValueError("Invalid GitHub topics")
    for workflow in Path(".github/workflows").glob("*.yml"):
        if not yaml.safe_load(workflow.read_text()).get("jobs"):
            raise ValueError("Workflow lacks jobs")
    check_documentation()
    check_archive(Path("docs/archive/history.tar.gz"), Path("docs/archive/index.json"))
    check_archive(
        Path("reports/archive/research-2026-11-10.tar.gz"), Path("reports/archive/index.json")
    )


def check_archive(archive: Path, index: Path) -> None:
    """Check original member bytes, membership and publication safety without extraction."""
    records = json.loads(index.read_text())
    if isinstance(records, dict):
        records = records["records"]
    expected = {row["path"]: row["sha256"] for row in records}
    with tarfile.open(archive, "r:gz") as tar:
        files = [member for member in tar.getmembers() if member.isfile()]
        if len(files) != len(expected) or {m.name for m in files} != set(expected):
            raise ValueError("Archive membership mismatch: " + str(archive))
        for member in tar.getmembers():
            if (
                member.name.startswith("/")
                or ".." in Path(member.name).parts
                or member.issym()
                or member.islnk()
            ):
                raise ValueError("Unsafe archive member")
        for member in files:
            stream = tar.extractfile(member)
            assert stream is not None
            payload = stream.read()
            if hashlib.sha256(payload).hexdigest() != expected[member.name]:
                raise ValueError("Archive checksum mismatch: " + member.name)
            if secret_findings(payload.decode(errors="replace")):
                raise ValueError("Potential secret in archive member (redacted): " + member.name)


def visible_markdown(path: Path) -> str:
    return re.sub(r"(?ms)^```.*?^```[^\n]*", "", path.read_text())


def heading_anchors(path: Path) -> set[str]:
    text = visible_markdown(path)
    anchors = set(re.findall(r'<a\s+(?:id|name)=["\']([^"\']+)', text))
    counts: dict[str, int] = {}
    for heading in re.findall(r"(?m)^#{1,6}\s+(.+?)\s*#*\s*$", text):
        slug = re.sub(r"[^\w -]", "", heading.replace("`", "").lower()).replace(" ", "-")
        count = counts.get(slug, 0)
        anchors.add(slug + ("-" + str(count) if count else ""))
        counts[slug] = count + 1
    return anchors


def check_documentation() -> None:
    """Validate active Markdown links, images, reference targets and local anchors."""
    names = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"], text=True
    ).splitlines()
    for name in names:
        doc = Path(name)
        if doc.suffix != ".md" or not doc.is_file():
            continue
        content = visible_markdown(doc)
        targets = re.findall(r"!?\[[^\]]*\]\((<[^>]+>|[^)]+)\)", content)
        targets += re.findall(r"(?m)^\s{0,3}\[[^\]]+\]:\s*(<[^>]+>|\S+)", content)
        for target in targets:
            target = (
                target[1 : target.index("> ") if "> " in target else target.index(">")]
                if target.startswith("<")
                else target.split(' "')[0].split(" '")[0]
            )
            if re.match(r"[a-zA-Z][a-zA-Z0-9+.-]*:", target):
                continue
            location, _, anchor = unquote(target).partition("#")
            path = doc.parent / location if location else doc
            if not path.exists():
                raise ValueError(f"Broken local link in {doc}: {target}")
            if anchor and path.suffix == ".md" and anchor not in heading_anchors(path):
                raise ValueError(f"Broken heading anchor in {doc}: {target}")
