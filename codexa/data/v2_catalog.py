"""Pinned source discovery and explicit bounded acquisition; never trains."""

import argparse
import hashlib
import json
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import yaml

from codexa.data.production import atomic, digest
from codexa.data.v2_acquire import acquire, acquire_xet
from codexa.data.v2_storage import GIB, Storage


def catalog(config: Path, destination: Path) -> dict[str, Any]:
    """Fetch file metadata only, then pin every selected LFS payload by SHA256."""
    policy = yaml.safe_load(config.read_text())
    entries = []
    for source in policy["sources"]:
        revision = source["revision"]
        if len(revision) != 40:
            raise ValueError("A full immutable revision is required")
        url: str | None = (
            f"https://huggingface.co/api/datasets/{source['repo']}/tree/{revision}?recursive=true&limit=1000"
        )
        files = []
        while url:
            with urllib.request.urlopen(url, timeout=90) as response:
                files.extend(json.load(response))
                link = response.headers.get("Link", "")
            url = None
            for part in link.split(","):
                if 'rel="next"' in part:
                    candidate = part[part.index("<") + 1 : part.index(">")]
                    if urllib.parse.urlsplit(candidate).hostname != "huggingface.co":
                        raise ValueError("Unexpected metadata pagination host")
                    url = candidate
        selected = []
        for file in files:
            path = file["path"]
            directories = source.get("directories", [source.get("directory", "")])
            eligible = path in source.get("files", []) or any(
                d and path.startswith(d + "/") and path.endswith(".parquet") for d in directories
            )
            if file["type"] != "file" or not eligible or path in source.get("exclude", []):
                continue
            lfs = file.get("lfs", {})
            sha = lfs.get("oid", "").removeprefix("sha256:")
            if len(sha) != 64:
                raise ValueError("Source payload has no pinned SHA256: " + path)
            selected.append(source | {"path": path, "bytes": file["size"], "sha256": sha})
        selected.sort(
            key=lambda x: hashlib.sha256((str(policy["seed"]) + x["path"]).encode()).hexdigest()
        )
        entries.extend(selected)
    result = {"config": policy, "sources": entries}
    if destination.exists() and json.loads(destination.read_text()) != result:
        raise ValueError("Registry is immutable; use a fresh preparation root for changed sources")
    atomic(destination, result)
    return result


def select(storage: Storage, limit_gib: float, per_source: int) -> dict[str, Any]:
    """Deterministic explicit batch. Counts already retained and partial payloads."""
    if not 0 < limit_gib * GIB <= storage.raw_maximum or per_source < 1:
        raise ValueError("Explicit batch limit must be within the configured raw ceiling")
    registry = json.loads((storage.root / "source-registry.json").read_text())["sources"]
    admissions = json.loads((storage.root / "source-admission.json").read_text())["sources"]
    usage = Storage(storage.root / "raw").usage()
    chosen = []
    counts: dict[str, int] = {}
    for spec in registry:
        approval = admissions.get(spec["name"], {})
        if (
            approval.get("status") != "admitted"
            or not approval.get("target_share", spec["target_share"])
            or "valid" in spec["path"]
        ):
            continue
        if approval.get("path_prefixes") and not any(
            spec["path"].startswith(p) for p in approval["path_prefixes"]
        ):
            continue
        name = spec["name"] + "--" + spec["path"].replace("/", "--")
        path = storage.root / "raw" / name
        if path.exists() or counts.get(spec["name"], 0) >= per_source:
            continue
        partial = path.with_name(name + ".partial")
        additional = spec["bytes"] - (partial.stat().st_size if partial.exists() else 0)
        if usage + additional > limit_gib * GIB:
            continue
        usage += additional
        chosen.append(spec)
        counts[spec["name"]] = counts.get(spec["name"], 0) + 1
    return {"sources": chosen, "maximum_raw_bytes": int(limit_gib * GIB), "projected_bytes": usage}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("catalog", "select", "acquire-batch", "profile", "extend-registry")
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/data/v2_sources.yaml"))
    parser.add_argument("--batch", type=Path)
    parser.add_argument("--limit-gib", type=float, default=8)
    parser.add_argument("--per-source", type=int, default=1)
    parser.add_argument("--transport", choices=("http", "xet"), default="http")
    args = parser.parse_args(argv)
    storage = Storage.load(args.root)
    storage.check()
    if args.action == "catalog":
        result = catalog(args.config, args.root / "source-registry.json")
        print(
            json.dumps(
                {
                    "files": len(result["sources"]),
                    "registry_sha256": digest(args.root / "source-registry.json"),
                }
            )
        )
    elif args.action == "select":
        if args.batch is None or args.batch.exists():
            parser.error("A new --batch file is required")
        atomic(args.batch, select(storage, args.limit_gib, args.per_source))
    elif args.action == "acquire-batch":
        if args.batch is None:
            parser.error("--batch required; no automatic full-corpus acquisition")
        specs = json.loads(args.batch.read_text())["sources"]
        registry = json.loads((args.root / "source-registry.json").read_text())["sources"]
        for spec in specs:
            # Pilot manifests may predate mixture changes; source identity must match.
            if not any(
                all(
                    spec.get(k) == r.get(k)
                    for k in ("name", "repo", "revision", "path", "sha256", "bytes")
                )
                for r in registry
            ):
                raise ValueError("Batch contains an unregistered payload")
            (acquire_xet if args.transport == "xet" else acquire)(storage, spec)
    elif args.action == "extend-registry":
        from codexa.data.v2_index import Index
        from codexa.data.v2_registry import extend

        if args.batch is None:
            parser.error("--batch must name an explicitly reviewed additional source registry")
        historical = Path(
            json.loads((args.root / "reports/historical-index.json").read_text())["input"]
        )
        index = Index(storage, historical)
        try:
            print(json.dumps(extend(index, json.loads(args.batch.read_text())), indent=2))
        finally:
            index.close()
    else:
        from codexa.data.v2_profile import profile

        for receipt in sorted((args.root / "raw").glob("*.receipt.json")):
            print(json.dumps(profile(storage, json.loads(receipt.read_text()))), flush=True)


if __name__ == "__main__":
    main()
