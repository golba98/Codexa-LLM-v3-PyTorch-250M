"""Verify that complete blinded documents are exactly the model's retained token input."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from codexa.data.language_protocol import SUPPORTED_VERSIONS
from codexa.data.production import atomic, digest, verify
from codexa.data.v2_storage import Storage
from codexa.tokenizer.bpe import CodexaTokenizer


def compare_documents(
    snapshot: Path,
    manifest: dict[str, Any],
    selected: dict[str, str],
    tokenizer: Any,
    split: str = "train",
) -> int:
    checked = set()
    with (snapshot / "documents.jsonl").open() as listing:
        rows = (r for r in map(json.loads, listing) if r["split"] == split)
        for entry in manifest["splits"][split]:
            data = np.memmap(snapshot / entry["file"], dtype="<u2", mode="r")
            offsets = np.memmap(snapshot / entry["index"], dtype="<u8", mode="r")
            for i, offset in enumerate(offsets):
                row = next(rows)
                if row["id"] not in selected:
                    continue
                end = int(offsets[i + 1]) if i + 1 < len(offsets) else len(data)
                expected = [2, *tokenizer.encode(selected[row["id"]]), 3]
                if len(expected) != row["tokens"] or not np.array_equal(
                    data[int(offset) : end], expected
                ):
                    raise ValueError(
                        "Blinded document text differs from actual retained training tokens"
                    )
                checked.add(row["id"])
        if next(rows, None) is not None or checked != set(selected):
            raise ValueError("Sampled document identity is missing or offset mapping differs")
    return len(checked)


def check_census(root: Path, snapshot: Path, census: Path, tokenizer: Path) -> dict[str, Any]:
    """Prove all admission candidates, including held-out records, match native input."""
    Storage.load(root).check(64 * 1024**2)
    manifest = verify(snapshot)
    if manifest["tokenizer_hash"] != digest(tokenizer):
        raise ValueError("Frozen tokenizer mismatch")
    protocol = json.loads((census / "protocol.json").read_text())
    if protocol["parent_manifest_sha256"] != digest(snapshot / "manifest.json"):
        raise ValueError("Census belongs to another snapshot")
    for name, sha in json.loads((census / "packet-integrity.json").read_text()).items():
        if Path(name).name != name or digest(census / name) != sha:
            raise ValueError("Frozen census changed")
    visible = {
        row["id"]: row for row in map(json.loads, (census / "blind.jsonl").read_text().splitlines())
    }
    selected: dict[str, dict[str, str]] = {split: {} for split in manifest["splits"]}
    for row in map(json.loads, (census / "mapping.jsonl").read_text().splitlines()):
        packet = visible.pop(row["id"])
        bucket = selected[row["split"]]
        if packet["coverage"] != "full document" or row["document_id"] in bucket:
            raise ValueError("Incomplete or duplicate census document")
        bucket[row["document_id"]] = packet["text"]
    if visible:
        raise ValueError("Unmapped census documents")
    encoder = CodexaTokenizer.load(tokenizer)
    counts = {
        split: compare_documents(snapshot, manifest, bucket, encoder, split)
        for split, bucket in selected.items()
    }
    return {
        "status": "passed",
        "manifest_sha256": digest(snapshot / "manifest.json"),
        "tokenizer_sha256": digest(tokenizer),
        "packet_integrity_sha256": digest(census / "packet-integrity.json"),
        "documents": sum(counts.values()),
        "split_documents": counts,
        "verifier_sha256": digest(Path(__file__)),
        "scope": "Every full admission candidate re-encoded and compared to its exact retained BOS/content/EOS sequence in its original split.",
    }


def review_directories(review: Path) -> tuple[Path, ...]:
    """Keep the historical mathematics extension mandatory for its protocol."""
    protocol = json.loads((review / "protocol.json").read_text())
    if protocol.get("version") in SUPPORTED_VERSIONS:
        from codexa.data.language_protocol import validate_authorization, validate_specification

        validate_specification(protocol)
        authorization = Path(protocol["authorization_path"])
        if digest(authorization) != protocol["authorization_sha256"]:
            raise ValueError("Language protocol authorization changed")
        validate_authorization(authorization, protocol)
        return (review,)
    from codexa.data.quality_protocol import validate_specification

    validate_specification(protocol)
    return (review, review / "extended")


def check(root: Path, snapshot: Path, review: Path, tokenizer: Path) -> dict[str, Any]:
    Storage.load(root).check(64 * 1024**2)
    manifest = verify(snapshot)
    if manifest["tokenizer_hash"] != digest(tokenizer):
        raise ValueError("Frozen tokenizer mismatch")
    encoder = CodexaTokenizer.load(tokenizer)
    arrays = [
        np.memmap(snapshot / e["file"], dtype="<u2", mode="r") for e in manifest["splits"]["train"]
    ]
    ends = np.cumsum([len(a) for a in arrays])
    total = int(ends[-1])
    selected: dict[str, str] = {}
    packets = []
    windows = 0
    positions: set[int] = set()
    for directory in review_directories(review):
        protocol = json.loads((directory / "protocol.json").read_text())
        if protocol["manifest_sha256"] != digest(snapshot / "manifest.json"):
            raise ValueError("Review belongs to another snapshot")
        integrity = json.loads((directory / "packet-integrity.json").read_text())
        for name, sha in integrity.items():
            if Path(name).name != name or digest(directory / name) != sha:
                raise ValueError("Frozen packet changed")
        rendering = json.loads((directory / "rendering.json").read_text())
        if rendering["rendered_blind_sha256"] != digest(
            directory / "rendered-blind.jsonl"
        ) or rendering["original_blind_sha256"] != digest(directory / "blind.jsonl"):
            raise ValueError("Rendered packet changed")
        visible_rows = list(
            map(json.loads, (directory / "rendered-blind.jsonl").read_text().splitlines())
        )
        mapping = list(map(json.loads, (directory / "mapping.jsonl").read_text().splitlines()))
        visible = {r["id"]: r for r in visible_rows}
        if (
            len(visible) != len(visible_rows)
            or set(visible) != {r["id"] for r in mapping}
            or len(mapping) != len(visible)
        ):
            raise ValueError("Review rendering identities differ from mapping")
        for row in mapping:
            if row["kind"] == "window":
                position = row["position"]
                if (
                    type(position) is not int
                    or not 0 <= position < total
                    or position in positions
                    or total < 1024
                ):
                    raise ValueError("Invalid or duplicate sampled token position")
                if protocol.get("version") in SUPPORTED_VERSIONS and position > total - 1024:
                    raise ValueError("Language window position exceeds the uniform eligible range")
                positions.add(position)
                start = min(position, total - 1024)
                ids: list[int] = []
                while len(ids) < 1024:
                    i = int(np.searchsorted(ends, start, side="right"))
                    offset = start - (int(ends[i - 1]) if i else 0)
                    part = arrays[i][offset : offset + 1024 - len(ids)].tolist()
                    ids.extend(part)
                    start += len(part)
                if encoder.decode(ids, skip_special_tokens=False) != visible[row["id"]]["text"]:
                    raise ValueError(
                        "Rendered review window differs from actual retained training tokens"
                    )
                windows += 1
                continue
            if row["document_id"] in selected or visible[row["id"]]["coverage"] != "full document":
                raise ValueError("Overlapping panels or incomplete document rendering")
            selected[row["document_id"]] = visible[row["id"]]["text"]
        packets.append(
            {
                "packet_integrity_sha256": digest(directory / "packet-integrity.json"),
                "rendered_blind_sha256": digest(directory / "rendered-blind.jsonl"),
            }
        )
    count = compare_documents(snapshot, manifest, selected, encoder)
    return {
        "status": "passed",
        "manifest_sha256": digest(snapshot / "manifest.json"),
        "tokenizer_sha256": digest(tokenizer),
        "documents": count,
        "windows": windows,
        "packets": packets,
        "verifier_sha256": digest(Path(__file__)),
        "scope": "Every full sampled document and rendered 1024-token window compared independently against actual retained BOS/content/EOS token input.",
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "snapshot", "tokenizer", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--review", type=Path)
    mode.add_argument("--census", type=Path)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    result = (
        check_census(a.root, a.snapshot, a.census, a.tokenizer)
        if a.census is not None
        else check(a.root, a.snapshot, a.review, a.tokenizer)
    )
    atomic(a.output, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "documents": result["documents"],
                "windows": result.get("windows", 0),
            }
        )
    )


if __name__ == "__main__":
    main()
