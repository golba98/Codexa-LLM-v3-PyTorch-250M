"""Preregister, draw and adjudicate a source-blinded continuation corpus review."""

import argparse
import bisect
import hashlib
import json
import random
import sqlite3
import zlib
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from codexa.data import language_protocol
from codexa.data.production import atomic, digest, verify
from codexa.data.quality_protocol import VERSION as STRICT_VERSION
from codexa.data.quality_protocol import adjudicate as strict_adjudicate
from codexa.data.quality_protocol import specification, validate_specification
from codexa.data.v2_index import Protection
from codexa.data.v2_storage import Storage
from codexa.tokenizer.bpe import CodexaTokenizer


def inspection_exclusions(
    censuses: list[Path],
) -> tuple[set[str], list[tuple[str, int]], list[dict[str, str]]]:
    """Bind all inspected census records, including incompletely judged candidates."""
    excluded_ids: set[str] = set()
    excluded_texts: list[tuple[str, int]] = []
    packets = []
    for census in censuses:
        integrity_path = census / "packet-integrity.json"
        integrity = json.loads(integrity_path.read_text())
        if not {"mapping.jsonl", "blind.jsonl"} <= integrity.keys():
            raise ValueError("Census packet lacks inspected identities or text")
        for name, sha in integrity.items():
            if digest(census / name) != sha:
                raise ValueError("Inspected census packet changed")
        mappings = [
            json.loads(line) for line in (census / "mapping.jsonl").read_text().splitlines()
        ]
        texts = [json.loads(line) for line in (census / "blind.jsonl").read_text().splitlines()]
        mapping_ids = [row["id"] for row in mappings]
        text_ids = [row["id"] for row in texts]
        if (
            len(set(mapping_ids)) != len(mapping_ids)
            or set(mapping_ids) != set(text_ids)
            or len(text_ids) != len(set(text_ids))
        ):
            raise ValueError("Inspected census identities do not match its text")
        excluded_ids.update(row["document_id"] for row in mappings)
        excluded_texts.extend((row["text"], 64) for row in texts)
        packets.append({"path": str(census), "integrity_sha256": digest(integrity_path)})
    return excluded_ids, excluded_texts, packets


def overlaps_inspected_window(
    start: int, length: int, intervals: list[tuple[int, int]], starts: list[int]
) -> bool:
    """Reject even a short overlap with an inspected document."""
    interval = bisect.bisect_left(starts, start + length) - 1
    return interval >= 0 and intervals[interval][1] > start


def require_known_census_exclusions(root: Path, supplied: list[Path]) -> None:
    """Fail closed when a known local curation packet was omitted."""
    known = {
        path.parent.resolve()
        for path in (root / "semantic-admission").glob("*/packet-integrity.json")
    }
    if not known <= {path.resolve() for path in supplied}:
        raise ValueError("Every known inspected census must be excluded from strict review")


def development_exclusions(
    roots: list[Path], database: Path
) -> tuple[set[str], list[tuple[str, int]], list[dict[str, str]]]:
    """Recover inspected origins even when old development text was only an excerpt."""
    origins: set[tuple[str, int]] = set()
    identities: set[str] = set()
    passages: list[tuple[str, int]] = []
    files: list[dict[str, str]] = []
    for root in roots:
        paths = set((root / "reports/profiles").glob("*.jsonl"))
        for development in (root / "reports").glob("development*"):
            if development.is_dir():
                paths.update(development.rglob("*.jsonl"))
        examples = root / "reports/cleaning-examples.jsonl"
        if examples.exists():
            paths.add(examples)
        for path in sorted(paths):
            files.append({"path": str(path.resolve()), "sha256": digest(path)})
            header = path.with_suffix(".json")
            source_sha = ""
            if header.exists():
                files.append({"path": str(header.resolve()), "sha256": digest(header)})
                source_sha = json.loads(header.read_text()).get("source_sha256", "")
            for row in map(json.loads, path.read_text().splitlines()):
                if row.get("document_id"):
                    identities.add(row["document_id"])
                sha = row.get("source_sha256", source_sha)
                if sha and row.get("ordinal") is not None:
                    origins.add((sha, int(row["ordinal"])))
                for key in ("before", "after", "end", "text", "raw_text"):
                    if row.get(key):
                        passages.append((row[key], 64))
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as db:
        for ident, metadata in db.execute("SELECT hash,metadata FROM docs WHERE role='new'"):
            meta = json.loads(metadata)
            if (meta.get("source_sha256", ""), meta.get("ordinal", -1)) in origins:
                identities.add(bytes(ident).hex())
    return identities, passages, files


def adjudicate(
    mapping: list[dict[str, Any]],
    ratings: list[dict[str, Any]],
    protocol: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if protocol is not None and protocol.get("version") in language_protocol.SUPPORTED_VERSIONS:
        return language_protocol.adjudicate(mapping, ratings, protocol)
    if protocol is not None and protocol.get("version") == STRICT_VERSION:
        return strict_adjudicate(mapping, ratings, protocol)
    if protocol is not None and protocol.get("version") is not None:
        raise ValueError("Unknown quality protocol version")
    by_id = {row["id"]: row for row in mapping}
    if len(by_id) != len(mapping):
        raise ValueError("Duplicate sample identity")
    expected = Counter("windows" if r["kind"] == "window" else r["source"] for r in mapping)
    if expected.get("windows") != 300 or any(
        n != 40 for name, n in expected.items() if name != "windows"
    ):
        raise ValueError("Review panel dimensions differ from preregistration")
    seen: set[str] = set()
    panels: dict[str, Counter[str]] = {}
    critical = []
    for row in ratings:
        ident = row["id"]
        if ident in seen or ident not in by_id or not row.get("reason"):
            raise ValueError("Invalid, duplicate or unexplained rating")
        seen.add(ident)
        if any(type(row.get(k)) is not bool for k in ("usable", "severe", "critical")):
            raise ValueError("Explicit boolean labels required")
        sample = by_id[ident]
        panel = "windows" if sample["kind"] == "window" else sample["source"]
        c = panels.setdefault(panel, Counter())
        c["reviewed"] += 1
        c["usable"] += row["usable"]
        c["severe"] += row["severe"]
        c["failed"] += not row["usable"]
        if row["critical"]:
            critical.append(ident)
    blockers = ["critical finding: " + x for x in critical]
    for name, total in expected.items():
        c = panels.get(name, Counter())
        required = 270 if name == "windows" else 34
        severe_limit = 15 if name == "windows" else 2
        if c["failed"] > total - required:
            blockers.append(name + ": usable-English threshold mathematically impossible")
        if c["severe"] > severe_limit:
            blockers.append(name + ": severe-defect threshold exceeded")
    complete = len(seen) == len(mapping)
    return {
        "status": "failed" if blockers else "passed" if complete else "incomplete",
        "blockers": blockers,
        "reviewed": len(seen),
        "unreviewed": len(mapping) - len(seen),
        "panels": {k: dict(v) for k, v in panels.items()},
        "reviewer": "source-blinded assistant; not independent human approval",
        "training_allowed": complete and not blockers,
    }


def prepare(
    root: Path,
    snapshot: Path,
    tokenizer: Path,
    output: Path,
    seed: int,
    excluded_reviews: list[Path] | None = None,
    strict_kind: str | None = None,
    attempt: int = 0,
    excluded_censuses: list[Path] | None = None,
    authorization: Path | None = None,
) -> None:
    Storage.load(root).check(64 * 1024**2)
    if strict_kind is not None:
        require_known_census_exclusions(root, excluded_censuses or [])
    if strict_kind == "language":
        if authorization is None:
            raise ValueError("Explicit language protocol authorization required")
        authorized = json.loads(authorization.read_text())
        for history in authorized.get("historical_roots", []):
            historical_root = Path(history)
            require_known_census_exclusions(historical_root, excluded_censuses or [])
            known = {
                p.parent.resolve()
                for p in (historical_root / "reviews").rglob("packet-integrity.json")
            }
            if not known <= {p.resolve() for p in excluded_reviews or []}:
                raise ValueError("Every known historical review must be excluded")
    policy_path = snapshot / "policy.json"
    if strict_kind is not None and policy_path.exists():
        policy = json.loads(policy_path.read_text())
        if policy.get("semantic_admission_decision_sha256"):
            raise ValueError(
                "Fresh approval cannot sample a positive semantic census: "
                "all admitted affected-source documents were previously inspected"
            )
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    excluded_ids: set[str] = set()
    excluded_texts: list[tuple[str, bool | int]] = []
    excluded_packets = []
    for review in excluded_reviews or []:
        integrity = json.loads((review / "packet-integrity.json").read_text())
        for name, sha in integrity.items():
            if digest(review / name) != sha:
                raise ValueError("Historical review packet changed")
        excluded_packets.append(
            {"path": str(review), "integrity_sha256": digest(review / "packet-integrity.json")}
        )
        for row in map(json.loads, (review / "mapping.jsonl").read_text().splitlines()):
            if row["kind"] == "document":
                excluded_ids.add(row["document_id"])
        for row in map(json.loads, (review / "blind.jsonl").read_text().splitlines()):
            excluded_texts.append((row["text"], 64))
    census_ids, census_texts, census_packets = inspection_exclusions(excluded_censuses or [])
    excluded_ids.update(census_ids)
    excluded_texts.extend(census_texts)
    development_files: list[dict[str, str]] = []
    if strict_kind == "language":
        development_ids, development_texts, development_files = development_exclusions(
            [root, *(Path(p) for p in authorized["historical_roots"])],
            root / "indexes/documents.sqlite",
        )
        excluded_ids.update(development_ids)
        excluded_texts.extend(development_texts)
    excluded_spans = Protection.build(excluded_texts) if excluded_texts else None
    protocol: dict[str, Any] = {
        "seed": seed,
        "snapshot": str(snapshot),
        "manifest_sha256": digest(snapshot / "manifest.json"),
        "windows": 300,
        "window_tokens": 1024,
        "documents_per_source": 40,
        "maximum_full_document_tokens": 8192,
        "usable_window_minimum": 270,
        "usable_per_source_minimum": 34,
        "severe_window_maximum": 15,
        "severe_per_source_maximum": 2,
        "critical_maximum": 0,
        "early_stop": False,
        "complete_required": True,
        "excluded_reviews": excluded_packets,
        "excluded_inspection_censuses": census_packets,
        "excluded_development_files": development_files,
        "exclusion_policy": "Exclude prior panel document identities and reject new windows with any verified historical-review 64-word/punctuation-token span.",
        "rubric": "Usable: readable connected English with meaningful topic progression. Fail: dominant boilerplate, incoherent extraction, disconnected fragments, or repeated text. Severe: extraction/loop damage dominates the reviewed text. Ignore cropping artifacts at random window edges. Critical: observed protected-evaluation leak or sensitive identifier; public biography names alone are not critical. Record mathematical correctness separately; language approval does not verify factuality.",
        "blinding": "Hide source names, URLs, filter decisions and scores; style can still reveal origins.",
        "reviewer": "assistant judgment, not independent human approval",
    }
    if strict_kind is not None:
        if strict_kind == "language":
            manifest = verify(snapshot)
            protocol.update(
                language_protocol.specification(
                    sorted(manifest["source_tokens"]),
                    attempt,
                    version=authorized.get("version", ""),
                )
            )
            assert authorization is not None
            protocol.update(
                authorization_path=str(authorization.resolve()),
                authorization_sha256=digest(authorization),
            )
            language_protocol.validate_authorization(authorization, protocol)
        else:
            protocol.update(specification(strict_kind, attempt))
        if seed != protocol["seed"]:
            raise ValueError("Sampling seed differs from preregistration")
        protocol["reviewer"] = "two independent source-blinded evaluation agents"
    atomic(output / "protocol.json", protocol)
    manifest = verify(snapshot)
    if digest(tokenizer) != manifest["tokenizer_hash"]:
        raise ValueError("Tokenizer mismatch")
    tok = CodexaTokenizer.load(tokenizer)
    rng = random.Random(seed)
    panel_size = protocol["documents_per_source"]
    candidate_size = panel_size + 256 if strict_kind else panel_size
    sampled: dict[str, list[dict[str, Any]]] = {}
    counts: Counter[str] = Counter()
    excluded_ranges: list[tuple[int, int]] = []
    source_totals: Counter[str] = Counter()
    source_token_totals: Counter[str] = Counter()
    source_excluded: Counter[str] = Counter()
    source_excluded_tokens: Counter[str] = Counter()
    train_position = 0
    with (snapshot / "documents.jsonl").open() as listing:
        for line in listing:
            row = json.loads(line)
            if row["split"] != "train":
                continue
            start = train_position
            train_position += row["tokens"]
            source_totals[row["source"]] += 1
            source_token_totals[row["source"]] += row["tokens"]
            if row["id"] in excluded_ids:
                excluded_ranges.append((start, train_position))
                source_excluded[row["source"]] += 1
                source_excluded_tokens[row["source"]] += row["tokens"]
                continue
            source = row["source"]
            if strict_kind is not None and source not in protocol["sources"]:
                continue
            counts[source] += 1
            bucket = sampled.setdefault(source, [])
            if len(bucket) < candidate_size:
                bucket.append(row)
            else:
                j = rng.randrange(counts[source])
                if j < candidate_size:
                    bucket[j] = row
    if strict_kind == "language":
        for source in protocol["sources"]:
            if (
                source_excluded[source] > source_totals[source] * 0.05
                or source_excluded_tokens[source] > source_token_totals[source] * 0.05
            ):
                raise ValueError("Inspection exclusions distort the language corpus distribution")
    db = sqlite3.connect(f"file:{root}/indexes/documents.sqlite?mode=ro", uri=True)
    packet = []
    excluded_document_spans = []
    passage_candidates: Counter[str] = Counter()
    passage_exclusions: Counter[str] = Counter()
    for source, rows in sorted(sampled.items()):
        if len(rows) < panel_size:
            raise ValueError("Insufficient unseen documents for source panel")
        if strict_kind:
            rng.shuffle(rows)
        selected = 0
        for row in rows:
            record = db.execute(
                "SELECT payload FROM docs WHERE hash=?", (bytes.fromhex(row["id"]),)
            ).fetchone()
            if record is None:
                raise ValueError("Missing retained document")
            text = zlib.decompress(record[0]).decode()
            passage_candidates[source] += 1
            if strict_kind and excluded_spans is not None and excluded_spans.matches(text):
                excluded_document_spans.append(row["id"])
                passage_exclusions[source] += 1
                continue
            ids = tok.encode(text)
            coverage = "full document"
            if strict_kind is None and len(ids) > 8192:
                starts = (0, len(ids) // 2 - 512, len(ids) - 1024)
                text = "\n[EXCERPT BREAK]\n".join(tok.decode(ids[s : s + 1024]) for s in starts)
                coverage = "beginning/middle/end excerpts; not full-document review"
            packet.append(
                {
                    "kind": "document",
                    "source": source,
                    "document_id": row["id"],
                    "coverage": coverage,
                    "text": text,
                }
            )
            selected += 1
            if selected == panel_size:
                break
        if selected != panel_size:
            raise ValueError("Insufficient independent documents in predetermined candidate pool")
        if (
            strict_kind == "language"
            and passage_exclusions[source] > passage_candidates[source] * 0.05
        ):
            raise ValueError("Inspected-passage exclusions distort the source document sample")
    db.close()
    entries = manifest["splits"]["train"]
    arrays = [np.memmap(snapshot / e["file"], dtype="<u2", mode="r") for e in entries]
    ends = np.cumsum([len(a) for a in arrays])
    total = int(ends[-1])
    if total < 1024:
        raise ValueError("Training stream is too small for the preregistered windows")
    if train_position != total:
        raise ValueError("Train document positions do not match token stream")
    excluded_starts = [start for start, _ in excluded_ranges]
    accepted_windows, attempts, positions = 0, 0, set()
    overlap_positions = []
    while accepted_windows < protocol["windows"]:
        attempts += 1
        if attempts > 100000:
            raise ValueError("Unable to sample independent windows within bounded attempts")
        position = (
            rng.randrange(total - 1024 + 1) if strict_kind == "language" else rng.randrange(total)
        )
        if position in positions:
            continue
        positions.add(position)
        start = min(position, total - 1024)
        if overlaps_inspected_window(start, 1024, excluded_ranges, excluded_starts):
            overlap_positions.append(position)
            continue
        remaining, parts = 1024, []
        while remaining:
            i = int(np.searchsorted(ends, start, side="right"))
            offset = start - (int(ends[i - 1]) if i else 0)
            part = arrays[i][offset : offset + remaining].tolist()
            parts.extend(part)
            remaining -= len(part)
            start += len(part)
        text = tok.decode(parts)
        if excluded_spans is not None and excluded_spans.matches(text):
            overlap_positions.append(position)
            continue
        accepted_windows += 1
        packet.append(
            {
                "kind": "window",
                "source": "hidden-mixture",
                "position": position,
                "coverage": "1024-token random window; may cross document boundaries",
                "text": text,
            }
        )
    rng.shuffle(packet)
    if strict_kind == "language" and len(overlap_positions) > attempts * 0.05:
        raise ValueError("Inspected-passage exclusions distort the training-window sample")
    hidden, visible = [], []
    for i, row in enumerate(packet):
        ident = f"R{i + 1:04d}"
        hidden.append({"id": ident, **{k: v for k, v in row.items() if k != "text"}})
        visible.append(
            {"id": ident, "kind": row["kind"], "coverage": row["coverage"], "text": row["text"]}
        )
    for name, rows in (("mapping.jsonl", hidden), ("blind.jsonl", visible)):
        with (output / name).open("x") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    atomic(
        output / "sampling-log.json",
        {
            "eligible_source_documents": dict(counts),
            "source_documents": dict(source_totals),
            "source_tokens": dict(source_token_totals),
            "excluded_source_documents": dict(source_excluded),
            "excluded_source_tokens": dict(source_excluded_tokens),
            "excluded_historical_document_ids": len(excluded_ids),
            "excluded_document_passage_matches": excluded_document_spans,
            "document_passage_candidates": dict(passage_candidates),
            "document_passage_exclusions": dict(passage_exclusions),
            "window_draws": attempts,
            "window_overlap_rejections": overlap_positions,
            "accepted_windows": accepted_windows,
        },
    )
    atomic(
        output / "packet-integrity.json",
        {
            p.name: digest(p)
            for p in (
                output / "protocol.json",
                output / "mapping.jsonl",
                output / "blind.jsonl",
                output / "sampling-log.json",
            )
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "render", "finalize"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--seed", type=int, default=2026110501)
    parser.add_argument("--exclude-review", type=Path, action="append", default=[])
    parser.add_argument("--exclude-census", type=Path, action="append", default=[])
    parser.add_argument("--strict-kind", choices=["primary", "extended", "language"])
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--attempt", type=int, default=0)
    args = parser.parse_args()
    if args.action == "prepare":
        if args.snapshot is None or args.tokenizer is None:
            parser.error("prepare requires --snapshot and --tokenizer")
        prepare(
            args.root,
            args.snapshot,
            args.tokenizer,
            args.output,
            args.seed,
            args.exclude_review,
            args.strict_kind,
            args.attempt,
            args.exclude_census,
            args.authorization,
        )
    elif args.action == "render":
        if args.tokenizer is None:
            parser.error("render requires --tokenizer")
        render(args.root, args.output, args.tokenizer)
    else:
        Storage.load(args.root).check()
        integrity = json.loads((args.output / "packet-integrity.json").read_text())
        for name, sha in integrity.items():
            if digest(args.output / name) != sha:
                raise ValueError("Review packet changed")
        ratings = [json.loads(s) for s in (args.output / "ratings.jsonl").read_text().splitlines()]
        mapping = [json.loads(s) for s in (args.output / "mapping.jsonl").read_text().splitlines()]
        protocol = json.loads((args.output / "protocol.json").read_text())
        if protocol.get("complete_required") and len(ratings) != len(mapping):
            raise ValueError("Complete every preregistered sample before finalizing this review")
        rendering = args.output / "rendering.json"
        if protocol.get("complete_required"):
            view = json.loads(rendering.read_text())
            if view["original_blind_sha256"] != digest(args.output / "blind.jsonl") or view[
                "rendered_blind_sha256"
            ] != digest(args.output / "rendered-blind.jsonl"):
                raise ValueError("Native-boundary review rendering changed")
        if protocol.get("version") in {STRICT_VERSION, *language_protocol.SUPPORTED_VERSIONS}:
            if protocol["version"] in language_protocol.SUPPORTED_VERSIONS:
                language_protocol.validate_authorization(
                    Path(protocol["authorization_path"]), protocol
                )
            else:
                validate_specification(protocol)
            from scripts.datasets.combine_quality_reviews import verify_reviewers

            verify_reviewers(args.output)
        result = adjudicate(mapping, ratings, protocol)
        result["ratings_sha256"] = hashlib.sha256(
            (args.output / "ratings.jsonl").read_bytes()
        ).hexdigest()
        result["protocol_sha256"] = integrity["protocol.json"]
        if rendering.exists():
            result["rendering_sha256"] = digest(rendering)
        path = args.output / "decision.json"
        if path.exists():
            raise FileExistsError(path)
        atomic(path, result)
        print(json.dumps(result, indent=2))


def render(root: Path, review: Path, tokenizer: Path) -> None:
    """Expose real BOS/EOS markers so packed documents are not mistaken for merged text."""
    Storage.load(root).check(16 * 1024**2)
    protocol = json.loads((review / "protocol.json").read_text())
    snapshot = Path(protocol["snapshot"])
    if digest(snapshot / "manifest.json") != protocol["manifest_sha256"]:
        raise ValueError("Sampled dataset identity changed")
    manifest = json.loads((snapshot / "manifest.json").read_text())
    if digest(tokenizer) != manifest["tokenizer_hash"]:
        raise ValueError("Frozen tokenizer changed")
    tok = CodexaTokenizer.load(tokenizer)
    positions = {
        r["id"]: r["position"]
        for r in map(json.loads, (review / "mapping.jsonl").read_text().splitlines())
        if r["kind"] == "window"
    }
    arrays = [
        np.memmap(snapshot / e["file"], dtype="<u2", mode="r") for e in manifest["splits"]["train"]
    ]
    ends = np.cumsum([len(a) for a in arrays])
    total = int(ends[-1])
    output = review / "rendered-blind.jsonl"
    with output.open("x") as f:
        for row in map(json.loads, (review / "blind.jsonl").read_text().splitlines()):
            if row["kind"] == "window":
                start = min(positions[row["id"]], total - 1024)
                ids: list[int] = []
                while len(ids) < 1024:
                    i = int(np.searchsorted(ends, start, side="right"))
                    offset = start - (int(ends[i - 1]) if i else 0)
                    part = arrays[i][offset : offset + 1024 - len(ids)].tolist()
                    ids.extend(part)
                    start += len(part)
                if tok.decode(ids) != row["text"]:
                    raise ValueError("Rendered window differs from the sampled tokens")
                row["text"] = tok.decode(ids, skip_special_tokens=False)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    atomic(
        review / "rendering.json",
        {
            "original_blind_sha256": digest(review / "blind.jsonl"),
            "rendered_blind_sha256": digest(output),
            "renderer_source_sha256": digest(Path(__file__)),
            "policy": "Only actual BOS/EOS IDs are made visible in packed windows. Source identities remain hidden. No mathematical text is edited. Judge prose within real document boundaries; genuine EOS transitions and window-edge crops are not extraction defects.",
        },
    )


if __name__ == "__main__":
    main()
