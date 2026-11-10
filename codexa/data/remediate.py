"""Versioned rejection-only remediation of an immutable dataset-v2 snapshot."""

import argparse
import ast
import copy
import fcntl
import hashlib
import json
import os
import re
import shutil
import sqlite3
import struct
import zlib
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from codexa.data.production import atomic, digest, verify
from codexa.data.quality_findings import findings
from codexa.data.v2_quality import canonical
from codexa.data.v2_storage import Storage

VERSION = "prose-math-remediation-6"


def essential_figure_dependency(text: str) -> bool:
    """Distinguish missing inputs from optional illustrations of supplied explanations."""
    visual_task = re.search(
        r"(?i)the graph shows|(?:find|calculate|determine|what|which|how many)[^.!?\n]{0,120}(?:shaded|marked|labelled|from the graph|from the diagram|in the figure)",
        text,
    )
    if not visual_task:
        return False
    # Counts of numbers or explanation words cannot establish that a missing
    # visual's essential inputs are supplied. Such dependencies remain unsafe.
    # A supplied line equation can answer a slope question, but cannot replace
    # a separate shaded region, figure or graph elsewhere in the document.
    line_task = re.search(
        r"(?i)the graph shows (?:the )?line\s+\$?y\s*=\s*-?\d*(?:\.\d+)?\s*x\s*[+-]\s*\d+\$?"
        r"\s*[.!?]\s*(?:find|calculate|determine|what is)[^.!?\n]{0,60}\b(?:slope|gradient|intercept)\b",
        text,
    )
    return not (
        line_task
        and not re.search(
            r"(?i)\b(?:shaded|marked|labelled|diagram|figure)\b|the graph shows",
            text[: line_task.start()] + text[line_task.end() :],
        )
    )


def prove_narrowing(old_source: str, new_source: str) -> None:
    """Permit decision reuse only when one predicate is provably narrowed by AND."""

    def function(source: str) -> ast.FunctionDef:
        return next(
            n
            for n in ast.parse(source).body
            if isinstance(n, ast.FunctionDef) and n.name == "rejection"
        )

    before, after = copy.deepcopy(function(old_source)), copy.deepcopy(function(new_source))

    def branch(fn: ast.FunctionDef) -> ast.If:
        matches = [
            n
            for n in fn.body
            if isinstance(n, ast.If)
            and any(
                isinstance(x, ast.Return)
                and isinstance(x.value, ast.Constant)
                and x.value.value == "missing_essential_figure"
                for x in n.body
            )
        ]
        if len(matches) != 1:
            raise ValueError("Expected one figure-dependency branch")
        return matches[0]

    old_if, new_if = branch(before), branch(after)
    if (
        not isinstance(new_if.test, ast.BoolOp)
        or not isinstance(new_if.test.op, ast.And)
        or len(new_if.test.values) != 2
        or ast.dump(new_if.test.values[0]) != ast.dump(old_if.test)
    ):
        raise ValueError("Cached decisions require a provably narrower figure predicate")
    old_if.test = new_if.test = ast.Constant(value=True)
    if ast.dump(before) != ast.dump(after):
        raise ValueError("Other rejection predicates changed; full rescreen required")


def split_records(path: Path, split: str) -> Any:
    with path.open() as listing:
        for line in listing:
            row = json.loads(line)
            if row["split"] == split:
                yield row


def prove_quality_narrowing(old_source: str, new_source: str) -> None:
    """Prove literal-alternative removal and optional AND-narrowed arithmetic rejection."""
    old = "(?:download|buy|purchase)"
    new = "(?:buy|purchase)"
    if old_source.count(old) != 1:
        raise ValueError("Cached acceptance reuse requires an exact literal-alternative narrowing")
    before = ast.parse(old_source.replace(old, new))
    after = ast.parse(new_source)
    if ast.dump(before) == ast.dump(after):
        return

    def arithmetic_branch(tree: ast.Module) -> ast.If:
        matches = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.If)
            and any(
                isinstance(value, ast.Constant)
                and value.value == "explicit_arithmetic_contradiction"
                for statement in node.body
                for value in ast.walk(statement)
            )
            and isinstance(node.test, ast.BoolOp)
        ]
        if len(matches) != 1:
            raise ValueError("Expected one arithmetic rejection branch")
        return matches[0]

    a, b = arithmetic_branch(before), arithmetic_branch(after)
    if (
        not isinstance(a.test, ast.BoolOp)
        or not isinstance(b.test, ast.BoolOp)
        or not isinstance(a.test.op, ast.And)
        or not isinstance(b.test.op, ast.And)
        or len(b.test.values) <= len(a.test.values)
    ):
        raise ValueError("Arithmetic cache reuse requires additional AND conditions")
    b.test.values = b.test.values[: len(a.test.values)]
    if ast.dump(before) != ast.dump(after):
        raise ValueError("Other quality predicates changed; full rescreen required")


def prove_screen_reuse(previous: Path) -> None:
    policy = json.loads((previous / "policy.json").read_text())
    old_source = previous / "policy-source.py"
    if digest(old_source) != policy["code_sha256"]:
        raise ValueError("Cached screening source changed")
    for name in ("rejection", "essential_figure_dependency"):
        old = next(
            n
            for n in ast.parse(old_source.read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name == name
        )
        new = next(
            n
            for n in ast.parse(Path(__file__).read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name == name
        )
        if ast.dump(old) != ast.dump(new):
            raise ValueError("Cached acceptance predicates changed")
    old_quality = previous / "quality-source.py"
    if (
        digest(old_quality) != policy["quality_code_sha256"]
        or digest(previous / "math-source.py") != policy["math_code_sha256"]
    ):
        raise ValueError("Cached helper source changed")
    if digest(Path(__file__).with_name("math_integrity.py")) != policy["math_code_sha256"]:
        raise ValueError("Symbolic screening changed; complete rescreen required")
    prove_quality_narrowing(
        old_quality.read_text(), Path(__file__).with_name("quality_findings.py").read_text()
    )


def durable_bytes(path: Path, payload: bytes | bytearray) -> None:
    with path.open("xb") as out:
        out.write(payload)
        out.flush()
        os.fsync(out.fileno())


def reuse_verified_file(source: Path, target: Path, storage: Storage) -> None:
    """Link immutable files only; otherwise use copy-on-write or an independent copy."""
    with source.open("rb") as src:
        try:
            flags = struct.unpack("I", fcntl.ioctl(src.fileno(), 0x80086601, bytes(4)))[0]
        except OSError:
            flags = 0
        if flags & 0x10:  # FS_IMMUTABLE_FL
            try:
                os.link(source, target)
                return
            except OSError:
                # Some filesystems forbid linking immutable inodes.
                pass
        storage.check(source.stat().st_size + 64 * 1024**2)
        with target.open("xb") as dst:
            try:
                fcntl.ioctl(dst.fileno(), 0x40049409, src.fileno())  # FICLONE
            except OSError:
                src.seek(0)
                dst.seek(0)
                dst.truncate()
                shutil.copyfileobj(src, dst, length=1024 * 1024)
            dst.flush()
            os.fsync(dst.fileno())


def rejection(text: str, source: str | None = None) -> str | None:
    """Reject damaged representations, not mathematical topics or hard notation."""
    issues = findings(text, source)
    if issues:
        return issues[0]["reason"]
    white = re.findall(r"\\textcolor\s*\{\s*w\s*h\s*i\s*t\s*e\s*\}\s*\{", text)
    if len(white) >= 2:
        return "decorative_tex_padding"
    if source in {"openwebmath", "finemath-4plus"}:
        # These are raw AsciiMath rendering controls, not normal TeX or color
        # highlights. Reject the document; never try to reconstruct its algebra.
        if len(re.findall(r"(?i)color\(white\)", text)) >= 2:
            return "unrendered_asciimath_padding"
        # Count empty promised equations only when a new prose paragraph follows.
        # A displayed formula, including ordinary TeX, breaks this pattern.
        absent = re.findall(
            r"(?im)(?:is (?:written|given) by|such that|as follows|"
            r"(?:can be|is) (?:written|expressed) as|(?:to|will) give)\s*:?\s*\n\s*\n"
            r"(?=(?:Where|Here|In the|Both of|The |Thus |Hence |[A-Z][a-z]+ potentials)\b)",
            text,
        )
        if len(absent) >= 3:
            return "repeated_missing_equations"
        if re.search(r"(?i)given in the (?:above|below) molecule", text) and re.search(
            r"(?i)(?:bonds|structures?|molecules?) (?:which )?are shown (?:below|above)", text
        ):
            return "missing_molecular_structure"
    if re.search(
        r"(?i)(?:try.{0,35}free classes|give the .{0,25}exam.{0,30}scholarship|"
        r"join.{0,25}learning program)",
        text,
    ):
        return "promotional_solution_contamination"
    if (
        len(
            re.findall(
                r"(?im)^\s*(?:Open in App|View More|Explore more|Suggest Corrections)\s*$", text
            )
        )
        >= 3
    ):
        return "navigation_contamination"
    headings = list(re.finditer(r"(?m)^(#{1,6})\s+([^\n]+)", text))
    primary = next((i for i, h in enumerate(headings) if len(h[1]) == 1), None)
    if primary is not None and primary >= 2:
        return "merged_exercise_teasers"
    if re.search(
        r"(?i)(?:the graph shows|(?:figures?|diagram) (?:below|above)|"
        r"(?:the|this) (?:area|region|part|portion) (?:of .{0,50})?is shaded|"
        r"what proportion.{0,80}is shaded|the counter is measuring|"
        r"the dot in the middle of the circle)",
        text,
    ) and essential_figure_dependency(text):
        return "missing_essential_figure"
    if re.search(r":\s*$", text) or re.search(r"(?i)(?:are as follows|is as follows)\s*$", text):
        return "incomplete_explanation"
    missing = re.findall(
        r"(?i)\b(?:depth of|area of|velocity of|temperature of|density was|elevation)\s*(?=[,.)])|"
        r"\b(?:depth of|approximately|temperature of|velocity of)[ ]{2,}(?:beneath|from|north|south|east|west)|"
        r"\ba[ ]{2,}by[ ]{2,}area\b",
        text,
    )
    if len(missing) >= 2:
        return "missing_measurements"
    if len(
        re.findall(r"(?i)described the[^\n]{0,55}(?:building|room)", text)
    ) >= 2 and not re.search(r'["“]', text):
        return "empty_promised_descriptions"
    return None


def rebuild(
    root: Path,
    parent: Path,
    output: Path,
    reuse_screen: Path | None = None,
    semantic_admission: Path | None = None,
) -> dict[str, Any]:
    storage = Storage.load(root)
    storage.check(5 * 1024**3)
    if not output.resolve().is_relative_to((root / "snapshots").resolve()):
        raise ValueError("New snapshot must stay on the preparation volume")
    if output.exists():
        raise FileExistsError(output)
    manifest = verify(parent)
    semantic_records = None
    semantic_texts: dict[str, str] = {}
    if semantic_admission is not None:
        from codexa.data.semantic import load_admission

        if reuse_screen is not None or not semantic_admission.resolve().is_relative_to(
            (root / "semantic-admission").resolve()
        ):
            raise ValueError(
                "Semantic admission must be an explicit local census without cached screening"
            )
        semantic_records = load_admission(semantic_admission, digest(parent / "manifest.json"))
        mapping = {
            row["id"]: row["document_id"]
            for row in map(
                json.loads, (semantic_admission / "mapping.jsonl").read_text().splitlines()
            )
        }
        for row in map(json.loads, (semantic_admission / "blind.jsonl").read_text().splitlines()):
            ident = mapping[row["id"]]
            if ident in semantic_texts or not isinstance(row["text"], str):
                raise ValueError("Invalid semantic source text")
            semantic_texts[ident] = row["text"]
        if set(semantic_texts) != set(semantic_records):
            raise ValueError("Semantic text identity differs from admissions")
    parent_audit = json.loads((parent / "audit.json").read_text())
    if parent_audit["manifest_sha256"] != digest(parent / "manifest.json"):
        raise ValueError("Parent audit changed")
    prior_rejected: set[str] | None = None
    if reuse_screen is not None:
        prior = verify(reuse_screen)
        previous_policy = json.loads((reuse_screen / "policy.json").read_text())
        if prior["identity"]["remediation_parent_manifest"] != digest(
            parent / "manifest.json"
        ) or previous_policy["version"] not in {
            "prose-math-remediation-1",
            "prose-math-remediation-4",
        }:
            raise ValueError("Incompatible cached screening parent or revision")
        previous_source = reuse_screen / "policy-source.py"
        if digest(previous_source) != previous_policy["code_sha256"]:
            raise ValueError("Cached screening source changed")
        if previous_policy["version"] == "prose-math-remediation-1":
            prove_narrowing(previous_source.read_text(), Path(__file__).read_text())
        else:
            prove_screen_reuse(reuse_screen)
        if digest(reuse_screen / "quarantine.jsonl") != prior["identity"]["quarantine_sha256"]:
            raise ValueError("Cached quarantine ledger changed")
        prior_rejected = {
            json.loads(line)["id"]
            for line in (reuse_screen / "quarantine.jsonl").read_text().splitlines()
        }
    output.mkdir(parents=True)
    durable_bytes(output / "policy-source.py", Path(__file__).read_bytes())
    quality_source = Path(__file__).with_name("quality_findings.py")
    durable_bytes(output / "quality-source.py", quality_source.read_bytes())
    math_source = Path(__file__).with_name("math_integrity.py")
    durable_bytes(output / "math-source.py", math_source.read_bytes())
    representation_source = Path(__file__).with_name("representation.py")
    durable_bytes(output / "representation-source.py", representation_source.read_bytes())
    if semantic_admission is not None:
        evidence_names = {
            "protocol.json",
            "packet-integrity.json",
            "blind.jsonl",
            "mapping.jsonl",
            "sampling.json",
            "token-verification.json",
            "curator-a.jsonl",
            "curator-b.jsonl",
            "admission.jsonl",
            "decision.json",
        } | set(json.loads((semantic_admission / "packet-integrity.json").read_text()))
        for name in sorted(evidence_names):
            durable_bytes(output / ("semantic-" + name), (semantic_admission / name).read_bytes())
        semantic_source = Path(__file__).with_name("semantic.py")
        durable_bytes(output / "semantic-source.py", semantic_source.read_bytes())
    atomic(
        output / "policy.json",
        {
            "version": "positive-semantic-remediation-6" if semantic_admission else VERSION,
            "semantic_admission_decision_sha256": digest(semantic_admission / "decision.json")
            if semantic_admission
            else None,
            "semantic_code_sha256": digest(Path(__file__).with_name("semantic.py"))
            if semantic_admission
            else None,
            "code_sha256": digest(Path(__file__)),
            "quality_code_sha256": digest(quality_source),
            "math_code_sha256": digest(math_source),
            "representation_code_sha256": digest(representation_source),
            "operation": "whole-document rejection only; no retained text changes",
            "reused_screen_manifest_sha256": digest(reuse_screen / "manifest.json")
            if reuse_screen
            else None,
            "reuse_proof": "Proven predicate narrowing; all previous rejections rescreened, prior acceptances unchanged"
            if reuse_screen
            else None,
        },
    )
    db = sqlite3.connect(f"file:{root}/indexes/documents.sqlite?mode=ro", uri=True)
    rejected_ids: set[str] = set()
    semantic_seen: set[str] = set()
    reasons: Counter[str] = Counter()
    tokens: Counter[str] = Counter()
    chars: Counter[str] = Counter()
    documents: Counter[str] = Counter()
    counts: Counter[str] = Counter()
    listing = (parent / "documents.jsonl").open()
    current: Any = json.loads(next(listing))
    with (output / "quarantine.jsonl").open("x") as quarantine:
        for ident, payload, metadata in db.execute(
            "SELECT hash,payload,metadata FROM docs WHERE role='new' ORDER BY id"
        ):
            if ident.hex() != current["id"]:
                continue
            meta = json.loads(metadata)
            affected = current["source"] not in {"wikipedia-en", "tinystories-gpt4"}
            disposition = (
                semantic_records.get(current["id"])
                if semantic_records is not None and affected
                else None
            )
            text = (
                zlib.decompress(payload).decode()
                if (disposition is not None and disposition["status"] == "admitted")
                or (
                    semantic_records is None
                    and affected
                    and (prior_rejected is None or current["id"] in prior_rejected)
                )
                else None
            )
            if text is not None:
                if hashlib.sha256(canonical(text).encode()).digest() != ident:
                    raise ValueError("Source database text identity changed")
                if disposition is not None and text != semantic_texts[current["id"]]:
                    raise ValueError(
                        "Screened text differs from independently verified review text"
                    )
            reason = rejection(text, current["source"]) if text is not None else None
            if semantic_records is not None and affected:
                if disposition is None:
                    reason = "semantic_review_pending"
                else:
                    semantic_seen.add(current["id"])
                    if any(disposition[k] != current[k] for k in ("source", "split", "tokens")):
                        raise ValueError("Semantic document identity differs from its parent")
                    if disposition["status"] != "admitted":
                        reason = "semantic_" + disposition["status"]
            if reason:
                rejected_ids.add(current["id"])
                reasons[current["source"] + ":" + reason] += 1
                quarantine.write(
                    json.dumps(
                        current
                        | {
                            "reason": reason,
                            "findings": findings(text, current["source"])
                            if text is not None
                            else [],
                            "semantic_disposition": semantic_records.get(current["id"])
                            if semantic_records is not None
                            else None,
                            "source_record": json.loads(metadata),
                            "preserved_text_database": str(root / "indexes/documents.sqlite"),
                        }
                    )
                    + "\n"
                )
            else:
                counts[current["split"]] += current["tokens"]
                if current["split"] == "train":
                    tokens[current["source"]] += current["tokens"]
                    chars[current["source"]] += (
                        len(text) if text is not None else meta["characters"]
                    )
                    documents[current["source"]] += 1
            line = next(listing, None)
            if line is None:
                current = None
                break
            current = json.loads(line)
        quarantine.flush()
        os.fsync(quarantine.fileno())
    db.close()
    listing.close()
    if current is not None:
        raise ValueError("Parent listing does not correspond to reference database")
    if semantic_records is not None and semantic_seen != set(semantic_records):
        raise ValueError("A semantic disposition is absent from the verified parent")
    atomic(
        output / "screening.json",
        {
            "reasons": dict(reasons),
            "rejected_documents": len(rejected_ids),
            "source_training_tokens": dict(tokens),
        },
    )
    print(
        json.dumps(
            {
                "screen_complete": True,
                "rejected": len(rejected_ids),
                "train_tokens": counts["train"],
            }
        ),
        flush=True,
    )
    splits: dict[str, list[dict[str, Any]]] = {s: [] for s in manifest["splits"]}
    # Align each split's listing with the original offset stream, preserving order and split.
    split_rows: dict[str, Any] = {}
    for split in splits:
        split_rows[split] = split_records(parent / "documents.jsonl", split)
    for split, entries in manifest["splits"].items():
        rows = split_rows[split]
        for entry in entries:
            data = np.memmap(parent / entry["file"], dtype="<u2", mode="r")
            offsets = np.memmap(parent / entry["index"], dtype="<u8", mode="r")
            selections = [next(rows) for _ in offsets]
            if all(r["id"] not in rejected_ids for r in selections):
                reuse_verified_file(parent / entry["file"], output / entry["file"], storage)
                reuse_verified_file(parent / entry["index"], output / entry["index"], storage)
                splits[split].append(entry.copy())
                continue
            buffer, new_offsets = bytearray(), []
            for i, row in enumerate(selections):
                end = int(offsets[i + 1]) if i + 1 < len(offsets) else len(data)
                start = int(offsets[i])
                if end - start != row["tokens"]:
                    raise ValueError("Parent mapping/offset mismatch")
                if row["id"] in rejected_ids:
                    continue
                new_offsets.append(len(buffer) // 2)
                buffer.extend(data[start:end].tobytes())
            if not buffer:
                continue
            index_bytes = np.asarray(new_offsets, dtype="<u8").tobytes()
            if reuse_screen is not None:
                reusable = next(
                    (r for r in prior["splits"][split] if r["file"] == entry["file"]), None
                )
                if (
                    reusable
                    and reusable["sha256"] == hashlib.sha256(buffer).hexdigest()
                    and reusable["index_sha256"] == hashlib.sha256(index_bytes).hexdigest()
                ):
                    reuse_verified_file(
                        reuse_screen / reusable["file"], output / reusable["file"], storage
                    )
                    reuse_verified_file(
                        reuse_screen / reusable["index"], output / reusable["index"], storage
                    )
                    splits[split].append(reusable.copy())
                    continue
            storage.check(len(buffer) + 64 * 1024**2)
            path = output / entry["file"]
            durable_bytes(path, buffer)
            ix = output / entry["index"]
            durable_bytes(ix, index_bytes)
            splits[split].append(
                entry
                | {
                    "tokens": len(buffer) // 2,
                    "documents": len(new_offsets),
                    "sha256": digest(path),
                    "index_sha256": digest(ix),
                }
            )
    with (
        (output / "documents.jsonl").open("x") as out,
        (parent / "documents.jsonl").open() as original,
    ):
        for line in original:
            if json.loads(line)["id"] not in rejected_ids:
                out.write(line)
        out.flush()
        os.fsync(out.fileno())
    result = manifest | {
        "dataset_name": "Codexa-v3-" + output.name,
        "splits": splits,
        "source_tokens": dict(tokens),
        "documents_sha256": digest(output / "documents.jsonl"),
        "identity": manifest["identity"]
        | {
            "remediation_parent_manifest": digest(parent / "manifest.json"),
            "remediation_policy": digest(output / "policy.json"),
            "quarantine_sha256": digest(output / "quarantine.jsonl"),
        },
    }
    atomic(output / "manifest.json", result)
    verify(output)
    audit = {
        "snapshot": str(output),
        "manifest_sha256": digest(output / "manifest.json"),
        "split_tokens": dict(counts),
        "source_training_tokens": dict(tokens),
        "source_training_characters": dict(chars),
        "source_training_documents": dict(documents),
        "rejected_documents": dict(reasons),
        "all_accepted_documents_screened_against_final_policy": semantic_records is None,
        "admitted_affected_documents_rescreened": semantic_records is not None,
        "unchanged_source_decisions_inherited": ["wikipedia-en", "tinystories-gpt4"],
        "retained_tokens_copied_exactly_from_verified_parent": True,
        "all_accepted_documents_reconstructed": False,
        "reconstruction_policy": "Exact byte subsequences from a fully reconstructed parent; independent token audit required",
        "parent_audit_sha256": digest(parent / "audit.json"),
        "training_started": False,
    }
    atomic(output / "audit.json", audit)
    return audit


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reuse-screen", type=Path)
    p.add_argument("--semantic-admission", type=Path)
    a = p.parse_args()
    print(
        json.dumps(
            rebuild(a.root, a.parent, a.output, a.reuse_screen, a.semantic_admission), indent=2
        )
    )


if __name__ == "__main__":
    main()
