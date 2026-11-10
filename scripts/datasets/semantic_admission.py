"""Prepare a bounded semantic-admission census, distinct from approval review panels."""

import argparse
import json
import random
import re
import sqlite3
import zlib
from collections import Counter
from pathlib import Path
from typing import Any

from codexa.data.production import atomic, digest, verify
from codexa.data.quality_protocol import CHECKS
from codexa.data.semantic import (
    SOURCES,
    assessed,
    cleared,
    load_admission,
    primary_evidence_ids,
    require_primary_evidence,
    verify_candidate_input,
)
from codexa.data.v2_index import Protection
from codexa.data.v2_storage import Storage


def merge_shards(output: Path, assignment_path: Path) -> None:
    """Assemble actual judgments in packet order, enforcing declared actor ownership."""
    mapping = [json.loads(x) for x in (output / "mapping.jsonl").read_text().splitlines()]
    assignments = json.loads(assignment_path.read_text())["shards"]
    ownership = {}
    files: dict[str, dict[str, str]] = {}
    for record in mapping:
        ident = record["id"]
        matches = [a for a in assignments if a["range"][0] <= ident <= a["range"][1]]
        if len(matches) != 1:
            raise ValueError("Each census identity needs exactly one declared shard")
        owner = matches[0]
        if owner["a"] == owner["b"]:
            raise ValueError("Two distinct actors required for every census identity")
        ownership[ident] = owner
        for slot in ("a", "b"):
            name = owner[slot + "_file"]
            if Path(name).name != name or name in {"curator-a.jsonl", "curator-b.jsonl"}:
                raise ValueError("Raw shard must be a separate local file")
            files.setdefault(name, {})[ident] = owner[slot]
    raw = {}
    for name, actors in files.items():
        rows = assessed(
            [json.loads(x) for x in (output / name).read_text().splitlines()], set(actors)
        )
        if any(row["judge_id"] != actors[ident] for ident, row in rows.items()):
            raise ValueError("Census judgment actor differs from declared independent owner")
        raw[name] = rows
    destinations = [output / ("curator-" + slot + ".jsonl") for slot in ("a", "b")]
    if any(path.exists() for path in destinations):
        raise FileExistsError("Preserve existing merged semantic judgments")
    for slot, path in zip(("a", "b"), destinations):
        with path.open("x") as stream:
            for record in mapping:
                ident = record["id"]
                row = raw[ownership[ident][slot + "_file"]][ident]
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    atomic(
        output / "merge-provenance.json",
        {
            "scope": "Reordering actual raw judgments only; no generated labels or rationales",
            "assignment_file": str(assignment_path),
            "assignment_sha256": digest(assignment_path),
            "raw_shard_hashes": {name: digest(output / name) for name in files},
            "merged_hashes": {path.name: digest(path) for path in destinations},
            "documents_per_judge": len(mapping),
        },
    )


def prepare(
    root: Path,
    parent: Path,
    output: Path,
    excluded_reviews: list[Path],
    plan: Path | None = None,
    excluded_censuses: list[Path] | None = None,
) -> None:
    Storage.load(root).check(128 * 1024**2)
    if output.exists():
        raise FileExistsError(output)
    manifest = verify(parent)
    configuration = json.loads(plan.read_text()) if plan is not None else None
    budgets = dict.fromkeys(SOURCES, 1000)
    seed, heldout, prefix = 2026110611, 50, "S"
    if configuration is not None:
        budgets = configuration["source_train_budgets"]
        seed = configuration["seed"]
        heldout = configuration["heldout_candidate_budget_per_source_split"]
        prefix = configuration["sample_prefix"]
        if (
            configuration.get("version") != "semantic-admission-extension-1"
            or configuration.get("parent_manifest_sha256") != digest(parent / "manifest.json")
            or not isinstance(budgets, dict)
            or not budgets
            or not set(budgets) <= SOURCES
            or any(type(n) is not int or n <= 0 for n in budgets.values())
            or type(seed) is not int
            or seed < 0
            or type(heldout) is not int
            or heldout <= 0
            or not isinstance(prefix, str)
            or re.fullmatch(r"[A-Z]{1,8}", prefix) is None
            or configuration.get("require_primary_evidence") is not True
            or not excluded_censuses
            or configuration.get("original_census_decision_sha256")
            not in {digest(p / "decision.json") for p in excluded_censuses}
        ):
            raise ValueError("Invalid frozen extension plan or missing original census exclusion")
    output.mkdir(parents=True)
    protocol: dict[str, Any] = {
        "version": "semantic-admission-extension-1" if configuration else "semantic-admission-1",
        "parent_manifest_sha256": digest(parent / "manifest.json"),
        "seed": seed,
        "train_candidate_budget_per_source": budgets if configuration else 1000,
        "heldout_candidate_budget_per_source_split": heldout,
        "sources": sorted(budgets),
        "scope": "Census of a bounded candidate corpus, not approval scores or exhaustive review of the parent. Unadmitted affected records remain pending/rejected; unaffected sources inherit the verified parent.",
        "required_checks": list(CHECKS),
        "reviewers": 2,
        "full_coverage_required": True,
        "rubric": "Identify the educational purpose, assumptions, required inputs and every worked solution. Assess notation, completeness, visuals, correctness, education, coherence and contamination. Retain explicit learner-error/correction dialogues and coherent multi-exercise lessons. Admit only substantive self-contained coherent material with no unresolved defect; unsupported or unreviewed content is not admitted. A mathematical solution requires checking its reasoning and conditions. Niche factual uncertainty requires a primary-reference citation or an explicit uncertain judgment. Do not ban topics, forums, difficulty levels or valid LaTeX. Record topic, level and format for a separate diversity audit. Admission is not independent approval: different agents must subsequently draw fresh blinded primary and extended approval panels, excluding all historical approval/development packets.",
        "decision_policy": "Both independent reviewers must clear every applicable check; preserve disagreements and uncertainty as unresolved. No default acceptance, quota filling or substitution of easy documents.",
        "excluded_review_packet_hashes": [],
    }
    if plan is not None:
        (output / "sampling-plan.json").write_bytes(plan.read_bytes())
        (output / "admission-preparation-source.py").write_bytes(Path(__file__).read_bytes())
        protocol["sampling_plan_sha256"] = digest(plan)
        protocol["require_primary_evidence"] = True
        protocol["sample_prefix"] = prefix
    excluded_ids: set[str] = set()
    excluded_texts: list[tuple[str, int]] = []
    for review in excluded_reviews:
        integrity = json.loads((review / "packet-integrity.json").read_text())
        for name, sha in integrity.items():
            if digest(review / name) != sha:
                raise ValueError("Historical packet changed")
        protocol["excluded_review_packet_hashes"].append(digest(review / "packet-integrity.json"))
        excluded_ids.update(
            row["document_id"]
            for row in map(json.loads, (review / "mapping.jsonl").read_text().splitlines())
            if row["kind"] == "document"
        )
        excluded_texts.extend(
            (row["text"], 64)
            for row in map(json.loads, (review / "blind.jsonl").read_text().splitlines())
        )
    census_hashes = []
    for census in excluded_censuses or []:
        load_admission(census, digest(parent / "manifest.json"))
        census_hashes.append(digest(census / "decision.json"))
        excluded_ids.update(
            row["document_id"]
            for row in map(json.loads, (census / "mapping.jsonl").read_text().splitlines())
        )
        excluded_texts.extend(
            (row["text"], 64)
            for row in map(json.loads, (census / "blind.jsonl").read_text().splitlines())
        )
    if census_hashes:
        protocol["excluded_census_decision_hashes"] = census_hashes
    atomic(output / "protocol.json", protocol)
    protection = Protection.build(excluded_texts)
    rng = random.Random(protocol["seed"])
    counts: Counter[tuple[str, str]] = Counter()
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    with (parent / "documents.jsonl").open() as listing:
        for row in map(json.loads, listing):
            if row["source"] not in budgets or row["id"] in excluded_ids:
                continue
            key = (row["source"], row["split"])
            size = budgets[row["source"]] if row["split"] == "train" else heldout
            capacity = size + 512
            counts[key] += 1
            bucket = buckets.setdefault(key, [])
            if len(bucket) < capacity:
                bucket.append(row)
            else:
                index = rng.randrange(counts[key])
                if index < capacity:
                    bucket[index] = row
    db = sqlite3.connect(f"file:{root}/indexes/documents.sqlite?mode=ro", uri=True)
    packet = []
    overlap = []
    for (source, split), rows in sorted(buckets.items()):
        rng.shuffle(rows)
        size = budgets[source] if split == "train" else heldout
        admitted_candidates = 0
        for row in rows:
            payload = db.execute(
                "SELECT payload FROM docs WHERE hash=?", (bytes.fromhex(row["id"]),)
            ).fetchone()
            if payload is None:
                raise ValueError("Missing candidate text")
            text = zlib.decompress(payload[0]).decode()
            if protection.matches(text):
                overlap.append(row["id"])
                continue
            packet.append(
                {
                    "document_id": row["id"],
                    "source": source,
                    "split": split,
                    "tokens": row["tokens"],
                    "text": text,
                }
            )
            admitted_candidates += 1
            if admitted_candidates == size:
                break
    db.close()
    if configuration is not None and len(packet) != sum(budgets.values()) + heldout * (
        len(manifest["splits"]) - 1
    ) * len(budgets):
        raise ValueError("Insufficient independent candidates for the frozen extension strata")
    rng.shuffle(packet)
    with (
        (output / "mapping.jsonl").open("x") as mapping,
        (output / "blind.jsonl").open("x") as blind,
    ):
        for i, row in enumerate(packet):
            ident = f"{prefix}{i + 1:05d}"
            mapping.write(
                json.dumps({"id": ident, **{k: v for k, v in row.items() if k != "text"}}) + "\n"
            )
            blind.write(
                json.dumps(
                    {"id": ident, "coverage": "full document", "text": row["text"]},
                    ensure_ascii=False,
                )
                + "\n"
            )
    atomic(
        output / "sampling.json",
        {
            "eligible_before_passage_checks": {
                s + ":" + split: n for (s, split), n in counts.items()
            },
            "overlap_exclusions": overlap,
            "candidate_documents": len(packet),
            "all_selected_documents_require_two_actual_assessments": True,
            "parent_tokenizer_sha256": manifest["tokenizer_hash"],
        },
    )
    names = ["protocol.json", "mapping.jsonl", "blind.jsonl", "sampling.json"]
    if plan is not None:
        names.extend(("sampling-plan.json", "admission-preparation-source.py"))
    atomic(output / "packet-integrity.json", {name: digest(output / name) for name in names})
    print(
        json.dumps(
            {
                "candidate_documents": len(packet),
                "status": "awaiting_independent_admission_assessments",
            }
        )
    )


def finalize(output: Path) -> None:
    integrity = json.loads((output / "packet-integrity.json").read_text())
    for name, sha in integrity.items():
        if digest(output / name) != sha:
            raise ValueError("Admission census changed")
    mapping = [json.loads(x) for x in (output / "mapping.jsonl").read_text().splitlines()]
    expected = {r["id"] for r in mapping}
    parent_sha = json.loads((output / "protocol.json").read_text())["parent_manifest_sha256"]
    token_proof_sha = verify_candidate_input(output, parent_sha, len(mapping))
    judges = [
        assessed([json.loads(x) for x in (output / name).read_text().splitlines()], expected)
        for name in ("curator-a.jsonl", "curator-b.jsonl")
    ]
    required_evidence = primary_evidence_ids(
        json.loads((output / "protocol.json").read_text()), expected
    )
    for record in mapping:
        if record["id"] in required_evidence:
            for judge in judges:
                if cleared(judge[record["id"]], record["source"]):
                    require_primary_evidence(judge[record["id"]])
    if any(judges[0][ident]["judge_id"] == judges[1][ident]["judge_id"] for ident in expected):
        raise ValueError("Two independent semantic judges required")
    counts: Counter[str] = Counter()
    with (output / "admission.jsonl").open("x") as ledger:
        for row in mapping:
            reviews = [judge[row["id"]] for judge in judges]
            if reviews[0]["judge_id"] == reviews[1]["judge_id"]:
                raise ValueError("Two independent semantic judges required")
            status = (
                "admitted"
                if all(cleared(r, row["source"]) for r in reviews)
                else "unresolved"
                if any(
                    "uncertain" in r["checks"].values() or not r["coverage_complete"]
                    for r in reviews
                )
                else "rejected"
            )
            counts[row["source"] + ":" + row["split"] + ":" + status] += 1
            ledger.write(
                json.dumps(
                    row
                    | {
                        "status": status,
                        "reasons": [r["reason"] for r in reviews],
                        "topics": [r["topic"] for r in reviews],
                        "levels": [r["level"] for r in reviews],
                        "formats": [r["format"] for r in reviews],
                    }
                )
                + "\n"
            )
    atomic(
        output / "decision.json",
        {
            "scope": "semantic admission, not approval review",
            "parent_manifest_sha256": json.loads((output / "protocol.json").read_text())[
                "parent_manifest_sha256"
            ],
            "counts": dict(counts),
            "census_documents": len(mapping),
            "unreviewed": 0,
            "admission_sha256": digest(output / "admission.jsonl"),
            "packet_integrity_sha256": digest(output / "packet-integrity.json"),
            "token_verification_sha256": token_proof_sha,
            "reviewer_hashes": {
                n: digest(output / n) for n in ("curator-a.jsonl", "curator-b.jsonl")
            },
        },
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["prepare", "merge", "finalize"])
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--parent", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--exclude-review", type=Path, action="append", default=[])
    p.add_argument("--assignments", type=Path)
    p.add_argument("--plan", type=Path)
    p.add_argument("--exclude-census", type=Path, action="append", default=[])
    a = p.parse_args()
    if a.action == "prepare":
        if a.parent is None:
            p.error("--parent required")
        prepare(a.root, a.parent, a.output, a.exclude_review, a.plan, a.exclude_census)
    elif a.action == "merge":
        if a.assignments is None:
            p.error("--assignments required")
        Storage.load(a.root).check()
        merge_shards(a.output, a.assignments)
    else:
        Storage.load(a.root).check()
        finalize(a.output)


if __name__ == "__main__":
    main()
