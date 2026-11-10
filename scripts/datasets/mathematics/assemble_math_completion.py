"""Freeze an additive, explicitly unapproved mathematics recovery snapshot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer

SCHOOL = Path("/mnt/codexa/pytorch/codexa-v3/school-math")
PARENT = SCHOOL / "curriculum-corpus-v2/candidate-06"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def dump(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    for part in ("records", "manifests", "reports", "exports"):
        (out / part).mkdir()
    inputs = {
        "foundation": SCHOOL / "foundation-completion-v1/teaching-02/records.jsonl",
        "junior": SCHOOL / "completion-v1/workers/junior-2011/records-final/records.jsonl",
        "original": SCHOOL / "completion-middle-v1/reviewed-v3/records.jsonl",
    }
    senior_root = SCHOOL / "completion-v1/workers/senior-html"
    pages = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted((senior_root / "processed").glob("*.json"))
    ]
    source_rows: list[dict[str, Any]] = []
    for group in ("foundation", "junior"):
        for row in rows(inputs[group]):
            source_rows.append({"recovery_batch": group, "original_record": row})
    for page in pages:
        for row in page["records"]:
            source_rows.append({"recovery_batch": "senior_online", "original_record": row})
    original = rows(inputs["original"])
    for name, collection in [
        ("source-additions.jsonl", source_rows),
        ("original-additions.jsonl", original),
    ]:
        with (out / "records" / name).open("w", encoding="utf-8") as stream:
            for row in collection:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    old_fingerprints: dict[tuple[int, str], str] = {}
    with (PARENT / "records/source-canonical.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            old = json.loads(line)
            text = old.get("original_text") or old.get("normalized_text")
            grade = old.get("grade")
            if isinstance(text, str) and isinstance(grade, int) and text.strip():
                old_fingerprints.setdefault(
                    (grade, sha(" ".join(text.split()).encode())), str(old.get("record_id"))
                )
    seen = dict(old_fingerprints)
    overlaps = []
    ids: dict[str, str] = {}
    id_conflicts = []
    by_grade: Counter[int] = Counter()
    by_batch: Counter[str] = Counter()
    for envelope in source_rows:
        row = envelope["original_record"]
        grade = int(row["grade"])
        by_grade[grade] += 1
        by_batch[envelope["recovery_batch"]] += 1
        rid = str(row.get("record_id") or row.get("id"))
        text = (
            row.get("original_text")
            or row.get("original_source_text")
            or row.get("normalized_text")
            or ""
        )
        fingerprint = sha(" ".join(text.split()).encode())
        key = (grade, fingerprint)
        if text.strip() and key in seen:
            overlaps.append(
                {
                    "record_id": rid,
                    "matches_record_id": seen[key],
                    "grade": grade,
                    "text_sha256": fingerprint,
                    "decision": "possible exact-text duplicate; context and lineage review pending",
                }
            )
        elif text.strip():
            seen[key] = rid
        if rid in ids:
            id_conflicts.append(
                {
                    "record_id": rid,
                    "first_batch": ids[rid],
                    "second_batch": envelope["recovery_batch"],
                }
            )
        else:
            ids[rid] = envelope["recovery_batch"]
    dump(out / "manifests/possible-exact-text-matches.json", overlaps)
    dump(out / "manifests/id-collisions.json", id_conflicts)
    queue = json.loads((senior_root / "queue.json").read_text(encoding="utf-8"))
    completed = {p["source_id"]: p for p in pages}
    with (out / "manifests/senior-section-queue.jsonl").open("w", encoding="utf-8") as stream:
        for section in queue["sections"]:
            result = completed.get(section["source_id"])
            row = {
                **section,
                "status": result["status"] if result else "pending",
                "reason": result.get("reason") if result else None,
            }
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    # Frozen prior exports remain authoritative. New data have not passed their gates.
    for name in (
        "content-checked-additions",
        "rights-cleared-additions",
        "verified-solution-additions",
        "finally-admitted-additions",
    ):
        (out / "exports" / (name + ".jsonl")).write_text("", encoding="utf-8")
    original_by_grade = Counter(int(r["grade"]) for r in original)
    tokenizer_path = Path(
        "/mnt/codexa/pytorch/codexa-v3/pretraining/20261103-500m-233201/tokenizer/tokenizer.json"
    )
    tokenizer_hash = sha(tokenizer_path.read_bytes())
    if tokenizer_hash != "60a5e06447542b9462d458577f4a3056551c135e5aa1b8859d39624a3eee8771":
        raise ValueError("Frozen Codexa tokenizer does not match candidate-06")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    checked_tokens: Counter[int] = Counter()
    checked_counts: Counter[int] = Counter()
    for row in original:
        if row.get("record_type") != "worked_example":
            continue
        text = "\n".join(str(row.get(key) or "") for key in ("question", "solution", "answer"))
        grade = int(row["grade"])
        checked_tokens[grade] += len(tokenizer.encode(text, add_special_tokens=False).ids)
        checked_counts[grade] += 1
    dump(
        out / "manifests/native-token-counts.json",
        {
            "tokenizer_path": str(tokenizer_path),
            "tokenizer_sha256": tokenizer_hash,
            "representation": "question + newline + solution + newline + answer; no BOS/EOS or sequence separators",
            "scope": "independently question-answer reviewed original worked candidates, still held; not training-eligible tokens",
            "reviewed_worked_candidates_by_grade": dict(checked_counts),
            "reviewed_candidate_tokens_by_grade": dict(checked_tokens),
            "training_eligible_tokens": 0,
        },
    )
    with (out / "reports/grade-additions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "grade",
                "new_source_candidates",
                "new_original_candidates",
                "new_content_checked",
                "new_question_answer_reviewed_worked_candidates",
                "reviewed_worked_candidate_tokens_held",
                "new_rights_cleared",
                "new_verified_solutions",
                "new_finally_admitted",
                "new_training_eligible_tokens",
            ]
        )
        for grade in range(1, 13):
            writer.writerow(
                [
                    grade,
                    by_grade[grade],
                    original_by_grade[grade],
                    0,
                    checked_counts[grade],
                    checked_tokens[grade],
                    0,
                    0,
                    0,
                    0,
                ]
            )
    input_hashes = {str(path): sha(path.read_bytes()) for path in inputs.values()}
    input_hashes[str(PARENT / "manifests/manifest.json")] = sha(
        (PARENT / "manifests/manifest.json").read_bytes()
    )
    input_hashes[str(senior_root / "queue.json")] = sha((senior_root / "queue.json").read_bytes())
    for path in sorted((senior_root / "processed").glob("*.json")):
        input_hashes[str(path)] = sha(path.read_bytes())
    result = {
        "parent_candidate": str(PARENT),
        "parent_is_immutable": True,
        "coverage_and_objectives": "Parent inventories remain unverified and are not copied or reclassified.",
        "source_additions": dict(by_batch),
        "source_additions_by_grade": dict(by_grade),
        "original_additions": len(original),
        "page_dispositions": dict(Counter(p["status"] for p in pages)),
        "page_count": len(pages),
        "queue_count": len(queue["sections"]),
        "exact_text_matches_pending": len(overlaps),
        "id_collisions_pending": len(id_conflicts),
        "rights_cleared_additions": 0,
        "admitted_additions": 0,
        "training_eligible_tokens_additions": 0,
        "input_sha256": input_hashes,
        "record_gates": "All additions are held. Arithmetic equality checks alone do not verify questions, explanations, overlap or rights.",
    }
    result["output_sha256"] = {
        str(p.relative_to(out)): sha(p.read_bytes()) for p in sorted(out.rglob("*")) if p.is_file()
    }
    dump(out / "manifests/snapshot.json", result)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("input_sha256", "output_sha256")},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
