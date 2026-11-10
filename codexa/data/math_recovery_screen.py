"""Screen held synthetic candidates against known prior and protected material."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from codexa.data.school_math_corpus import arithmetic_holdout
from codexa.evaluation.protection import QuestionProtection
from codexa.tokenizer.bpe import CodexaTokenizer


def canon(text: str) -> str:
    return " ".join(text.casefold().split())


def read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def screen(
    rows: list[dict[str, Any]],
    checks: list[dict[str, Any]],
    previous: list[Path],
    pool: Path,
    freeze: Path,
    benchmark: Path,
    tokenizer: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    policy = json.loads(freeze.read_text(encoding="utf-8"))
    if policy["pool_sha256"] != sha(pool):
        raise ValueError("protected assessment pool does not match frozen hash")
    evaluations = read(pool)
    reserved_families = set(policy["reserved_families"])
    if reserved_families != {r["family"] for r in evaluations}:
        raise ValueError("protected family registry does not match the pool")
    protected_questions = {canon(r["question"]) for r in evaluations}
    old_questions = {canon(r["question"]) for path in previous for r in read(path)}
    evidence = {r["id"]: r for r in checks}
    if len(evidence) != len(checks):
        raise ValueError("duplicate check identities")
    matcher = QuestionProtection(benchmark)
    encoder = CodexaTokenizer.load(tokenizer)
    counts: Counter[str] = Counter()
    grades: dict[str, Counter[str]] = {}
    seen: set[str] = set()
    results = []
    for row in rows:
        ident = row["id"]
        if ident in seen:
            raise ValueError("duplicate candidate identity")
        seen.add(ident)
        question = row["question"]
        flags = []
        if canon(question) in old_questions:
            flags.append("exact_duplicate_prior_synthetic")
        if canon(question) in protected_questions:
            flags.append("exact_reserved_assessment_question")
        if row["problem_family"] in reserved_families:
            flags.append("reserved_family")
        match = matcher.matches(question)
        if match:
            flags.append("existing_fixture_possible_overlap")
        if arithmetic_holdout(question):
            flags.append("existing_arithmetic_holdout")
        if ident not in evidence or evidence[ident].get("errors"):
            flags.append("independent_arithmetic_check_failed_or_missing")
        counts.update(flags)
        payload = question + "\n" + "\n".join(row["solution_steps"]) + "\n" + row["answer"]
        if row["answer_unit"]:
            payload += " " + row["answer_unit"]
        tokens = len(encoder.encode(payload))
        grade = grades.setdefault(str(row["grade"]), Counter())
        grade["records"] += 1
        grade["unflagged_records"] += not flags
        grade["problem_solution_text_tokens_all_held"] += tokens
        results.append(
            {
                "id": ident,
                "flags": flags,
                "fixture_id": match,
                "token_count": tokens,
                "status": "held_caps_and_full_evaluation_boundary_pending",
            }
        )
    if set(evidence) != seen:
        raise ValueError("check identities do not match candidates")
    report = {
        "generated": len(rows),
        "flags": counts,
        "by_grade": grades,
        "distinct_mathematical_structures": len({r["problem_family"] for r in rows}),
        "training_admitted": 0,
        "input_sha256": {str(path): sha(path) for path in [*previous, pool, freeze, tokenizer]},
        "benchmark_sha256": {name: sha(benchmark / name) for name in ("dev.jsonl", "test.jsonl")},
        "limitations": [
            "All records held; CAPS, source-independent evaluation quality and official exam screening incomplete.",
            "Exact/heuristic overlap checks cannot rule out paraphrases or related mathematical families.",
            "The assessment pool was authored and seen by the generator author: exposed development material, not a protected unseen pool.",
            "One provisional assessment per grade is not a complete competency evaluation.",
            "Step equality checks do not independently establish every reasoning step.",
        ],
        "token_counts": "question, solution steps, answer and optional unit; no special tokens",
    }
    return results, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "checks", "pool", "freeze", "benchmark", "tokenizer", "output", "report"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--previous", type=Path, action="append", required=True)
    args = parser.parse_args()
    results, report = screen(
        read(args.input),
        read(args.checks),
        args.previous,
        args.pool,
        args.freeze,
        args.benchmark,
        args.tokenizer,
    )
    report["input_sha256"][str(args.input)] = sha(args.input)
    report["input_sha256"][str(args.checks)] = sha(args.checks)
    with args.output.open("x", encoding="utf-8") as stream:
        for result in results:
            stream.write(json.dumps(result, sort_keys=True) + "\n")
    with args.report.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"screened": len(results), "flags": report["flags"], "admitted": 0}))


if __name__ == "__main__":
    main()
