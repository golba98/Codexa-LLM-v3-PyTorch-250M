"""Summarize preserved language evidence without inference or new subjective scores."""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from codexa.data.production import digest


def analyze(root: Path, output: Path) -> dict[str, Any]:
    review = root / "benchmarks/language-baseline-review"
    scores = json.loads((review / "score.json").read_text())["records"]
    mapping = [json.loads(line) for line in (review / "mapping.jsonl").read_text().splitlines()]
    directories = {Path(row["output"]) for row in mapping}
    if len(directories) != 1:
        raise ValueError("Expected the single preserved parent checkpoint panel")
    generations = directories.pop() / "outputs.jsonl"
    rows = [json.loads(line) for line in generations.read_text().splitlines()]
    by_key = {row["key"]: row for row in rows}
    if len(rows) != 400 or len(scores) != 100 or len(by_key) != len(rows):
        raise ValueError("Preserved baseline coverage or identities differ")
    dimensions = {}
    for name in ("grammar", "coherence", "relevance", "completeness", "task_success"):
        counts = Counter(row["scores"][name] for row in scores)
        dimensions[name] = {"failed": counts[0], "partial": counts[1], "full": counts[2]}
    tasks = {}
    for task in sorted({row["task"] for row in scores}):
        panel = [row for row in scores if row["task"] == task]
        tasks[task] = {
            "cases": len(panel),
            "fully_successful": sum(row["scores"]["task_success"] == 2 for row in panel),
            "partly_successful": sum(row["scores"]["task_success"] == 1 for row in panel),
        }
    # Selection is fixed by fixture identity, independent of ratings or apparent quality.
    example_keys = [f"language-v1-test-00-{task}:sample" for task in tasks]
    evidence = {row["key"]: row for row in scores}
    result = {
        "version": "codexa-language-baseline-analysis-1",
        "inputs": {str(p): digest(p) for p in (generations, review / "score.json")},
        "generations": len(rows),
        "severe_exact_loops": sum(row["severe_loop"] for row in rows),
        "token_limit": sum(row["stop_reason"] == "token_limit" for row in rows),
        "reviewed_cases": len(scores),
        "independent_contexts": 20,
        "dimensions": dimensions,
        "tasks": tasks,
        "reviewer_disagreements": sum(row["disagreement"] for row in scores),
        "examples": [
            {
                "key": key,
                "prompt": by_key[key]["prompt"],
                "response": by_key[key]["text"],
                "truncated": by_key[key]["truncated"],
                "scores": evidence[key]["scores"],
                "independent_reasons": evidence[key]["reasons"],
            }
            for key in example_keys
        ],
        "interpretation": {
            "present": "Some locally grammatical sentences and partial passage relevance/task success.",
            "absent_in_panel": "Sustained coherent complete responses across all five tasks.",
            "understanding": "Comprehension, answer selection and free-running generation failures cannot be causally separated by these outputs alone.",
            "instruction_and_context": "Instruction and dialogue tasks fail, but these tasks also require coherent generation; they do not isolate memory capacity.",
            "rubric": "Grammar and partial-success scores retain partial capabilities. Universal coherence failure and observed loops are not explained solely by strict all-dimension acceptance.",
        },
        "limitations": [
            "100 reviewed outputs share 20 scenarios; the other 300 outputs have automatic metrics only.",
            "Exact-loop detection misses semantic and longer-period repetition.",
            "A token limit alone does not establish incoherence or incomplete meaning.",
            "No new ratings, training, forward passes or competence claims were produced.",
        ],
        "optimizer_updates": 0,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.root, args.output)
    print(json.dumps({key: value for key, value in result.items() if key != "examples"}))


if __name__ == "__main__":
    main()
