"""Conservative answer proxies and an independently implemented math-fixture oracle."""

import argparse
import json
import re
import unicodedata
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any

from codexa.data.production import atomic, digest


def normalize(text: str) -> str:
    text = "".join(
        c for c in unicodedata.normalize("NFKD", text.casefold()) if not unicodedata.combining(c)
    )
    return " ".join(re.findall(r"[a-z0-9]+", text))


def numeric_answer(text: str) -> Fraction | None:
    """First explicit answer only. No searching later text for a matching number."""
    text = text.strip().removeprefix("$").strip()
    text = re.sub(r"^(?:x\s*=\s*|the answer is\s*|answer:\s*)", "", text, flags=re.IGNORECASE)
    match = re.match(r"([-+]?\d+(?:\.\d+)?(?:\s*/\s*[-+]?\d+)?)(?=$|\s|[.,;!?)$])", text)
    if not match:
        return None
    try:
        return Fraction(match[1].replace(" ", ""))
    except (ValueError, ZeroDivisionError):
        return None


def free_correct(item: dict[str, Any], text: str) -> bool:
    if item["kind"] == "math":
        return numeric_answer(text) == Fraction(item["answer"])
    expected = normalize(item["answer"])
    actual = normalize(text)
    return actual == expected or actual.startswith(expected + " ")


def math_oracle(item: dict[str, Any]) -> str:
    """Reparse the frozen question with SymPy, not the fixture's Fraction generator."""
    import sympy as sp

    q = item["question"]
    values = [int(x) for x in re.findall(r"\d+", q)]
    topic = item["topic"]
    if topic == "arithmetic":
        value = sp.Add(*values)
    elif topic in ("word_problem", "geometry"):
        value = sp.Mul(*values)
    elif topic == "fraction":
        value = sp.Rational(*values)
    elif topic == "algebra":
        a, b, result = values
        x = sp.Symbol("x")
        value = sp.solve(sp.Eq(a * x + b, result), x)[0]
    elif topic == "function":
        a, exponent, b, argument = values
        x = sp.Symbol("x")
        value = (a * x**exponent + b).subs(x, argument)
    else:
        raise ValueError("Unknown math topic")
    return str(value)


def score(root: Path, output: Path) -> dict[str, Any]:
    fixture_root = root / "benchmarks/v3"
    fixture_manifest = json.loads((fixture_root / "manifest.json").read_text())
    for filename, sha in fixture_manifest["files"].items():
        if digest(fixture_root / filename) != sha:
            raise ValueError("Frozen fixtures changed")
    items = {
        r["id"]: r
        for split in ("dev", "test")
        for r in (
            json.loads(line)
            for line in (fixture_root / (split + ".jsonl")).read_text().splitlines()
        )
    }
    math = [x for x in items.values() if x["kind"] == "math"]
    errors = [x["id"] for x in math if math_oracle(x) != x["answer"]]
    if errors:
        raise ValueError("Independent math oracle disagrees: " + str(errors))
    groups: dict[str, list[bool]] = defaultdict(list)
    rows = []
    for line in (output / "outputs.jsonl").read_text().splitlines():
        row = json.loads(line)
        item = items[row["item_id"]]
        if "text" not in row or item["kind"] not in ("math", "fact", "context"):
            continue
        correct = free_correct(item, row["text"])
        groups[f"{item['split']}:{item['kind']}:{row['setting']}"].append(correct)
        rows.append(
            {
                "key": row["key"],
                "first_answer_proxy_correct": correct,
                "requires_rubric_review": True,
            }
        )
    report = {
        "fixture_manifest_sha256": digest(fixture_root / "manifest.json"),
        "outputs_sha256": digest(output / "outputs.jsonl"),
        "independent_sympy_oracle": {"checked": len(math), "errors": errors},
        "scores": {
            key: {"n": len(v), "correct": sum(v), "accuracy": sum(v) / len(v)}
            for key, v in groups.items()
        },
        "limitations": "First-answer proxy, not paragraph coherence or demonstrated reasoning. Lexical variants/number words can be false negatives; correct opening answers followed by contradictions can be false positives. Candidate scores are reported separately.",
    }
    atomic(output / "answer-proxies.json", report)
    (output / "answer-proxy-items.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(score(args.root, args.output), indent=2))


if __name__ == "__main__":
    main()
