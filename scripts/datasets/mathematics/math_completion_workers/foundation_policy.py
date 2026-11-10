"""Emit a small, explicitly provisional ATP evidence ledger; never a curriculum denominator."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

BASE = Path(
    "/mnt/codexa/pytorch/codexa-v3/school-math/curriculum-corpus-v2/workers/foundation/policy"
)
ITEMS = (
    (
        1,
        1,
        "counting-in-ones",
        "Count forwards and backwards in 1s, from any number between 1-20 and describe the sequence",
        None,
    ),
    (2, 1, "counting-in-ones", "Count forwards and backwards in 1s from any number to 100", None),
    (
        3,
        1,
        "counting-in-ones",
        "Count forwards and backwards in 1s from any number between 0 to 200",
        None,
    ),
    (
        2,
        5,
        "multiply-by-two-and-five",
        "Multiply numbers 1 to 10 by 2 and 5",
        (
            "01d169ecd25a1a623d512ca30a97306ea371123930419f13ad2096a101fbe84a",
            "4f4cf32ad995b47a585fdd120965138391d8f7da29d4ec1d318e8311836bc5ed",
            "a4c8f2d0be677d56587887d6ee054286df6f3f8e97d84a5d68663554bbcc4b30",
            "497b59cc33eea0befd8e3d5460f4bf6dcee9e1f26b92cc73c75a86239aa8c94d",
        ),
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    output = parser.parse_args().output
    sources = {r["grade"]: r for r in json.loads((BASE / "atp-sources.json").read_text())}
    records = []
    for grade, page, label, quote, links in ITEMS:
        source = sources[grade]
        pdf = Path(source["path"])
        if hashlib.sha256(pdf.read_bytes()).hexdigest() != source["sha256"]:
            raise ValueError(f"Unexpected policy file hash: {pdf}")
        text = subprocess.check_output(
            ["pdftotext", "-raw", "-f", str(page), "-l", str(page), str(pdf), "-"], text=True
        )
        if quote not in " ".join(text.split()):
            raise ValueError(f"Expected passage absent: grade {grade} page {page}")
        records.append(
            {
                "objective_id": f"provisional-atp-g{grade}-{label}",
                "grade": grade,
                "description": quote,
                "policy_source_url": source["url"],
                "policy_sha256": source["sha256"],
                "policy_pdf_page": page,
                "extraction_evidence": "pdftotext -raw; whitespace normalized for exact substring match",
                "term": 1 if page == 1 else 2,
                "status": "provisional; independently reviewed curriculum mapping pending",
                "classification": "core-versus-revision-unresolved",
                "applicability": "2026 circular and subsequent amendments require reconciliation",
                "linked_source_record_ids": list(links or ()),
                "links_status": "potential skills match; independently confirm source and context",
                "coverage_status": "not established",
            }
        )
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in records)
    )


if __name__ == "__main__":
    main()
