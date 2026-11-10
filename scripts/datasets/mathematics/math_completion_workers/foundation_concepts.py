"""Compare prior concept extraction with layout mode; never clear review gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

INPUT = Path(
    "/mnt/codexa/pytorch/codexa-v3/school-math/curriculum-corpus-v2/workers/foundation/guide-concepts-run-02/teacher-concepts.jsonl"
)


def norm(s: str) -> str:
    return " ".join(s.casefold().split())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    dest = parser.parse_args().output
    if dest.exists():
        raise FileExistsError(dest)
    results = []
    for line in INPUT.read_text().splitlines():
        r = json.loads(line)
        pdf = Path(r["source_pdf"])
        if hashlib.sha256(pdf.read_bytes()).hexdigest() != r["source_hash"]:
            raise ValueError(f"source hash mismatch: {pdf}")
        page = re.fullmatch(r"PDF page (\d+)", r["source_page"])
        if page is None:
            raise ValueError("unexpected page identifier")
        text = subprocess.check_output(
            ["pdftotext", "-layout", "-f", page[1], "-l", page[1], str(pdf), "-"], text=True
        )
        results.append(
            {
                "record_id": r["record_id"],
                "layout_text_agrees": norm(r["original_text"]) in norm(text),
                "method": "pdftotext layout-mode comparison; not independent visual or mathematical review",
            }
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in results))
    print(
        json.dumps(
            {
                "records": len(results),
                "matches": sum(r["layout_text_agrees"] for r in results),
                "input_sha256": hashlib.sha256(INPUT.read_bytes()).hexdigest(),
                "output_sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
            }
        )
    )


if __name__ == "__main__":
    main()
