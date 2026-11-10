"""Positive-only historical overlap evidence from the compact passage index."""

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

from codexa.data.production import atomic, digest
from codexa.data.v2_quality import WORD, rolling


def audit(root: Path, historical: Path) -> dict[str, Any]:
    fixtures = root / "benchmarks/v3"
    reference = sqlite3.connect(f"file:{root}/indexes/documents.sqlite?mode=ro", uri=True)
    old = sqlite3.connect(f"file:{historical}?mode=ro&immutable=1", uri=True)
    checked = short = capped = 0
    matched: dict[str, list[dict[str, Any]]] = {}
    for split in ("dev", "test"):
        for line in (fixtures / (split + ".jsonl")).read_text().splitlines():
            item = json.loads(line)
            texts = [item["prompt"], item.get("question", ""), *item.get("candidates", [])]
            for text in dict.fromkeys(t for t in texts if t):
                words = WORD.findall(text.lower())
                if len(words) < 13:
                    short += 1
                    continue
                checked += 1
                found = False
                for position, value in enumerate(rolling(words, 13)):
                    candidates = reference.execute(
                        "SELECT d.hash,a.pos FROM anchors a JOIN docs d ON d.id=a.doc WHERE a.hash=? AND d.role=? LIMIT 65",
                        (int(value).to_bytes(8, "little"), "history"),
                    ).fetchall()
                    if len(candidates) > 64:
                        capped += 1
                    for ident, offset in candidates[:64]:
                        source_text = old.execute(
                            "SELECT text FROM docs WHERE id=?", (ident.hex(),)
                        ).fetchone()[0]
                        source_words = WORD.findall(source_text.lower())
                        if source_words[offset : offset + 13] != words[position : position + 13]:
                            continue
                        matched.setdefault(item["id"], []).append(
                            {
                                "source_id": ident.hex(),
                                "query_position": position,
                                "source_position": offset,
                                "verified_identical_word_punctuation_tokens": 13,
                            }
                        )
                        found = True
                        break
                    if found:
                        break
    reference.close()
    old.close()
    report = {
        "fixture_manifest_sha256": digest(fixtures / "manifest.json"),
        "checked_texts_at_least_13_tokens": checked,
        "short_texts_not_covered": short,
        "candidate_capped_queries": capped,
        "items_with_verified_shared_13_token_spans": len(matched),
        "matches": matched,
        "scope": "Positive evidence only. Historical index stores winnowed anchors, not all 13-grams; negatives do not establish unseen wording. Short/common grammar/factual items and semantic paraphrases are not exhaustively audited. Shared spans need not mean an entire test answer was trained on. Authored fixtures are independent of acquired source selection; ordinary factual knowledge is expected from pretraining.",
        "training_started": False,
    }
    atomic(root / "benchmarks/historical-overlap.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--historical", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root, args.historical)
    print(json.dumps({k: v for k, v in result.items() if k != "matches"}, indent=2))


if __name__ == "__main__":
    main()
