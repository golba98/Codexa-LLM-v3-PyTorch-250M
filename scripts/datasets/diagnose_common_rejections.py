"""Reconstruct the previous Common Corpus rejection ledger without acquiring data."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from codexa.data.language import cleaned
from codexa.data.production import digest
from codexa.data.v2_quality import canonical


def diagnose(root: Path, output: Path) -> dict[str, Any]:
    import pyarrow.parquet as pq

    profile = root / "reports/development-common-verified-v2"
    header = json.loads((profile / "common-corpus.json").read_text())
    spec = header["sources"][0]
    payload = Path(spec["local_path"])
    if digest(payload) != spec["sha256"]:
        raise ValueError("Original source payload changed")
    inspected = profile / "common-corpus-inspected.jsonl"
    selected = {json.loads(line)["ordinal"] for line in inspected.read_text().splitlines()}
    counts: Counter[str] = Counter()
    labels: Counter[str] = Counter()
    cross: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
    identities: Counter[str] = Counter()
    ordinal = 0
    for batch in pq.ParquetFile(payload).iter_batches(batch_size=32):
        for row in batch.to_pylist():
            ordinal += 1
            if ordinal not in selected:
                continue
            _, reason = cleaned(row, spec)
            reason = reason or "retained"
            counts[reason] += 1
            for field in ("collection", "language", "language_type", "license"):
                cross[field][str(row.get(field)), reason] += 1
            if reason == "common_unresolved_license":
                labels[str(row.get("license"))] += 1
            identities[hashlib.sha256(canonical(row["text"]).encode()).hexdigest()] += 1
    expected = {
        k: v
        for k, v in header["counts"].items()
        if k not in ("scanned", "accepted_tokens_with_boundaries", "accepted_text_bytes")
    }
    expected["retained"] = expected.pop("filter_accepted")
    if dict(counts) != expected:
        raise ValueError("Historical rejection ledger does not reconstruct")
    result = {
        "version": "codexa-common-rejection-diagnosis-1",
        "input_sha256": digest(inspected),
        "source_sha256": spec["sha256"],
        "records": sum(counts.values()),
        "first_failure_counts": dict(counts),
        "cross_tabs": {
            field: [
                {"value": value, "first_failure": reason, "count": count}
                for (value, reason), count in sorted(values.items())
            ]
            for field, values in cross.items()
        },
        "licence_rejection_labels": dict(labels),
        "exact_duplicate_excess_in_sample": sum(n - 1 for n in identities.values()),
        "interpretation": {
            "source_selection": "Written-English, date and collection gates dominate; these are not prose-quality ratings.",
            "licence_mapping": "CC-By is a recognizable family with missing version; CC-By-SA and Open Parliament are outside the previous allowlist, not inherently unlicensed. Generic Various labels remain unresolved.",
            "public_domain": "Nine Public Domain labels and three CC0 labels reached retention; zero approval means original-rights evidence was not admitted. Two article-text CC-BY grants were mislabelled data-only CC0.",
            "completeness": "36 terminal-check failures; surviving prose was separately reviewed. A footer can cause this predicate to fail without proving the body is incomplete.",
            "length": "No first-failure length rejection was recorded; earlier gates censor later quality and length checks.",
            "duplicates": "Exact normalized duplicates measured here were not an original rejection gate. Near duplicates were assessed only on the retained panel, not all 10,000.",
        },
        "training_allowed": False,
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
    result = diagnose(args.root, args.output)
    print(json.dumps({k: v for k, v in result.items() if k != "cross_tabs"}))


if __name__ == "__main__":
    main()
