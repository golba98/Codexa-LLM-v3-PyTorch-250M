"""Bounded synthetic ingestion measurement. Outputs are not educational data."""

import argparse
import json
from pathlib import Path

from codexa.data.production import curate

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--documents", type=int, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
if not 1 <= a.documents <= 100000:
    raise ValueError("Bounded to 100000 records")
a.output.mkdir(parents=True, exist_ok=False)
sources = []
for split in ["train", "val", "test"]:
    path = a.output / (split + ".jsonl")
    with path.open("w") as f:
        for i in range(a.documents // 3):
            f.write(
                json.dumps(
                    {
                        "text": f"{split} record {i} "
                        + " ".join("vocabulary" + str(j) for j in range(50))
                    }
                )
                + "\n"
            )
    sources.append(
        {
            "name": split,
            "split": split,
            "paths": [str(path)],
            "license": "CC0",
            "snapshot": "original-test",
            "attribution": "original synthetic memory fixture",
        }
    )
import time

start = time.perf_counter()
curate(
    {"sources": sources, "near_duplicates": False, "protect_spans": False},
    str(a.output / "curated"),
)
result = {
    "documents": (a.documents // 3) * 3,
    "seconds": time.perf_counter() - start,
    "peak_rss_kib": int(
        next(
            line.split()[1]
            for line in Path("/proc/self/status").read_text().splitlines()
            if line.startswith("VmHWM:")
        )
    ),
    "scope": "synthetic ingestion only, no tokenization or model training",
}
(a.output / "measurement.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
