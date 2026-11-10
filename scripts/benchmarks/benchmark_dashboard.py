"""Bounded synthetic history load test; does not train a model or report model performance."""

import argparse
import json
import time
from pathlib import Path

from codexa.dashboard.server import History

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--records", type=int, default=10000)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
if not 1 <= a.records <= 1000000:
    raise ValueError("Choose 1–1000000 synthetic records")
if a.output.exists():
    raise FileExistsError(a.output)
a.output.mkdir(parents=True)
path = a.output / "metrics.jsonl"
with path.open("w") as f:
    for i in range(a.records):
        f.write(
            json.dumps(
                {
                    "step": i,
                    "loss": 1 / (1 + i / 1000),
                    "tokens": i * 64,
                    "tokens_per_second": 10000,
                }
            )
            + "\n"
        )
h = History(a.output, a.output / "index")
key = next(iter(h.runs()))
start = time.perf_counter()
h.view(key, path)
first = time.perf_counter() - start
start = time.perf_counter()
while h.positions[key] < path.stat().st_size:
    h.update(key, path)
index_seconds = time.perf_counter() - start
start = time.perf_counter()
view = h.view(key, path)
query = time.perf_counter() - start
start = time.perf_counter()
History(a.output, a.output / "index").view(key, path)
restart = time.perf_counter() - start
memory = None
if Path("/proc/self/status").exists():
    memory = {
        line.split(":")[0]: float(line.split()[1]) / 1024
        for line in Path("/proc/self/status").read_text().splitlines()
        if line.startswith(("VmRSS:", "VmHWM:"))
    }
r = {
    "synthetic_records": a.records,
    "indexed_records": view["records"],
    "returned_rows": len(view["rows"]),
    "first_view_seconds": first,
    "remaining_index_seconds": index_seconds,
    "history_query_seconds": query,
    "snapshot_restart_seconds": restart,
    "process_memory_mib": memory,
    "scope": "Synthetic history backend only; not a model throughput measurement",
}
(a.output / "measurements.json").write_text(json.dumps(r, indent=2))
print(json.dumps(r, indent=2))
