#!/usr/bin/env python3
"""Independent SymPy exact-value recheck of stored equalities; no semantic claim."""

import argparse
import hashlib
import json
import re
from pathlib import Path

import sympy

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
p = args.input
if args.output.exists():
    raise FileExistsError("Use a fresh evidence output")
raw = p.read_bytes()
records = [json.loads(line) for line in raw.decode().splitlines()]
assert len({r["record_id"] for r in records}) == len(records)
checks = []
for r in records:
    if r["record_type"] != "worked_example":
        continue
    left, right = r["solution"].split(" = ")
    assert re.fullmatch(r"[\d+*/(). -]+", left) and re.fullmatch(r"[\d+*/(). -]+", right)
    x = sympy.sympify(left, rational=True)
    y = sympy.sympify(right, rational=True)
    assert sympy.simplify(x - y) == 0, r["record_id"]
    checks.append(
        {
            "record_id": r["record_id"],
            "method": "independent SymPy exact rational equality",
            "result": "passed",
            "limits": "Expression equality only; wording correspondence and explanatory completeness are not independently checked.",
        }
    )
receipt = {
    "input_sha256": hashlib.sha256(raw).hexdigest(),
    "sympy_version": sympy.__version__,
    "expressions_checked": len(checks),
    "failed": 0,
    "records": checks,
}
args.output.write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps({k: v for k, v in receipt.items() if k != "records"}, indent=2))
