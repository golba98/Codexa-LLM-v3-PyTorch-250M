"""Every mapped exclusion and ambiguous family must remain held."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path


def test_mapping_holds_excluded_and_unknown(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text(
        "\n".join(
            json.dumps(x)
            for x in [
                {"id": "a", "problem_family": "arithmetic.missing_addend"},
                {"id": "b", "problem_family": "algebra.substitution"},
                {"id": "c", "problem_family": "unknown.new"},
            ]
        )
        + "\n"
    )
    exclusions = tmp_path / "exclusions.json"
    exclusions.write_text(json.dumps({"forbidden_generator_families": ["addition"]}))
    (tmp_path / "freeze-manifest.json").write_text(
        json.dumps(
            {"family_exclusions_sha256": hashlib.sha256(exclusions.read_bytes()).hexdigest()}
        )
    )
    output = tmp_path / "output"
    script = (
        Path(__file__).parents[1]
        / "scripts"
        / "datasets"
        / "mathematics"
        / "map_recovery_families.py"
    )
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--input",
            str(source),
            "--exclusions",
            str(exclusions),
            "--output",
            str(output),
        ],
        check=True,
    )
    rows = [json.loads(line) for line in (output / "family-flags.jsonl").read_text().splitlines()]
    assert [x["reason"] for x in rows] == [
        "excluded_broad_family",
        "ambiguous_or_unmapped_family",
        "ambiguous_or_unmapped_family",
    ]
    assert all(x["disposition"] == "held" for x in rows)

    report = json.loads((output / "mapping-report.json").read_text())
    assert report["input_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert report["exclusions_sha256"] == hashlib.sha256(exclusions.read_bytes()).hexdigest()
    exclusions.write_text("{}")
    failure = subprocess.run(
        [
            sys.executable,
            str(script),
            "--input",
            str(source),
            "--exclusions",
            str(exclusions),
            "--output",
            str(tmp_path / "bad"),
        ],
        check=False,
        capture_output=True,
    )
    assert failure.returncode != 0
    assert b"hash mismatch" in failure.stderr
