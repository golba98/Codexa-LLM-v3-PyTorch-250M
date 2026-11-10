import json
from pathlib import Path

import pytest

from scripts.index_math_collections_v2 import band, index


def test_band_indexes_keep_unmapped_and_do_not_duplicate_same_band(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    synthetic = tmp_path / "synthetic.jsonl"
    source.write_text(
        "".join(
            json.dumps(x) + "\n"
            for x in [
                {"record_id": "one", "grades": [4, 5]},
                {"record_id": "two", "grades": []},
                {"record_id": "three", "grades": [7, 8]},
            ]
        )
    )
    synthetic.write_text(json.dumps({"record_id": "generated", "grade": 12}) + "\n")
    rows = {r["collection"]: r for r in index(source, synthetic)}
    assert rows["B:4-7"]["source_record_ids"] == ["one", "three"]
    assert rows["C:8-9"]["source_record_ids"] == ["three"]
    assert rows["D:10-12"]["synthetic_record_ids"] == ["generated"]
    assert rows["unmapped-grade"]["source_record_ids"] == ["two"]
    with pytest.raises(ValueError):
        band(13)
