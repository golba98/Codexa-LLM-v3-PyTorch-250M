"""Fail-closed document rights and provenance for the new language corpus."""

import hashlib
import json
from unittest.mock import patch

import pytest

from codexa.data.common_corpus import POLICY, REPO, REVISION, covered, eligible, grouping
from codexa.data.language import (
    V2_VERSION,
    RangeReader,
    development_charge,
    source_rows,
    source_shares,
)


def source():
    return {
        "name": "common-corpus",
        "repo": REPO,
        "revision": REVISION,
        "adapter": POLICY,
        "category": "general",
        "target_share": 1.0,
        "sha256": "a" * 64,
    }


def document():
    return {
        "identifier": "https://example.org/articles/trees.txt_1",
        "collection": "Nature essays",
        "open_type": "Open Science",
        "language": "English",
        "language_type": "Written",
        "license": "CC-BY-4.0",
        "date": 2022,
        "title": "Trees",
        "text": "Trees have roots and branches. Roots take in water from the soil.",
    }


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"license": "Various open science"}, "common_unresolved_license"),
        ({"license": "CC-By-SA"}, "common_unresolved_license"),
        ({"date": 1908}, "common_not_modern"),
        ({"date": None}, "common_not_modern"),
        ({"language": "French"}, "common_not_written_english"),
        ({"collection": "USPTO"}, "common_excluded_collection"),
        ({"collection": "SEC filings"}, "common_excluded_collection"),
        ({"text": "Solve the equation x + y = 2."}, "common_math_or_missing_text"),
        ({"title": "An introduction to algebra"}, "common_math_or_missing_text"),
    ],
)
def test_rejects_ineligible_records(change, reason):
    assert eligible(document() | change, source()) == reason


def test_eligibility_never_confers_rights_and_coverage_requires_original_proof(tmp_path):
    row, spec = document(), source()
    assert eligible(row, spec) is None
    assert not covered(row, spec, {"status": "approved"})
    proof = tmp_path / "original-license.txt"
    proof.write_text("Original fictional fixture license evidence.")
    grant = {
        "collection": row["collection"],
        "licenses": [row["license"]],
        "scope": "all_admitted_documents",
        "verification": "original_license_verified",
        "attribution": "Original author",
        "basis_url": "https://example.org/license",
        "origin_prefixes": ["https://example.org/articles/"],
        "proof": {"path": str(proof), "sha256": hashlib.sha256(proof.read_bytes()).hexdigest()},
    }
    evidence = {
        "status": "approved",
        "repo": REPO,
        "revision": REVISION,
        "policy": POLICY,
        "reviewer": "fixture reviewer",
        "coverage": [grant],
    }
    assert covered(row, spec, evidence)
    assert not covered(row | {"identifier": "https://other.example/tree"}, spec, evidence)
    assert not covered(row | {"license": "CC0"}, spec, evidence)
    proof.write_text("Changed evidence")
    assert not covered(row, spec, evidence)


def test_parts_of_origin_share_split_group():
    assert grouping(document()) == grouping(
        document() | {"identifier": "https://example.org/articles/trees.txt_2"}
    )


def test_local_parquet_preserves_metadata_and_samples_across_full_payload(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    path = tmp_path / "sample.parquet"
    rows = [document() | {"identifier": f"https://example.org/articles/{i}"} for i in range(100)]
    pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=20)
    spec = source() | {
        "local_path": str(path),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "_development_limit": 4,
    }
    result = list(source_rows(spec, tmp_path / "unused", 0))
    assert [ordinal for ordinal, _ in result] == [13, 38, 63, 88]
    assert all(
        row["license"] == "CC-BY-4.0" and row["collection"] == "Nature essays" for _, row in result
    )


def test_v2_source_shares_support_multiple_shards_and_reject_drift():
    decision = {
        "version": V2_VERSION,
        "policy": POLICY,
        "sources": [source(), source() | {"sha256": "b" * 64}],
    }
    assert source_shares(decision) == {"common-corpus": 1.0}
    decision["sources"][1]["target_share"] = 0.5
    with pytest.raises(ValueError, match="Inconsistent"):
        source_shares(decision)
    decision["sources"][1]["target_share"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        source_shares(decision)


def test_interrupted_range_is_charged_before_network_and_survives_failure(tmp_path):
    root = tmp_path
    (root / "raw").mkdir()
    cache = root / "cache" / "partial"
    reader = RangeReader("https://example.invalid/pinned", 100, cache, 100)
    with (
        patch("urllib.request.urlopen", side_effect=KeyboardInterrupt),
        pytest.raises(KeyboardInterrupt),
    ):
        reader.read(40)
    receipt = json.loads((cache / "attempted-ranges.json").read_text())
    assert receipt["attempted_bytes"] == 40
    assert reader.transferred == 0
    assert development_charge(root) == 40
    (root / "raw" / "staged.parquet").write_bytes(b"x" * 10)
    assert development_charge(root) == 50


def test_acquisition_caps_charge_interrupted_requests_and_require_frozen_admission(
    tmp_path, monkeypatch
):
    import yaml

    from codexa.data.language import acquire_next
    from codexa.data.production import atomic, digest
    from codexa.data.v2_storage import GIB, Storage

    monkeypatch.setattr(Storage, "check", lambda self, allocation=0: None)
    Storage(tmp_path, maximum=24 * GIB, raw_maximum=16 * GIB).initialize()
    spec = source() | {
        "path": "data/fixture.parquet",
        "bytes": 10,
        "license": "Document rights required",
        "filter": {"require_terminal": True},
    }
    config = tmp_path / "sources.yaml"
    config.write_text(yaml.safe_dump({"version": V2_VERSION, "sources": [spec]}))
    attempted = tmp_path / "cache/interrupted/attempted-ranges.json"
    attempted.parent.mkdir()
    atomic(attempted, {"attempted_bytes": 16 * GIB})
    with patch("codexa.data.v2_acquire.acquire") as transfer:
        with pytest.raises(OSError, match="Cumulative"):
            acquire_next(tmp_path, tmp_path, config)
        transfer.assert_not_called()
    atomic(attempted, {"attempted_bytes": GIB})
    with patch("codexa.data.v2_acquire.acquire") as transfer:
        with pytest.raises(ValueError, match="admission"):
            acquire_next(tmp_path, tmp_path, config)
        transfer.assert_not_called()

    development = tmp_path / "reports/development/comparison.json"
    development.parent.mkdir()
    atomic(development, {"fixture": "bounded development completed"})
    proof = tmp_path / "reports/original-license.txt"
    proof.write_text("Original CC0 fictional test fixture proof.")
    evidence = tmp_path / "reports/rights.json"
    atomic(
        evidence,
        {
            "status": "approved",
            "repo": REPO,
            "revision": REVISION,
            "policy": POLICY,
            "reviewer": "fixture reviewer",
            "coverage": [
                {
                    "collection": "Nature essays",
                    "licenses": ["CC0-1.0"],
                    "scope": "all_admitted_documents",
                    "verification": "original_license_verified",
                    "attribution": "Original fixtures",
                    "basis_url": "https://example.org/license",
                    "origin_prefixes": ["https://example.org/articles/"],
                    "proof": {"path": str(proof), "sha256": digest(proof)},
                }
            ],
        },
    )
    admitted_spec = spec | {
        "rights_status": "approved",
        "rights_evidence": {"path": str(evidence), "sha256": digest(evidence)},
    }
    admission = tmp_path / "admission.json"
    decision = {
        "version": V2_VERSION,
        "status": "admitted",
        "policy": POLICY,
        "frozen_development_sha256": digest(development),
        "sources": [admitted_spec],
    }
    atomic(admission, decision)
    with patch(
        "codexa.data.v2_acquire.acquire", return_value=tmp_path / "raw/fixture.parquet"
    ) as transfer:
        report = acquire_next(tmp_path, tmp_path, config, admission)
        assert report["training_allowed"] is False
        transfer.assert_called_once()
    admitted_spec["filter"] = {"require_terminal": False}
    atomic(admission, decision)
    with patch("codexa.data.v2_acquire.acquire") as transfer:
        with pytest.raises(ValueError, match="filter differs"):
            acquire_next(tmp_path, tmp_path, config, admission)
        transfer.assert_not_called()
    admitted_spec["filter"] = spec["filter"]
    atomic(admission, decision)
    proof.write_text("Changed original source proof")
    with patch("codexa.data.v2_acquire.acquire") as transfer:
        with pytest.raises(ValueError, match="rights coverage"):
            acquire_next(tmp_path, tmp_path, config, admission)
        transfer.assert_not_called()
