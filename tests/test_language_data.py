import hashlib
import io
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from codexa.data.language import RangeReader, cleaned, load_sources, prepare, profile


class Response(io.BytesIO):
    status = 206

    def __init__(self, payload: bytes, header: str):
        super().__init__(payload)
        self.headers = {"Content-Range": header}


def test_ranges_validate_identity_and_budget(tmp_path: Path) -> None:
    reader = RangeReader("https://example.invalid/pinned", 10, tmp_path / "cache", 5)
    with patch("urllib.request.urlopen", return_value=Response(b"abc", "bytes 0-2/10")):
        assert reader.read(3) == b"abc"
    assert reader.tell() == 3
    assert reader.requests[0]["sha256"] == hashlib.sha256(b"abc").hexdigest()
    with pytest.raises(OSError, match="budget"):
        reader.read(3)
    assert reader.seek(-2, 2) == 8
    with (
        patch("urllib.request.urlopen", return_value=Response(b"xx", "bytes 0-1/10")),
        pytest.raises(ValueError, match="changed"),
    ):
        reader.read(2)


def test_server_ignoring_ranges_fails_before_read(tmp_path: Path) -> None:
    response = Response(b"entire huge shard", "bytes 0-2/10")
    response.status = 200
    reader = RangeReader("https://example.invalid/pinned", 10, tmp_path / "cache", 5)
    with (
        patch("urllib.request.urlopen", return_value=response),
        pytest.raises(ValueError, match="ignored"),
    ):
        reader.read(3)
    assert reader.transferred == 0


def test_source_payload_pins_required(tmp_path: Path) -> None:
    config = tmp_path / "sources.yaml"
    config.write_text("sources:\n- name: bad\n  revision: main\n  sha256: unknown\n")
    with pytest.raises(ValueError, match="Pinned"):
        load_sources(config, tmp_path)


def test_source_specific_filter_preserves_normal_math() -> None:
    text = "A recipe is a set of instructions for preparing food. A cook can change the quantities to serve more people. For example, doubling a recipe changes 2 cups into 4 cups. Measurements help cooks obtain consistent results."
    cleaned_text, error = cleaned(
        {"text": text},
        {"name": "cosmopedia-v2", "category": "general", "filter": {"require_terminal": True}},
    )
    assert error is None
    assert "2 cups into 4 cups" in cleaned_text
    _, error = cleaned(
        {"text": text[:-1]},
        {"name": "cosmopedia-v2", "category": "general", "filter": {"require_terminal": True}},
    )
    assert error == "admission_incomplete_prose"


def test_preparation_refuses_missing_rights_before_outputs(tmp_path: Path) -> None:
    admission = tmp_path / "admission.json"
    admission.write_text(json.dumps({"version": "codexa-language-data-1", "status": "pending"}))
    with patch("codexa.data.language.Storage.load") as storage:
        with pytest.raises(ValueError, match="development"):
            prepare(tmp_path, tmp_path, tmp_path / "tokenizer", tmp_path / "fixtures", admission)
        storage.return_value.check.assert_called_once()
    assert not (tmp_path / "snapshots").exists()


def test_profile_rejects_unbounded_comparison(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="bounded"):
        profile(tmp_path, tmp_path, tmp_path, tmp_path, 10001)


def test_interleaving_prevents_first_payload_consuming_quota() -> None:
    from codexa.data.language import prepared_rows

    def rows(spec, cache, budget):
        for ordinal in (1, 2):
            yield ordinal, {"text": spec["name"]}

    with patch("codexa.data.language.source_rows", side_effect=rows):
        result = list(prepared_rows([{"name": "a"}, {"name": "b"}]))
    assert [(spec["name"], ordinal) for spec, ordinal, _ in result] == [
        ("a", 1),
        ("b", 1),
        ("a", 2),
        ("b", 2),
    ]


def test_multi_payload_comparison_keeps_total_row_budget(tmp_path: Path) -> None:
    from codexa.data.language import comparison_rows

    def rows(spec, cache, budget):
        for ordinal in range(10):
            yield ordinal, {"text": spec["sha256"]}

    with patch("codexa.data.language.source_rows", side_effect=rows):
        result = list(comparison_rows([{"sha256": "a"}, {"sha256": "b"}], tmp_path, 5, 0))
    assert len(result) == 5
    assert [spec["sha256"] for spec, _, _ in result] == ["a", "a", "a", "b", "b"]


def test_reference_excludes_untrained_candidates_and_is_readonly(tmp_path: Path) -> None:
    import sqlite3

    import numpy as np

    from codexa.data.language import Reference

    (tmp_path / "indexes").mkdir()
    history = tmp_path / "historical.sqlite"
    with sqlite3.connect(history) as db:
        db.execute("CREATE TABLE docs(id TEXT,text TEXT)")
    reference = tmp_path / "indexes/documents.sqlite"
    with sqlite3.connect(reference) as db:
        db.executescript(
            "CREATE TABLE docs(id INTEGER,hash BLOB,role TEXT,payload BLOB); CREATE TABLE bands(band BLOB,doc INTEGER); CREATE TABLE anchors(hash BLOB,doc INTEGER,pos INTEGER);"
        )
        for n, role in enumerate(("history", "evaluation", "new"), 1):
            db.execute("INSERT INTO docs VALUES(?,?,?,?)", (n, bytes([n]) * 32, role, b""))
    old = Reference(tmp_path, history)
    try:
        assert (
            old.duplicate(bytes([1]) * 32, "text", np.array([], dtype=np.uint64), b"x" * 512, [])
            == "exact_document"
        )
        assert (
            old.duplicate(bytes([2]) * 32, "text", np.array([], dtype=np.uint64), b"x" * 512, [])
            == "exact_document"
        )
        assert (
            old.duplicate(bytes([3]) * 32, "text", np.array([], dtype=np.uint64), b"x" * 512, [])
            is None
        )
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            old.db.execute("DELETE FROM ref.docs")
    finally:
        old.close()


def test_rights_evidence_cannot_be_a_bare_approval_label(tmp_path: Path) -> None:
    from codexa.data.language import rights_evidence

    with pytest.raises(ValueError, match="reviewed"):
        rights_evidence({"rights_status": "approved", "rights_evidence": "reputable publisher"})
    evidence = tmp_path / "rights.json"
    evidence.write_text('{"status":"approved"}')
    spec = {
        "rights_status": "approved",
        "rights_evidence": {
            "path": str(evidence),
            "sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
        },
    }
    assert rights_evidence(spec)["path"] == str(evidence)
    evidence.write_text('{"status":"pending"}')
    with pytest.raises(ValueError, match="changed"):
        rights_evidence(spec)


@pytest.mark.parametrize(
    "version,adapter",
    [
        ("codexa-language-data-1", None),
        ("codexa-language-data-2", None),
        ("codexa-language-data-3", "codexa-dclm-english-1"),
        ("codexa-language-data-3", "codexa-pressbooks-english-1"),
        ("codexa-language-data-4", None),
    ],
)
def test_prepare_and_independent_reconstruction_then_token_tamper(
    tmp_path: Path, monkeypatch, version, adapter
) -> None:
    import random
    import sqlite3
    import string

    from codexa.data import language
    from codexa.data.production import atomic, digest
    from codexa.data.v2_index import Index, Protection
    from codexa.data.v2_storage import Storage
    from codexa.evaluation.language import freeze
    from codexa.tokenizer.bpe import CodexaTokenizer

    monkeypatch.setattr(Storage, "check", lambda self, allocation=0: None)
    root, reference = tmp_path / "language", tmp_path / "reference"
    Storage(root).initialize()
    Storage(reference).initialize()
    historical = tmp_path / "parent/curated/documents.sqlite"
    historical.parent.mkdir(parents=True)
    with sqlite3.connect(historical) as db:
        db.execute("CREATE TABLE docs(id TEXT PRIMARY KEY,text TEXT)")
    atomic(historical.parent / "manifest.json", {"fixture": "historical"})
    (historical.parent.parent / "tokens").mkdir()
    atomic(historical.parent.parent / "tokens/manifest.json", {"fixture": "parent-dataset"})
    index = Index(Storage(reference), historical)
    index.close()
    atomic(reference / "reports/historical-index.json", {"input": str(historical)})
    Protection.build([]).save(reference / "indexes/protected-v2.npz")
    old_fixtures = reference / "benchmarks/v3"
    old_fixtures.mkdir(parents=True)
    for split in ("dev", "test"):
        (old_fixtures / (split + ".jsonl")).write_text("")
    fixtures = root / "benchmarks/language-v1"
    freeze(fixtures)
    development = root / "reports/development"
    development.mkdir()
    atomic(development / "comparison.json", {"fixture": "development"})
    rights = root / "reports/rights.json"
    atomic(rights, {"status": "approved", "scope": "original CC0 test fixtures"})
    specs, source_rows_by_name, all_text = [], {}, []
    source_mix = (
        (("wikipedia-en", 0.6), ("tinystories-gpt4", 0.15), ("cosmopedia-v2", 0.25))
        if version == language.VERSION
        else (("dclm-edu" if adapter == "codexa-dclm-english-1" else "pressbooks-filtered", 1.0),)
        if version == language.V3_VERSION
        else (("tinystories-gpt4", 1.0),)
        if version == language.V4_VERSION
        else (("common-corpus", 1.0),)
    )
    for source, share in source_mix:
        rows = []
        for split in ("train", "val", "test"):
            group_number = 0
            while True:
                group = f"{source}-{split}-{group_number}"
                if version == language.V3_VERSION:
                    group = "https://example.org/" + group
                bucket = (
                    int(hashlib.sha256(("language-split-1:" + group).encode()).hexdigest()[:16], 16)
                    / 2**64
                )
                actual_split = "test" if bucket < 1 / 62 else "val" if bucket < 2 / 62 else "train"
                if split == actual_split:
                    break
                group_number += 1
            rng = random.Random(group)
            words = ["".join(rng.choices(string.ascii_lowercase, k=8)) for _ in range(120)]
            text = ". ".join(" ".join(words[i : i + 30]) for i in range(0, 120, 30)) + "."
            row = {"text": text, "id": group}
            if version == language.V2_VERSION:
                row.update(
                    identifier=group,
                    collection="Original prose fixtures",
                    open_type="Open Science",
                    language="English",
                    language_type="Written",
                    date=2022,
                    license="CC0-1.0",
                )
            if version == language.V3_VERSION:
                if adapter == "codexa-dclm-english-1":
                    row.update(
                        url=group,
                        edu_int_score=4,
                        edu_score=3.8,
                        language="en",
                        language_score=0.99,
                    )
                else:
                    row["metadata"] = {
                        "book_url": group,
                        "url": group + "/chapter/",
                        "license": "https://creativecommons.org/publicdomain/zero/1.0/",
                        "title": "Original fictional prose",
                        "subject": "Language",
                    }
            rows.append(row)
            all_text.append(text)
        source_rows_by_name[source] = rows
        specs.append(
            {
                "name": source,
                "category": "general",
                "sha256": hashlib.sha256(source.encode()).hexdigest(),
                "revision": "a" * 40,
                "license": "CC0-1.0 original test fixtures",
                "local_path": str(tmp_path / source),
                "target_share": share,
                "rights_status": "approved",
                "rights_evidence": {"path": str(rights), "sha256": digest(rights)},
            }
        )
    if version == language.V2_VERSION:
        from codexa.data.common_corpus import POLICY, REPO, REVISION

        proof = root / "reports/original-license.txt"
        proof.write_text("CC0 original test fixtures; not external source clearance.")
        atomic(
            rights,
            {
                "status": "approved",
                "repo": REPO,
                "revision": REVISION,
                "policy": POLICY,
                "reviewer": "fixture-only reviewer",
                "coverage": [
                    {
                        "collection": "Original prose fixtures",
                        "licenses": ["CC0-1.0"],
                        "scope": "all_admitted_documents",
                        "verification": "original_license_verified",
                        "attribution": "Codexa original fixtures",
                        "basis_url": "https://example.org/fixture-license",
                        "origin_prefixes": ["common-corpus-"],
                        "proof": {"path": str(proof), "sha256": digest(proof)},
                    }
                ],
            },
        )
        for spec in specs:
            spec.update(repo=REPO, revision=REVISION, adapter=POLICY)
            spec["rights_evidence"]["sha256"] = digest(rights)
    if version == language.V3_VERSION:
        from codexa.data.language_sources import ADAPTERS, POLICY

        proof = root / "reports/original-license.txt"
        proof.write_text("Original CC0 fictional fixtures; not external source approval.")
        coverage = []
        for row in source_rows_by_name[specs[0]["name"]]:
            grant = {
                "license": "CC0-1.0",
                "scope": "original_documents"
                if adapter == "codexa-dclm-english-1"
                else "original_book",
                "verification": "original_license_verified",
                "attribution": "Original Codexa fixture author",
                "basis_url": "https://example.org/fixture-license",
                "proof": {"path": str(proof), "sha256": digest(proof)},
            }
            if adapter == "codexa-dclm-english-1":
                grant["origin_prefixes"] = [row["url"]]
            else:
                grant["book_url"] = row["metadata"]["book_url"]
            coverage.append(grant)
        repo, revision = ADAPTERS[adapter]
        atomic(
            rights,
            {
                "status": "approved",
                "repo": repo,
                "revision": revision,
                "adapter": adapter,
                "policy": POLICY,
                "reviewer": "original fixture reviewer",
                "coverage": coverage,
            },
        )
        for spec in specs:
            spec.update(repo=repo, revision=revision, adapter=adapter)
            spec["rights_evidence"]["sha256"] = digest(rights)
    tokenizer = root / "tokenizer.json"
    if version == language.V4_VERSION:
        from codexa.data import language_sources

        specs[0]["category"] = "narrative"
        # This test exercises native packing/reconstruction with original fixtures;
        # the real creator-evidence checks have separate rejection/acceptance tests.
        monkeypatch.setattr(language_sources, "validate_narrative_rights", lambda spec, proof: True)
        monkeypatch.setattr(language, "stage_narrative_validation", lambda index, reference: 0)
    CodexaTokenizer.train(all_text, 512).save(tokenizer)
    monkeypatch.setattr(language, "TOKENIZER_SHA", digest(tokenizer))
    monkeypatch.setattr(language, "SPLIT_TARGETS", {"train": 100, "val": 100, "test": 100})
    monkeypatch.setattr(language, "V3_MIN_TRAIN_TOKENS", 100)
    monkeypatch.setattr(language, "V3_MIN_EVALUATION_TOKENS", 100)

    def source_rows(spec, cache, budget):
        yield from enumerate(source_rows_by_name[spec["name"]], 1)

    monkeypatch.setattr(language, "source_rows", source_rows)
    admission = root / "source-admission.json"
    atomic(
        admission,
        {
            "version": version,
            "policy": "codexa-common-modern-english-1"
            if version == language.V2_VERSION
            else "codexa-language-sources-3"
            if version == language.V3_VERSION
            else "codexa-tinystories-creator-cdla-1"
            if version == language.V4_VERSION
            else None,
            "status": "admitted",
            "educational_source": "cosmopedia-v2",
            "frozen_development_sha256": digest(development / "comparison.json"),
            "sources": specs,
        },
    )
    prepared = language.prepare(root, reference, tokenizer, fixtures, admission)
    assert prepared["training_allowed"] is False
    snapshot = root / "snapshots/language-60m"
    proof = language.verify_snapshot(root, snapshot, reference, tokenizer, fixtures, admission)
    assert proof["status"] == "passed"
    assert proof["documents_reconstructed"] == (9 if version == language.VERSION else 3)
    assert all(
        proof[key]
        for key in (
            "all_tokens_reencoded",
            "protection_verified",
            "split_dedup_verified",
            "prior_exposure_verified",
            "rights_verified",
        )
    )
    (snapshot / "independent-verification.json").rename(root / "reports/original-proof.json")
    if version == language.V3_VERSION:
        original_manifest = (snapshot / "manifest.json").read_text()
        changed_manifest = json.loads(original_manifest)
        changed_manifest["identity"]["split_targets"]["train"] -= 1
        atomic(snapshot / "manifest.json", changed_manifest)
        with pytest.raises(ValueError, match="token targets differ"):
            language.verify_snapshot(root, snapshot, reference, tokenizer, fixtures, admission)
        (snapshot / "manifest.json").write_text(original_manifest)
        with sqlite3.connect(root / "indexes/documents.sqlite") as db:
            number, metadata_text = db.execute(
                "SELECT id,metadata FROM docs WHERE split='train' ORDER BY id LIMIT 1"
            ).fetchone()
            changed = json.loads(metadata_text)
            if adapter == "codexa-dclm-english-1":
                changed["language_source"]["edu_int_score"] = 5
            else:
                changed["language_source"]["metadata"]["title"] = "Forged title"
            db.execute("UPDATE docs SET metadata=? WHERE id=?", (json.dumps(changed), number))
        with pytest.raises(ValueError, match="Original language metadata"):
            language.verify_snapshot(root, snapshot, reference, tokenizer, fixtures, admission)
        with sqlite3.connect(root / "indexes/documents.sqlite") as db:
            db.execute("UPDATE docs SET metadata=? WHERE id=?", (metadata_text, number))
    with sqlite3.connect(root / "indexes/documents.sqlite") as db:
        number, original_metadata = db.execute(
            "SELECT id,metadata FROM docs WHERE split='train' ORDER BY id LIMIT 1"
        ).fetchone()
        changed = json.loads(original_metadata)
        candidate = 0
        while True:
            group = f"forged-origin-{candidate}"
            bucket = (
                int(hashlib.sha256(("language-split-1:" + group).encode()).hexdigest()[:16], 16)
                / 2**64
            )
            if bucket >= 2 / 62:
                break
            candidate += 1
        changed["id"] = group
        db.execute("UPDATE docs SET metadata=? WHERE id=?", (json.dumps(changed), number))
    with pytest.raises(ValueError, match="Original source grouping"):
        language.verify_snapshot(root, snapshot, reference, tokenizer, fixtures, admission)
    with sqlite3.connect(root / "indexes/documents.sqlite") as db:
        db.execute("UPDATE docs SET metadata=? WHERE id=?", (original_metadata, number))
    train_total = json.loads((snapshot / "manifest.json").read_text())["splits"]["train"][0][
        "tokens"
    ]
    monkeypatch.setattr(
        language, "SPLIT_TARGETS", {"train": int(train_total * 0.9), "val": 100, "test": 100}
    )
    with pytest.raises(ValueError, match="Frozen source quota|token targets differ"):
        language.verify_snapshot(root, snapshot, reference, tokenizer, fixtures, admission)
    monkeypatch.setattr(language, "SPLIT_TARGETS", {"train": 100, "val": 100, "test": 100})
    token_file = snapshot / "train-00000.bin"
    payload = bytearray(token_file.read_bytes())
    payload[4] ^= 1
    token_file.write_bytes(payload)
    with pytest.raises(ValueError, match="Shard integrity"):
        language.verify_snapshot(root, snapshot, reference, tokenizer, fixtures, admission)


def test_v4_reference_protects_prior_holdouts_and_keeps_untrained_training(tmp_path, monkeypatch):
    import random
    import sqlite3
    import string

    from codexa.data import language
    from codexa.data.v2_index import Index
    from codexa.data.v2_quality import features
    from codexa.data.v2_storage import Storage

    monkeypatch.setattr(Storage, "check", lambda self, allocation=0: None)
    reference = tmp_path / "reference"
    Storage(reference).initialize()
    historical = tmp_path / "historical.sqlite"
    with sqlite3.connect(historical) as db:
        db.execute("CREATE TABLE docs(id TEXT PRIMARY KEY,text TEXT)")
    index = Index(Storage(reference), historical)
    records = {}
    for seed, (role, split) in enumerate(
        (("new", "train"), ("new", "val"), ("new", "test"), ("evaluation", "train"))
    ):
        rng = random.Random(seed)
        words = ["".join(rng.choices(string.ascii_lowercase, k=8)) for _ in range(200)]
        text = " ".join(words) + "."
        ident = hashlib.sha256(text.encode()).digest()
        shingles, signature, anchors = features(text)
        number = index.insert(
            ident, "original-fixture", "narrative", split, text, signature, anchors, {}, 250, role
        )
        records[(role, split)] = (ident, text, shingles, signature, anchors, number)
    index.db.commit()
    index.close()
    legacy = language.Reference(reference, historical)
    current = language.Reference(reference, historical, include_candidate_holdouts=True)
    try:
        for key, (ident, text, shingles, signature, anchors, _) in records.items():
            result = current.duplicate(ident, text, shingles, signature, anchors)
            assert result == (None if key == ("new", "train") else "exact_document")
        ident, text, shingles, signature, anchors, _ = records[("new", "val")]
        assert legacy.duplicate(ident, text, shingles, signature, anchors) is None
        near_text = text.rsplit(" ", 1)[0] + " changedword."
        near_features = features(near_text)
        assert current.duplicate(
            hashlib.sha256(near_text.encode()).digest(), near_text, *near_features
        )
        expected = {row[-1] for key, row in records.items() if key != ("new", "train")}
        assert {row[0] for row in current.db.execute("SELECT DISTINCT doc FROM bands")} == expected
        assert {
            row[0] for row in current.db.execute("SELECT DISTINCT doc FROM anchors")
        } == expected
    finally:
        legacy.close()
        current.close()
    frozen = language.reference_holdout_identity(reference)
    with sqlite3.connect(reference / "indexes/documents.sqlite") as db:
        db.execute("UPDATE docs SET tokens=tokens+1 WHERE role='new' AND split='train'")
    assert language.reference_holdout_identity(reference) == frozen
    with sqlite3.connect(reference / "indexes/documents.sqlite") as db:
        db.execute("UPDATE docs SET tokens=tokens+1 WHERE role='new' AND split='val'")
    assert language.reference_holdout_identity(reference) != frozen


def test_v4_publisher_validation_is_reserved_and_hash_bound(tmp_path, monkeypatch):
    import sqlite3

    from codexa.data import language
    from codexa.data.v2_index import Index
    from codexa.data.v2_quality import features
    from codexa.data.v2_storage import Storage

    monkeypatch.setattr(Storage, "check", lambda self, allocation=0: None)
    reference = tmp_path / "reference"
    Storage(reference).initialize()
    text = "The curious bird watched the gardener carry a basket through the gate. " * 3
    path = reference / "raw" / language.NARRATIVE_VALIDATION_FILE
    path.write_text(text + "\n<|endoftext|>\n")
    monkeypatch.setattr(
        language, "NARRATIVE_VALIDATION_SHA", hashlib.sha256(path.read_bytes()).hexdigest()
    )
    monkeypatch.setattr(language, "NARRATIVE_VALIDATION_BYTES", path.stat().st_size)
    historical = tmp_path / "historical.sqlite"
    with sqlite3.connect(historical) as db:
        db.execute("CREATE TABLE docs(id TEXT PRIMARY KEY,text TEXT)")
    index = Index(Storage(reference), historical)
    try:
        assert language.stage_narrative_validation(index, reference) == 1
        ident = hashlib.sha256(language.canonical(text).encode()).digest()
        assert index.duplicate(ident, text, *features(text)) == "exact_document"
        assert index.db.execute("SELECT role,split FROM docs").fetchall() == [("evaluation", "val")]
        assert language.stage_narrative_validation(index, reference) == 0
        path.write_text(path.read_text().replace("bird", "fish"))
        with pytest.raises(ValueError, match="validation file changed"):
            language.stage_narrative_validation(index, reference)
    finally:
        index.close()
