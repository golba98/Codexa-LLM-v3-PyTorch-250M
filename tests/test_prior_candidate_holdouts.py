"""Original synthetic fixtures protect measured prior holdouts from new training."""

import hashlib
import json
import random
import sqlite3
import string
import zlib
from pathlib import Path
from typing import Any

import pytest

from codexa.data import language
from codexa.data.production import digest
from codexa.data.v2_index import Index
from codexa.data.v2_quality import canonical, features
from codexa.data.v2_storage import Storage


def original_text(seed: int) -> str:
    rng = random.Random(seed)
    return " ".join("".join(rng.choices(string.ascii_lowercase, k=8)) for _ in range(200)) + "."


def add(index: Index, text: str, split: str, role: str = "new") -> str:
    ident = hashlib.sha256(canonical(text).encode()).digest()
    _, signature, anchors = features(text)
    index.insert(
        ident, "original-test-fixture", "narrative", split, text, signature, anchors, {}, role=role
    )
    return ident.hex()


@pytest.fixture
def corpus(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(Storage, "check", lambda self, allocation=0: None)
    historical = tmp_path / "historical.sqlite"
    with sqlite3.connect(historical) as db:
        db.execute("CREATE TABLE docs(id TEXT PRIMARY KEY,text TEXT)")
    prior_root = tmp_path / "prior"
    Storage(prior_root).initialize()
    previous = Index(Storage(prior_root), historical)
    texts = {split: original_text(seed) for seed, split in enumerate(("train", "val", "test"))}
    ids = {split: add(previous, text, split) for split, text in texts.items()}
    previous.db.commit()
    previous.close()
    snapshot = prior_root / "snapshots" / "frozen"
    snapshot.mkdir(parents=True)
    listing = snapshot / "documents.jsonl"
    listing.write_text(
        "".join(json.dumps({"id": ids[split], "split": split}) + "\n" for split in texts)
    )
    manifest = snapshot / "manifest.json"
    manifest.write_text(json.dumps({"documents_sha256": digest(listing)}))
    decision = {
        "version": language.V4_VERSION,
        "prior_candidate_holdouts": [
            {
                "root": str(prior_root),
                "snapshot": str(snapshot),
                "manifest_sha256": digest(manifest),
            }
        ],
    }
    current_root = tmp_path / "current"
    Storage(current_root).initialize()
    current = Index(Storage(current_root), historical)
    # Native binary integrity is independently tested elsewhere; this fixture
    # isolates the helper's manifest/listing/body/role binding using a real Index.
    monkeypatch.setattr(
        language, "verify", lambda path: json.loads((path / "manifest.json").read_text())
    )
    state: dict[str, Any] = {
        "root": prior_root,
        "snapshot": snapshot,
        "manifest": manifest,
        "listing": listing,
        "decision": decision,
        "index": current,
        "texts": texts,
        "ids": ids,
    }
    try:
        yield state
    finally:
        current.close()


def refresh_binding(state: dict[str, Any]) -> None:
    state["manifest"].write_text(json.dumps({"documents_sha256": digest(state["listing"])}))
    state["decision"]["prior_candidate_holdouts"][0]["manifest_sha256"] = digest(state["manifest"])


def test_all_prior_holdouts_become_evaluation_and_block_exact_and_near_training(corpus):
    index, decision = corpus["index"], corpus["decision"]
    assert language.stage_prior_candidate_holdouts(index, decision) == 2
    assert index.db.execute("SELECT role,split FROM docs ORDER BY split").fetchall() == [
        ("evaluation", "test"),
        ("evaluation", "val"),
    ]
    for split, text in corpus["texts"].items():
        ident = bytes.fromhex(corpus["ids"][split])
        reason = index.duplicate(ident, text, *features(text))
        assert reason == (None if split == "train" else "exact_document")
        if split != "train":
            near = text.rsplit(" ", 1)[0] + " replacement."
            assert (
                index.duplicate(
                    hashlib.sha256(canonical(near).encode()).digest(), near, *features(near)
                )
                == "near_document"
            )
    rows = index.db.execute("SELECT hash,metadata FROM docs").fetchall()
    assert {bytes(row[0]).hex() for row in rows} == {corpus["ids"]["val"], corpus["ids"]["test"]}
    assert all(
        json.loads(row[1])["manifest_sha256"]
        == decision["prior_candidate_holdouts"][0]["manifest_sha256"]
        for row in rows
    )
    assert language.stage_prior_candidate_holdouts(index, decision) == 0


def test_absent_v4_prior_holdouts_is_noop(corpus):
    assert (
        language.stage_prior_candidate_holdouts(corpus["index"], {"version": language.V4_VERSION})
        == 0
    )
    assert corpus["index"].db.execute("SELECT COUNT(*) FROM docs").fetchone() == (0,)


@pytest.mark.parametrize("version", [language.VERSION, language.V2_VERSION, language.V3_VERSION])
def test_prior_holdouts_require_v4_admission(corpus, version):
    corpus["decision"]["version"] = version
    with pytest.raises(ValueError, match="explicit version 4"):
        language.stage_prior_candidate_holdouts(corpus["index"], corpus["decision"])


@pytest.mark.parametrize("changed", ["manifest", "listing"])
def test_manifest_and_listing_are_hash_bound(corpus, changed):
    corpus[changed].write_text(corpus[changed].read_text() + " ")
    with pytest.raises(ValueError, match=f"{changed} changed"):
        language.stage_prior_candidate_holdouts(corpus["index"], corpus["decision"])


def test_prior_snapshot_must_stay_inside_its_root(corpus):
    corpus["decision"]["prior_candidate_holdouts"][0]["root"] = str(corpus["root"] / "unrelated")
    with pytest.raises(ValueError, match="outside its recorded root"):
        language.stage_prior_candidate_holdouts(corpus["index"], corpus["decision"])


def test_missing_prior_index_is_not_created(corpus):
    path = corpus["root"] / "indexes/documents.sqlite"
    path.unlink()
    with pytest.raises(sqlite3.OperationalError, match="unable to open"):
        language.stage_prior_candidate_holdouts(corpus["index"], corpus["decision"])
    assert not path.exists()


@pytest.mark.parametrize("change", ["split", "omitted_listing", "omitted_index", "extra_index"])
def test_prior_listing_and_index_require_exact_holdout_coverage(corpus, change):
    rows = list(map(json.loads, corpus["listing"].read_text().splitlines()))
    with sqlite3.connect(corpus["root"] / "indexes/documents.sqlite") as db:
        if change == "split":
            rows[-1]["split"] = "val"
        elif change == "omitted_listing":
            rows = rows[:-1]
        elif change == "omitted_index":
            db.execute("DELETE FROM docs WHERE hash=?", (bytes.fromhex(corpus["ids"]["test"]),))
        else:
            # The prior train row cannot silently be substituted for a listed holdout.
            db.execute("UPDATE docs SET split='val' WHERE split='train'")
    corpus["listing"].write_text("".join(json.dumps(row) + "\n" for row in rows))
    refresh_binding(corpus)
    with pytest.raises(ValueError, match="differs from listing|incomplete"):
        language.stage_prior_candidate_holdouts(corpus["index"], corpus["decision"])


def test_modified_prior_body_cannot_retain_old_document_identity(corpus):
    with sqlite3.connect(corpus["root"] / "indexes/documents.sqlite") as db:
        db.execute(
            "UPDATE docs SET payload=? WHERE split='val'",
            (zlib.compress(original_text(99).encode()),),
        )
    with pytest.raises(ValueError, match="text changed"):
        language.stage_prior_candidate_holdouts(corpus["index"], corpus["decision"])
