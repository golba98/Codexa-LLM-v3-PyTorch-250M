import hashlib
import json
import sqlite3

import numpy as np
import pytest

from codexa.data.v2_index import Index, Protection
from codexa.data.v2_quality import bands, canonical, clean, features, shared_passage
from codexa.data.v2_storage import GIB, Storage

PROSE = (
    "A gardener planted several tomato seeds in a sunny part of her garden. "
    "She watered the soil each morning and checked that the young plants had enough light. "
    "After several weeks, green leaves appeared and the plants began to grow taller. "
)


def test_preparation_pytest_cache_location(tmp_path):
    import os
    import subprocess
    import sys

    from codexa.data.v2_storage import environment

    root = tmp_path / "volume with spaces"
    work = tmp_path / "working"
    work.mkdir()
    fixture = work / "test_example.py"
    fixture.write_text("def test_example():\n    assert 2 + 2 == 4\n")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(fixture)],
        cwd=work,
        env=os.environ | environment(root),
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (root / "cache/pytest/v/cache/nodeids").is_file()
    assert not (work / ".pytest_cache").exists()


def test_cleaning_preserves_prose_and_math():
    result, errors, stats = clean(PROSE + "\nYou must be logged in to post a comment.")
    assert not errors and "logged in" not in result and stats["removed_characters"] > 0
    assert clean(PROSE + "\ufffd")[1] == ["corrupt_unicode"]
    assert clean("The teacher explained this important idea to the class. " * 20)[1] == [
        "repeated_sentences"
    ]
    math = PROSE + "\nThe formula $x^2 + y^2 = z^2$ describes a right triangle."
    assert not clean(math, "math")[1]
    assert not clean(PROSE + "\nPhone: +27 12 345 6789")[1]
    assert "345 6789" not in clean(PROSE + "\nPhone: +27 12 345 6789")[0]
    exercise = "Share this observation with your class, then compare your sketches of leaf shapes."
    assert exercise in clean(exercise + "\n" + PROSE)[0]
    assert "Share on Facebook" not in clean(PROSE + "\nShare on Facebook")[0]


def test_wikipedia_bare_is_opening_gate_is_narrow():
    from codexa.data.v2_policy import admission_error

    approval = {"reject_bare_is_opening": True}
    assert admission_error("is a Japanese artist with a long career.", {}, approval) == (
        "admission_missing_subject"
    )
    assert admission_error("History is a study of the past.", {}, approval) is None
    assert admission_error("Is this a useful mathematical question?", {}, {}) is None


def test_compact_similarity_and_offset_passages():
    text = " ".join(f"educationalword{chr(97 + i // 26)}{chr(97 + i % 26)}" for i in range(160))
    a, signature, anchors = features(text)
    assert len(signature) == 512 and len(bands(signature)) == 16
    b, signature2, anchors2 = features("an introduction " + text + " a conclusion")
    assert len(np.intersect1d(a, b)) / len(np.union1d(a, b)) > 0.85
    common = [(p, q) for h, p in anchors for k, q in anchors2 if h == k]
    assert common and any(
        shared_passage(text, "an introduction " + text + " a conclusion", p, q) for p, q in common
    )
    assert signature == features(text)[1] and len(signature2) == 512


def test_protected_items_not_bare_answers(tmp_path):
    protection = Protection.build([("Which city is the national capital of France?", True)])
    assert protection.matches(
        "Question: Which city is the national capital of France? Answer: Paris."
    )
    assert not protection.matches("Paris is a city in France. " + PROSE)
    protection.save(tmp_path / "protected.npz")
    assert Protection.load(tmp_path / "protected.npz").matches(
        "Which city is the national capital of France?"
    )


def test_index_resume_and_historical_rejection(tmp_path, monkeypatch):
    monkeypatch.setattr(Storage, "check", lambda *a, **k: None)
    storage = Storage(tmp_path / "v2")
    storage.initialize()
    old = tmp_path / "old.sqlite"
    db = sqlite3.connect(old)
    db.execute("CREATE TABLE docs(id TEXT PRIMARY KEY,text TEXT)")
    ident = hashlib.sha256(canonical(PROSE).encode()).hexdigest()
    db.execute("INSERT INTO docs VALUES(?,?)", (ident, PROSE))
    db.commit()
    db.close()
    (tmp_path / "manifest.json").write_text(json.dumps({"fixture": True}))
    index = Index(storage, old)
    index.reference(workers=1)
    shingles, signature, anchors = features(PROSE)
    assert (
        index.duplicate(bytes.fromhex(ident), PROSE, shingles, signature, anchors)
        == "exact_document"
    )
    index.close()
    index = Index(storage, old)
    index.reference(workers=1)
    assert index.value("historical_documents") == 1
    index.close()


def test_storage_identity_and_reserve(tmp_path, monkeypatch):
    monkeypatch.setattr("subprocess.check_output", lambda *a, **k: "wrong\n")
    with pytest.raises(OSError, match="volume"):
        Storage(tmp_path).check()
    monkeypatch.setattr("subprocess.check_output", lambda *a, **k: "expected\n")
    monkeypatch.setattr("shutil.disk_usage", lambda *a: type("Usage", (), {"free": 40 * GIB})())
    with pytest.raises(OSError, match="reserve"):
        Storage(tmp_path, "expected").check(1)


def test_explicit_raw_ceiling_persists_and_bounds_batch(tmp_path, monkeypatch):
    from codexa.data.v2_catalog import select

    monkeypatch.setattr(Storage, "check", lambda *a, **k: None)
    root = tmp_path / "v2"
    storage = Storage(root, raw_maximum=45 * GIB)
    storage.initialize()
    loaded = Storage.load(root)
    assert loaded.raw_maximum == 45 * GIB
    (root / "source-registry.json").write_text(
        json.dumps(
            {
                "sources": [
                    {"name": "reviewed", "path": "a.parquet", "bytes": 2 * GIB, "target_share": 1.0}
                ]
            }
        )
    )
    (root / "source-admission.json").write_text(
        json.dumps({"sources": {"reviewed": {"status": "admitted", "target_share": 1.0}}})
    )
    assert len(select(loaded, 40, 1)["sources"]) == 1
    with pytest.raises(ValueError, match="configured raw ceiling"):
        select(loaded, 46, 1)


def test_frozen_benchmark_counts_and_unique_prompts(tmp_path):
    from collections import Counter

    from codexa.evaluation.fixtures import freeze

    freeze(tmp_path)
    base = tmp_path / "benchmarks/v3"
    splits = {}
    for split in ("dev", "test"):
        rows = [json.loads(x) for x in (base / (split + ".jsonl")).read_text().splitlines()]
        counts = Counter(x["kind"] for x in rows)
        assert counts == {
            "sentence": 100,
            "paragraph": 100,
            "context": 100,
            "vocabulary": 100,
            "fact": 100,
            "grammar": 200,
            "math": 300,
        }
        for kind in ("paragraph", "context", "fact", "vocabulary"):
            values = [x["prompt"] for x in rows if x["kind"] == kind]
            assert len(set(values)) == len(values)
        assert len({x["answer"] for x in rows if x["kind"] == "context"}) == 5
        splits[split] = {x["id"] for x in rows}
    assert splits["dev"].isdisjoint(splits["test"])
    assert freeze(tmp_path)["files"] == freeze(tmp_path)["files"]


def test_repetition_detector_and_prefix_metric():
    from codexa.evaluation.benchmark import repetition

    assert repetition([7, 8, 9] * 20)["severe_loop"]
    assert not repetition(list(range(64)))["severe_loop"]


def test_phase_policies_without_updates(prepared):
    import copy
    import random

    import torch

    from codexa.architecture.model import Codexa
    from codexa.config import Config
    from codexa.continuation import phase_state
    from codexa.training.optimizer import make_optimizer
    from codexa.training.scheduler import make_scheduler

    _data, tokenizer, _ = prepared
    config = Config()
    model = Codexa(config.model)
    optimizer = make_optimizer(model, config.training)
    scheduler = make_scheduler(optimizer, config.training)
    # Synthetic finite moment tensors; no backward or optimizer update is used.
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            optimizer.state[parameter] = {
                "step": torch.tensor(3.0),
                "exp_avg": torch.zeros_like(parameter),
                "exp_avg_sq": torch.ones_like(parameter) * 0.2,
            }
    parent = {
        "version": 1,
        "config": config.to_dict(),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": 3,
        "tokens": 768,
        "microsteps": 3,
        "sampler": {"algorithm": "feistel-v1", "size": 100, "seed": 42, "epoch": 0, "position": 3},
        "rng": {"python": random.getstate(), "torch": torch.get_rng_state(), "cuda": []},
        "fingerprints": {"dataset": "old", "tokenizer": "old"},
        "tokenizer_json": __import__("pathlib").Path(tokenizer).read_text(),
        "device_type": "cpu",
        "bf16": False,
    }
    sha = hashlib.sha256(parent["tokenizer_json"].encode()).hexdigest()
    new = copy.deepcopy(config)
    new.training.lr = 0.001
    for policy in ("preserve", "reset"):
        result = phase_state(parent, new, "new-dataset", sha, 100, policy, "parent-sha")
        assert result["step"] == 0 and result["tokens"] == 768
        assert result["sampler"]["position"] == 0
        assert result["phase"]["parent_updates"] == 3
        assert all(torch.equal(v, parent["model"][k]) for k, v in result["model"].items())
        assert bool(result["optimizer"]["state"]) == (policy == "preserve")
        if policy == "preserve":
            assert all(float(v["step"]) == 3 for v in result["optimizer"]["state"].values())
        assert result["scheduler"]["last_epoch"] == 0
    with pytest.raises(ValueError, match="Tokenizer"):
        phase_state(parent, new, "new", "bad", 100, "preserve", "parent")
    same = phase_state(parent, new, "old", sha, 100, "preserve", "parent")
    assert same["sampler"] == parent["sampler"]
    budget = (
        new.training.steps
        * new.training.batch_size
        * new.training.sequence
        * new.training.accumulation
        - 1
    )
    exact = phase_state(parent, new, "new", sha, 100, "preserve", "parent", budget)
    assert exact["phase"]["additional_target_tokens"] == budget
    assert exact["phase"]["parent_scheduler"] == parent["scheduler"]
    with pytest.raises(ValueError, match="budget"):
        phase_state(parent, new, "new", sha, 100, "preserve", "parent", 0)
    malformed = copy.deepcopy(parent)
    first = next(iter(malformed["optimizer"]["state"].values()))
    first["step"] = torch.tensor(2.0)
    with pytest.raises(ValueError, match="counter"):
        phase_state(malformed, new, "new", sha, 100, "preserve", "parent")
    malformed = copy.deepcopy(parent)
    first = next(iter(malformed["optimizer"]["state"].values()))
    first["exp_avg_sq"].view(-1)[0] = -1
    with pytest.raises(ValueError, match="Negative"):
        phase_state(malformed, new, "new", sha, 100, "preserve", "parent")
    # Synthetic post-reset state; no backward pass or optimizer update is performed.
    after_reset = phase_state(parent, new, "new", sha, 100, "reset", "parent")
    after_reset["step"] = 2
    after_reset["optimizer"] = copy.deepcopy(parent["optimizer"])
    for value in after_reset["optimizer"]["state"].values():
        value["step"] = torch.tensor(2.0)
    preserved_reset = phase_state(after_reset, new, "new", sha, 100, "preserve", "reset-parent")
    assert preserved_reset["phase"]["parent_updates"] == 5
    assert preserved_reset["phase"]["optimizer_origin_updates"] == 3
    new.model.context = 32
    with pytest.raises(ValueError, match="Architecture"):
        phase_state(parent, new, "new", sha, 100, "preserve", "parent")


def test_prepare_shards_and_resume_without_training(prepared, tmp_path, monkeypatch):
    from pathlib import Path

    import pyarrow as pa
    import pyarrow.parquet as pq

    from codexa.data.production import digest, verify
    from codexa.data.v2 import prepare
    from codexa.evaluation.fixtures import freeze

    monkeypatch.setattr(Storage, "check", lambda *a, **k: None)
    storage = Storage(tmp_path / "v2")
    storage.initialize()
    freeze(storage.root)
    old = tmp_path / "history.sqlite"
    db = sqlite3.connect(old)
    db.execute("CREATE TABLE docs(id TEXT PRIMARY KEY,split TEXT,text TEXT)")
    ident = hashlib.sha256(canonical(PROSE).encode()).hexdigest()
    db.execute("INSERT INTO docs VALUES(?,?,?)", (ident, "train", PROSE))
    db.commit()
    db.close()
    (tmp_path / "manifest.json").write_text("{}")
    index = Index(storage, old)
    index.reference(1)
    index.close()
    text = (
        "The research team recorded changes in sunlight across a remote valley. "
        "Their instruments collected measurements at regular intervals throughout the season. "
        "Each evening the scientists compared the latest measurements with earlier records. "
        "They kept careful notes about cloud cover so that unusual readings could be explained."
    )
    src = storage.root / "raw" / "fixture--source.parquet"
    pq.write_table(
        pa.table({"text": [text], "url": ["https://example.org/original-research"]}), src
    )
    spec = {
        "name": "fixture",
        "path": "source.parquet",
        "sha256": digest(src),
        "bytes": src.stat().st_size,
        "revision": "a" * 40,
        "category": "general",
        "target_share": 1.0,
        "license": "CC0-1.0",
    }
    src.with_name(src.name + ".receipt.json").write_text(json.dumps(spec))
    (storage.root / "source-registry.json").write_text(json.dumps({"sources": [spec]}))
    (storage.root / "source-admission.json").write_text(
        json.dumps(
            {
                "cleaner": __import__("codexa.data.v2_quality", fromlist=["VERSION"]).VERSION,
                "sources": {"fixture": {"status": "admitted", "scope": "CC0 test"}},
            }
        )
    )
    tok = Path(prepared[1])
    import codexa.data.v2 as pipeline

    real_atomic = pipeline.atomic

    def interrupted_manifest(path, value):
        if path.name == "manifest.json" and path.parent.name == "tokens":
            raise OSError("simulated interruption after database/shard commit")
        return real_atomic(path, value)

    monkeypatch.setattr(pipeline, "atomic", interrupted_manifest)
    with pytest.raises(OSError, match="simulated interruption"):
        prepare(storage, old, tok, 10000, 1, digest(tok), 512)
    durable = [(p.name, p.read_bytes()) for p in (storage.root / "tokens").glob("*.bin")]
    assert durable
    monkeypatch.setattr(pipeline, "atomic", real_atomic)
    result = prepare(storage, old, tok, 10000, 1, digest(tok), 512)
    assert durable == [(p.name, p.read_bytes()) for p in (storage.root / "tokens").glob("*.bin")]
    assert result["training_tokens"] > 0
    verified = verify(storage.root / "tokens")
    assert verified["vocab_size"] == 512
    saved = [(p.name, p.read_bytes()) for p in (storage.root / "tokens").glob("*.bin")]
    repeated = prepare(storage, old, tok, 10000, 1, digest(tok), 512)
    assert repeated["training_tokens"] == result["training_tokens"]
    assert saved == [(p.name, p.read_bytes()) for p in (storage.root / "tokens").glob("*.bin")]
    from codexa.data.v2_finalize import finalize
    from scripts.audit_v2_snapshot import audit

    snapshot = finalize(storage.root, tok, "fixture")
    assert snapshot["all_accepted_documents_reconstructed"]
    token_audit = audit(storage.root / "snapshots/fixture", storage.root)
    assert token_audit["unk_count"] == token_audit["pad_count"] == 0
    assert token_audit["bos_count"] == token_audit["eos_count"]
    assert snapshot["split_tokens"]["train"] == result["training_tokens"]
    with pytest.raises(FileExistsError, match="immutable"):
        finalize(storage.root, tok, "fixture")


def test_question_paraphrases_and_ordinary_prose(tmp_path):
    from codexa.evaluation.fixtures import freeze
    from codexa.evaluation.protection import QuestionProtection

    freeze(tmp_path)
    protection = QuestionProtection(tmp_path / "benchmarks/v3")
    assert protection.matches("Which city is the capital of France?")
    assert protection.matches("What is the national capital city of France?")
    assert not protection.matches("Paris is the capital city of France. It has many museums.")


def test_answer_proxies_and_independent_math_oracle(tmp_path):
    from fractions import Fraction

    from codexa.evaluation.fixtures import freeze
    from codexa.evaluation.scoring import free_correct, math_oracle, numeric_answer

    freeze(tmp_path)
    for split in ("dev", "test"):
        for line in (tmp_path / "benchmarks/v3" / (split + ".jsonl")).read_text().splitlines():
            item = json.loads(line)
            if item["kind"] == "math":
                assert math_oracle(item) == item["answer"]
    assert numeric_answer("x = 3/4. That is the result.") == Fraction(3, 4)
    assert numeric_answer("3+4") is None
    assert numeric_answer("I considered 7, but the answer is 5.") is None
    assert free_correct({"kind": "fact", "answer": "Brasilia"}, "Brasília is the capital.")
    assert not free_correct({"kind": "fact", "answer": "Paris"}, "Parish is a word.")


def test_admission_gates_and_writer_exclusion(tmp_path, monkeypatch):
    from codexa.data.v2_policy import admission_error

    assert (
        admission_error(
            PROSE, {"url": "https://evil.example/socratic.org"}, {"url_domains": ["socratic.org"]}
        )
        == "admission_domain"
    )
    assert (
        admission_error(PROSE, {"url": "https://socratic.org/a"}, {"url_domains": ["socratic.org"]})
        is None
    )
    assert (
        admission_error(PROSE + "\n" + ("abc" * 100), {}, {"maximum_word_length": 64})
        == "admission_scrambled_word"
    )
    monkeypatch.setattr(Storage, "check", lambda *a, **k: None)
    storage = Storage(tmp_path / "data")
    storage.initialize()
    old = tmp_path / "old.sqlite"
    sqlite3.connect(old).close()
    index = Index(storage, old)
    try:
        with pytest.raises(RuntimeError, match="writer"):
            Index(storage, old)
    finally:
        index.close()


def test_text_oversize_is_rejected_not_truncated(tmp_path):
    from codexa.data.v2 import records

    src = tmp_path / "stories.txt"
    src.write_text(PROSE + "\n" + "A" * 1024**2 + "\n<|endoftext|>\n" + PROSE + "\n<|endoftext|>\n")
    rows = list(records(src, {}))
    assert rows[0][1] == {"error": "oversize"}
    assert rows[1][1]["text"].strip() == PROSE.strip()


def test_pinned_http_range_recovery_and_ignored_range(tmp_path, monkeypatch):
    import io

    from codexa.data.v2_acquire import acquire

    monkeypatch.setattr(Storage, "check", lambda *a, **k: None)
    storage = Storage(tmp_path / "v2")
    storage.initialize()
    payload = b"verified bounded source data\n" * 30
    spec = {
        "name": "fixture",
        "repo": "original/fixture",
        "revision": "a" * 40,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "path": "sample.txt",
        "license": "CC0-1.0",
        "category": "general",
    }
    path = storage.root / "raw/fixture--sample.txt"
    partial = path.with_name(path.name + ".partial")
    partial.write_bytes(payload[:17])

    class Response(io.BytesIO):
        status = 206

        def __init__(self, data):
            super().__init__(data)
            self.headers = {"Content-Range": f"bytes 17-{len(payload) - 1}/{len(payload)}"}

    monkeypatch.setattr("urllib.request.urlopen", lambda request, **kw: Response(payload[17:]))
    assert acquire(storage, spec).read_bytes() == payload
    assert (
        json.loads(path.with_name(path.name + ".receipt.json").read_text())["verified_sha256"]
        == spec["sha256"]
    )
    spec = spec | {"path": "ignored.txt"}
    other = storage.root / "raw/fixture--ignored.txt.partial"
    other.write_bytes(payload[:17])
    Response.status = 200
    with pytest.raises(ValueError, match="range recovery"):
        acquire(storage, spec)
    assert other.read_bytes() == payload[:17]


def test_append_only_registry_transaction_and_recovery(tmp_path, monkeypatch):
    import copy

    from codexa.data.production import atomic, digest
    from codexa.data.v2_registry import extend, recover, validate_extension

    monkeypatch.setattr(Storage, "check", lambda *a, **k: None)
    storage = Storage(tmp_path / "v2")
    storage.initialize()
    old_db = tmp_path / "old.sqlite"
    sqlite3.connect(old_db).close()
    original = {
        "name": "first",
        "repo": "original/fixture",
        "revision": "a" * 40,
        "sha256": "b" * 64,
        "path": "one.parquet",
        "bytes": 100,
        "license": "CC0-1.0",
        "category": "general",
    }
    extra = original | {"name": "second", "sha256": "c" * 64, "path": "two.parquet"}
    first = {"config": {"seed": 42, "sources": [{"name": "first"}]}, "sources": [original]}
    addition = {"config": {"seed": 42, "sources": [{"name": "second"}]}, "sources": [extra]}
    path = storage.root / "source-registry.json"
    atomic(path, first)
    old_bytes = path.read_bytes()
    index = Index(storage, old_db)
    try:
        index.set("preparation_identity", {"source_registry": digest(path), "tokenizer": "frozen"})
        index.db.commit()
        transaction = extend(index, addition)
        assert json.loads(path.read_text())["sources"] == [original, extra]
        assert index.value("preparation_identity")["tokenizer"] == "frozen"
        # Simulate interruption between durable journal and replacing the top-level registry.
        index.set("registry_pending", transaction)
        index.db.commit()
        path.write_bytes(old_bytes)
        recover(index)
        assert digest(path) == transaction["new"]
        bad = copy.deepcopy(json.loads(path.read_text()))
        bad["sources"][0]["bytes"] = 99
        with pytest.raises(ValueError, match="prior pinned"):
            validate_extension(first, bad)
        assert index.value("registry_revisions") == [transaction]
    finally:
        index.close()


def test_admission_navigation_and_template_quarantine():
    from codexa.data.v2_policy import admission_error, admission_text

    raw = (
        "Courses\nCourses for Kids\nFree study material\nStore\n\n# Cuboid volume\n"
        + PROSE
        + "\n[asy]draw(A--B);[/asy]"
    )
    text = admission_text(raw, {"strip_educational_navigation": True})
    assert text.startswith("# Cuboid volume") and "draw(A" not in text
    assert PROSE.strip() in text
    assert (
        admission_error("# Search by Topic\n" + PROSE, {}, {"reject_extraction_markers": True})
        == "admission_extraction_template"
    )


def test_accelerated_receipt_recovery_without_redownload(tmp_path, monkeypatch):
    from codexa.data.v2_acquire import acquire_xet

    monkeypatch.setattr(Storage, "check", lambda *a, **k: None)
    storage = Storage(tmp_path / "v2")
    storage.initialize()
    payload = b"completed verified public fixture"
    spec = {
        "name": "fixture",
        "repo": "original/fixture",
        "revision": "a" * 40,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "path": "completed.txt",
        "license": "CC0-1.0",
        "category": "general",
    }
    path = storage.root / "raw/fixture--completed.txt"
    path.write_bytes(payload)
    assert acquire_xet(storage, spec) == path
    assert (
        json.loads(path.with_name(path.name + ".receipt.json").read_text())["verified_sha256"]
        == spec["sha256"]
    )


def test_phase_import_rejects_wrong_volume_before_model_load(tmp_path, monkeypatch):
    import torch

    from codexa.config import Config
    from codexa.training.trainer import Trainer

    cfg = Config()
    stub = object.__new__(Trainer)
    stub.config = cfg
    stub.fingerprints = {}
    stub.device = torch.device("cpu")
    stub.bf16 = False
    stub.output = tmp_path / "future-run"
    state = {
        "version": 1,
        "config": cfg.to_dict(),
        "fingerprints": {},
        "device_type": "cpu",
        "bf16": False,
        "phase": {"volume_uuid": "incorrect-volume", "minimum_free_bytes": 40 * GIB},
    }
    monkeypatch.setattr("codexa.training.trainer.load", lambda path: state)
    monkeypatch.setattr("subprocess.check_output", lambda *a, **k: "actual-volume\n")
    # Fails before there is even a model to import, and never enters a training loop.
    with pytest.raises(OSError, match="volume"):
        stub.resume("synthetic-fixture")
