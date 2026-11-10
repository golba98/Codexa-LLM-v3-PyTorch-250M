"""Small in-memory streams exercise recovery without optimizer updates."""

import gzip
import hashlib
import io
import json

import pytest
import torch

from codexa.config import Config
from codexa.data.streaming import StreamingData
from codexa.tokenizer.bpe import BOS, EOS, CodexaTokenizer
from codexa.training.trainer import Trainer


def text_for(split, n):
    found = []
    i = 0
    while len(found) < n:
        text = f"Document {i}: a coherent fixture about the natural world. " * 3
        digest = hashlib.sha256(text.encode()).hexdigest()
        if StreamingData.document_split(digest) == split:
            found.append(text)
        i += 1
    return found


def make_stream(tok, rows, split="train", opener=None, policy="fixture-v1"):
    payload = gzip.compress(b"".join(json.dumps({"text": t}).encode() + b"\n" for t in rows))
    return StreamingData(
        [{"url": "fixture", "etag": "fixed"}],
        tok,
        16,
        41,
        split,
        lambda row: True,
        lambda text: "protected" in text,
        opener=opener or (lambda shard: io.BytesIO(payload)),
        policy_identity=policy,
    )


def test_stream_resume_boundaries_and_duplicate_protection(prepared):
    tok = CodexaTokenizer.load(prepared[1])
    texts = text_for("train", 3)
    stream = make_stream(tok, [texts[0], texts[0], "protected text", *texts[1:]])
    first = stream.next(2)
    state = stream.state_dict()
    expected = stream.next(12)
    resumed = make_stream(tok, [texts[0], texts[0], "protected text", *texts[1:]])
    resumed.load_state_dict(state)
    assert resumed.next(12) == expected
    assert first[0][0] == BOS
    assert first[0][-1] == first[1][0]
    assert any(EOS in row for row in first + expected)
    with pytest.raises(ValueError, match="identity"):
        make_stream(tok, texts, policy="different").load_state_dict(state)


def test_retry_and_failed_batch_are_transactional(prepared):
    tok = CodexaTokenizer.load(prepared[1])
    texts = text_for("train", 1)
    payload = gzip.compress((json.dumps({"text": texts[0]}) + "\n").encode())
    calls = 0

    def opener(shard):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("temporary")
        return io.BytesIO(payload)

    stream = make_stream(tok, texts, opener=opener)
    assert stream.next(1)
    assert calls == 2
    before = stream.state_dict()
    with pytest.raises(EOFError):
        stream.next(1000)
    assert stream.state_dict() == before


def test_split_duplicates_and_protected_text(prepared):
    tok = CodexaTokenizer.load(prepared[1])
    train = text_for("train", 3)
    heldout = text_for("val", 1)[0]
    rows = [train[0], train[0], "protected text", heldout, *train[1:]]
    stream = make_stream(tok, rows)
    actual = []
    while True:
        try:
            actual.extend(stream.next(1)[0][:-1])
        except EOFError:
            break
    expected = []
    for text in train:
        expected.extend([BOS, *tok.encode(text), EOS])
    assert actual == expected[: len(actual)]
    assert stream.seen == {hashlib.sha256(t.encode()).hexdigest() for t in train}


def test_trainer_forward_checkpoint_and_validation_replay(prepared, tmp_path, monkeypatch):
    monkeypatch.delenv("CODEXA_NO_TRAINING", raising=False)
    tok = CodexaTokenizer.load(prepared[1])
    train = make_stream(tok, text_for("train", 3))
    val = make_stream(tok, text_for("val", 3), split="val")
    cfg = Config()
    cfg.training.sequence = 16
    cfg.training.validation_batches = 2
    trainer = Trainer(
        cfg, "unused", prepared[1], str(tmp_path / "stream"), "cpu", stream_data=(train, val)
    )
    before = {k: v.clone() for k, v in trainer.model.state_dict().items()}
    assert trainer.validate() == trainer.validate()
    trainer.train_data.batch(trainer.sampler.next(2), trainer.device)
    trainer.checkpoint()
    expected = trainer.sampler.next(2)
    restored = Trainer(
        cfg,
        "unused",
        prepared[1],
        str(tmp_path / "restore"),
        "cpu",
        stream_data=(
            make_stream(tok, text_for("train", 3)),
            make_stream(tok, text_for("val", 3), split="val"),
        ),
    )
    restored.resume(str(tmp_path / "stream/latest.pt"))
    assert restored.sampler.next(2) == expected
    with pytest.raises(RuntimeError, match="Streaming training is blocked"):
        restored.train(ui="none")
    assert all(torch.equal(v, trainer.model.state_dict()[k]) for k, v in before.items())
    assert not trainer.optimizer.state


def test_interrupted_read_replays_from_committed_record(prepared):
    tok = CodexaTokenizer.load(prepared[1])
    rows = text_for("train", 1000)
    payload = gzip.compress(b"".join(json.dumps({"text": t}).encode() + b"\n" for t in rows))
    active = []
    failures = []

    class Interruption(io.BytesIO):
        fail = False

        def read(self, size=-1):
            if self.fail:
                self.fail = False
                failures.append(True)
                raise OSError("interrupted during read")
            return super().read(32 if size < 0 else min(size, 32))

    def opener(shard):
        item = Interruption(payload)
        active.append(item)
        return item

    stream = make_stream(tok, rows, opener=opener)
    baseline = make_stream(tok, rows)
    assert stream.next(5) == baseline.next(5)
    active[-1].fail = True
    assert stream.next(1500) == baseline.next(1500)
    assert failures
    assert stream.state_dict() == baseline.state_dict()


def test_pinned_stream_configuration_rejects_unsafe_inputs():
    from codexa.data.stream_probe import validate_config

    config = {
        "dataset": "common-pile/comma_v0.1_training_dataset",
        "revision": "a" * 40,
        "shards": [{"path": "wikimedia/part.jsonl.gz", "etag": "b" * 64}],
    }
    assert "/resolve/" + "a" * 40 + "/wikimedia/" in validate_config(config)[0]["url"]
    config["shards"][0]["path"] = "../private.jsonl.gz"
    with pytest.raises(ValueError, match="path"):
        validate_config(config)


def test_configured_probe_is_bounded_and_not_approved(prepared, monkeypatch):
    from codexa.data import stream_probe

    rows = text_for("train", 5)
    payload = gzip.compress(b"".join(json.dumps({"text": t}).encode() + b"\n" for t in rows))
    monkeypatch.setattr(stream_probe, "open_provider", lambda _: io.BytesIO(payload))
    config = {
        "dataset": "common-pile/comma_v0.1_training_dataset",
        "revision": "a" * 40,
        "shards": [{"path": "wikimedia/part.jsonl.gz", "etag": "b" * 64}],
    }
    report = stream_probe.probe(config, prepared[1], 2, 16, 4096)
    assert report["target_tokens"] == 32
    assert report["status"] == "development-only-not-approved"
    assert report["compressed_bytes"] <= 4096
    with pytest.raises(RuntimeError, match="byte budget"):
        stream_probe.probe(config, prepared[1], 2, 16, 8)


def test_training_cli_rejects_unapproved_stream(tmp_path, monkeypatch):
    import sys

    from codexa.cli import main

    output = tmp_path / "must-not-be-created"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "codexa",
            "train",
            "--stream-config",
            "unapproved.json",
            "--tokenizer",
            "unused",
            "--output",
            str(output),
        ],
    )
    with pytest.raises(FileNotFoundError):
        main()
    assert not output.exists()


def test_provider_rejects_changed_etag(monkeypatch):
    from codexa.data import streaming

    class Response(io.BytesIO):
        def __init__(self):
            super().__init__(b"")
            self.headers = {"ETag": '"' + "c" * 64 + '"', "Content-Length": "0"}
            self.status = 200

    response = Response()
    monkeypatch.setattr(streaming, "urlopen", lambda *args, **kwargs: response)
    shard = {
        "url": "https://huggingface.co/datasets/org/repo/resolve/" + "a" * 40 + "/part.jsonl.gz",
        "etag": "b" * 64,
    }
    with pytest.raises(ValueError, match="identity changed"):
        streaming.open_provider(shard)
    assert response.closed


def test_native_online_duplicate_metadata_rollback_and_restore():
    from codexa.data.stream_admission import OnlineDuplicates

    text = " ".join(f"word{i}" for i in range(160))
    ledger = OnlineDuplicates([{"url": "fixture", "etag": "fixed"}])
    ledger.begin()
    assert ledger.add(text, 0, 1, {"license": "fixture"}) is None
    ledger.commit()
    saved = ledger.state_dict()
    assert text.encode() not in saved["database"]
    ledger.begin()
    assert ledger.add(text, 0, 2) == "exact_document"
    ledger.rollback()
    assert ledger.db.execute("SELECT COUNT(*) FROM docs").fetchone()[0] == 1
    restored = OnlineDuplicates([{"url": "fixture", "etag": "fixed"}])
    restored.load_state_dict(saved)
    assert (
        restored.db.execute("SELECT attribution FROM docs").fetchone()[0]
        == '{"license": "fixture"}'
    )
    assert restored.db.execute("SELECT shard,record FROM docs").fetchone() == (0, 1)


def test_wikimedia_metadata_policy():
    from codexa.data.stream_admission import wikimedia_article

    row = {
        "id": "0-1",
        "text": "ordinary prose " * 80,
        "metadata": {
            "wiki": "wikipedia.com",
            "namespace": "0",
            "url": "https://wikipedia.com/wiki/Fixture",
            "license": "Creative Commons - Attribution Share-Alike - https://creativecommons.org/licenses/by-sa/4.0/",
        },
    }
    assert wikimedia_article(row)
    row["metadata"]["namespace"] = "1"
    assert not wikimedia_article(row)


def test_configured_trainer_policy_checkpoint_path(prepared, tmp_path):
    import random
    import sqlite3

    from codexa.data.production import digest
    from codexa.data.stream_policy import configured_streams
    from codexa.data.v2_index import Protection

    root = tmp_path / "reference"
    (root / "indexes").mkdir(parents=True)
    db = root / "indexes/documents.sqlite"
    with sqlite3.connect(db) as conn:
        conn.executescript(
            "CREATE TABLE docs(id INTEGER,hash BLOB,role TEXT,split TEXT); CREATE TABLE bands(band BLOB,doc INTEGER); CREATE TABLE anchors(hash BLOB,doc INTEGER,pos INTEGER);"
        )
    old = tmp_path / "history.sqlite"
    with sqlite3.connect(old) as conn:
        conn.execute("CREATE TABLE docs(id TEXT,text TEXT)")
    protection = tmp_path / "protected.npz"
    Protection.build(
        [
            (
                "This deliberately protected evaluation sentence must never appear in streamed language training material.",
                True,
            )
        ]
    ).save(protection)
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    for split in ("dev", "test"):
        (fixtures / f"{split}.jsonl").write_text(
            json.dumps({"id": split, "question": "What is the protected fixture answer?"}) + "\n"
        )
    files = [db, old, protection, fixtures / "dev.jsonl", fixtures / "test.jsonl"]
    review_ids = tmp_path / "review-ids.json"
    review_ids.write_text("[]")
    files.append(review_ids)
    from pathlib import Path

    import codexa.data.stream_policy as policy_module

    policy_path = Path(policy_module.__file__)
    cfg = {
        "dataset": "common-pile/wikimedia_filtered",
        "revision": "0641bb84bd9b7162bcddf8be7836822161a9a342",
        "shards": [{"path": "wikimedia-0000.json.gz", "etag": "a" * 64}],
        "tokenizer_sha256": digest(prepared[1]),
        "references": [
            {"role": role, "root": str(root), "historical": str(old)}
            for role in ("historical", "prior50m", "prior100m")
        ],
        "protected_indexes": [str(protection)],
        "fixtures": str(fixtures),
        "exclusion_files": [{"path": str(p), "sha256": digest(p)} for p in files],
        "review_ids": str(review_ids),
        "policy_sources": [
            {"path": str(p), "sha256": digest(p)}
            for p in (
                policy_path,
                policy_path.with_name("stream_admission.py"),
                policy_path.with_name("streaming.py"),
                policy_path.with_name("v2_index.py"),
                policy_path.with_name("v2_quality.py"),
                policy_path.with_name("language_protocol.py"),
                policy_path.parents[1] / "evaluation/protection.py",
            )
        ],
    }
    path = tmp_path / "stream.json"
    path.write_text(json.dumps(cfg))
    rng = random.Random(44)
    rows = []
    for i in range(400):
        text = (
            " ".join("".join(rng.choices("abcdefghijklmnopqrstuvwxyz", k=8)) for _ in range(110))
            + "."
        )
        rows.append(
            {
                "id": f"0-{i}",
                "text": text,
                "metadata": {
                    "wiki": "wikipedia.com",
                    "namespace": "0",
                    "url": f"https://wikipedia.com/wiki/{i}",
                    "license": "Creative Commons - Attribution Share-Alike - https://creativecommons.org/licenses/by-sa/4.0/",
                },
            }
        )
    payload = gzip.compress(b"".join(json.dumps(row).encode() + b"\n" for row in rows))
    opener = lambda shard: io.BytesIO(payload)
    train, val = configured_streams(
        str(path), prepared[1], 16, 42, require_launch=False, opener=opener
    )
    config = Config()
    config.training.sequence = 16
    config.training.validation_batches = 1
    trainer = Trainer(
        config,
        "unused",
        prepared[1],
        str(tmp_path / "actual-policy"),
        "cpu",
        stream_data=(train, val),
    )
    trainer.sampler.next(2)
    trainer.checkpoint()
    expected = trainer.sampler.next(2)
    train2, val2 = configured_streams(
        str(path), prepared[1], 16, 42, require_launch=False, opener=opener
    )
    resumed = Trainer(
        config,
        "unused",
        prepared[1],
        str(tmp_path / "policy-resume"),
        "cpu",
        stream_data=(train2, val2),
    )
    resumed.resume(str(tmp_path / "actual-policy/latest.pt"))
    assert resumed.sampler.next(2) == expected
    assert trainer.validate() == trainer.validate()
    assert not trainer.optimizer.state
    with pytest.raises(ValueError, match="private assessments"):
        configured_streams(str(path), prepared[1], 16, 42, opener=opener)


def test_stream_phase_preserves_native_parent_without_updates(prepared, tmp_path):
    from codexa.continuation import phase_state
    from codexa.data.packing import fingerprint
    from codexa.training.checkpoint import load

    config = Config()
    config.training.sequence = 16
    parent_trainer = Trainer(
        config, prepared[0], prepared[1], str(tmp_path / "parent-fixture"), "cpu"
    )
    parent_trainer.checkpoint()
    parent_path = tmp_path / "parent-fixture/latest.pt"
    parent = load(parent_path)
    tok = CodexaTokenizer.load(prepared[1])
    stream = make_stream(tok, text_for("train", 3))
    update = config.training.batch_size * config.training.sequence * config.training.accumulation
    config.training.steps = (1_000_000_000 + update - 1) // update
    state = phase_state(
        parent,
        config,
        stream.identity,
        fingerprint(prepared[1]),
        None,
        "preserve",
        fingerprint(parent_path),
        1_000_000_000,
        stream_sampler=stream.state_dict(),
    )
    assert state["sampler"] == stream.state_dict()
    assert state["optimizer"] == parent["optimizer"]
    assert state["phase"]["additional_target_tokens"] == 1_000_000_000
    assert torch.equal(state["phase"]["parent_sampler"]["order"], parent["sampler"]["order"])
    assert all(torch.equal(v, state["model"][k]) for k, v in parent["model"].items())


def test_billion_stream_budget_requires_explicit_launch_authorization():
    from codexa.data.language_protocol import V5_VERSION, authorized_budget

    record = {
        "additional_target_tokens": 1_000_000_000,
        "cumulative_target_tokens": 1_500_006_912,
        "user_decision": "synthetic fixture only",
        "training_launch_authorized": False,
    }
    assert not authorized_budget(record, V5_VERSION)
    record["training_launch_authorized"] = True
    assert authorized_budget(record, V5_VERSION)


@pytest.mark.parametrize("version", ["codexa-language-quality-5", "codexa-language-quality-7"])
def test_stream_launch_recomputes_reviews_and_requires_bound_authorization(
    tmp_path, monkeypatch, version
):
    from codexa.data import language_protocol, stream_policy
    from codexa.data.language_protocol import V7_VERSION, specification, specification_hash
    from codexa.data.production import digest
    from codexa.data.stream_policy import launch_approval

    # Entirely synthetic gate evidence; this is never a source approval artifact.
    rubric_fixture = {"kind": "Original CC0 synthetic rubric fixture"}
    original_fixture = {"kind": "Original CC0 synthetic panel fixture"}
    if version == V7_VERSION:
        for name, value, attribute in (
            ("rubric.json", rubric_fixture, "GENERAL_RUBRIC_SHA256"),
            ("original-panel.json", original_fixture, "GENERAL_PANEL_SHA256"),
        ):
            fixture = tmp_path / name
            fixture.write_text(json.dumps(value))
            monkeypatch.setattr(language_protocol, attribute, digest(fixture))
            if attribute == "GENERAL_RUBRIC_SHA256":
                monkeypatch.setattr(stream_policy, attribute, digest(fixture))
    config = {
        "dataset": "common-pile/wikimedia_filtered",
        "revision": "0641bb84bd9b7162bcddf8be7836822161a9a342",
        "training_config_identity": "b" * 64,
        "parent_checkpoint_sha256": "c" * 64,
        "assessment_roots": ["protected-math-recovery-v1", "protected-math-recovery-v2"],
    }
    protocol = specification(["wikimedia-wikipedia"], version=version)
    protocol["stream_config_sha256"] = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode()
    ).hexdigest()
    protocol["training_config_identity"] = config["training_config_identity"]
    mapping = [
        {
            "id": f"D{i}",
            "kind": "document",
            "source": "wikimedia-wikipedia",
            "document_id": f"fixture-{i}",
            "coverage": "full document",
        }
        for i in range(100)
    ]
    mapping += [
        {"id": f"W{i}", "kind": "window", "source": "wikimedia-wikipedia"} for i in range(300)
    ]
    ratings = [
        {
            "id": row["id"],
            "usable": True,
            "severe": False,
            "critical": False,
            "coverage_complete": True,
            "checks": {
                key: "pass"
                for key in (
                    "notation",
                    "completeness",
                    "visuals",
                    "correctness",
                    "education",
                    "coherence",
                    "contamination",
                )
            },
            "reason": "Original synthetic fixture assessment",
        }
        for row in mapping
    ]

    if version == V7_VERSION:
        ratings = [
            {
                "id": row["id"],
                "kind": row["kind"],
                "band": "U1",
                "decision": "usable",
                "scope": "eligible",
                "coverage_complete": True,
                "severe": False,
                "critical": False,
                "confirmed_source_defect": True,
                "defect_categories": ["localized_fixture_typo"],
                "observed_evidence": ["Original CC0 fixture"],
                "reason": "A local fixture typo does not destroy its central English explanation.",
                "residual_utility": "The original fixture remains intelligible.",
            }
            for row in mapping
        ]

    def bound(name, value, lines=False):
        path = tmp_path / name
        path.write_text(
            "".join(json.dumps(x) + "\n" for x in value) if lines else json.dumps(value)
        )
        return {"path": str(path), "sha256": digest(path)}

    auth = {
        "version": version,
        "status": "pending",
        "specification_sha256": specification_hash(protocol),
        "sources": protocol["sources"],
        "additional_target_tokens": 1_000_000_000,
        "cumulative_target_tokens": 1_500_006_912,
        "training_launch_authorized": False,
        "user_decision": "",
        "historical_roots": ["fixture"],
        "replaces": protocol["replaces"],
        "stream_config_sha256": protocol["stream_config_sha256"],
        "training_config_identity": config["training_config_identity"],
    }
    evidence = {
        "mapping": bound("mapping.jsonl", mapping, True),
        "reviewers": [
            dict(bound(f"reviewer-{i}.jsonl", ratings, True), identity=f"fixture-{i}")
            for i in (1, 2)
        ],
        "capacity": bound(
            "capacity.json",
            {
                "source_bindings": protocol["source_bindings"],
                "stream_config_sha256": protocol["stream_config_sha256"],
                "lower_95_eligible_train_tokens": 1_000_000_001,
                "inspection_representativeness": {
                    "status": "passed-on-bounded-evidence",
                    "maximum_permitted_fraction": 0.05,
                    "estimated_inspection_token_loss_fraction": 0.0,
                    "estimated_inspection_document_loss_fraction": 0.0,
                },
            },
        ),
        "compute": bound(
            "compute.json",
            {
                "status": "passed",
                "optimizer_schema_verified": True,
                "checkpoint_resume_verified": True,
                "cuda_memory_verified": True,
                "training_config_identity": config["training_config_identity"],
                "parent_sha256": config["parent_checkpoint_sha256"],
            },
        ),
        "rights": bound(
            "rights.json",
            {
                "status": "approved",
                "scope": "internal-pretraining",
                "source_bindings": protocol["source_bindings"],
                "reviewer": "fixture",
                "original_rights_basis": "Original CC0 fixture",
                "attribution_policy": "fixture notices",
            },
        ),
    }
    if version == V7_VERSION:
        evidence["rubric"] = bound("rubric.json", rubric_fixture)
        evidence["original_panel"] = bound("original-panel.json", original_fixture)
        evidence["quality_authorization"] = bound(
            "quality-authorization.json",
            {
                "status": "authorized-for-reevaluation-only",
                "source": config["dataset"],
                "revision": config["revision"],
                "approved_application_sha256": language_protocol.GENERAL_RUBRIC_SHA256,
                "user_decision": "Synthetic fixture quality approval only",
            },
        )
    evidence["authorization"] = bound("authorization.json", auth)
    protocol["authorization_path"] = str((tmp_path / "authorization.json").resolve())
    protocol["authorization_sha256"] = evidence["authorization"]["sha256"]
    evidence["protocol"] = bound("protocol.json", protocol)
    config["approval"] = evidence
    with pytest.raises(ValueError, match="authorization"):
        launch_approval(config)
    auth.update(
        status="authorized",
        training_launch_authorized=True,
        user_decision="Synthetic test authorization only",
    )
    evidence["authorization"] = bound("authorization.json", auth)
    protocol["authorization_sha256"] = evidence["authorization"]["sha256"]
    evidence["protocol"] = bound("protocol.json", protocol)
    launch_approval(config)
    ratings[0]["critical"] = True
    if version == V7_VERSION:
        ratings[0].update(band="C", decision="reject")
    else:
        ratings[0]["usable"] = False
    evidence["reviewers"][1] = dict(bound("reviewer-2.jsonl", ratings, True), identity="fixture-2")
    with pytest.raises(ValueError, match="quality"):
        launch_approval(config)


def test_frozen_assessment_screen_preserves_existing_thresholds(tmp_path):
    from codexa.data.production import digest
    from codexa.evaluation.protection import FrozenAssessmentProtection
    from scripts.screen_reserved_math import normalise, shape

    text = "How many of the original 37 items in 5 boxes remain after giving away 9 items?"
    root = tmp_path / "protected-fixture"
    root.mkdir()
    pool = root / "private-assessment.jsonl"
    pool.write_text(json.dumps({"id": "original-fixture", "question": text}) + "\n")
    families = root / "family-exclusions.json"
    families.write_text('{"forbidden_generator_families":[]}')
    (root / "freeze-manifest.json").write_text(
        json.dumps(
            {
                "private_assessment_sha256": digest(pool),
                "family_exclusions_sha256": digest(families),
            }
        )
    )
    guard = FrozenAssessmentProtection([root])
    assert guard.normalise(text) == normalise(text)
    assert guard.shape(text) == shape(text)
    assert guard.matches(text)
    assert guard.matches(text.replace("37", "48").replace("9", "6"))
    assert not guard.matches(
        "Ordinary English discusses gardens, weather and the changing seasons."
    )
    pool.write_text("{}\n")
    with pytest.raises(ValueError, match="changed"):
        FrozenAssessmentProtection([root])


def test_network_eof_at_gzip_member_boundary_cannot_skip_records(prepared, monkeypatch):
    from codexa.data import streaming

    monkeypatch.setattr(streaming.time, "sleep", lambda _: None)
    tok = CodexaTokenizer.load(prepared[1])
    rows = text_for("train", 2)
    members = [gzip.compress((json.dumps({"text": text}) + "\n").encode()) for text in rows]
    full = b"".join(members)
    calls = []

    def opener(shard):
        data = members[0] if not calls else full
        calls.append(True)
        return streaming.CompleteResponse(io.BytesIO(data), len(full))

    stream = make_stream(tok, rows, opener=opener)
    baseline = make_stream(tok, rows)
    count = (sum(len(tok.encode(t)) + 2 for t in rows) - 1) // 16
    assert stream.next(count) == baseline.next(count)
    assert len(calls) == 2
    assert stream.state_dict() == baseline.state_dict()


def test_general_english_source_admission_and_versioned_budget():
    from codexa.data.language_protocol import V6_VERSION, authorized_budget, specification
    from codexa.data.stream_admission import stackexchange_discussion
    from codexa.data.stream_policy import StackExchangeAdmission

    license = "Creative Commons - Attribution Share-Alike - https://creativecommons.org/licenses/by-sa/4.0/"
    row = {
        "id": "42",
        "text": "An original CC0 synthetic explanation of language and everyday life. " * 20,
        "metadata": {
            "site": "english.stackexchange.com",
            "url": "https://english.stackexchange.com/questions/42",
            "license": license,
            "all_licenses": [license],
        },
    }
    assert stackexchange_discussion(row)
    policy = object.__new__(StackExchangeAdmission)
    identity = policy.group(row)
    row["metadata"]["site"] = "history.stackexchange.com"
    row["metadata"]["url"] = "https://history.stackexchange.com/questions/42"
    assert stackexchange_discussion(row)
    assert policy.group(row) != identity
    for site in ("math.stackexchange.com", "stackoverflow.com", "english.meta.stackexchange.com"):
        row["metadata"]["site"] = site
        row["metadata"]["url"] = f"https://{site}/questions/42"
        assert not stackexchange_discussion(row)
    row["metadata"]["site"] = "english.stackexchange.com"
    row["metadata"]["url"] = "https://english.stackexchange.com/questions/42"
    row["metadata"]["all_licenses"].append("unknown")
    assert not stackexchange_discussion(row)
    spec = specification(["stackexchange-general-english"], version=V6_VERSION)
    assert spec["usable_window_minimum"] == 270
    assert spec["usable_per_source_minimum"] == 90
    assert (
        spec["source_bindings"]["stackexchange-general-english"]["repo"]
        == "common-pile/stackexchange_filtered"
    )
    record = {
        "additional_target_tokens": 1_000_000_000,
        "cumulative_target_tokens": 1_500_006_912,
        "user_decision": "synthetic fixture",
    }
    assert not authorized_budget(record, V6_VERSION)
    record["training_launch_authorized"] = True
    assert authorized_budget(record, V6_VERSION)


def test_wikipedia_scope_filter_uses_title_and_intro_without_rewriting():
    from codexa.data.stream_admission import wikipedia_navigation

    text = "An original CC0 example is a useful account of history and places. " * 15
    row = {"metadata": {"title": "Original example"}, "text": text}
    assert not wikipedia_navigation(row)
    row["metadata"]["title"] = "Original example (disambiguation)"
    assert wikipedia_navigation(row)
    row["metadata"]["title"] = "Original example"
    row["text"] = (
        "Original example may refer to:\n * one original fictional place\n * another original fictional place"
    )
    assert wikipedia_navigation(row)
    row["text"] = text + "The author may also refer to a later event in ordinary prose."
    assert not wikipedia_navigation(row)
    assert row["text"] == text + "The author may also refer to a later event in ordinary prose."
