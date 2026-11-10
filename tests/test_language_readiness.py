"""Language authorization and evidence remain mandatory at continuation readiness."""

import json
from pathlib import Path
from typing import Any

import pytest

from codexa.data.language_protocol import (
    V2_VERSION,
    V3_VERSION,
    V4_VERSION,
    VERSION,
    specification,
    specification_hash,
)
from codexa.data.production import digest
from scripts.datasets.verify_review_packet import review_directories
from scripts.training.continuation_healthcheck import language_freshness_gate, language_quality_gate


def write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def authorization(tmp_path, version=VERSION):
    sources = [
        {V3_VERSION: "dclm-edu", V4_VERSION: "tinystories-gpt4"}.get(version, "wikipedia-en")
    ]
    protocol = specification(sources, version=version)
    path = tmp_path / "authorization.json"
    history = tmp_path / "history"
    history.mkdir()
    write(
        path,
        {
            "version": version,
            "status": "authorized",
            "specification_sha256": specification_hash(protocol),
            "sources": protocol["sources"],
            "additional_target_tokens": 50_000_000,
            "user_decision": "Explicit test authorization fixture",
            "historical_roots": [str(history)],
            "replaces": protocol["replaces"],
        },
    )
    protocol.update(authorization_path=str(path), authorization_sha256=digest(path))
    return protocol, path


def test_language_has_one_review_packet_only_after_authorization(tmp_path):
    protocol, path = authorization(tmp_path)
    review = tmp_path / "reviews" / "new"
    write(review / "protocol.json", protocol)
    assert review_directories(review) == (review,)
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="authorization changed"):
        review_directories(review)


def test_unknown_protocol_cannot_remove_legacy_extension(tmp_path):
    review = tmp_path / "review"
    write(review / "protocol.json", {"version": "invented", "sources": ["wikipedia-en"]})
    with pytest.raises(ValueError):
        review_directories(review)


def test_language_gate_requires_authorized_manifest_and_reconstruction(tmp_path):
    protocol, _ = authorization(tmp_path)
    data = tmp_path / "snapshots" / "new"
    write(data / "manifest.json", {"test": "manifest"})
    protocol["manifest_sha256"] = "stale"
    manifest = {"source_tokens": {"wikipedia-en": 100_000}, "tokenizer_hash": "frozen"}
    with pytest.raises(ValueError, match="another dataset"):
        language_quality_gate(data, tmp_path / "review", manifest, protocol)
    protocol["manifest_sha256"] = digest(data / "manifest.json")
    write(
        data / "independent-verification.json",
        {
            "status": "passed",
            "manifest_sha256": protocol["manifest_sha256"],
            "tokenizer_sha256": "frozen",
            "no_added_or_rewritten_text": True,
        },
    )
    with pytest.raises(ValueError, match="Independent language corpus"):
        language_quality_gate(data, tmp_path / "review", manifest, protocol)


def eligibility(tmp_path):
    protocol, _ = authorization(tmp_path)
    data = tmp_path / "snapshots" / "new"
    review = tmp_path / "reviews" / "new"
    manifest = {"source_tokens": {"wikipedia-en": 100_000}}
    log = {
        "accepted_windows": 300,
        "source_documents": {"wikipedia-en": 1000},
        "source_tokens": manifest["source_tokens"],
        "excluded_source_documents": {"wikipedia-en": 10},
        "excluded_source_tokens": {"wikipedia-en": 1000},
        "eligible_source_documents": {"wikipedia-en": 990},
    }
    write(review / "sampling-log.json", log)
    return data, review, manifest, protocol, log


def test_freshness_rechecks_representativeness_and_missing_history(tmp_path):
    data, review, manifest, protocol, log = eligibility(tmp_path)
    language_freshness_gate(data, review, manifest, protocol)
    log["excluded_source_tokens"]["wikipedia-en"] = 6000
    write(review / "sampling-log.json", log)
    with pytest.raises(ValueError, match="distorts"):
        language_freshness_gate(data, review, manifest, protocol)
    log["excluded_source_tokens"]["wikipedia-en"] = 1000
    write(review / "sampling-log.json", log)
    history = Path(
        json.loads(Path(protocol["authorization_path"]).read_text())["historical_roots"][0]
    )
    write(history / "reviews" / "earlier" / "packet-integrity.json", {})
    with pytest.raises(ValueError, match="omitted a known"):
        language_freshness_gate(data, review, manifest, protocol)


def test_language_manifest_cannot_relabel_failed_legacy_protocol(tmp_path):
    from codexa.data.quality_protocol import specification as legacy_specification

    review = tmp_path / "review"
    write(review / "protocol.json", legacy_specification())
    assert review_directories(review) == (review, review / "extended")


def test_language_baseline_requires_every_preserved_generated_response(tmp_path, monkeypatch):
    from codexa.evaluation.language import SETTINGS, freeze, load_fixtures
    from scripts.training import continuation_healthcheck as health

    monkeypatch.setattr(health, "language_baseline_review_gate", lambda root: None)

    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text("frozen tokenizer")
    monkeypatch.setattr(health, "TOKENIZER", tokenizer)
    fixtures = tmp_path / "benchmarks/language-v1"
    freeze(fixtures)
    output = tmp_path / "benchmarks" / ("language-baseline-" + health.PARENT_SHA[:16])
    records = [
        {"key": row["id"] + ":" + setting, "settings": SETTINGS}
        for row in load_fixtures(fixtures)
        for setting in ("greedy", "sample")
    ]
    output.mkdir()
    (output / "outputs.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
    identity = {
        "checkpoint_sha256": health.PARENT_SHA,
        "fixture_manifest_sha256": digest(fixtures / "manifest.json"),
        "settings": SETTINGS,
        "device": "cuda",
    }
    verification = {
        **identity,
        "complete": True,
        "records": 400,
        "expected_records": 400,
        "tokenizer_sha256": digest(tokenizer),
        "outputs_sha256": digest(output / "outputs.jsonl"),
        "weights_unchanged": True,
        "backward_calls": 0,
        "optimizer_updates": 0,
        "forbidden_attempts": [],
    }
    write(output / "identity.json", identity)
    write(output / "verification.json", verification)
    write(
        tmp_path / "benchmarks/language-pair-budget.json",
        {"limit_seconds": 14400, "used_seconds": 120},
    )
    assert len(health.language_baseline_gate(tmp_path)["comparison_keys"]) == 400
    (output / "outputs.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records[:-1]))
    verification["outputs_sha256"] = digest(output / "outputs.jsonl")
    write(output / "verification.json", verification)
    with pytest.raises(ValueError, match="baseline is missing or stale"):
        health.language_baseline_gate(tmp_path)


def test_baseline_cannot_train_without_independent_response_review(tmp_path):
    from scripts.training.continuation_healthcheck import language_baseline_review_gate

    with pytest.raises(FileNotFoundError):
        language_baseline_review_gate(tmp_path)


@pytest.mark.parametrize("changed", ["fixtures", "input"])
def test_baseline_review_must_bind_current_fixtures_and_output(tmp_path, changed):
    from codexa.evaluation.language import freeze, load_fixtures
    from scripts.training import continuation_healthcheck as health

    fixtures = tmp_path / "benchmarks/language-v1"
    freeze(fixtures)
    output = tmp_path / "benchmarks" / ("language-baseline-" + health.PARENT_SHA[:16])
    output.mkdir()
    (output / "outputs.jsonl").write_text("{}\n")
    write(
        tmp_path / "benchmarks/baseline-review-plan.json",
        {
            "selection": {"split": "test", "setting": "sample", "seed": 42, "max_tokens": 256},
            "reviewers_required": 2,
            "fixture_manifest_sha256": digest(fixtures / "manifest.json"),
        },
    )
    manifest = {
        "records": 100,
        "fixture_manifest_sha256": digest(fixtures / "manifest.json"),
        "selection": {"split": "test", "setting": "sample"},
        "inputs": [
            {"path": str(output.resolve()), "outputs_sha256": digest(output / "outputs.jsonl")}
        ],
    }
    if changed == "fixtures":
        manifest["fixture_manifest_sha256"] = "stale"
    else:
        manifest["inputs"][0]["path"] = str(tmp_path / "another-old-baseline")
    packet = tmp_path / "benchmarks/language-baseline-review"
    write(packet / "manifest.json", manifest)
    write(
        packet / "score.json",
        {
            "records": [
                {"key": row["id"] + ":sample", "checkpoint_sha256": health.PARENT_SHA}
                for row in load_fixtures(fixtures)
                if row["split"] == "test"
            ],
            "reviewers": {},
        },
    )
    with pytest.raises(ValueError, match="preregistered coverage"):
        health.language_baseline_review_gate(tmp_path)


def test_native_window_verification_detects_rehashed_rendering_tampering(tmp_path, monkeypatch):
    import numpy as np

    from scripts.datasets import verify_review_packet as packet

    snapshot, review = tmp_path / "snapshot", tmp_path / "review"
    snapshot.mkdir()
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text("frozen")
    tokens = np.arange(1100, dtype="<u2")
    tokens.tofile(snapshot / "train.bin")
    manifest = {"tokenizer_hash": digest(tokenizer), "splits": {"train": [{"file": "train.bin"}]}}
    write(snapshot / "manifest.json", manifest)
    write(review / "protocol.json", {"manifest_sha256": digest(snapshot / "manifest.json")})
    write(review / "packet-integrity.json", {})
    (review / "blind.jsonl").write_text("{}\n")
    (review / "mapping.jsonl").write_text(
        json.dumps({"id": "W1", "kind": "window", "position": 2}) + "\n"
    )

    class Encoder:
        def decode(self, ids, skip_special_tokens=False):
            return "native:" + str(ids[0]) + ":" + str(ids[-1])

    class Disk:
        def check(self, allocation):
            pass

    monkeypatch.setattr(packet.Storage, "load", lambda root: Disk())
    monkeypatch.setattr(packet, "verify", lambda path: manifest)
    monkeypatch.setattr(packet.CodexaTokenizer, "load", lambda path: Encoder())
    monkeypatch.setattr(packet, "compare_documents", lambda *args: 0)
    monkeypatch.setattr(packet, "review_directories", lambda path: (review,))

    def render(text):
        (review / "rendered-blind.jsonl").write_text(json.dumps({"id": "W1", "text": text}) + "\n")
        write(
            review / "rendering.json",
            {
                "original_blind_sha256": digest(review / "blind.jsonl"),
                "rendered_blind_sha256": digest(review / "rendered-blind.jsonl"),
            },
        )

    render("native:2:1025")
    assert packet.check(tmp_path, snapshot, review, tokenizer)["windows"] == 1
    render("Readable replacement which does not match the model input.")
    with pytest.raises(ValueError, match="differs from actual retained training tokens"):
        packet.check(tmp_path, snapshot, review, tokenizer)


def test_new_development_inspection_invalidates_approval_inventory(tmp_path):
    data, review, manifest, protocol, _ = eligibility(tmp_path)
    write(tmp_path / "reports/development/additional.jsonl", {"text": "new inspection"})
    with pytest.raises(ValueError, match="newly inspected development"):
        language_freshness_gate(data, review, manifest, protocol)


@pytest.mark.parametrize("version", [V2_VERSION, V3_VERSION, V4_VERSION])
def test_successor_review_directory_still_requires_bound_authorization(tmp_path, version):
    protocol, path = authorization(tmp_path, version)
    review = tmp_path / "reviews" / "new"
    write(review / "protocol.json", protocol)
    assert review_directories(review) == (review,)
    record = json.loads(path.read_text())
    record["version"] = VERSION
    write(path, record)
    protocol["authorization_sha256"] = digest(path)
    write(review / "protocol.json", protocol)
    with pytest.raises(ValueError, match="authorization"):
        review_directories(review)


@pytest.mark.parametrize("version", [V2_VERSION, V3_VERSION, V4_VERSION])
def test_successor_readiness_dispatch_does_not_fall_back_to_legacy(tmp_path, monkeypatch, version):
    from scripts.training import continuation_healthcheck as health

    protocol, _ = authorization(tmp_path, version)
    review = tmp_path / "reviews" / "new"
    write(review / "protocol.json", protocol)
    calls = []
    monkeypatch.setattr(health, "language_quality_gate", lambda *args: calls.append(args))
    monkeypatch.setattr(
        health,
        "legacy_quality_gate",
        lambda *args: pytest.fail("Language successors must not use historical mathematics gates"),
    )
    assert health.quality_gate(tmp_path, review, {}) == protocol
    assert len(calls) == 1


@pytest.mark.parametrize("version", [V2_VERSION, V3_VERSION, V4_VERSION])
def test_successor_postrun_uses_language_comparison(tmp_path, monkeypatch, version):
    from scripts.training import continuation_healthcheck as health

    record = {"checks": {"quality_protocol": version}}
    write(tmp_path / "control.json", record)
    calls = []
    monkeypatch.setattr(health, "language_postrun", lambda *args: calls.append(args))
    health.postrun(tmp_path)
    assert calls == [(tmp_path, record)]


@pytest.mark.parametrize(
    "budget,counters",
    [(50_000_000, (1526, 550006912, 24415)), (100_000_000, (3052, 600006912, 48829))],
)
def test_exact_authorized_budget_accounting(budget, counters):
    from scripts.training.continuation_healthcheck import expected_accounting

    assert expected_accounting(budget) == counters
    updates, _, microsteps = counters
    assert (updates - 1) * 32768 < budget <= updates * 32768
    assert (microsteps - 1) * 2048 < budget <= microsteps * 2048
    assert budget - (updates - 1) * 32768 == (28800 if budget == 50_000_000 else 24832)


def test_run_budget_rejects_unscoped_or_arbitrary_extensions():
    from scripts.training.continuation_healthcheck import run_budget

    assert run_budget({"checks": {}}) == 50_000_000
    assert (
        run_budget({"checks": {"additional_targets": 100_000_000, "quality_protocol": V4_VERSION}})
        == 100_000_000
    )
    for budget, version in (
        (100_000_000, V3_VERSION),
        (150_000_000, V4_VERSION),
        (True, V4_VERSION),
    ):
        with pytest.raises(ValueError, match="outside the authorized"):
            run_budget({"checks": {"additional_targets": budget, "quality_protocol": version}})


@pytest.mark.parametrize("budget", [50_000_000, 100_000_000])
def test_preflight_requires_bundle_budget_to_match_bound_authorization(
    tmp_path, monkeypatch, budget
):
    from types import SimpleNamespace

    from codexa import continuation, repository
    from scripts.training import continuation_healthcheck as health

    protocol, auth_path = authorization(tmp_path, V4_VERSION)
    if budget == 100_000_000:
        record = json.loads(auth_path.read_text())
        record.update(
            additional_target_tokens=budget,
            supersedes_additional_target_tokens=50_000_000,
            cumulative_target_tokens=600_006_912,
        )
        write(auth_path, record)
        protocol["authorization_sha256"] = digest(auth_path)
    data, bundle, review = (tmp_path / name for name in ("data", "bundle", "review"))
    write(data / "source-verification.json", {"status": "passed", "source": {"digest": "code"}})
    write(data / "manifest.json", {})
    write(bundle / "verification.json", {"checkpoint_sha256": "bundle"})
    write(review / "decision.json", {})
    monkeypatch.setattr(health.Storage, "load", lambda root: SimpleNamespace(check=lambda _: None))
    monkeypatch.setattr(
        health.shutil, "disk_usage", lambda root: SimpleNamespace(free=100 * health.GIB)
    )
    monkeypatch.setattr(repository, "source_identity", lambda: {"digest": "code"})
    monkeypatch.setattr(health, "verify", lambda _: {"tokenizer_hash": "frozen"})
    monkeypatch.setattr(health, "quality_gate", lambda *args: protocol)
    monkeypatch.setattr(
        health, "language_baseline_gate", lambda _: {"comparison_keys": ["protected"]}
    )
    report: dict[str, Any] = {
        "phase": {
            "additional_target_tokens": budget,
            "optimizer_policy": "preserve",
            "parent_tokens": 500006912,
            "parent_updates": 15259,
        },
        "optimizer_parameter_states": 170,
        "checkpoint_sha256": "bundle",
    }
    monkeypatch.setattr(continuation, "validate", lambda *args: report)
    checks = health.preflight(tmp_path, data, bundle, review, require_smoke=False)
    assert checks["additional_targets"] == budget
    assert checks["cumulative_target"] == 500006912 + budget
    report["phase"]["additional_target_tokens"] = (
        100_000_000 if budget == 50_000_000 else 50_000_000
    )
    with pytest.raises(ValueError, match="exact authorized target budget"):
        health.preflight(tmp_path, data, bundle, review, require_smoke=False)


def test_deferred_postrun_does_not_launch_evaluation(tmp_path, monkeypatch):
    from scripts.training import continuation_healthcheck as health

    monkeypatch.setattr(
        health.subprocess, "run", lambda *args, **kwargs: pytest.fail("Unexpected evaluation")
    )
    health.queued_postrun(tmp_path, {"defer_postrun": True})
    assert json.loads((tmp_path / "postrun-deferred.json").read_text()) == {
        "status": "awaiting_user",
        "training_complete": True,
    }


def test_historical_postrun_still_launches_read_only_evaluation(tmp_path, monkeypatch):
    from scripts.training import continuation_healthcheck as health

    calls = []
    monkeypatch.setattr(
        health.subprocess, "run", lambda *args, **kwargs: calls.append((args, kwargs))
    )
    health.queued_postrun(tmp_path, {})
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[0][-3:] == ["postrun", "--run", str(tmp_path)]
    assert kwargs["check"] is True
    assert kwargs["env"]["CODEXA_NO_TRAINING"] == "1"
    assert not (tmp_path / "postrun-deferred.json").exists()


@pytest.mark.parametrize("budget,defer", [(50_000_000, False), (100_000_000, True)])
def test_launch_persists_authorized_budget_and_explicit_evaluation_deferral(
    tmp_path, monkeypatch, budget, defer
):
    from types import SimpleNamespace

    from scripts.training import continuation_healthcheck as health

    parent = tmp_path / "pretraining/parent/final/checkpoint.pt"
    parent.parents[2].mkdir()
    monkeypatch.setattr(health, "PARENT", parent)
    monkeypatch.setattr(
        health,
        "preflight",
        lambda *args: {"additional_targets": budget, "quality_protocol": V4_VERSION},
    )
    commands = []
    monkeypatch.setattr(
        health.subprocess, "run", lambda command, **kwargs: commands.append(command)
    )
    monkeypatch.setattr(
        health.subprocess,
        "Popen",
        lambda command, **kwargs: (commands.append(command), SimpleNamespace(pid=123))[1],
    )
    health.launch(tmp_path, tmp_path, tmp_path, tmp_path, defer_postrun=defer)
    controls = list(parent.parents[2].glob("*/control.json"))
    assert len(controls) == 1
    record = json.loads(controls[0].read_text())
    assert record["checks"]["additional_targets"] == budget
    assert record["defer_postrun"] is defer
    assert record["session"].startswith(f"codexa-v3-exact{budget // 1_000_000}m-")
    assert f"--title=Codexa v3 — exact {budget // 1_000_000}M continuation" in commands[-1]
