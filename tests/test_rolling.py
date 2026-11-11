"""Recovery transactions exercise real serialization and bounded CPU optimization."""

import copy
import io
import json

import pytest
import torch

from codexa.config import Config, TrainConfig
from codexa.training import rolling
from codexa.training.rolling import RollingCheckpoints, equal_state
from codexa.training.terminal.__main__ import read_events
from codexa.training.trainer import Trainer


def payload(step):
    return {"step": step, "tokens": step * 32, "model": {"weight": torch.ones(2) * step}}


def test_rolling_preserves_only_two_and_final(tmp_path):
    original = tmp_path / "original.pt"
    original.write_bytes(b"protected original")
    other = tmp_path / "other-run.pt"
    other.write_bytes(b"unrelated")
    manager = RollingCheckpoints(tmp_path, reserve=0)
    entries = [manager.save(payload(i)) for i in range(1, 5)]
    assert not (tmp_path / entries[0]["file"]).exists()
    assert not (tmp_path / entries[1]["file"]).exists()
    assert [e["step"] for e in manager.record["retained"]] == [4, 3]
    final = manager.save(payload(5), final=True)
    assert final["file"] == "final.pt"
    assert manager.record["final"] == manager.record["latest"]
    assert [e["step"] for e in manager.record["retained"]] == [5, 4]
    assert len(list(tmp_path.glob("checkpoint-history/*.json"))) == 5
    assert original.read_bytes() == b"protected original" and other.read_bytes() == b"unrelated"
    assert manager.save(payload(5), final=True) == final


@pytest.mark.parametrize("failure", ["write", "verify", "manifest", "space"])
def test_failed_transaction_never_deletes_recovery(tmp_path, monkeypatch, failure):
    manager = RollingCheckpoints(tmp_path, reserve=0)
    manager.save(payload(1))
    manager.save(payload(2))
    manifest = manager.manifest.read_bytes()
    paths = [manager.path(e) for e in manager.record["retained"]]

    def fail(*args, **kwargs):
        raise OSError("injected transaction failure")

    if failure == "space":
        manager.reserve = 2**63
    elif failure == "write":
        monkeypatch.setattr(torch, "save", fail)
    elif failure == "verify":
        monkeypatch.setattr(rolling, "equal_state", fail)
    else:
        monkeypatch.setattr(rolling, "write_json", fail)
    with pytest.raises(OSError):
        manager.save(payload(3))
    assert manager.manifest.read_bytes() == manifest
    assert all(p.is_file() for p in paths)
    assert not list(tmp_path.glob(".pending-*"))


def test_corrupt_latest_falls_back_and_path_escape_is_rejected(tmp_path):
    manager = RollingCheckpoints(tmp_path, reserve=0)
    previous = manager.save(payload(1))
    latest = manager.save(payload(2))
    manager.path(latest).write_bytes(b"corruption")
    with pytest.warns(UserWarning, match="previous recovery"):
        assert manager.latest() == manager.path(previous)
    manager.save(payload(3))
    assert [e["step"] for e in manager.record["retained"]] == [3, 1]
    with pytest.raises(ValueError, match="belong"):
        manager.path({"file": "../original.pt"})
    with pytest.raises(ValueError, match="belong"):
        manager.path({"file": "other.pt"})


def test_best_has_no_optimizer_and_replaces_atomically(tmp_path, monkeypatch):
    manager = RollingCheckpoints(tmp_path, reserve=0)
    manager.save_best(payload(1))
    before = (tmp_path / "best-weights.pt").read_bytes()
    monkeypatch.setattr(rolling, "equal_state", lambda *a: (_ for _ in ()).throw(ValueError()))
    with pytest.raises(ValueError):
        manager.save_best(payload(2))
    assert (tmp_path / "best-weights.pt").read_bytes() == before


def test_replacement_of_corrupt_final_keeps_verified_previous(tmp_path):
    manager = RollingCheckpoints(tmp_path, reserve=0)
    previous = manager.save(payload(1))
    final = manager.save(payload(2), final=True)
    manager.path(final).write_bytes(b"corrupt final")
    replacement = manager.save(payload(2), final=True)
    assert manager.verify(replacement).name == "final.pt"
    assert manager.verify(previous).is_file()
    assert [e["step"] for e in manager.record["retained"]] == [2, 1]


def test_manifest_failure_can_retry_same_step(tmp_path, monkeypatch):
    manager = RollingCheckpoints(tmp_path, reserve=0)
    manager.save(payload(1))
    original_writer = rolling.write_json
    monkeypatch.setattr(rolling, "write_json", lambda *a: (_ for _ in ()).throw(OSError()))
    with pytest.raises(OSError):
        manager.save(payload(2))
    monkeypatch.setattr(rolling, "write_json", original_writer)
    manager.save(payload(2))
    assert manager.latest().is_file()
    assert len(list(tmp_path.glob("recovery-*.pt"))) == 2


def test_rolling_exact_resume_and_partial_final(prepared, tmp_path, monkeypatch):
    from codexa.data.v2_storage import Storage

    monkeypatch.delenv("CODEXA_NO_TRAINING", raising=False)
    monkeypatch.setattr(Storage, "check", lambda *a: None)
    data, tok, _ = prepared
    config = Config()
    config.model.dropout = 0.15
    config.training.steps = 4
    config.training.warmup = 1
    config.training.sequence = 16
    config.training.batch_size = 1
    config.training.accumulation = 3
    config.training.checkpoint_every_tokens = 48
    config.training.validate_every = 2
    phase = {
        "parent_tokens": 500,
        "parent_updates": 0,
        "additional_target_tokens": 163,
        "volume_uuid": "fixture",
        "minimum_free_bytes": 0,
    }

    def trainer(name):
        t = Trainer(config, data, tok, str(tmp_path / name), "cpu")
        t.phase, t.tokens = copy.deepcopy(phase), 500
        t.recovery.reserve = 0
        return t

    full = trainer("full")
    full.train(ui="none", handle_signals=False)
    partial = trainer("partial")
    partial.train(until=2, ui="none", handle_signals=False)
    resumed = trainer("partial")
    resumed.resume(str(partial.output))
    resumed.train(ui="none", handle_signals=False)
    assert full.tokens == resumed.tokens == 663
    assert full.step == resumed.step == 4
    for a, b in [
        (full.model.state_dict(), resumed.model.state_dict()),
        (full.optimizer.state_dict(), resumed.optimizer.state_dict()),
        (full.scheduler.state_dict(), resumed.scheduler.state_dict()),
        (full.sampler.state_dict(), resumed.sampler.state_dict()),
    ]:
        equal_state(a, b)
    manifest = json.loads((resumed.output / "recovery.json").read_text())
    assert manifest["final"]["file"] == "final.pt"
    assert len(manifest["retained"]) == 2
    best = torch.load(resumed.output / "best-weights.pt", weights_only=True)
    assert "optimizer" not in best
    assert best["validation_loss"] == resumed.best_validation_loss


def test_follower_waits_for_complete_records_and_reconnects():
    row = {
        "schema_version": 1,
        "run_id": "run",
        "session_id": "first",
        "seq": 1,
        "event": "started",
        "step": 0,
        "tokens": 0,
        "timestamp": "2026-11-10",
        "elapsed_seconds": 0.0,
        "data": {},
    }
    encoded = json.dumps(row).encode()
    stream = io.BytesIO(encoded)
    assert read_events(stream) == [] and stream.tell() == 0
    stream = io.BytesIO(encoded + b"\n" + encoded[:10])
    events = read_events(stream)
    assert len(events) == 1 and events[0].session_id == "first"
    assert stream.tell() == len(encoded) + 1


@pytest.mark.parametrize(
    "kwargs",
    [{"checkpoint_keep": 1}, {"checkpoint_every_tokens": 0}, {"checkpoint_every_tokens": True}],
)
def test_invalid_checkpoint_policy(kwargs):
    with pytest.raises(ValueError):
        TrainConfig(**kwargs)


def test_current_stream_budget_rejects_old_billion_authorization():
    from codexa.data.language_protocol import V7_VERSION, authorized_budget

    record = {
        "additional_target_tokens": 500_000_000,
        "cumulative_target_tokens": 1_000_006_912,
        "training_launch_authorized": True,
        "user_decision": "Explicit fixture authorization",
    }
    assert authorized_budget(record, V7_VERSION)
    record.update(additional_target_tokens=1_000_000_000, cumulative_target_tokens=1_500_006_912)
    assert not authorized_budget(record, V7_VERSION)


def test_stream_preparation_refuses_billion_before_any_io(tmp_path):
    from codexa.continuation import prepare_stream

    with pytest.raises(ValueError, match="500M budget"):
        prepare_stream(
            tmp_path / "run",
            tmp_path / "parent",
            tmp_path / "stream",
            tmp_path / "tokenizer",
            tmp_path / "config",
            "fixture",
            1_000_000_000,
        )
