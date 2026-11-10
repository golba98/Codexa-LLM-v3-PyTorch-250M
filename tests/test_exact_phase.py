"""Bounded CPU optimization validates exact-budget recovery; no real weights used."""

import copy

import torch

from codexa.config import Config
from codexa.data.v2_storage import Storage
from codexa.training.trainer import Trainer


def test_exact_phase_partial_update_and_resume(prepared, tmp_path, monkeypatch):
    monkeypatch.delenv("CODEXA_NO_TRAINING", raising=False)
    monkeypatch.setattr(Storage, "check", lambda *a, **kw: None)
    data, tok, _ = prepared
    config = Config()
    config.training.steps = 3
    config.training.warmup = 1
    config.training.sequence = 16
    config.training.batch_size = 1
    config.training.accumulation = 3
    config.model.dropout = 0.15
    phase = {
        "parent_tokens": 500,
        "parent_updates": 0,
        "additional_target_tokens": 115,
        "volume_uuid": "fixture",
        "minimum_free_bytes": 0,
    }
    full = Trainer(config, data, tok, str(tmp_path / "full"), "cpu")
    full.phase, full.tokens = copy.deepcopy(phase), 500
    full.train(handle_signals=False, ui="none")
    expected = {k: v.clone() for k, v in full.model.state_dict().items()}
    expected_optimizer = copy.deepcopy(full.optimizer.state_dict())
    partial = Trainer(config, data, tok, str(tmp_path / "partial"), "cpu")
    partial.phase, partial.tokens = copy.deepcopy(phase), 500
    partial.train(until=2, handle_signals=False, ui="none")
    resumed = Trainer(config, data, tok, str(tmp_path / "resume"), "cpu")
    resumed.resume(str(tmp_path / "partial/latest.pt"))
    rows = resumed.train(handle_signals=False, ui="none")
    assert resumed.tokens == full.tokens == 615
    assert resumed.step == full.step == 3
    assert resumed.microsteps == full.microsteps == 8
    assert resumed.phase["unused_reserved_window_targets"] == 13
    assert rows[0]["tokens"] == 615
    for name, value in full.sampler.state_dict().items():
        actual = resumed.sampler.state_dict()[name]
        assert torch.equal(value, actual) if isinstance(value, torch.Tensor) else value == actual
    assert resumed.scheduler.state_dict() == full.scheduler.state_dict()
    for name, value in resumed.model.state_dict().items():
        assert torch.equal(value, expected[name])
    for key, old in expected_optimizer["state"].items():
        for name, value in old.items():
            actual = resumed.optimizer.state_dict()["state"][key][name]
            assert (
                torch.equal(value, actual) if isinstance(value, torch.Tensor) else value == actual
            )
