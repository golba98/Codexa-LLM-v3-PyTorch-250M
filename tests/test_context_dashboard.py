import json
import threading
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest
import torch

from codexa.architecture.model import Codexa
from codexa.architecture.rope import RoPE
from codexa.config import Config
from codexa.dashboard.server import History, serve
from codexa.inference.generate import generate
from codexa.training.callbacks.telemetry import EventBus
from codexa.training.trainer import Trainer


def test_cache_and_positions():
    torch.manual_seed(42)
    m = Codexa(Config().model).eval()
    x = torch.randint(512, (1, 12))
    full = m(x)
    _, cache = m.forward_cached(x[:, :5], capacity=16)
    y, cache = m.forward_cached(x[:, 5:9], cache)
    torch.testing.assert_close(y, full[:, 8:9], atol=1e-6, rtol=1e-5)
    y, cache = m.forward_cached(x[:, 9:], cache)
    torch.testing.assert_close(y, full[:, -1:], atol=1e-6, rtol=1e-5)
    assert cache.length == 12
    assert generate(m, [2, 7], 12, greedy=True) == generate(
        m, [2, 7], 12, greedy=True, use_cache=False
    )
    with pytest.raises(ValueError):
        generate(m, [2] * 60, 8)
    with pytest.raises(ValueError):
        m.forward_cached(x, cache)
    sample = torch.randn(1, 1, 1, 64)
    torch.testing.assert_close(RoPE(64, factor=16)(sample, 16), RoPE(64)(sample, 1))
    rope = RoPE(64)
    v = torch.randn(1, 1, 32768, 64)
    rot = rope(v)
    assert torch.isfinite(rot).all()
    torch.testing.assert_close(rot[:, :, 32760:], rope(v[:, :, 32760:], 32760))
    for n in (2048, 4096, 8192, 16384, 32768):
        c = Config.load(f"configs/context/250m-{n}.yaml")
        assert Codexa.count_without_allocation(c.model) == 248687488


def test_telemetry_and_server(prepared, tmp_path):
    data, tok, out = prepared
    t = Trainer(Config(), data, tok, out, "cpu")
    t.train(until=3, handle_signals=False)
    rows = [
        json.loads(s)
        for s in (__import__("pathlib").Path(out) / "events.jsonl").read_text().splitlines()
    ]
    assert rows[0]["event"] == "started" and rows[-1]["event"] == "paused"
    assert len([r for r in rows if r["event"] == "step_completed"]) == 3
    assert (
        next(r for r in rows if r["event"] == "checkpoint_saved")["data"]["verification_status"]
        == "atomic_save_only"
    )
    t2 = Trainer(Config(), data, tok, out, "cpu")
    t2.resume(out + "/latest.pt")
    t2.train(until=5, handle_signals=False)
    server = serve(tmp_path, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        runs = json.load(urlopen(base + "/api/runs"))
        assert runs
        view = json.load(urlopen(base + "/api/run?id=" + runs[0]["id"]))
        assert view["records"] > 0
        assert b"CODEXA" in urlopen(base).read()
        with pytest.raises(HTTPError):
            urlopen(base + "/../../README.md")
    finally:
        server.shutdown()
        server.server_close()


def test_failure_and_partial(tmp_path):
    class Bad:
        def on_event(self, event):
            raise RuntimeError("test")

    bus = EventBus(tmp_path, {}, (Bad(),))
    with pytest.warns(UserWarning):
        bus.emit("started", 0, 0)
    bus.emit("step_completed", 1, 64, loss=2.0)
    bus.emit("step_completed", 1, 64, loss=3.0)
    bus.close()
    p = tmp_path / "events.jsonl"
    with p.open("ab") as f:
        f.write(b"{bad")
    bus = EventBus(tmp_path, {})
    bus.emit("started", 1, 64)
    bus.close()
    h = History(tmp_path, tmp_path / "cache")
    key = next(iter(h.runs()))
    v = h.view(key, p)
    assert v["corrupt"] == 1 and v["records"] == 4
    replacement = tmp_path / "replacement"
    replacement.write_text(json.dumps({"step": 2, "loss": 0.25}) + "\n")
    replacement.replace(p)
    v = h.view(key, p)
    assert v["records"] == 1 and v["rows"][0]["loss"] == 0.25


def test_checkpoint_gradient_parity():
    a = Codexa(Config().model)
    b = Codexa(Config().model)
    b.load_state_dict(a.state_dict())
    b.gradient_checkpointing = True
    x = torch.randint(512, (1, 8))
    a(x).sum().backward()
    b(x).sum().backward()
    for p, q in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(p.grad, q.grad)


def test_live_restart(prepared, tmp_path):
    import time
    from pathlib import Path

    class Slow:
        def on_event(self, event):
            if event.event == "step_completed":
                time.sleep(0.01)

    data, tok, out = prepared
    trainer = Trainer(Config(), data, tok, out, "cpu", callbacks=(Slow(),))
    errors = []

    def training():
        try:
            trainer.train(until=100, handle_signals=False)
        except Exception as e:  # noqa: BLE001 - propagate worker failures to test
            errors.append(e)

    worker = threading.Thread(target=training)
    worker.start()
    deadline = time.monotonic() + 5
    while not (Path(out) / "events.jsonl").exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    for _ in range(2):
        server = serve(tmp_path, 0)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{server.server_port}"
        runs = json.load(urlopen(base + "/api/runs"))
        view = json.load(urlopen(base + "/api/run?id=" + runs[0]["id"]))
        assert view["records"] > 0
        with urlopen(base + "/api/events?id=" + runs[0]["id"], timeout=5) as stream:
            line = stream.readline()
            assert line.startswith(b"data: ")
        server.shutdown()
        server.server_close()
    worker.join(10)
    assert not worker.is_alive() and not errors and trainer.step == 100


def test_transition_and_rollback(prepared, tmp_path):
    from pathlib import Path

    import yaml

    from codexa.context import main

    data, tok, out = prepared
    old = Trainer(Config(), data, tok, out, "cpu")
    old.train(until=3, handle_signals=False)
    config = Config()
    config.model.context = 128
    config.training.sequence = 128
    cfg = tmp_path / "stage.yaml"
    cfg.write_text(yaml.safe_dump(config.to_dict()))
    target = tmp_path / "stage.pt"
    main(
        [
            "transition",
            "--checkpoint",
            out + "/latest.pt",
            "--config",
            str(cfg),
            "--output",
            str(target),
            "--reset-data-order",
        ]
    )
    new = Trainer(config, data, tok, str(tmp_path / "new"), "cpu")
    new.resume(str(target))
    assert new.step == 3 and new.tokens == 384
    new.train(until=4, handle_signals=False)
    assert new.tokens == 640
    rows = [json.loads(s) for s in Path(tmp_path / "new/events.jsonl").read_text().splitlines()]
    assert any(r["event"] == "context_changed" for r in rows)
    root = tmp_path / "rollback"
    bus = EventBus(root, {})
    bus.emit("started", 0, 0)
    bus.emit("step_completed", 20, 1280, loss=1.0)
    bus.close()
    bus = EventBus(root, {})
    bus.emit("started", 10, 640)
    bus.emit("step_completed", 11, 704, loss=2.0)
    bus.close()
    h = History(root, tmp_path / "cache-rollback")
    k = next(iter(h.runs()))
    v = h.view(k, root / "events.jsonl")
    assert not any(r["step"] == 20 for r in v["rows"])


def test_sampled_parity_and_cache_owner():
    m = Codexa(Config().model).eval()
    assert generate(m, [2, 7], 10, seed=4, top_k=12, top_p=0.9) == generate(
        m, [2, 7], 10, seed=4, top_k=12, top_p=0.9, use_cache=False
    )
    _, cache = m.forward_cached(torch.tensor([[2, 7]]), capacity=16)
    with pytest.raises(ValueError):
        Codexa(Config().model).eval().forward_cached(torch.tensor([[8]]), cache)
