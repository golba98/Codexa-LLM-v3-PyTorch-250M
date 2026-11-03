import copy
import random

import numpy as np
import pytest
import torch

from codexa.architecture.model import Codexa
from codexa.config import Config
from codexa.data.packing import PackedData, Sampler, read_documents, split_documents
from codexa.inference.generate import filter_logits, from_checkpoint, generate
from codexa.mathematics.arithmetic import accuracy, problems
from codexa.tokenizer.bpe import CodexaTokenizer
from codexa.training.checkpoint import load, save_atomic
from codexa.training.trainer import Trainer, causal_loss


def test_tokenizer_packing(prepared):
    data, path, _ = prepared
    tok = CodexaTokenizer.load(path)
    for text in ["Hello, café!\n  x² ≤ π × √2 ≠ 0\t🙂", "", "a  b\r\nc", "0123456789 − ÷ ∑"]:
        assert tok.decode(tok.encode(text)) == text
    ds = PackedData(data + "/train.bin", 64)
    x, y = ds.batch([0, 1], torch.device("cpu"))
    torch.testing.assert_close(x[:, 1:], y[:, :-1])
    a, b = split_documents(read_documents("examples/corpus"))
    assert not set(a) & set(b)
    assert len(a) + len(b) == 12


def compare(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            compare(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            compare(x, y)
    else:
        assert a == b


def test_resume_exact(prepared, tmp_path):
    data, tok, out = prepared
    c = Config()
    c.model.dropout = 0.15
    c.training.accumulation = 2
    full = Trainer(c, data, tok, out, "cpu")
    full.train(until=6, handle_signals=False)
    expected = {k: v.clone() for k, v in full.model.state_dict().items()}
    opt = copy.deepcopy(full.optimizer.state_dict())
    sched = full.scheduler.state_dict()
    draws = (random.random(), np.random.rand(), torch.rand(3))
    partial = Trainer(c, data, tok, str(tmp_path / "partial"), "cpu")
    partial.train(until=3, handle_signals=False)
    resumed = Trainer(c, data, tok, str(tmp_path / "resumed"), "cpu")
    resumed.resume(str(tmp_path / "partial/latest.pt"))
    resumed.train(until=6, handle_signals=False)
    compare(expected, resumed.model.state_dict())
    compare(opt, resumed.optimizer.state_dict())
    compare(sched, resumed.scheduler.state_dict())
    compare(full.sampler.state_dict(), resumed.sampler.state_dict())
    assert resumed.tokens == full.tokens == 6 * 2 * 2 * 64
    assert random.random() == draws[0] and np.random.rand() == draws[1]
    assert torch.equal(torch.rand(3), draws[2])
    assert isinstance(
        from_checkpoint(str(tmp_path / "resumed/latest.pt"), "Hi", "cpu", max_tokens=2), str
    )
    c.training.lr *= 2
    bad = Trainer(c, data, tok, str(tmp_path / "bad"), "cpu")
    with pytest.raises(ValueError):
        bad.resume(str(tmp_path / "partial/latest.pt"))


def test_atomic_failure(tmp_path, monkeypatch):
    p = tmp_path / "safe.pt"
    save_atomic({"a": 1}, p)

    def fail(*args, **kwargs):
        raise OSError("simulated failed write")

    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError):
        save_atomic({"a": 2}, p)
    assert load(p) == {"a": 1}
    assert list(tmp_path.iterdir()) == [p]


def test_accumulation():
    torch.manual_seed(3)
    a = Codexa(Config().model)
    b = copy.deepcopy(a)
    x = torch.randint(512, (4, 8))
    y = torch.randint(512, (4, 8))
    causal_loss(a(x), y).backward()
    for i in range(4):
        (causal_loss(b(x[i : i + 1]), y[i : i + 1]) / 4).backward()
    for p, q in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(p.grad, q.grad, atol=1e-7, rtol=1e-4)


def test_generation():
    m = Codexa(Config().model)
    assert generate(m, [2, 7], 5, greedy=True) == generate(m, [2, 7], 5, greedy=True, seed=123)
    assert generate(m, [2], 10, top_k=8, top_p=0.9) == generate(m, [2], 10, top_k=8, top_p=0.9)
    first = generate(m, [2], 1, greedy=True)[-1]
    assert len(generate(m, [2], 20, greedy=True, eos=first)) == 2
    with pytest.raises(ValueError):
        generate(m, list(range(70)), 3, eos=-1)
    s = filter_logits(torch.tensor([4.0, 3.0, 2.0, 1.0]), top_k=2)
    assert torch.isfinite(s).sum() == 2
    assert torch.isfinite(filter_logits(torch.tensor([20.0, 0.0, -1.0]), top_p=0.5)).sum() == 1
    with pytest.raises(ValueError):
        generate(m, [2], temperature=0)


def test_math():
    pools = [problems(s, n) for s, n in [("train", 200), ("val", 20), ("test", 20)]]
    assert pools[0] == problems("train", 200)
    ids = [{p.id for p in pool} for pool in pools]
    assert not ids[0] & ids[1] and not ids[0] & ids[2] and not ids[1] & ids[2]
    for pool in pools:
        for p in pool:
            a, b = p.operands
            assert p.answer == (a + b if p.topic == "addition" else a - b)
            assert str(p.answer) in p.steps[-1]
    rows = [p.to_dict() for p in pools[2]]
    assert accuracy(rows, {p.id: str(p.answer) for p in pools[2]})["overall"]["accuracy"] == 1
    assert accuracy(rows, {})["overall"]["accuracy"] == 0
    with pytest.raises(ValueError):
        problems("test", 10000)


def test_sampler():
    a = Sampler(5, 1)
    a.next(7)
    state = a.state_dict()
    expected = a.next(13)
    b = Sampler(5, 99)
    b.load_state_dict(state)
    assert b.next(13) == expected


def test_overfit():
    torch.manual_seed(42)
    m = Codexa(Config().model)
    x = torch.tensor([[2, 7, 8, 9, 7, 8, 9, 3]])
    y = torch.tensor([[7, 8, 9, 7, 8, 9, 3, 2]])
    opt = torch.optim.AdamW(m.parameters(), lr=0.003)
    initial = float(causal_loss(m(x), y).detach())
    for _ in range(100):
        opt.zero_grad()
        loss = causal_loss(m(x), y)
        loss.backward()
        opt.step()
    final = float(causal_loss(m(x), y).detach())
    print({"overfit_initial": initial, "overfit_final": final})
    assert final < 1 and final < initial * 0.2


def test_graceful_signal(prepared, monkeypatch):
    import os
    import signal

    data, tok, out = prepared
    trainer = Trainer(Config(), data, tok, out, "cpu")
    previous_handler = signal.getsignal(signal.SIGINT)

    def signal_backward(gradient):
        os.kill(os.getpid(), signal.SIGINT)
        return gradient

    trainer.model.embedding.weight.register_hook(signal_backward)
    trainer.train(until=4)
    state = load(out + "/latest.pt")
    assert state["step"] == trainer.step == 1
    assert signal.getsignal(signal.SIGINT) == previous_handler


def test_validation_preserves_rng(prepared):
    from codexa.training.checkpoint import rng_state

    data, tok, out = prepared
    trainer = Trainer(Config(), data, tok, out, "cpu")
    before = rng_state()
    trainer.validate()
    compare(before, rng_state())
    assert trainer.model.training
