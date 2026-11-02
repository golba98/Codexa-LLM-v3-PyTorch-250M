import pytest
import torch
from torch.nn import functional as F

from codexa.architecture.attention import CausalAttention
from codexa.architecture.model import Codexa
from codexa.architecture.normalization import RMSNorm
from codexa.architecture.rope import RoPE
from codexa.architecture.transformer import SwiGLU
from codexa.config import Config, ModelConfig
from codexa.training.trainer import causal_loss


@pytest.mark.parametrize("name,count", [("smoke", 133440), ("50m", 49295872), ("250m", 248687488)])
def test_counts(name, count):
    c = Config.load(f"configs/{name}.yaml").model
    assert Codexa.count_without_allocation(c) == Codexa.expected_parameters(c) == count


def test_forward_tying_causal():
    model = Codexa(ModelConfig()).eval()
    x = torch.randint(512, (2, 12))
    other = x.clone()
    other[:, 6:] = torch.randint(512, (2, 6))
    assert model(x).shape == (2, 12, 512)
    torch.testing.assert_close(model(x)[:, :6], model(other)[:, :6], rtol=0, atol=0)
    assert model.embedding.weight is model.lm_head.weight
    with pytest.raises(ValueError):
        model(torch.ones(1, 65, dtype=torch.long))


def test_attention_reference():
    c = ModelConfig()
    attn = CausalAttention(c).eval()
    x = torch.randn(2, 9, 64)
    q, k, v = attn.qkv(x).reshape(2, 9, 3, 4, 16).permute(2, 0, 3, 1, 4).unbind(0)
    q, k = attn.rope(q), attn.rope(k)
    scores = q @ k.transpose(-2, -1) / 4
    scores.masked_fill_(~torch.ones(9, 9, dtype=torch.bool).tril(), -torch.inf)
    ref = attn.out((scores.softmax(-1) @ v).transpose(1, 2).reshape(2, 9, 64))
    assert attn(x).shape == x.shape
    torch.testing.assert_close(attn(x), ref)


def test_rope():
    rope = RoPE(4)
    x = torch.randn(2, 3, 5, 4)
    y = rope(x)
    torch.testing.assert_close(x.square().sum(-1), y.square().sum(-1))
    torch.testing.assert_close(x[:, :, 0], y[:, :, 0])
    angle = torch.tensor(1.0)
    torch.testing.assert_close(
        y[:, :, 1, 0], x[:, :, 1, 0] * angle.cos() - x[:, :, 1, 1] * angle.sin()
    )


def test_norm_swiglu():
    x = torch.randn(2, 4, 64)
    norm = RMSNorm(64)
    torch.testing.assert_close(norm(x), x / (x.square().mean(-1, keepdim=True) + 1e-6).sqrt())
    ff = SwiGLU(64, 176)
    torch.testing.assert_close(
        ff(x),
        F.linear(F.silu(F.linear(x, ff.gate.weight)) * F.linear(x, ff.up.weight), ff.down.weight),
    )


def test_loss_grad_step():
    m = Codexa(ModelConfig())
    x = torch.randint(512, (2, 10))
    y = torch.randint(512, (2, 10))
    logits = m(x)
    loss = causal_loss(logits, y)
    torch.testing.assert_close(loss, F.cross_entropy(logits.reshape(-1, 512), y.reshape(-1)))
    before = m.embedding.weight.detach().clone()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    torch.optim.AdamW(m.parameters()).step()
    assert not torch.equal(before, m.embedding.weight)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda():
    m = Codexa(ModelConfig()).cuda()
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_bf16_supported()):
        loss = causal_loss(
            m(torch.randint(512, (2, 16), device="cuda")),
            torch.randint(512, (2, 16), device="cuda"),
        )
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters())


@pytest.mark.parametrize("values", [{"heads": 3}, {"dim": 63}, {"dropout": 1.0}, {"context": 0}])
def test_invalid(values):
    with pytest.raises(ValueError):
        ModelConfig(**values)
