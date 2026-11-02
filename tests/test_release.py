import json
import subprocess
import sys

import pytest
import torch

from codexa.architecture.model import Codexa
from codexa.config import Config
from codexa.release.artifacts import digest, export_model, load_export, secret_findings
from codexa.release.preflight import preflight
from codexa.repository import check_repository, source_identity
from codexa.tokenizer.bpe import CodexaTokenizer


def artifact(prepared, tmp_path):
    _, tok, _ = prepared
    m = Codexa(Config().model).eval()
    meta = {
        "stage": "fixture",
        "version": "0.0.0-fixture.1",
        "training_origin": "random_initialization",
        "training_tokens": 0,
        "datasets": "CC0 BPE fixtures only",
        "compute": "CPU initialization",
        "source_commit": source_identity()["commit"],
        "evaluation": "Roundtrip only",
        "uses": "Serialization fixture",
        "weight_license": "Not approved for distribution",
    }
    p = export_model(m, CodexaTokenizer.load(tok), meta, tmp_path / "export")
    return m, p


def test_roundtrip(prepared, tmp_path):
    m, p = artifact(prepared, tmp_path)
    loaded, tok, _ = load_export(p)
    x = torch.tensor([[2, 7, 8, 3]])
    torch.testing.assert_close(m(x), loaded(x), rtol=0, atol=0)
    assert loaded.embedding.weight is loaded.lm_head.weight
    assert loaded.parameter_count() == 133440
    assert tok.decode(tok.encode("π ≤ 2")) == "π ≤ 2"
    code = "from codexa.release.artifacts import load_export; import sys; m,_,_=load_export(sys.argv[1]); print(m.parameter_count())"
    assert (
        subprocess.check_output([sys.executable, "-c", code, str(p)], text=True).strip() == "133440"
    )
    with pytest.raises(FileExistsError):
        export_model(m, tok, {}, p)
    report = tmp_path / "verification.json"
    report.write_text(json.dumps({"status": "passed", "source": source_identity()}))
    result = preflight(str(p), str(report))
    assert result["validation"] == "passed" and not result["publish_ready"]
    assert not result["publication_performed"]


@pytest.mark.parametrize(
    "change", ["checksum", "schema", "path", "secret", "extra", "shape", "tokenizer"]
)
def test_rejects(prepared, tmp_path, change):
    _, p = artifact(prepared, tmp_path)
    if change == "checksum":
        (p / "README.md").write_text("tampered")
    elif change == "extra":
        (p / "extra.pt").write_bytes(b"bad")
    else:
        manifest = json.loads((p / "manifest.json").read_text())
        if change == "path":
            manifest["../bad"] = manifest.pop("README.md")
        else:
            name = (
                "config.json"
                if change in ("schema", "shape")
                else "tokenizer.json"
                if change == "tokenizer"
                else "README.md"
            )
            if change == "secret":
                (p / name).write_text("gh" + "p_" + "a" * 40)
            else:
                data = json.loads((p / name).read_text())
                if change == "schema":
                    data["schema_version"] = 999
                elif change == "shape":
                    data["model"]["heads"] = 2
                    data["model"]["dim"] = 32
                    from codexa.config import ModelConfig

                    data["parameters"] = Codexa.expected_parameters(ModelConfig(**data["model"]))
                else:
                    data["added_tokens"][0]["content"] = "wrong"
                (p / name).write_text(json.dumps(data))
            manifest[name] = {"sha256": digest(p / name), "bytes": (p / name).stat().st_size}
        (p / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises((ValueError, RuntimeError)):
        load_export(p)


def test_policy_and_scan():
    check_repository()
    assert secret_findings("hf_" + "a" * 40)
    assert not secret_findings("ordinary prose")


def test_existing_output_guard(prepared):
    from codexa.training.trainer import Trainer

    data, tok, out = prepared
    a = Trainer(Config(), data, tok, out, "cpu")
    a.train(until=1, handle_signals=False)
    with pytest.raises(FileExistsError):
        Trainer(Config(), data, tok, out, "cpu").train(until=2, handle_signals=False)
