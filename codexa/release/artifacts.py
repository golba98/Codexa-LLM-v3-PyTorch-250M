import hashlib
import json
import re
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from tokenizers import Tokenizer

from codexa.architecture.model import Codexa
from codexa.config import ModelConfig
from codexa.tokenizer.bpe import CodexaTokenizer

PAYLOAD = {
    "LICENSE",
    "model.safetensors",
    "config.json",
    "tokenizer.json",
    "generation_config.json",
    "metadata.json",
    "README.md",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def secret_findings(text: str) -> bool:
    return bool(
        re.search(
            r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{30,}|hf_[A-Za-z0-9]{30,}|AKIA[A-Z0-9]{16}",
            text,
        )
    )


def export_model(
    model: Codexa, tokenizer: CodexaTokenizer, metadata: dict[str, Any], output: str | Path
) -> Path:
    from safetensors.torch import save_model

    target = Path(output)
    if target.exists():
        raise FileExistsError(target)
    if tokenizer.backend.get_vocab_size() != model.config.vocab_size:
        raise ValueError("Tokenizer vocabulary mismatch")
    if model.embedding.weight is not model.lm_head.weight:
        raise ValueError("Output weights must be tied")
    if metadata.get("training_origin") != "random_initialization":
        raise ValueError("Training origin must be explicit")
    if metadata.get("stage") not in ("fixture", "experimental", "base") or not metadata.get(
        "version"
    ):
        raise ValueError("Stage and version required")
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".export-", dir=target.parent))
    try:
        save_model(model, str(tmp / "model.safetensors"))
        config = {
            "schema_version": 1,
            "architecture": "codexa-decoder-v1",
            "model": asdict(model.config),
            "parameters": model.parameter_count(),
        }
        (tmp / "config.json").write_text(json.dumps(config, indent=2))
        tokenizer.save(tmp / "tokenizer.json")
        (tmp / "generation_config.json").write_text(
            json.dumps({"bos_token_id": 2, "eos_token_id": 3, "pad_token_id": 0})
        )
        (tmp / "metadata.json").write_text(json.dumps(metadata, indent=2))
        sections = {
            "Status and version": f"{metadata['stage']} / {metadata['version']}",
            "Architecture": config,
            "Training origin": "random initialization; custom PyTorch architecture",
            "Tokenizer": f"Local byte BPE, vocabulary {model.config.vocab_size}; context {model.config.context}",
            "Training tokens": metadata.get("training_tokens", "UNKNOWN"),
            "Datasets and provenance": metadata.get("datasets", "UNKNOWN"),
            "Compute and hardware": metadata.get("compute", "UNKNOWN"),
            "Source and reproducibility": metadata.get("source_commit", "UNKNOWN"),
            "Evaluation": metadata.get("evaluation", "UNKNOWN"),
            "Capabilities and uses": metadata.get("uses", "UNKNOWN"),
            "Limitations, bias and safety": "No demonstrated general competence. Synthetic fixtures are serialization tests only. Outputs may be incorrect or harmful.",
            "Inference requirements": f"Codexa custom loader and PyTorch; FP32 weights approximately {model.parameter_count() * 4} bytes plus runtime memory. No Transformers or other runtime compatibility is claimed.",
            "Weight licensing": metadata.get("weight_license", "UNREVIEWED"),
            "Links": "https://github.com/golba98/Codexa-LLM-v3-PyTorch-250M — docs/RELEASE_PROCESS.md",
        }
        (tmp / "README.md").write_text(
            "# Codexa local model artifact\n\n"
            + "".join(
                f"## {k}\n\n{json.dumps(v) if isinstance(v, (dict, list)) else v}\n\n"
                for k, v in sections.items()
            )
        )
        (tmp / "LICENSE").write_text(
            metadata.get(
                "weight_license_text",
                "No weight distribution license approved. Local fixture only.\n",
            )
        )
        manifest = {p.name: {"sha256": digest(p), "bytes": p.stat().st_size} for p in tmp.iterdir()}
        (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2))
        tmp.rename(target)
        return target
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)


def validate_files(path: Path) -> dict[str, Any]:
    if path.is_symlink() or any(p.is_symlink() for p in path.iterdir()):
        raise ValueError("Symlinks forbidden")
    manifest = json.loads((path / "manifest.json").read_text())
    if set(manifest) != PAYLOAD or {p.name for p in path.iterdir()} != PAYLOAD | {"manifest.json"}:
        raise ValueError("Unexpected or missing artifact files")
    for name, record in manifest.items():
        if Path(name).name != name or not (path / name).is_file():
            raise ValueError("Invalid manifest path")
        p = path / name
        if p.stat().st_size != record["bytes"] or digest(p) != record["sha256"]:
            raise ValueError("Checksum mismatch: " + name)
        if p.suffix != ".safetensors" and secret_findings(p.read_text()):
            raise ValueError("Potential secret detected (redacted)")
    config = json.loads((path / "config.json").read_text())
    if config.get("schema_version") != 1 or config.get("architecture") != "codexa-decoder-v1":
        raise ValueError("Unsupported export schema")
    c = ModelConfig(**config["model"])
    if Codexa.expected_parameters(c) != config["parameters"]:
        raise ValueError("Parameter count mismatch")
    return config


def load_export(
    path: str | Path, device: str = "cpu"
) -> tuple[Codexa, CodexaTokenizer, dict[str, Any]]:
    from safetensors import safe_open
    from safetensors.torch import load_model

    root = Path(path)
    config = validate_files(root)
    c = ModelConfig(**config["model"])
    # Verify header dimensions before allocating real model weights.
    with torch.device("meta"):
        skeleton = Codexa(c)
    expected = skeleton.state_dict()
    with safe_open(root / "model.safetensors", framework="pt") as f:
        keys = set(f.keys())
        alias = {"embedding.weight", "lm_head.weight"}
        if keys - alias != set(expected) - alias or not keys & alias:
            raise ValueError("Unexpected tensor names")
        for k in keys:
            if tuple(f.get_slice(k).get_shape()) != tuple(expected[k].shape):
                raise ValueError("Tensor shape mismatch")
    tok = CodexaTokenizer(Tokenizer.from_file(str(root / "tokenizer.json")))
    if tok.backend.get_vocab_size() != c.vocab_size:
        raise ValueError("Tokenizer mismatch")
    model = Codexa(c).to(device)
    load_model(model, str(root / "model.safetensors"), strict=True, device=device)
    if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise ValueError("Nonfinite exported parameters")
    model.eval()
    return model, tok, json.loads((root / "metadata.json").read_text())
