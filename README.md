# Codexa v3

A custom PyTorch decoder-only language model research project, trained from random initialization. The primary architecture has **248,687,488 parameters**: 24 layers, width 896, 14 heads, SwiGLU, RMSNorm, RoPE and tied embeddings. Configurations support 2K–32K context; architectural support does not establish trained long-context capability.

**No pretrained public model is available.** CPU/CUDA smoke training, exact CPU recovery, inference, Rich terminal monitoring, a separate local browser dashboard and SafeTensors exports are implemented. Production corpus approval and sustained pretraining remain future work.

V1 explored PyTorch; v2 explored NumPy/CuPy. V3 is a separate implementation.

## Install

Python 3.11–3.13; Linux is verified. Use the CPU lock for development or the CUDA lock for the recorded RTX 4080 environment:

```sh
uv venv --python 3.13 .venv
uv pip sync requirements-cpu-lock.txt
uv pip install --no-deps -e .
source .venv/bin/activate
```

Optional Parquet ingestion requires `uv pip install -e '.[data]'`. Existing dependency pins are retained. CUDA wheels include their runtime; install an appropriate NVIDIA driver separately.

## Quick start

The original small corpus is an engineering fixture, not a generalization benchmark:

```sh
codexa tokenizer train --corpus examples/corpus --vocab-size 512 --output outputs/tokenizers/demo/tokenizer.json
codexa data prepare --corpus examples/corpus --tokenizer outputs/tokenizers/demo/tokenizer.json --output outputs/datasets/demo
codexa train --config configs/smoke.yaml --data outputs/datasets/demo --tokenizer outputs/tokenizers/demo/tokenizer.json --output outputs/experiments/demo --device cpu --until 50 --ui terminal
codexa train --config configs/smoke.yaml --data outputs/datasets/demo --tokenizer outputs/tokenizers/demo/tokenizer.json --output outputs/experiments/demo --device cpu --resume outputs/experiments/demo/latest.pt
codexa generate --checkpoint outputs/experiments/demo/latest.pt --prompt 'Morning light' --max-tokens 16 --greedy
```

Training selects the full Rich TUI automatically on a capable TTY. `--ui none` is headless; Ctrl+C safely checkpoints after the current update. `codexa dashboard` starts the independent browser viewer. No port is required for terminal monitoring.

![Terminal training](docs/images/terminal-training-full.png)

## Production data workflow

Use a reviewed source manifest, or explicitly fetch the small pinned WikiText-2 verification sample (under 100 MiB), then curate and prepare immutable uint16 shards:

```sh
codexa data sample --output outputs/datasets/wikitext-sample
codexa data curate --config outputs/datasets/wikitext-sample/config.yaml --output outputs/datasets/curated
codexa tokenizer train --dataset outputs/datasets/curated --vocab-size 16384 --output outputs/tokenizers/candidate.json
codexa tokenizer evaluate --dataset outputs/datasets/curated --tokenizer outputs/tokenizers/candidate.json
codexa tokenizer freeze --tokenizer outputs/tokenizers/candidate.json --output outputs/tokenizers/frozen
codexa data prepare --dataset outputs/datasets/curated --tokenizer outputs/tokenizers/frozen/tokenizer.json --output outputs/datasets/prepared
codexa data verify --data outputs/datasets/prepared
```

Edit source paths/provenance before using the production example. See [datasets](docs/DATASETS.md) for limits, contamination controls and candidate status. A candidate tokenizer is not an approved production freeze. Never launch the 100M-token pilot or 5B-token run without explicit authorization.

## Development and documentation

```sh
python scripts/verify.py
sh scripts/smoke.sh
codexa benchmark --config configs/smoke.yaml --device cpu --sequence 64
```

- [Architecture](docs/ARCHITECTURE.md)
- [Training, monitoring and recovery](docs/TRAINING.md)
- [Datasets and tokenizer](docs/DATASETS.md)
- [Research roadmap](docs/ROADMAP.md)
- [Release process](docs/RELEASE.md)
- [Measured benchmarks](docs/benchmarks/README.md)

Source code is Apache-2.0; original fixtures are CC0. External datasets and model weights require separate licensing review. See [contributing](CONTRIBUTING.md), [security](SECURITY.md) and [third-party notices](THIRD_PARTY_NOTICES.md). Historical engineering records are preserved in the [archive index](docs/archive/index.json). No weights or large data belong in Git.
