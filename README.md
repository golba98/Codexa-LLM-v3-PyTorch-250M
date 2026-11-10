# Codexa v3

Codexa v3 is a custom PyTorch decoder-only language model built from random initialization. It supports research into general language learning and, in a separate workstream, Grade 1–12 mathematics. The primary model has **248,687,488 parameters**: 24 layers, width 896, 14 attention heads, SwiGLU, RMSNorm, RoPE and tied embeddings. The implementation lives in [codexa/architecture](codexa/architecture/).

An internal run completed **500,006,912 training-token targets**. Its checkpoint passes recorded integrity and validation checks, but generation still suffers from repetition, factual errors and poor response coherence. **No pretrained public weights are available.** Configured 2K–32K context support does not establish learned long-context capability.

The owner has authorized a future **1B additional-token language continuation** using the reviewed, pinned English Wikipedia source. Training is prohibited during this repository cleanup and has not started. The saved machine launch record remains pending; it must be finalized separately with all gates intact. See [continuation readiness](reports/language/continuation-readiness.md), [completed training](reports/training/base-500m.md) and [language evaluation](reports/evaluation/language-baseline.md).

Mathematics corpus development proceeds separately. Source extracts, curriculum mappings, synthetic candidates and protected reviews are retained outside Git; they do not constitute an approved mathematics training corpus. See [mathematics status](reports/datasets/mathematics-corpus.md).

## Install

Python 3.11–3.13; Linux is verified. From the repository root:

```sh
python -m pip install uv==0.13.5
uv venv --python 3.13 .venv
uv pip sync requirements-cpu-lock.txt
uv pip install --no-deps -e .
source .venv/bin/activate
codexa --help
```

Use `requirements-lock.txt` for the recorded CUDA environment. Optional Parquet ingestion uses the `data` extra; installation details and dependency boundaries are in [CONTRIBUTING](CONTRIBUTING.md). Existing pins are retained.

## Inference

For a trusted existing native checkpoint with its embedded tokenizer:

```sh
codexa generate --checkpoint /path/to/checkpoint.pt --prompt 'Morning light' --max-tokens 32 --greedy
```

Replace the checkpoint path with an available local artifact. Public users must provide their own checkpoint; the repository does not download weights. SafeTensors exports use Codexa's loader; see the [release guide](docs/release.md). Random initialization and tiny smoke fixtures are engineering checks, not capability demonstrations.

## Development

```sh
python scripts/verify.py
```

This runs repository checks, Ruff lint/format checks, mypy and pytest. Training-dependent tests use tiny disposable CPU fixtures; there is no production training launch. For guarded forward-only verification:

```sh
python scripts/verify.py --forward-only
```

Model profiles remain at `configs/250m.yaml`, `configs/50m.yaml` and `configs/smoke.yaml`; context profiles are in [configs/context](configs/context/) and source configurations in [configs/data](configs/data/). Script locations and bounded smoke instructions are in [CONTRIBUTING](CONTRIBUTING.md).

## Find your way

| Topic | Start here |
| --- | --- |
| All documentation and workflows | [Documentation index](docs/index.md) |
| Model implementation and parameter accounting | [Architecture](docs/architecture.md) |
| Training, inference, monitoring and recovery | [Training guide](docs/training.md) |
| Sources, tokenizer, provenance and mathematics | [Dataset guide](docs/datasets.md) |
| Current results and historical failures | [Report index](reports/README.md) |
| Measured engineering performance | [Benchmark report](reports/benchmarks/recorded-performance.md) |
| Milestones and remaining limitations | [Roadmap](docs/roadmap.md) |
| Export formats and publication gates | [Release guide](docs/release.md) |

Code is Apache-2.0; original fixtures are CC0. Dataset and weight permissions require separate review. See [third-party notices](THIRD_PARTY_NOTICES.md), [security](SECURITY.md) and [contributing](CONTRIBUTING.md). Native checkpoints, optimizer states, private assessments and downloaded corpora belong outside Git. V3 is independent of the earlier PyTorch v1 and NumPy/CuPy v2 projects.
