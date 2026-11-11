# Documentation

Codexa v3 builds a decoder-only language model from random initialization. Use these guides to understand the implementation, run the local fixtures or review the research.

[Project overview](../README.md) · [Research reports](../reports/README.md) · [Contributing](../CONTRIBUTING.md)

## Choose a starting point

| I want to… | Start here |
| --- | --- |
| Install the project and try the CLI | [README quick start](../README.md#quick-start) |
| Run a small fixture or recover a run | [Training and inference](training.md) |
| Understand the custom Transformer | [Architecture](architecture.md) |
| Inspect sources, tokenizer and split protections | [Datasets](datasets.md) |
| See what is complete and what comes next | [Roadmap](roadmap.md) |
| Export and validate a local artifact | [Release](release.md) |
| Change the code or regenerate reports | [Contributing](../CONTRIBUTING.md) |

## The five guides

- **[Architecture](architecture.md)** explains dimensions, decoder equations, initialization, parameter accounting and the next-token objective.
- **[Training](training.md)** covers CPU fixtures, trusted-checkpoint inference, native recovery, streaming and monitoring.
- **[Datasets](datasets.md)** covers the frozen tokenizer, pinned language source, provenance, evaluation separation and mathematics corpus work.
- **[Roadmap](roadmap.md)** separates language learning, conversational post-training, mathematics and eventual publication.
- **[Release](release.md)** distinguishes native recovery files from inference exports and explains local validation and licensing.

These guides describe the implementation and supported workflows. Dated measurements and assessments belong in reports.

## Results and evidence

The [report index](../reports/README.md) groups findings separately from historical decisions. Start with the [completed base run](../reports/training/base-500m.md), [language evaluation](../reports/evaluation/language-baseline.md) or [recorded benchmarks](../reports/benchmarks/recorded-performance.md).

The [old 1B readiness assessment](../reports/language/continuation-readiness.md) describes the decision state on 10 November 2026. The current continuation budget is 500M additional tokens; use the [training guide](training.md#language-continuation) for that configuration.

## Historical records

The [research archive](../reports/archive/index.json) and [earlier engineering archive](archive/index.json) retain original documents and checksums. Archived commands and launch suggestions describe their original context. Follow the current guides for executable workflows.
