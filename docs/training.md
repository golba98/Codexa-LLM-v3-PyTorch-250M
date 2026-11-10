# Training, inference and recovery

The [completed 500M run](../reports/training/base-500m.md) is the preserved baseline. The future 1B continuation is owner-authorized, but this cleanup prohibits launching it. The saved source-bound machine authorization remains pending. [Current readiness](../reports/language/continuation-readiness.md) separates quality, rights, streaming, compute, authorization and execution state.

## Bounded engineering smoke

Run from the repository root in the pinned environment:

```sh
sh scripts/training/smoke.sh
```

This fits a fixture tokenizer, packs the original [CC0 corpus](../examples/corpus/PROVENANCE.md), trains the **133,440-parameter smoke model** on CPU, stops/resumes, and generates text. It uses a fresh `outputs/experiments/smoke.*` directory and preserves results. It is not a production training command or a generalization benchmark. Full verification uses similarly disposable tiny fixtures.

## Native checkpoints and exact recovery

Native checkpoint version 1 retains model, AdamW, scheduler, sampler, CPU/CUDA/Python/NumPy RNG, counters, configuration and tokenizer identity. Restore a trusted native checkpoint into a fresh output directory with the same compatible configuration and data:

```sh
codexa train --help
codexa generate --checkpoint /path/to/checkpoint.pt --prompt 'Morning light' --max-tokens 32 --greedy
```

For actual training, the existing CLI accepts `--resume`, `--data`, `--tokenizer`, `--config` and `--output`; review their exact meanings with `--help` before a separately authorized run. Changing a completed schedule or dataset requires explicit continuation preparation, not silently modifying an exact-resume configuration. SafeTensors inference exports do not contain optimizer recovery state.

`--checkpoint-volume-uuid UUID` checks filesystem identity and a 20 GiB reserve before writes. Phase-specific storage policies can require a larger reserve. Never create a fallback directory when the intended volume is absent. `--deterministic` enables deterministic PyTorch algorithms; CUDA replay additionally needs `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Exact replay is bounded by tested software/hardware compatibility.

## Streaming continuation

The working streaming loader is in `codexa/data/streaming.py`; `stream_policy.py` binds the pinned source, tokenizer, exclusions, protected evaluations, rights, quality, capacity and compute evidence. Seven source hashes are part of that approval. Paths and source bytes are preserved by the cleanup.

Forward-only provider checks use `CODEXA_NO_TRAINING=1`, `--forward-only` and a mandatory byte cap no larger than 64 MiB. They perform no optimizer updates. No provider checks or downloads are required during this cleanup. Existing evidence records native next-batch resume and fixed-validation replay; the [readiness report](../reports/language/continuation-readiness.md) explains measurement limits.

`codexa continuation stream-prepare` prepares a phase only after source/config-bound launch authorization passes. It does not train. No launch bundle or replacement authorization record is created by repository maintenance. The legacy developer healthcheck covers historical 50M/100M array-based phases; it is not the 1B streaming launcher. Dataset quality approval never implies launch authorization.

## Monitoring

Training selects the Rich terminal UI on a capable TTY; `--ui none` is headless. Ctrl+C checkpoints after the current update. Terminal rendering runs separately so renderer failures do not stop optimization. `codexa dashboard --help` describes the independent localhost browser viewer and telemetry options.

![Terminal training interface](images/terminal-training-full.png)

## Evaluation and context

Generation enforces context budgets and supports per-request KV caching. Configured profiles range from 2K to 32K; the baseline trained at 2,048. Experimental linear RoPE scaling is explicit. Configuration support does not establish trained long-context reasoning.

Protected evaluation tools live in `codexa/evaluation`; analysis/plotting tools are in `scripts/evaluation`. Use identical prompts and decoding settings for paired evaluation, independent checkpoint-blinded scoring, and distinct grammar/coherence/relevance/completeness/task-success dimensions. Preserve raw outputs privately. Automatic answer proxies and corpus loss are not human factual-accuracy or conversational-capability scores.

See the [language baseline](../reports/evaluation/language-baseline.md), [benchmarks](../reports/benchmarks/recorded-performance.md), [roadmap](roadmap.md), and [archived detailed training records](../reports/archive/index.json).
