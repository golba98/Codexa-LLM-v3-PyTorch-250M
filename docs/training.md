# Training, inference and recovery

The [completed 500M run](../reports/training/base-500m.md) is the preserved baseline. The current authorization supersedes the future 1B proposal with exactly **500,000,000 additional tokens**, ending at **1,000,006,912 cumulative targets**. The parent checkpoint and tokenizer remain immutable. Launch requires fresh configuration-bound optimizer/resume evidence; the [historical readiness report](../reports/language/continuation-readiness.md) records the earlier pending state.

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

The working streaming loader is in `codexa/data/streaming.py`; `stream_policy.py` binds the pinned source, tokenizer, exclusions, protected evaluations, rights, quality, capacity and compute evidence. Seven source hashes are part of that approval. Policy changes require fresh source hashes and evidence bindings; original quality judgments and exclusion payloads remain preserved.

Forward-only provider checks use `CODEXA_NO_TRAINING=1`, `--forward-only` and a mandatory byte cap no larger than 64 MiB. They perform no optimizer updates. Live training streams pinned provider bytes without building a replacement corpus. Existing evidence records native next-batch resume and fixed-validation replay; the [readiness report](../reports/language/continuation-readiness.md) explains measurement limits.

`codexa continuation stream-prepare` prepares a phase only after source/config-bound launch authorization passes. It does not train. The current Wikimedia phase requires exactly 500M targets. The legacy developer healthcheck covers historical 50M/100M array-based phases; it is not the current streaming launcher. Dataset quality approval never implies launch authorization.

Production streaming uses a disposable 2 GiB cache of the exact compressed provider blocks under the run's `stream-cache` directory. Duplicate checks and resume can replay already fetched bytes locally. Missing blocks use bounded 1 MiB HTTP ranges checked against the pinned object ETag, exact content range and length; cached blocks carry SHA-256 checksums. Eviction removes cache blocks owned by that cache, subject to the 40 GiB free-space reserve. This cache follows the original byte stream and has no checkpoint or token-array state.

## Rolling recovery

Set `training.checkpoint_every_tokens: 50000000` and `training.checkpoint_keep: 2` for this continuation. Token milestones are measured from the phase parent and saved after the first completed update crossing them. Omitted token frequency preserves legacy step-based saving. The reviewed continuation keeps AdamW moments, LR 0.0001/minimum 0.00001, 100 warmup updates and a fresh 15,259-update cosine schedule. The final update consumes only the remaining 25,856 targets.

Rolling native files are read back and compared with the complete live state, including optimizer, scheduler, RNG and streaming cursor/buffer. A checksummed `recovery.json` atomically identifies latest and previous; only obsolete files owned by that run are removed afterward. Lightweight history receipts remain. Saves require the intended volume and space for the new transaction above the 40 GiB reserve. A failed transaction stops training without deleting existing recovery points. Unrecoverable errors can lose updates since the last verified save.

`--resume /path/to/run/training` resolves the newest verified retained file, warning and falling back to previous if latest fails integrity. Normal file paths and native v1 checkpoint payloads remain supported; absent new configuration fields use their defaults. `best-weights.pt` is a private weights-only artifact, not a full recovery checkpoint or public export. Completion publishes `final.pt` as the latest full recovery file without a redundant optimizer copy.

## Monitoring

Training selects the Rich terminal UI on a capable TTY; `--ui none` is headless. Ctrl+C checkpoints after the current update. Terminal rendering runs separately so renderer failures do not stop optimization. `codexa dashboard --help` describes the independent localhost browser viewer and telemetry options.

For a persistent independent view using the same original Rich renderer:

```sh
python -m codexa.training.terminal --run /path/to/run/training
```

The viewer replays and follows real `events.jsonl` records, including resumed sessions; it has no training controls. Run training and viewer in separate tmux sessions so closing the visible viewer leaves optimization running. The existing checkpoint panel shows the next token milestone, retained recovery steps and disk usage. The performance panel shows update throughput and effective session throughput, which includes loading stalls, validation and saves. ETA uses elapsed time between recent updates. Reattach the launch viewer with `tmux attach-session -t codexa-v3-language-500m-viewer`.

The application retains scheduled validation loss, stops at exactly 500M additional tokens, verifies the final checkpoint and exits. No automatic generation evaluation, mathematics training or follow-on experiment is scheduled. Codex ends its task after confirming actual production optimizer updates and the visible independent viewer.

![Terminal training interface](images/terminal-training-full.png)

## Evaluation and context

Generation enforces context budgets and supports per-request KV caching. Configured profiles range from 2K to 32K; the baseline trained at 2,048. Experimental linear RoPE scaling is explicit. Configuration support does not establish trained long-context reasoning.

Protected evaluation tools live in `codexa/evaluation`; analysis/plotting tools are in `scripts/evaluation`. Use identical prompts and decoding settings for paired evaluation, independent checkpoint-blinded scoring, and distinct grammar/coherence/relevance/completeness/task-success dimensions. Preserve raw outputs privately. Automatic answer proxies and corpus loss are not human factual-accuracy or conversational-capability scores.

See the [language baseline](../reports/evaluation/language-baseline.md), [benchmarks](../reports/benchmarks/recorded-performance.md), [roadmap](roadmap.md), and [archived detailed training records](../reports/archive/index.json).
