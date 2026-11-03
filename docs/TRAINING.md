# Training and evaluation

Use the executable quick start in the root README. Existing corpus commands and uint32 smoke datasets remain supported. Production preparation yields a versioned sharded uint16 manifest; `--data` detects the format. Model configuration and loss are unchanged.

## Training and recovery

Use `codexa train --config CONFIG --data DATA --tokenizer TOKENIZER --output NEW_RUN --device cpu|cuda|auto`. Stop at a bounded global step with `--until 50`; resume with `--resume RUN/latest.pt` using identical configuration, data, tokenizer, device type and precision. Ctrl+C/SIGTERM finish the current optimizer update and save safely. Exceptions retain the preceding successful checkpoint.

Training uses AdamW, warmup/cosine decay, gradient accumulation and clipping. BF16 autocast is used on supported CUDA devices; FP32 parameters/gradients/Adam states remain. Unsupported precision falls back to FP32. Sequences use exactly one next-token shift and permit causal context across BOS/EOS document boundaries; no padding is introduced.

Native checkpoint version remains 1. Model, optimizer, scheduler, RNG, sampler cursor/order, token counts, tokenizer JSON and fingerprints are restored. New sharded datasets use a six-round Feistel permutation with cycle walking: every window appears once per epoch without allocating a full permutation. Algorithm/seed/epoch/cursor are checkpointed. Legacy datasets retain the original permutation-array sampler, preserving exact old CPU recovery. Dataset changes require a new run, not silent resume. Loading is single-process; no multiprocessing prefetch is claimed.

## Terminal dashboard

Rich is optional (`uv pip install -e '.[terminal]'`) and included in the locked development environments. `--ui auto` selects the full panel interface on a capable POSIX TTY; `--ui terminal` selects it explicitly; `--ui none` is headless. Missing Rich, redirected output and dumb terminals fall back safely. `NO_COLOR` is respected. Linux is verified.

The interface adapts v1's rounded two-column panels, cyan/slate palette, lifecycle badges, progress bar, sparklines and compact fallback. Original MIT attribution is in THIRD_PARTY_NOTICES and the bundled V1_LICENSE. V2's single-line display was inspected but not copied. Kitty launch gates, supervisor/notification controls and raw keyboard input were deliberately not transplanted.

Panels show raw/smoothed loss, validation/best loss, perplexity, applied LR, pre-clipping gradient norm, optimizer, throughput, timing, batch/accumulation, dataset, loading duration, context and checkpoint events. Epoch is unavailable for legacy shuffled windows. Unknown metrics remain unavailable. The mean covers at most 60 observed session losses; peak memory is the maximum observed per-step process allocation in that session. GPU utilization/temperature/power are optional physical-GPU samples; CPU/RAM samples come from Linux /proc. Samples are asynchronous and timestamped. `auto` attention is a configured policy, not proof of a particular kernel.

The renderer runs in a separate process, consumes existing telemetry callbacks and needs no network port or model access. Nonblocking snapshots refresh around four times per second. Closing/failing the renderer does not stop training. Ctrl+C belongs to the trainer; there are no pause/delete or q controls. Normal shutdown restores screen/cursor; failure restoration is best effort if the terminal cannot accept bytes. Raw events remain persisted independently. A narrow screen selects compact panels. A typical full layout is 120×36.

Dataset curation also emits progress through the same EventBus and renderer. Record counts are real; unknown totals do not receive invented percentages or ETAs. Loading duration measures host batch assembly/transfer-call wall time, not isolated asynchronous GPU transfer latency.

## Browser viewer and telemetry

`codexa dashboard --list` lists historical runs. `codexa dashboard --run ID` serves the independent read-only browser viewer at localhost:8765. It never deserializes checkpoints or controls training. SQLite indexing, incremental JSONL reads and bounded chart histories avoid repeated full-log scans. Decimation can miss rare spikes; raw records are retained. Live updates can reset chart zoom. Run discovery may require reloading the browser.

Events use schema 1 with run/session IDs, ordered sequence numbers, UTC timestamps, step/token counts and a data object. Lifecycle events cover startup, optimizer steps, validation, saves, interruption, failure and completion. Metrics use seconds, tokens/second, memory bytes, percent, Celsius and watts. Missing measurements are null. Fsync occurs periodically and at lifecycle boundaries. Observer failures are isolated. Checkpoint `atomic_save_only` records a successful atomic write, not independent reload verification. A pathological storage stall remains an existing synchronous I/O limitation.

## Context and evaluation

The default primary context remains 2048. Explicit profiles cover 4096/8192/16384/32768; linear factor-16 RoPE interpolation is experimental and opt-in. KV caches use absolute positions and bounded request capacity. Prompt plus generation budget must fit; inference never silently truncates. Cached and uncached generation parity is tested.

Use `codexa context transition --checkpoint PATH --config NEXT_CONFIG --output NEW_CHECKPOINT --reset-data-order` for explicit compatible context transitions. Weights/optimizer/scheduler/RNG/counts are retained; sequence-dependent data order is deliberately reset. Arbitrary schedule/optimizer changes are rejected. Review the new dataset format before context transitions; legacy transition behavior remains tested.

Use `codexa evaluate --checkpoint PATH --data DATA --split val|test --max-batches N` or `--export DIR`. Default evaluation is validation; test data must not become a tuning target. Token-weighted cross-entropy/perplexity identify evaluated tokens and dataset fingerprints. Arithmetic evaluation reports supplied integer predictions by grade/topic; it does not establish language or mathematical competence.

Architectural support, successful execution, memory feasibility, actual training and demonstrated long-context reasoning are different claims. Historical 32K inference succeeded; resident-optimizer 32K training ran out of its explicit VRAM budget even with activation checkpointing. See benchmarks for exact scope. No long-context reasoning or trained base model is claimed.
