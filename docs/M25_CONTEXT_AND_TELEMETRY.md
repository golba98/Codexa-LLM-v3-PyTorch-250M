# M2.5 context, telemetry and dashboard

## Context contract

Primary model dimensions and count remain unchanged: 248,687,488 parameters. Existing `context` is the architectural maximum; training `sequence` remains independently validated. New inference settings include maximum sequence, token budget, cache use and precision. Original primary 2048 behavior and smoke/50M sizes are retained. Profiles in configs/context cover 2048/4096/8192/16384/32768. Unscaled RoPE remains default. The explicitly experimental linear profile divides position indices by 16 (2048 → 32768), changing positional semantics without changing weights. It is not a trained extension or a quality recommendation. See https://arxiv.org/abs/2306.15595.

RoPE uses FP32 absolute positions; no learned positional parameters or persistent trigonometric state are introduced. Cached inference uses preallocated per-request K/V arrays, FP32 by default or BF16 explicitly; capacity is bounded by prompt plus requested generation. Prefill is causal and cached chunks use lower-right causal bias. No rolling truncation remains: prompt plus budget must fit the explicit limit. This is an intentional overflow-policy change. CUDA long-sequence auto attention disables math fallback, reporting unavailable fused execution rather than allocating dense attention matrices. Activation checkpointing is opt-in, nonreentrant and RNG-preserving; training never uses inference caches.

`forward` still returns full logits by default; `last_logits_only=True` avoids unnecessary vocabulary projections for inference. `forward_cached` requires eval mode and returns logits plus KVCache. Caches become invalid on failed updates. Generation supports greedy/top-k/top-p/temperature/EOS and bounded cache capacity. Explicit `--precision bf16` or `auto` permits supported CUDA mixed precision. Cache precision changes are rejected. Arbitrary cross-model cache reuse is not supported.

Old native checkpoint v1 configuration is normalized through current defaults on loading. State keys and tied parameters remain unchanged. Context transitions are explicit: `codexa context transition --checkpoint PATH --config configs/context/250m-4096.yaml --output outputs/checkpoints/stage4k.pt --reset-data-order`. It preserves weights, optimizer, compatible scheduler and RNG state, global steps/tokens and recorded microsteps. Sequence-dependent sampler order is reset and recorded. Schedule/optimizer hyperparameter changes are rejected; starting a differently scheduled run requires separate reviewed initialization. Never overwrite an existing transition output.

## Observer API and persistence

`Trainer(..., callbacks=(observer,))` accepts objects implementing `on_event(TrainingEvent)`. Observers receive independent data snapshots; they must be lightweight. Exceptions disable a failing observer with a warning, without swallowing core training failures. Checkpointing and evaluation remain trainer operations. Closing a viewer never controls the trainer.

A run stores `run.json`, `summary.json`, and append-only `events.jsonl` alongside compatible legacy metrics. Records have schema_version 1, stable run ID, session UUID, session-monotonic seq, unique event ID, event name, global step, consumed target tokens, UTC timestamp, elapsed seconds and data. Step completion is deduplicated within a session. A new `started` record establishes a resume boundary; effective charts hide superseded later records while raw bytes remain unchanged. A torn tail is preserved and subsequent writing begins on a new line. Readers count corruption. A process lock prevents competing writers.

Events: started, step_completed, validation_started/completed, checkpoint_started/saved, context_changed, paused, interrupted, failed, completed. An intentional --until stop is paused; completion requires the configured terminal step. Metrics use loss in nats/token, durations in seconds, rates in tokens/second, learning rates as scalars, gradient norm before clipping, memory bytes, temperature Celsius, utilization percent and power watts. GPU system samples identify physical GPU 0 and their timestamps; they are optional, distinct from PyTorch process allocation, and unavailable when probing fails. Missing metrics are null, not zero. Fsync occurs every 20 records and at save/terminal boundaries. Local storage writes are synchronous; storage stalls and arbitrary slow custom observers cannot be given a hard real-time guarantee. Persistence exceptions degrade monitoring visibly.

Checkpoint status `atomic_save_only` means a completed safe write, not independent reload verification. Validation records identify actual evaluated tokens; no mathematics scores are fabricated.

## Dashboard architecture and operation

```sh
codexa dashboard --list
codexa dashboard --run RUN_ID
codexa dashboard --root outputs/experiments --port 8765
sh scripts/smoke.sh
codexa context benchmark --mode both --output outputs/benchmarks/m25-context.json
codexa generate --checkpoint PATH --prompt 'Test' --max-tokens 16 --precision bf16
```

The localhost-only standard-library server serves bundled assets and SSE. The dashboard path avoids importing PyTorch. It never deserializes checkpoints or accepts mutation controls. Run IDs map to discovery under the explicitly configured root. Static assets are allowlisted; host/origin checks, traversal rejection, client limits and socket timeouts bound exposure. A derived SQLite index and cursors live under ignored outputs; appended data is read in 1MiB batches. Derived cursors track file identity and reset when logs are replaced or truncated. Raw telemetry is never modified. Completed indexed views are cached; large first-time histories show indexing progress. Legacy metrics are supported with unknown metadata and uncertain lineage rather than inferred checkpoint facts.

Charts use locally bundled uPlot 1.6.32 with no CDN or analytics. Loss and validation remain separate; sparse validation is shown as points. Charts can zoom; pairwise comparison shows configuration differences without implying capability deltas. Histories are decimated for display; rare spikes between sampled metric records can be missed, while raw records remain available. Non-step events are retained separately up to a display bound. Current live updates rebuild charts, so zoom is best used on archived runs. These are documented UI limitations, not modifications of the evidence.

Empty, failed, paused, interrupted, completed and stale displays use recorded events. There is no training pause/stop control. Current context evidence reflects explicit primary-model benchmark records, not proof that the viewed smoke model was trained at those lengths. Mathematics remains inactive without actual results.

Optional development hooks: `uv tool run --from pre-commit==4.3.0 pre-commit install`. Hooks run Ruff only; no GPU work, training or dataset preparation.

History-load benchmark: `.venv/bin/python scripts/benchmark_dashboard.py --records 1000000 --output outputs/dashboard-performance-NEW` uses synthetic records clearly separated from real runs. See the curated timing report under docs/benchmarks.

## Research interpretation

Architectural support, successful execution, memory feasibility, real training, and demonstrated retrieval/reasoning are distinct. Synthetic benchmark optimizer updates are not base pretraining. Long-context capability evaluation remains future work: frozen long-document loss, position-dependent retrieval, distractor/generalization sweeps and short-context regression after continued training. Next major milestone remains bounded-memory dataset/tokenizer preparation.
