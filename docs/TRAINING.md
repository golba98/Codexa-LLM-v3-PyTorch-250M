# Training and evaluation

**Wikimedia passes the authorized general-English quality reevaluation:** two independent full reviews yield a conservative 94/100 usable documents and 293/300 usable windows, with one severe document, three severe windows and zero critical findings. Thresholds are unchanged. The original reviews and all changed judgments remain preserved. See [complete results and criteria](DATASETS.md#approved-general-english-rubric-wikimedia-pass). No training is authorized.

## Direct-streaming continuation readiness

The current requested budget is **1,000,000,000 additional native-token targets** from the preserved 500,006,912-token model, for 1,500,006,912 cumulative targets. Training is not authorized. The CLI now integrates pinned provider streams with the existing trainer, optimizer, tokenizer and native v1 checkpoint path. It enforces source quality/rights/exclusions, conservative capacity, compute evidence and a separate explicit budget/launch decision. See [source decisions](DATASETS.md#direct-language-streaming-and-source-decisions-2026-11-10).

Configure a reviewed source with its exact dataset revision, shard paths and transport ETags; tokenizer SHA; seven admission-policy source hashes; bound historical/held-out/protected/review indexes; and a canonical training-config identity. Metadata-only online duplicate and attribution state is checkpointed; no static token dataset is prepared. Validation/test groups are independent and deterministic. Source approval does not follow from a development callback, test fixture or merely setting a flag.

Use a fresh output directory for each bounded production-path check, without optimizer updates:

```sh
CODEXA_NO_TRAINING=1 codexa train --config TARGET_CONFIG --stream-config SOURCE_CONFIG \
  --tokenizer PRESERVED_TOKENIZER --parent-checkpoint PRESERVED_CHECKPOINT \
  --device cuda --forward-only --max-bytes 67108864 --stream-check-windows 128 \
  --output FRESH_CHECK_DIRECTORY
```

`--max-bytes` is mandatory for forward-only provider tests and capped at 64 MiB. Use `--resume FRESH_CHECK_DIRECTORY/stream-check.pt` in another fresh output directory to verify the next provider batch. These checkpoints have no continuation phase and are not launch bundles. First-connection latency, subsequent loader throughput, fixed validation, model forward and optimizer updates are reported separately. Tests exercise interrupted reads and transactional rollback without updates. Exact late-shard recovery can require replaying a gzip prefix; this is a known network cost, not a hidden token archive.

After every gate passes and explicit launch authorization is recorded, the separate `codexa continuation stream-prepare` command can prepare a fresh native v1 phase while retaining parent weights, Adam moments, tokenizer and RNG. It requires `--additional-tokens 1000000000`, `--parent-sha256`, a source/configuration-bound authorization, a fresh root and the expected mounted volume. It changes only the explicitly proposed phase schedule and sampler. It does not train. The source configuration is now approved for internal use, but a launch-ready native continuation bundle must wait for the separate explicit training authorization; none was created by this reevaluation.

Read-only compute checks restored all 170 Adam parameter states at step 15,259 with finite correctly shaped moments and nonnegative variances. A controlled synthetic English readiness check exercised sequence 2,048, microbatch 1, accumulation 16, BF16 autocast/FP32 loss, with the original gradient-checkpointing setting. Sixteen backward calls completed, with finite gradients, unchanged parameter version counters and unchanged Adam steps. Peak CUDA allocated was 7,567,977,984 bytes; reserved 8,166,309,888 bytes on the 16 GiB RTX 4080. **Optimizer updates: zero.** This establishes tested activation/gradient residency and schema compatibility; it does not measure Adam-update performance or training throughput. Evidence: `.agents/scratch/wikimedia-review-v1/cuda-accumulation-readiness.json`.

The actual production CLI read 128 training windows with the complete pinned admission policy, then resumed its native stream checkpoint for 2,048 more windows. The first new batch SHA256 exactly matched the pre-resume next-batch SHA256; fixed validation loss was identical (3.200244352221489). The resumed loading interval emitted 4,192,256 target positions in 176.6134 seconds: **23,736.9 positions/s** after the first batch, which was already buffered. This 176.6-second interval includes provider transfer, on-demand tokenization and production exclusions in one shard; it does not measure full-run or GPU training throughput. The initial check's first loader batch took 4.6624 seconds, followed by 260,096 targets in 5.0138 seconds; that shorter cached/buffered interval is not the sustained claim. Application payload reads were 17,956,924 and 29,753,434 bytes respectively, each below its 64 MiB cap. Evidence: `.agents/scratch/wikimedia-review-v1/live-resume.json`.

The first trial checkpoint's complete model tensors, Adam state and tokenizer were compared against the parent and were bitwise identical. It has 128 loader windows but zero training targets/updates and no continuation phase. Its 454,656-byte metadata ledger represented 167 admitted documents, 2,672 bands and 5,821 anchors; a naive linear projection is 1.73 GB at 1B targets, not a measured scale guarantee. A second live resume at 2,176 loader windows also matched its next-batch hash and identical validation loss; replay to the saved gzip cursor took 42.1368 seconds. The original parent/tokenizer hashes still match trusted receipts. Bounded sample checks and source approval must remain distinct.

The earlier forward-only Comma check measured finite BF16 logits and 1,071,160,832 allocated/1,180,696,576 reserved bytes; two post-warmup forwards took 0.02632/0.02628 seconds. It was inference-only and did not establish optimizer or training readiness. The complete Wikimedia panel now passes under the explicitly authorized general-English rubric. The pinned local source configuration is `.agents/scratch/wikimedia-approved-reevaluation-v1/approved-source-config.json`; its launch gate validates quality, rights, capacity and compute, then fails closed at the intentionally pending explicit 1B launch authorization. The revised bounded eligible-token estimate is 5.059B, with an approximate conditional 95% lower bound of 3.606B; see the dataset guide for its assumptions. The prior live throughput/resume evidence predates the additional explicit-navigation exclusion; local controlled tests verify that predicate and the current policy. No new provider benchmark is claimed. The complete forward-only verification passed 387 tests, Ruff check/format and mypy. Preserve all original checkpoints and all Grade 1–12 mathematics assets. The sections below retain historical 50M/100M plans and outcomes, not current launch instructions.

## Historical language-first milestone

Preserve the 500,006,912-target native v1 checkpoint, tokenizer, RNG, all 170 AdamW parameter states and historical evaluations. The next experiment is conditional on a passing explicitly authorized language protocol and all mechanical readiness checks. The TinyStories-only `codexa-language-quality-4` source binding and single exact-100M budget now have explicit user authorization; implementation alone never grants permission. This supersedes the historical 50M slot. The final optimizer update uses the existing tested partial-token budget path.

Reuse the existing phase-conversion policy: preserve weights and optimizer moments/counters, start a recorded new-data sampler and fresh warmup/cosine phase schedule (peak LR 1e-4, floor 1e-5, 100 warmup updates, context 2,048, accumulation 16, BF16). This is continued pretraining with a new schedule, not a replay of the completed parent schedule. Preserve strict data/tokenizer/source identities and exact within-phase CPU recovery.

Version 3 binds either DCLM-Edu or the filtered Pressbooks fallback to the same single continuation milestone; it does not authorize two runs. Source preparation is isolated under `language-continuation-20261110`, while the original baseline, evaluation budget and continuation launch root remain under `language-first-20261110`. Native corpus targets must support the actual reserved sampler windows after every exclusion; approximate acquisition yield is insufficient. The internal research rights scope does not authorize public weight or dataset release. Conversational preparation remains conditional on the existing base-language gate, and the user selected a separate concrete budget decision before SFT training.

The preserved extended language baseline includes protected reading comprehension, instruction following, explanations, summarization and conversational continuity: 100 development and 100 test items, references and rubrics, with actual greedy and seeded sampled completions through 256 tokens. Reuse all 400 outputs, stopping outcomes, settings and repetition measurements. Preserve old grammar and paragraph evidence, including severe sampled loops in 63/100 development and 70/100 test paragraphs. Existing likelihood scores do not establish coherent generation.

Use identical prompts/settings for paired evaluation, independent checkpoint-blinded scoring and separate grammaticality, coherence, relevance, completeness and task-success judgments. Preserve raw outputs and disclose truncation, disagreement and incomplete coverage. Serialize GPU use through the shared lease. The paired-evaluation budget is four hours; historical ledgers remain unchanged. Training readiness additionally requires source checks, rights, fresh approval, independent full token/split/protection verification and disposable full-shape CUDA smoke. A 50M milestone is not a fluency certificate; further token budgets require measured evidence and separate authorization.

The version-2 replacement-source implementation records authorization under `/mnt/codexa/pytorch/codexa-v3/language-recovery-20261110/authorizations`. Corpus preparation uses that new root; continuation bundles and baseline evaluation remain under the original language-first root, preserving the shared evaluation ledger. The bounded Common Corpus comparison does not establish an admissible production corpus, so no real-model update, CUDA smoke or successor checkpoint is claimed.

`python -m scripts.analyze_language_baseline --root ROOT --output NEW_JSON` summarizes the preserved 400 generations and existing independent ratings without inference or new scoring. The recorded analysis finds 276 exact-loop flags and 399 token-limit stops; among 100 reviewed responses, grammar is fully satisfied in seven and partly satisfied in 87, while task success is partly satisfied in 14. All 100 fail whole-response coherence. The report preserves full predetermined examples, both rationales and the limitation that the panel shares only 20 contexts. Comprehension, answer selection and generation failures are not causally isolated by these outputs.

The authorized future base-language gate requires 100 fresh protected development prose completions, at least 80 grammatical/coherent/complete responses and at most ten severe loops. It is separate from corpus approval and cannot be scored from the existing assistant-task panel. Conversational SFT remains deferred until this gate passes; no new assistant objective or special token IDs have been introduced. Later SFT must use original-rights-reviewed dialogue subsets, explicit role formatting and assistant-only loss masking, with a concrete pilot budget before training.

At the historical 23,055 tokens/s end-to-end rate, 50M/1B/2B/5B additional tokens would take approximately 0.60/12.05/24.10/60.25 hours, before new preparation and evaluation. These are estimates, not new throughput measurements. Choose any larger experiment only after paired held-out loss and actual generation evidence; none of the billion-token budgets is authorized.


The [full Codexa v3 diagnostic report](CODEXA_V3_DIAGNOSTIC_REPORT.md) investigates the completed 500M-token checkpoint using code inspection, independent numerical references, corpus/log analysis and frozen inference. It distinguishes local grammar and restricted factual discrimination from failed free-running coherence, documents the tested implementation paths and limitations, and defines proposed base-language gates before conversational fine-tuning. Supporting artifacts are retained at `/mnt/codexa/pytorch/codexa-v3/diagnostics/20261104-120458`. This investigation performed no training or checkpoint modification; its future-phase proposal is not an active run or training authorization.

Use the executable quick start in the root README. Existing corpus commands and uint32 smoke datasets remain supported. Production preparation yields a versioned sharded uint16 manifest; `--data` detects the format. Model configuration and loss are unchanged.

`python scripts/verify.py --output /ABSOLUTE/PATH/verification.json` directs the verification report to a chosen storage volume. Without `--output`, the existing `outputs/logs/verification.json` destination remains the default. Test temporary directories and tool caches can be redirected with `TMPDIR`, `PYTEST_ADDOPTS`, `RUFF_CACHE_DIR` and `MYPY_CACHE_DIR`.

The unchanged 500M checkpoint additionally measured loss **2.648682**, perplexity **14.135390**, over 999,424 complete validation targets from the new native narrative candidate. CUDA BF16 forward/FP32 CE took 16.912 seconds; weights remained unchanged with zero backward calls, optimizer updates or forbidden attempts. This source-specific baseline is recorded under `language-first-20261110/benchmarks/narrative-parent-loss-64e80dc835685c6a`; it is not evidence of improvement or conversational ability. The shared four-hour evaluation ledger was charged.

### Recorded language-first baseline — 2026-11-10

The unchanged 500M checkpoint completed all 400 fixed language generations in 857.626 seconds. The exact tandem-loop detector flagged 276/400 responses; 399/400 reached the 256-token limit. Peak inference CUDA allocation was 1,073,169,408 bytes and reservation 1,153,433,600 bytes; these are inference measurements, not training memory or throughput. Numerical loop detection can miss semantic repetition, and a token limit alone does not establish incoherence. Raw outputs, fixed non-cherry-picked examples and verification are in `benchmarks/language-baseline-54fca007f272fb37` under the language preparation root. Model/tokenizer hashes are unchanged and backward/optimizer/forbidden-call counts are zero.

The preregistered independent qualitative panel contains all 100 test cases under sampled seed-42 decoding. It has 20 correlated scenarios across five tasks. Independent ratings are saved separately under `benchmarks/language-baseline-review`; the launch gate requires both complete rating files and recomputes the bound combined scores. Poor baseline scores do not block improved-data pretraining, but missing baseline evidence does. Subsequent comparison replays historical 256-token paragraph prompts as well as the new tasks, with all work charged to the shared four-hour budget and no required mathematics generation.

### Protected language evaluation commands

```sh
.venv/bin/codexa language v1 evaluate \
  --root /mnt/codexa/pytorch/codexa-v3/language-first-20261110 \
  --fixtures /mnt/codexa/pytorch/codexa-v3/language-first-20261110/benchmarks/language-v1 \
  --checkpoint /mnt/codexa/pytorch/codexa-v3/pretraining/20261103-500m-233201/final/checkpoint.pt \
  --output /mnt/codexa/pytorch/codexa-v3/language-first-20261110/benchmarks/language-baseline-54fca007f272fb37 \
  --lease-root /mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814 \
  --seconds 5400
```

The command resumes only compatible, hash-bound saved records; it preserves the persistent combined evaluation budget. Use `language v1 packet --split test --setting sample` for the preregistered 100-case baseline review; preserve two independent files for `language v1 score`. The 100 task cases share 20 scenarios, so report correlated context groups rather than 100 independent subjects. Development and greedy outputs remain available outside this bounded qualitative panel. A paired post-run packet can include both checkpoints; independent scores remain distinct from automatic repetition measurements.


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

Architectural support, successful execution, memory feasibility, actual training and demonstrated long-context reasoning are different claims. Historical 32K inference succeeded; resident-optimizer 32K training ran out of its explicit VRAM budget even with activation checkpointing. See benchmarks for exact scope. These context probes do not establish trained long-context reasoning.

## External checkpoint volumes and deterministic pilots

`codexa train --checkpoint-volume-uuid UUID` checks the output filesystem before creating the run and before every checkpoint write. It fails if the mounted filesystem UUID differs or available space falls below 20 GiB. Atomic checkpoint temporaries use the checkpoint directory itself. The option is Linux-specific and opt-in; native checkpoint version 1 and legacy defaults are unchanged. Place the entire `--output` directory on the intended volume. For this workstation the approved volume is `/mnt/codexa`, filesystem UUID `279352d0-0011-4733-8348-6815a4308196`; new v3 work uses its `pytorch/codexa-v3/pretraining` subtree. Never create that path as a fallback when the volume is absent.

`--deterministic` enables PyTorch deterministic algorithms for a bounded recovery comparison. Set `CUBLAS_WORKSPACE_CONFIG=:4096:8` before launching CUDA processes. Unsupported deterministic operations fail explicitly. This does not establish cross-hardware or cross-version reproducibility. Terminal acceptance accepts `--timeout SECONDS`, `--checkpoint-volume-uuid UUID` and `--deterministic`; its default timeout remains 60 seconds.

### Measured real-data pilot

A fresh 248,687,488-parameter CUDA/BF16 pilot completed 3,031 updates and 99,319,808 target tokens at sequence 2,048, microbatch 1 and accumulation 16. An additional 20-update reference replay brought aggregate GPU training to 99,975,168 tokens, below the separately authorized 100M ceiling. Pilot-only overrides were 200 warmup updates, validation every 100 updates on 32 fixed batches, and checkpointing every 250 updates; primary defaults and 32K configurations are unchanged. All generated artifacts and checkpoint temporaries reside on the dedicated 512 GB volume.

A full-model 20-update reference and a 10+10 stop/resume matched model, optimizer, scheduler, sampler, counters and CPU/CUDA RNG state exactly. The actual Rich PTY run recorded validation/checkpoint events and restored the alternate screen, cursor and terminal settings. Training losses and gradients remained finite. Initial training loss was 9.88408; the final 60-update mean was 3.72844. Fixed-slice validation loss fell from 8.71679 at update 10 to 3.59504 at update 3,031, evaluating 65,536 tokens per event. This is a bounded validation slice, not a complete validation-set or external benchmark score; test text was not used for model evaluation.

Sustained optimizer throughput after update 200 was 23,264.95 tokens/s. Consecutive windows measured 22,525.11 / 23,485.32 / 23,652.51 tokens/s. Observed pilot wall time was 4,489.22 seconds, including its stop/resume, validation and checkpoint overhead, yielding 22,124.09 tokens/s. Peak allocated CUDA memory was 7.0953 GiB and maximum observed reserved memory 7.6289 GiB. Physical GPU memory includes unrelated desktop graphics and is a separate scope.

At the same pilot settings and cadence, estimates for 100M / 1B / 5B tokens are 1.26 / 12.56 / 62.78 hours including observed overhead, versus 1.19 / 11.94 / 59.70 hours of optimizer time. These are extrapolations from approximately 75 minutes, not multi-day measurements or estimates for 32K training. Different checkpoint/validation cadence, desktop load, thermals and data loading change the result. In particular, the primary configuration's more frequent saves are not covered by the pilot-cadence estimate.

The final native v1 checkpoint is 2,985,607,403 bytes. CPU reload verified finite model/optimizer tensors, parameter count, counters, tokenizer/data identities and a byte-identical immutable copy (SHA256 `e3164d9ddf6d855833e48e920f4cbc06c89560c0f8c838d4e814cd3497a18706`). The pilot schedule is complete; extending it requires a separately reviewed continuation rather than silently changing exact-resume configuration.

### Completed 500M-token run

On 2026-11-04 UTC, a separately authorized fresh 248,687,488-parameter run completed on the RTX 4080: 15,259 updates and 500,006,912 target tokens after rounding up to complete 32,768-token updates. The completed pilot's schedule and data fingerprint differ from this run, so exact pilot resumption was rejected. Its checkpoint is preserved; the new model started from random initialization with the same frozen tokenizer.

Run settings are sequence 2,048, microbatch 1, accumulation 16, BF16 autocast, seed 42, AdamW learning rate 0.0003, minimum learning rate 0.00003, 2,000 warmup updates, validation every 100 updates on 32 batches, and checkpoints every 250 updates. Architecture, objective, primary configuration defaults and native checkpoint version 1 are unchanged.

The workstation stores the entire run under `/mnt/codexa/pytorch/codexa-v3/pretraining/20261103-500m-233201`: datasets, tokenizer copy, caches, temporary files, checkpoints and logs. The volume is the encrypted ext4 filesystem on the HS-SSD-E100 512G. Run-local environment settings redirect Hugging Face, Python, PyTorch, CUDA, Triton, package and verification caches there. During training, the live Rich interface ran in Kitty attached to the persistent tmux session `codexa-v3-500m-20261103-500m-233201`; `NO_COLOR` was cleared for the explicitly requested colors. That session now shows the verified completion results.

The first ten updates belong to the actual token target. Their 2,985,607,339-byte checkpoint was independently reloaded on CPU and verified for finite model/optimizer tensors, architecture shapes, tied weights, configuration, fingerprints, optimizer/scheduler/sampler counters and token counts. SHA256: `c5499001be49944916cbb843c33c1f6612d71dbc41d86ea4575b732255df2f0d`. The same run then resumed and completed further CUDA updates with observed parameter changes. Initial optimizer throughput was approximately 23.4k tokens/s; startup observations are recorded separately from the sustained measurements below.

The run-local supervisor retains the current checkpoint and two previous verified snapshots. Recovery verifies saved SHA256 receipts and the unchanged training source, configuration, tokenizer and dataset. Fatal integrity/nonfinite/storage failures stop the worker; transient worker failures have at most three delayed retries. Checkpoint writes require the expected volume identity and a 20 GiB free-space reserve. Ordinary CLI exact-resume checks remain enforced.

On this workstation, reattach with `sh /mnt/codexa/pytorch/codexa-v3/pretraining/20261103-500m-233201/control/reattach.sh`. The run has completed and no training process remains. The recovery helper refuses a completed run; further training requires separate authorization and a compatible continuation plan. No export or publication was performed.

The completed log contains every update from 1 through 15,259 exactly once, 63 successful checkpoint events and 154 validation events, with no failed/interrupted updates or observer warnings. Losses and gradient norms remained finite. Final training loss was 3.140551; the final 60-update mean was 3.117484. The fixed 65,536-token validation slice fell from 9.641515 at update 10 to 3.005710 at the final update. Its best observed loss was 3.005181 at update 15,000; that snapshot is retained separately and represents 491,520,000 training tokens.

Supervisor wall time was 21,687.67 seconds (6h 1m 28s), including initial loading, the startup stop/resume, validation, checkpoint writing and audits: 23,054.90 end-to-end training tokens/s. Optimizer-only throughput after the 2,000-update warmup was 24,023.45 tokens/s. Consecutive windows measured 24,010.17 / 24,004.06 / 24,048.33 tokens/s. Maximum observed per-step allocated CUDA memory was 7.0171 GiB; maximum sampled reserved memory was 7.1582 GiB. These allocator measurements exclude unrelated desktop graphics and driver scope.

Independent post-run checks reloaded the final native v1 checkpoint on CPU, verified all model and optimizer tensors, tied weights, optimizer shapes/variances, counters, sampler, scheduler, RNG schema, embedded tokenizer, unchanged training source and all data shards. The 2,985,607,403-byte final checkpoint has SHA256 `54fca007f272fb3784c759d18db44912091227aedc233b62c9731514b1835c73` and is preserved at `final/checkpoint.pt` inside the run directory. Project verification passed 80 tests with one CUDA unit test skipped, plus Ruff check/format and mypy.

A separate CUDA/BF16 forward-only reload restored the saved CPU/CUDA RNG state exactly and reproduced the logged final validation-slice loss with zero difference. Evaluation of all complete validation windows covered 606,208 tokens: loss 3.048003, perplexity 21.073212. Evaluation left model weights unchanged and made zero optimizer updates. The test split was not model-evaluated. These checks establish checkpoint health and frozen-corpus loss; broader language, mathematics and long-context competence remain unbenchmarked. Machine-readable results are retained under the run's `checks/postrun-20261104-073346` and `final/postrun-report.json`.

### Conversation assessment after 500M tokens

**Assessment: conversational readiness is weak in the observed tests.** The model can produce English-like sentences, but repeated text, off-topic answers, factual errors and failed instructions prevent treating this checkpoint as a reliable chat assistant. This conclusion comes from a small qualitative probe on 2026-11-04, not a standardized benchmark or an estimated accuracy percentage.

The assessed checkpoint is the completed 248,687,488-parameter model at 500,006,912 training tokens, SHA256 `54fca007f272fb3784c759d18db44912091227aedc233b62c9731514b1835c73`. It received causal text pretraining on the educational-web/mathematics corpus. No conversation/instruction fine-tuning was performed. The successful checkpoint audit and validation perplexity of 21.073212 establish separate evidence about loading, optimization and frozen-corpus prediction loss; they do not measure dialogue quality.

#### Protocol

- Use the unchanged final checkpoint and frozen tokenizer, a 2,048-token context and CUDA/BF16 generation with a KV cache. No training, optimizer updates or checkpoint modifications are made.
- Test six conversation scenarios: greeting, a simple factual question, an exact two-word instruction, a short explanation, remembering personal details, and following up on activity suggestions. The last two scenarios each have two turns. Actual generated replies are included in the subsequent history; no successful assistant answers are supplied as examples.
- Format dialogue as plain `User: ...\nAssistant:` text. This is an evaluation convention, not a trained chat template or a special-token protocol.
- Run greedy decoding and one sampled configuration: temperature 0.7, top-k 40, top-p 0.9, seed 42. Each assistant reply has a 96-token generation cap.
- Add two plain text-completion controls, `The capital of France is` and `Ice melts when`, with a 64-token cap. Together, the two decoding configurations produce 20 continuations: 16 dialogue replies and four completion controls. All reached their generation cap without EOS.

#### Observed results

| Probe | Greedy decoding | Sampled decoding |
| --- | --- | --- |
| Greeting: “Hi! How are you?” | Begins with “Hi!” and loops on “I am a newbie, so I am a newbie.” | Produces mailbox/article metadata and a description of an unrelated website. |
| Capital of France, requested in one short sentence | Repeats the question and `Assistant:` label without answering. | Produces unrelated, repetitive text about the US presidency. |
| “Reply with exactly these two words: blue moon” | Repeats a modified version of the instruction with reply labels. | Produces unrelated NASA/Jet Propulsion Laboratory metadata and repetition. |
| Explain melting ice in two simple sentences | Repeats the instruction instead of explaining. | Produces writing-task instructions and repeated questions. |
| Remember “Lindi” and “cycling,” then retrieve both | Repeats the name but does not provide the requested name-and-hobby answer. | Switches to unrelated company ownership and cost descriptions. |
| Suggest two indoor activities, then compare equipment needs | Repeats “I can do it outdoors.” across both turns. | Mixes indoor/outdoor advice and repeats the earlier response instead of answering the comparison. |

The completion controls also exposed weaknesses outside the dialogue format. For `The capital of France is`, greedy decoding continued with “the city of Bologna”; sampling continued with “the city of Lucca.” Both answers are incorrect. For `Ice melts when`, greedy decoding repeated a sentence about the sun shining; sampling made incorrect claims about melting as water temperature drops below −50°F. These examples are observed model errors, not accepted factual statements.

#### Cache and precision control

A separate control generated 32 tokens for the greeting and capital-completion prompts, comparing cached and uncached decoding in FP32 and BF16. FP32 cached/uncached token sequences matched for both prompts and retained the same repetition and incorrect capital answer. BF16 sequences matched for the greeting but diverged later in the capital continuation. Both BF16 paths still began with the incorrect Bologna answer.

These observations show that the principal failures also occur with uncached FP32 inference. The BF16 difference is recorded without asserting its cause or claiming universal cache parity across precisions, prompts or sequence lengths.

#### Interpretation and limits

The observed failures span turn-taking, instruction-following, use of supplied context and basic factual accuracy. Changing to the tested sampled configuration did not make the responses reliable. Fluent fragments and recognizable vocabulary are insufficient evidence of useful conversation.

No dialogue/instruction fine-tuning has been assessed, so its eventual effect is unknown. The poor raw text completions also mean the failures should not be attributed solely to the absence of a chat template or fine-tuning. Further evaluation should measure base factuality and repetition alongside conversational behavior.

This probe uses a small, manually selected set of English prompts, one dialogue format and one sampling seed. Prompt overlap with the training corpus was not audited; these examples are not claimed to be unseen holdouts. No independent human rating, broad factuality benchmark, safety assessment, mathematics benchmark or long-conversation evaluation was performed. Generation caps can truncate replies. No numerical conversation score, reliability percentage or Grade 1–12 competence claim is supported. A future assistant-readiness decision requires broader, independently held-out evidence; this report does not authorize further training.

#### Evidence retained on the training drive

Artifacts are stored under `/mnt/codexa/pytorch/codexa-v3/pretraining/20261103-500m-233201/checks/conversation-20261104-094424`:

- `probe.py` and `outputs.json`: exact greedy prompts, generated continuations, token counts, EOS status and timings.
- `probe_sampled.py` and `sampled-outputs.json`: the corresponding sampled run, including its decoding settings.
- `cache_sanity.py` and `cache-sanity.json`: cached/uncached FP32/BF16 comparisons.
- `assessment.json`: the recorded qualitative assessment and its scope.

Raw outputs remain outside Git on the approved drive. The reviewed report is maintained here in the canonical training guide.

## Explicit continuation preparation

The initial conditional 50M continuation launch was blocked on 2026-11-05 by the blinded corpus-quality gate: three severe FineMath document defects exceeded the maximum of two in its fixed 40-document sample. No training, backward pass or optimizer update was performed in that initial gate-check phase. See [the historical gate decision and examples](CODEXA_V3_TRAINING_READINESS.md#historical-initial-continuation-gate-failed-on-2026-11-05). Subsequent systematic remediation produced another independently verified snapshot, but the complete fresh version-3 review still fails FineMath and FineWeb-Edu thresholds. The historical 500M checkpoint and both untrained continuation bundles remain preserved; no real-model continuation was launched.

See [training readiness](CODEXA_V3_TRAINING_READINESS.md) for the read-only baseline and separately callable `codexa continuation prepare/verify` operations. They create a fresh native-v1 phase bundle with identical model tensors, cumulative parent-token accounting, an explicit preserve/reset optimizer policy and a new schedule. An audited new dataset must record exclusion of the parent dataset. Native exact-resume fingerprint checks remain strict. Optional `phase` metadata persists parent/phase counters; phase checkpoint writes enforce the dedicated UUID and 40 GiB floor. Preparation sets `CODEXA_NO_TRAINING=1`; no training launcher is part of this workflow.

The selected dataset-v2 snapshot now contains 2,108,050,326 new training positions, with final manifest SHA256 `5cc849cc50680c86a50c40fe10aa6f56d0b0cfc77e1dcab5feb7ddba8c5dfddf`. On its 10,598,400 complete validation targets, the unchanged parent checkpoint measured forward-only loss 3.036705 and perplexity 20.836476. The on-disk 50M-token health-check continuation bundle passed parent/optimizer/weight/RNG/fingerprint and finite CPU-forward verification, with zero backward calls or optimizer updates. It is not a trained successor model. The 5B corpus expansion is deferred.


### Authorized exact-budget continuation gate

The remediation task authorizes an actual continuation only after a complete passing review and integrity/storage checks. Its explicit budget is exactly 50,000,000 new targets: 1,525 full updates plus a final 28,800-target update, ending at 550,006,912 cumulative targets. `codexa continuation prepare --additional-tokens 50000000 --output NEW_BUNDLE` preserves the original weights, AdamW state and RNG, records the completed parent scheduler, and creates the explicit new phase schedule. Native exact-resume fingerprints stay strict. Tiny CPU recovery tests cover the partial update; they do not touch real model weights.

The separately callable `scripts/continuation_healthcheck.py` requires the exact passing review, independent token proof and current source verification before opening tmux/GNOME Terminal and using CUDA. It saves step-two proof of actual Adam updates and changed weights, checkpoints and live logs on the dedicated volume, and queues read-only held-out/math/generation comparisons after successful completion. See [current readiness](CODEXA_V3_TRAINING_READINESS.md#remediation-status-and-exact-continuation-accounting). A pending or failed review is a hard launch blocker.

The latest candidate's 500 frozen ratings give 276/300 usable windows, FineMath 35/40 usable with three severe defects, FineWeb-Edu 32/40 usable, OpenWebMath 37/40 usable, and 40/40 usable each for TinyStories and Wikipedia. FineMath exceeds the severe limit of two and FineWeb-Edu misses the usable minimum of 34. A mechanically compatible bundle is insufficient: the launcher must refuse this candidate. No new model loss, perplexity, arithmetic accuracy, generations or training throughput can be reported without an actual permitted run.

### Strict quality gate and automatic launch

The 2026-11-06 authorization strengthens the quality requirements without changing the exact-50M objective. The launcher now requires the current `codexa-quality-4` primary panel, its additional 400-document assessment, two complete independently preserved ratings per record, matching immutable data identities, current source checks and a separately bounded full-shape CUDA smoke. Historical passing protocols cannot authorize a launch. Every affected source's full-document checks include correctness, completeness, visual dependencies and notation; readable wrong solutions cannot pass by language labels alone.

Production starts from a fresh verified bundle preserving the original 248,687,488 parameters, all 170 AdamW states and RNG. Smoke updates occur only on disposable state after quality approval. The existing explicit phase schedule and archived completed parent scheduler remain documented; a fresh phase schedule is not an exact continuation of the completed parent LR schedule. The final budget remains 550,006,912 cumulative targets.

The persistent tmux socket, run logs and checkpoints stay on `/mnt/codexa`. GNOME attachment is attempted without making GUI availability a prerequisite for retaining the session. The dashboard displays continuation tokens and lifetime updates alongside cumulative tokens, loss, throughput, validation, resources and recovery state. Paired post-run evaluation uses fresh per-checkpoint budgets with a combined initial four-hour CUDA envelope; historical evaluation ledgers remain unchanged. Incomplete results must remain explicitly incomplete.
