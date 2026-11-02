# M2.5 implementation report

## Architecture and compatibility
The primary model remains exactly **248,687,488** parameters. Context profiles cover 2048/4096/8192/16384/32768 using the original architecture. New profiles maintain 32768 effective target tokens/update by reducing accumulation as sequence length increases; original configs are unchanged. Unscaled RoPE and FP32 inference remain defaults; linear factor-16 interpolation is explicitly experimental. Absolute offsets, causal KV caches, last-position projection and opt-in activation checkpointing are implemented. Generation rejects over-budget requests instead of silently truncating.

The unchanged fixed-seed CPU forward SHA256 is be43fd6ccdff542397b182c8c81ea359ad99d21d04ed19bb2a23b20e85147f6d. A preserved original native v1 checkpoint restored step 200 and 25600 tokens successfully into a fresh temporary trainer; its original file was not modified. Exact resumed training remains tested. Default-aware config normalization supports missing newly added fields. State-dict parameter names and tied weights are unchanged. SafeTensors roundtrip and release tests remain in the suite.

## References, dashboard and telemetry
V1/V2 were inspected read-only; details and reuse/licensing decisions are in M25_REFERENCE_AUDIT.md. The dashboard uses real v3 metrics, a local standard-library server/SSE, derived SQLite history, bundled MIT uPlot, responsive dark panels and explicit inactive curriculum state. It never loads checkpoints or controls training. Dashboard/list commands avoid importing PyTorch.

Events include independent callback snapshots, run/session identity, ordered records, timestamps, measured step/validation/checkpoint fields, resume and context-transition events. Observer exceptions are isolated. Raw history remains append-only; corruption is counted, duplicate event identities suppressed, rollback branches hidden in the effective view. Atomic summaries, writer locks and derived cursors support interruption/restart. Optional GPU system sampling is bounded and separately timestamped. No missing measurements are fabricated.

Browser verification used the user's local Chrome connection: real 200-step smoke history, five rendered chart panels, desktop, 768px tablet and 390px mobile layouts; no narrow-layout DOM overflow was observed. Screenshots: ../images/m25-dashboard-desktop.jpg and ../images/m25-dashboard-mobile.jpg. The viewer is read-only at http://127.0.0.1:8765. Chart comparison is pairwise, not a four-run overlay. Decimation may miss rare metric spikes; raw records are retained. Live chart rebuilds can reset zoom; archived zoom remains useful. Those limitations are documented rather than presented as complete range-query functionality.

## Bounded GPU measurements
RTX 4080 16GB, PyTorch 2.11.0+cu130, BF16 autocast, batch one, accumulation one. Allocator budgets were approximately 79–82% of physical VRAM, leaving desktop headroom. OOM means failure within that explicit budget, not a proof that every possible 16GB setup fails. All models were randomly initialized for the benchmark and discarded. No research pretraining occurred.

| Context | Prefill ms | Decode ms/token | Inference peak GiB | Training without checkpointing | With checkpointing (or unchanged if not retried) | Resident-state update s | Training peak GiB |
|---|---:|---:|---:|---|---|---:|---:|
| 2,048 | 194.2 | 15.43 | 1.16 | ok | ok | 0.098 | 6.12 |
| 4,096 | 210.1 | 13.58 | 1.38 | ok | ok | 0.18 | 8.96 |
| 8,192 | 295.0 | 14.25 | 1.82 | oom | ok | 0.555 | 5.26 |
| 16,384 | 525.6 | 13.83 | 2.70 | oom | ok | 1.534 | 7.28 |
| 32,768 | 1148.5 | 14.25 | 4.45 | oom | oom | OOM | 11.19 |

Inference cache at 32K measured 2,818,572,288 bytes (2.625 GiB). Full allocated/reserved peaks, initialization, cold-step times, budgets and completed-update counts are preserved in ../benchmarks/m25-context.json. Prefill/decode rates can be derived from recorded lengths/timings; these single-case timings are not sustained averages.

**32K inference succeeded. 32K resident-optimizer training failed even with activation checkpointing:** one cold update succeeded but the next ran out of budget. This additional check corrected the misleading impression a single cold update would give. 16K with checkpointing completed both updates. Recommended current training fallback is 16K with accumulation two for a 32768-token effective update. Larger VRAM or separately verified exact loss/projection memory optimization should precede 32K training. Gradient accumulation alone cannot reduce one sequence's activation memory.

## Validation and limits
Baseline: 33 tests passed. Final suite: 40 tests, covering existing release/resume behavior plus cache/chunk/sample parity, ownership/bounds, 32K positions, profile counts, activation-checkpoint gradients, observer failures, corrupt tails, HTTP safety, rollback and stage transitions. `.venv/bin/ruff check .`, `.venv/bin/ruff format --check .`, `.venv/bin/mypy codexa`, and `.venv/bin/python scripts/verify.py` are the final gates. Exact final command outcomes are also retained under ignored outputs/logs/verification.json.

A CPU smoke run completed 200 real corpus updates, stopping/resuming at 50, producing unchanged final loss 0.0420020707 and validation prefix loss 0.6915101148. Tests restart the HTTP server while a real trainer continues to step 100, then verify completion. No viewer acknowledgement or dependency exists in the training loop. An additional actual CUDA trainer run completed 20 updates, emitted optional physical-GPU measurements and safely checkpointed at its intentional pause. Linux/CPU tests and available CUDA checks ran locally; hosted CI was not pushed or run.

A fresh 1,000,000-record synthetic run measured 86.9ms first view, 6.81s remaining indexing, 71.4ms history query, 9.0ms snapshot restart and 29.54MiB process RSS/high-water. Reproduce with `.venv/bin/python scripts/benchmark_dashboard.py --records 1000000 --output outputs/dashboard-performance-NEW`. The synthetic history test records first-view/index/query/restart latency and process VmRSS/VmHWM in ../benchmarks/m25-dashboard-performance.json. Earlier inherited resource high-water counters were discarded in favor of /proc/self process measurements. First-time large histories index progressively; there is no repeated whole-log polling. The browser also rendered the million-record fixture as five bounded charts and displayed its 1,000,000-record count without horizontal overflow; sustained interaction was not exhaustively profiled. Wheel/sdist builds succeeded and a separate wheel installation loaded dashboard assets outside the source tree without importing PyTorch. Release preflight independently reloads the local random fixture; publish readiness remains false by design. A stale verification report after changing screenshots was correctly rejected, then regenerated for the final source tree.

Remaining limits: no trained/evaluated 32K language capability; no long-context retrieval benchmark or production corpus; linear scaling is experimental; no arbitrary scheduler restart on stage transition; cache use is single-request/batch-shape specific; server is for trusted local use; synchronous storage or custom callbacks can stall on a pathological filesystem or user hook. Preflight remains nonpublishing; fixture validation is not model release readiness. The next major milestone is bounded-memory tokenizer/data preparation and general-language pretraining readiness. Nothing was pushed or published; earlier Codexa workspaces and valuable assets were not changed.

## Exact changed/new file inventory

```text
.pre-commit-config.yaml
MANIFEST.in
README.md
THIRD_PARTY_NOTICES.md
codexa/architecture/attention.py
codexa/architecture/model.py
codexa/architecture/rope.py
codexa/architecture/transformer.py
codexa/cli.py
codexa/config.py
codexa/context.py
codexa/dashboard/__init__.py
codexa/dashboard/server.py
codexa/dashboard/static/UPLOT_LICENSE
codexa/dashboard/static/app.js
codexa/dashboard/static/index.html
codexa/dashboard/static/style.css
codexa/dashboard/static/uplot.css
codexa/dashboard/static/uplot.js
codexa/inference/generate.py
codexa/training/callbacks/__init__.py
codexa/training/callbacks/telemetry.py
codexa/training/trainer.py
configs/context/250m-16384.yaml
configs/context/250m-2048.yaml
configs/context/250m-32768-linear.yaml
configs/context/250m-32768.yaml
configs/context/250m-4096.yaml
configs/context/250m-8192.yaml
docs/ARCHITECTURE.md
docs/M25_CONTEXT_AND_TELEMETRY.md
docs/benchmarks/m25-context.json
docs/benchmarks/m25-dashboard-performance.json
docs/engineering/M25_REFERENCE_AUDIT.md
docs/engineering/M25_REPORT.md
docs/images/m25-dashboard-desktop.jpg
docs/images/m25-dashboard-mobile.jpg
pyproject.toml
scripts/benchmark_dashboard.py
tests/test_m25.py
tests/test_pipeline.py
```
