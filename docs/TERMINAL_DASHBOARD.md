# Terminal training dashboard

The terminal viewer adapts the **original v1 Rich training dashboard**, rather than replacing it with the browser interface or a progress-only display. It uses the same v3 `TrainingEvent` callback stream that already persists JSONL telemetry. It opens no network listener and does not import or inspect model weights.

## Install and launch

Install `uv pip install -e '.[dev,release,terminal]'`, or sync the project's CPU/CUDA lockfile and install the editable package with `--no-deps`. Rich is an optional runtime extra; the lock profiles include its tested version and PTY verification dependencies. Core training does not require Rich.

Prepare the smoke tokenizer/data as shown in the README, then use a fresh output:

```sh
codexa train --config configs/smoke.yaml --data outputs/datasets/new-smoke --tokenizer outputs/tokenizers/new-smoke/tokenizer.json --output outputs/experiments/tui-smoke --device cpu --until 50 --ui terminal
codexa train --config configs/smoke.yaml --data outputs/datasets/new-smoke --tokenizer outputs/tokenizers/new-smoke/tokenizer.json --output outputs/experiments/tui-smoke --device cpu --resume outputs/experiments/tui-smoke/latest.pt --ui terminal
```

The existing required data, tokenizer and output arguments are unchanged. Replace the smoke config only after preparing compatible data/tokenizer and reviewing the training budget. The UI never authorizes a research run.

| Option | Behavior |
|---|---|
| `--ui auto` (default) | Full terminal view when stdout is a capable TTY, Rich is available and the OS is POSIX |
| `--ui terminal` | Explicit selection; falls back with a warning if unavailable |
| `--ui none` | Headless training, ordinary validation log output, no renderer process |
| `codexa dashboard` | Separate optional browser viewer; requires its own local HTTP server |

Redirecting output and `TERM=dumb` disable the live view. `NO_COLOR` is respected by Rich; unset it for colored badges and panels. Full layout fits 120×36; medium layout begins at 86×28; smaller windows show compact training/recovery panels. Tiny windows retain key metrics; some secondary fields are omitted. Resizing is detected each refresh.

**Ctrl+C retains the trainer's existing behavior:** finish the current optimizer update, write the safe native checkpoint, persist an interrupted event, then restore the terminal. The viewer never consumes stdin or installs raw/cbreak keyboard modes. There are no q/pause/delete controls: v1's q key closed a separate viewer, whereas here Ctrl+C belongs to training. SIGTERM also uses the existing graceful pathway.

## Metric meanings

- Header: actual model parameter count, run ID, phase/status, global step / configured total steps, percentage, global target tokens and session elapsed time. `--until` pauses partway through the original schedule; it does not redefine the budget. An ETA appears only after at least three optimizer observations and uses recent wall time between events. It is an estimate, not a completion promise. Unknown total work produces no percentage or ETA.
- Optimization: raw training cross-entropy; arithmetic mean of up to 60 observed session losses; latest validation cross-entropy; best observed validation; validation perplexity `exp(loss)`; applied learning rate; pre-clipping gradient norm; clipping indication; optimizer and schedule. Sparklines show the most recent 24 losses and rates, scaled to their own ranges.
- Performance: actual target tokens/second, inverse measured optimizer duration (steps/second), optimizer duration, effective batch sequences, accumulation and sequence/context limits. Packed training has no user-facing epoch boundary, so epoch is explicitly N/A. These rates exclude validation/checkpoint/UI work; ETA uses wall time between step events.
- Resources: GPU name, current process allocated/reserved bytes, highest observed per-step CUDA allocation in this session, optional physical-GPU utilization/temperature/power, optional host CPU utilization and RAM, process RSS, precision and configured attention backend. `auto` does not certify a particular selected kernel. Host `/proc` and optional nvidia-smi samples run asynchronously every two seconds; missing values display `—`/Unavailable. Host and physical-GPU metrics are shared-system samples, not exclusively the trainer. They carry timestamps in raw telemetry. The terminal renderer allocates no CUDA resources.
- Recovery: latest successful save identifier/step/time, actual file size and save duration, configured interval and next scheduled step, resume state, save status and validation progress. Native saves are `atomic_save_only`, not independently verified reloads. Save progress is unknown while writing; validation counts are real completed batches. Validation and checkpoint activity replace the running badge while those operations are active. Short operations may finish between display refreshes but remain in the events/history.

Best validation is now an optional native v1 checkpoint field. New checkpoints restore it on resume. Old v1 checkpoints without the field still load; best remains unavailable until an evaluation is observed. Smoothing and session peak memory restart on resume; total steps/tokens do not. Training weights, optimizer/scheduler, RNG and sampler semantics are unchanged.

## Reliability and data flow

`Trainer.train(ui=...)` attaches `TerminalDashboard.on_event` to the existing EventBus. The parent maintains bounded histories (60 losses/speeds, 30 timings, six recent events) and sends small snapshots through a nonblocking Unix socketpair. There is no bound address or network port. Four updates/second are sufficient for the live view; a final lifecycle snapshot is also sent. Full sockets drop display snapshots, not raw persisted training events.

A separate Python process renders Rich Live on the alternate screen, with no PyTorch imports, model access, CUDA synchronization, stdout redirection, input grabs, or global signal-handler changes in the trainer. Rendering stalls cannot hold the optimizer thread. Renderer exceptions/exits are isolated by the existing callback machinery, recorded as warnings, and leave training in persistence-only headless mode. The training process can continue while the renderer is unavailable. A renderer failure warning is persisted without a synchronous terminal write; subsequent metrics are kept in JSONL instead of writing into a possibly blocked/broken TTY. Raw `events.jsonl`, `metrics.jsonl`, run summaries and checkpoints remain separate from ANSI output.

The [Rich Live alternate-screen API](https://rich.readthedocs.io/en/stable/live.html#alternate-screen) restores the command prompt on normal exit; the worker also restores cursor visibility. Shutdown is bounded (up to about 2.5 seconds after training/checkpoint work); a stuck renderer is terminated then killed. Failure recovery attempts cursor/screen restoration through a separate nonblocking TTY descriptor. A physically blocked/disconnected terminal cannot be guaranteed to accept recovery bytes; training still proceeds without waiting indefinitely. The renderer exits if its owning trainer process disappears. This does not add remote management or training controls.

## Original reference audit and adaptation

Both earlier workspaces and component instructions were read before implementation; neither checkout, environment, history, asset nor run was modified.

| Reference | Verified implementation | Decision |
|---|---|---|
| v1 `LLM-From-Scratch/src/codexa_workspace/monitoring/dashboard.py` | Rich Live alternate screen, dynamic sizing, keyboard help/compact/details, separate read-only process | Preserve stable alternate-screen rendering and responsive panel behavior. Do not transplant termios input handling or separate Kitty launch requirements into an in-process training CLI. |
| v1 `.../monitoring/render.py` | Rounded panels, two columns, colored badges, progress, training/validation/resources/recovery groups, loss/speed sparklines and compact fallback | Selected visual foundation. Adapt palette, composition and small formatting/sparkline helpers to v3's actual fields. Retain MIT attribution/full license. |
| v1 `.../monitoring/reader.py` | Incremental tails, derived metrics and optional GPU sampling | Reuse the process-separation principle; replace status-file reader with existing v3 callbacks. No second telemetry collector. |
| v1 `LLM-Training/src/llm_training/viewer.py` and conversational training hooks | Plain ANSI dashboard and mandatory Kitty attachment for legacy operator launches | Inspected, rejected full-screen clear/print loop, legacy model labels, notifications, fixed memory thresholds and launch gates. Those policies belong to protected legacy training, not v3. |
| v2 `src/llm_numpy/training/progress.py` and trainer integration | Rate-limited single-line ANSI LiveProgress with redirected log fallback | Retain rate limiting/fallback concepts only. It lacks v1's full panel interface; no v2 code copied or zero-filled unknown metrics adopted. |

The earlier implementations were inspected as source, not run in their protected environments. Compatibility with their current jobs is not claimed. V1's MIT notice is retained at `codexa/training/terminal/V1_LICENSE` and acknowledged in THIRD_PARTY_NOTICES.md. No private v1/v2 logs or assets were imported.

## Verification

Run `.venv/bin/python scripts/verify.py` for regression, formatting, static analysis and automated PTY tests. The PTY tests launch actual CPU training, assert live rendered steps, validation/checkpoint events and alternate-screen/cursor/termios restoration. Separate tests cover dropped snapshots, dead renderers, callbacks, absent GPU data, narrow sizes, headless selection and interruption.

For an independently reviewable real run (dev and terminal extras installed):

```sh
python scripts/terminal_acceptance.py --output outputs/terminal-acceptance/NEW --data outputs/datasets/new-smoke --tokenizer outputs/tokenizers/new-smoke/tokenizer.json --ui auto
# Optional, bounded GPU acceptance; coordinate with other GPU work:
python scripts/terminal_acceptance.py --output outputs/terminal-acceptance/GPU-NEW --data outputs/datasets/new-smoke --tokenizer outputs/tokenizers/new-smoke/tokenizer.json --device cuda
```

The harness caps work at 200 optimizer steps and 60 seconds, uses a real PTY, and saves raw ANSI, parsed rendered cells/SVG, JSON evidence and real training outputs under the ignored directory. It enables colors only in its isolated test terminal. `--resize-to 60x24` tests resizing; `--interrupt-after 10` sends real SIGINT after observing progress; `--resume PATH --until 100` verifies continued training. The image in the README is rasterized from captured PTY cells, not a fabricated dashboard or separately simulated metric view. See the [implementation report](engineering/TERMINAL_DASHBOARD_REPORT.md).
