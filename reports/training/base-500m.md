<!-- Generated from the adjacent evidence record; edit the JSON and regenerate. -->
# Completed 500M base run

Record the completed native checkpoint and measured training scope.

| Date | Run | Model | Dataset | Status |
| --- | --- | --- | --- | --- |
| 2026-11-04 | 20261103-500m-233201 | native-v1 500M; parent 54fca007f272fb37 | Frozen 884.18M-position training corpus; native BPE 16,384 | PASS |

## Executive summary

- The authorized fresh-from-random-initialization run completed its rounded token budget. This is checkpoint and run-completion evidence, not a language-competence certificate.

## Key metrics

| Metric | Value | Scope |
| --- | --- | --- |
| Parameters | 248,687,488 | Recorded measurement |
| Targets / updates | 500,006,912 / 15,259 | Recorded measurement |
| Final fixed-slice loss | 3.005710 | 65,536 validation targets |
| Full-window validation loss / perplexity | 3.048003 / 21.073212 | 606,208 validation targets; test split not evaluated |
| End-to-end tokens/s | 23,054.90 | 21,687.67s including loading, replay, validation and checkpoints |
| Optimizer tokens/s | 24,023.45 | After 2,000-update warmup |
| Allocated / reserved GiB | 7.0171 / 7.1582 | Observed allocator peaks; excludes desktop/driver |

## Methodology

- RTX 4080, BF16 autocast, FP32 loss; context 2,048, microbatch 1, accumulation 16, seed 42.
- Independent CPU reload and forward-only CUDA validation replay checked the final native-v1 state.

## Results

- Final checkpoint SHA256: 54fca007f272fb3784c759d18db44912091227aedc233b62c9731514b1835c73.
- All 15,259 updates appear once; recorded losses and gradients remained finite.

## Failures and limitations

- Frozen-corpus loss does not establish broad language, mathematics or long-context competence.
- The completed schedule cannot be silently extended through exact resume.

## Supporting evidence

| Evidence | Artifact location | SHA256 |
| --- | --- | --- |
| Original historical document | `archive:research-2026-11-10.tar.gz!docs/TRAINING.md` | `98eaf69d31946d518d2facdb52db181ea07df299d5a81f24210b1c55e05731b9` |
| Preserved final checkpoint | `external:pretraining/20261103-500m-233201/final/checkpoint.pt` | `54fca007f272fb3784c759d18db44912091227aedc233b62c9731514b1835c73` |

`external:` locations are relative to the private artifact root; payloads are not distributed here.

## Required next actions

- Preserve final and best checkpoints, optimizer/RNG state and tokenizer.
- Use paired protected evaluations for any separately launched continuation.
