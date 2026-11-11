<!-- Generated from the adjacent evidence record; edit the JSON and regenerate. -->
# Completed 500M base run

The base run completed 500,006,912 targets with a validated native recovery checkpoint.

**Recorded status: PASS · 2026-11-04**

## Finding

- The authorized fresh-from-random-initialization run completed its rounded token budget. This is checkpoint and run-completion evidence, not a language-competence certificate.

## Run and scope

| Field | Recorded value |
| --- | --- |
| Date | 2026-11-04 |
| Run | 20261103-500m-233201 |
| Model | native-v1 500M; parent 54fca007f272fb37 |
| Dataset | Frozen 884.18M-position training corpus; native BPE 16,384 |
| Assessment scope | Record the completed native checkpoint and measured training scope. |

## Measurements

| Metric | Value | Scope |
| --- | --- | --- |
| Parameters | 248,687,488 | Recorded measurement |
| Targets / updates | 500,006,912 / 15,259 | Recorded measurement |
| Final fixed-slice loss | 3.005710 | 65,536 validation targets |
| Full-window validation loss / perplexity | 3.048003 / 21.073212 | 606,208 validation targets; test split not evaluated |
| End-to-end tokens/s | 23,054.90 | 21,687.67s including loading, replay, validation and checkpoints |
| Optimizer tokens/s | 24,023.45 | After 2,000-update warmup |
| Allocated / reserved GiB | 7.0171 / 7.1582 | Observed allocator peaks; excludes desktop/driver |

## Interpretation

- Final checkpoint SHA256: 54fca007f272fb3784c759d18db44912091227aedc233b62c9731514b1835c73.
- All 15,259 updates appear once; recorded losses and gradients remained finite.

## Limitations

- Frozen-corpus loss does not establish broad language, mathematics or long-context competence.
- The completed schedule cannot be silently extended through exact resume.

## Method

- RTX 4080, BF16 autocast, FP32 loss; context 2,048, microbatch 1, accumulation 16, seed 42.
- Independent CPU reload and forward-only CUDA validation replay checked the final native-v1 state.

## Next steps

- Preserve final and best checkpoints, optimizer/RNG state and tokenizer.
- Use paired protected evaluations for any separately launched continuation.

## Evidence

<details>
<summary>Artifact locations and original SHA-256 checksums</summary>

**Original historical document**

Location: `archive:research-2026-11-10.tar.gz!docs/TRAINING.md`

SHA-256: `98eaf69d31946d518d2facdb52db181ea07df299d5a81f24210b1c55e05731b9`

**Preserved final checkpoint**

Location: `external:pretraining/20261103-500m-233201/final/checkpoint.pt`

SHA-256: `54fca007f272fb3784c759d18db44912091227aedc233b62c9731514b1835c73`

`repository:` identifies a checked-in path; `archive:` identifies a checksummed archive member. `external:` is relative to the private artifact root; `local:` identifies ignored local evidence. Private payloads are not distributed here.

</details>
