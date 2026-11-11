<!-- Generated from the adjacent evidence record; edit the JSON and regenerate. -->
# 500M language baseline limitations

The 500M baseline still fails the assessed conversational-generation panel.

**Recorded status: FAIL · 2026-11-10**

## Finding

- The preserved checkpoint does not demonstrate reliable conversational generation in the assessed panel.

## Run and scope

| Field | Recorded value |
| --- | --- |
| Date | 2026-11-10 |
| Run | research-status-2026-11-10 |
| Model | native-v1 500M; parent 54fca007f272fb37 |
| Dataset | Protected original language-v1 fixtures |
| Assessment scope | Separate observed generation failures from corpus and engineering approvals. |

## Measurements

| Metric | Value | Scope |
| --- | --- | --- |
| Generated / reviewed outputs | 400 / 100 | Recorded measurement |
| Independent reviewed contexts | 20 | Recorded measurement |
| Exact-loop flags | 276 / 400 | Automatic exact-loop detector |
| Token-limit stops | 399 / 400 | Stopping outcome; not alone a quality failure |
| Fully grammatical reviewed outputs | 7 / 100 | 87 additionally partially satisfy grammar |
| Reviewed coherence failures | 100 / 100 | Recorded independent assessment |

## Interpretation

- Locally grammatical sentences and partial relevance occur, but sustained coherent complete responses were not demonstrated.

## Limitations

- Only 20 contexts support the reviewed panel; 300 outputs have automatic metrics only.
- This panel cannot causally separate comprehension from free-running generation failure.

## Method

- Reuse the preserved 400 generations and checkpoint-blinded independent judgments; no new scoring or inference.
- Report automatic stopping/repetition and subjective rubric dimensions separately.

## Next steps

- Keep SFT deferred until the separately specified base-language gate passes.
- After a future continuation, compare identical protected prompts and settings with blinded scoring.

## Evidence

<details>
<summary>Artifact locations and original SHA-256 checksums</summary>

**Original historical document**

Location: `archive:research-2026-11-10.tar.gz!docs/TRAINING.md`

SHA-256: `98eaf69d31946d518d2facdb52db181ea07df299d5a81f24210b1c55e05731b9`

**Original historical document**

Location: `archive:research-2026-11-10.tar.gz!docs/CODEXA_V3_DIAGNOSTIC_REPORT.md`

SHA-256: `62fe4f7e8e402ecebbf8a6607eb15a8620f6b597138c037258533361e4dac8dd`

`repository:` identifies a checked-in path; `archive:` identifies a checksummed archive member. `external:` is relative to the private artifact root; `local:` identifies ignored local evidence. Private payloads are not distributed here.

</details>
