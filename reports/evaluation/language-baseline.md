<!-- Generated from the adjacent evidence record; edit the JSON and regenerate. -->
# 500M language baseline limitations

Separate observed generation failures from corpus and engineering approvals.

| Date | Run | Model | Dataset | Status |
| --- | --- | --- | --- | --- |
| 2026-11-10 | research-status-2026-11-10 | native-v1 500M; parent 54fca007f272fb37 | Protected original language-v1 fixtures | FAIL |

## Executive summary

- The preserved checkpoint does not demonstrate reliable conversational generation in the assessed panel.

## Key metrics

| Metric | Value | Scope |
| --- | --- | --- |
| Generated / reviewed outputs | 400 / 100 | Recorded measurement |
| Independent reviewed contexts | 20 | Recorded measurement |
| Exact-loop flags | 276 / 400 | Automatic exact-loop detector |
| Token-limit stops | 399 / 400 | Stopping outcome; not alone a quality failure |
| Fully grammatical reviewed outputs | 7 / 100 | 87 additionally partially satisfy grammar |
| Reviewed coherence failures | 100 / 100 | Recorded independent assessment |

## Methodology

- Reuse the preserved 400 generations and checkpoint-blinded independent judgments; no new scoring or inference.
- Report automatic stopping/repetition and subjective rubric dimensions separately.

## Results

- Locally grammatical sentences and partial relevance occur, but sustained coherent complete responses were not demonstrated.

## Failures and limitations

- Only 20 contexts support the reviewed panel; 300 outputs have automatic metrics only.
- This panel cannot causally separate comprehension from free-running generation failure.

## Supporting evidence

| Evidence | Artifact location | SHA256 |
| --- | --- | --- |
| Original historical document | `archive:research-2026-11-10.tar.gz!docs/TRAINING.md` | `98eaf69d31946d518d2facdb52db181ea07df299d5a81f24210b1c55e05731b9` |
| Original historical document | `archive:research-2026-11-10.tar.gz!docs/CODEXA_V3_DIAGNOSTIC_REPORT.md` | `62fe4f7e8e402ecebbf8a6607eb15a8620f6b597138c037258533361e4dac8dd` |

`external:` locations are relative to the private artifact root; payloads are not distributed here.

## Required next actions

- Keep SFT deferred until the separately specified base-language gate passes.
- After a future continuation, compare identical protected prompts and settings with blinded scoring.
