<!-- Generated from the adjacent evidence record; edit the JSON and regenerate. -->
# 1B language continuation readiness

Show source admission, preserved baseline, authorization and remaining launch preparation independently.

| Date | Run | Model | Dataset | Status |
| --- | --- | --- | --- | --- |
| 2026-11-10 | research-status-2026-11-10 | native-v1 500M; parent 54fca007f272fb37 | common-pile/wikimedia_filtered@0641bb84bd9b7162bcddf8be7836822161a9a342 | BLOCKED |

## Readiness decisions

| Gate / decision | State |
| --- | --- |
| quality | PASS |
| internal rights | PASS |
| historical stream resume | PASS |
| historical compute scope | PASS |
| machine launch record | BLOCKED |
| cleanup execution | BLOCKED |
| Training authorized | YES |
| Training execution permitted | NO |
| Training started | NO |

## Executive summary

- The owner authorized the future 1B-token continuation in the cleanup request. This task explicitly prohibits starting it.
- The pinned English Wikipedia subset passes its approved general-English quality rubric. The saved machine authorization record remains pending and unchanged.

## Key metrics

| Metric | Value | Scope |
| --- | --- | --- |
| Current checkpoint targets | 500,006,912 | Recorded measurement |
| Additional targets | 1,000,000,000 | Owner-authorized future budget; not consumed |
| Cumulative target | 1,500,006,912 | Arithmetic budget, not achieved training |
| Usable documents | 94 / 100 | Conservative combination of two independent reviews |
| Usable windows | 293 / 300 | Conservative combination of two independent reviews |
| Severe findings | 1 document; 3 windows; 0 critical | Recorded review |
| Eligible training positions, lower bound | 3.606B | Approximate conditional 95% lower estimate; not full census |
| Resumed loader positions/s | 23,736.9 | Historical 176.6134s interval; loader only |
| CUDA allocated / reserved bytes | 7,567,977,984 / 8,166,309,888 | Recorded gradient residency check; zero optimizer updates |
| Training started | NO | No continuation launched by this cleanup |

## Methodology

- Reuse the pinned source panel, internal-use rights record, protected exclusions, native-token capacity estimate and prior compute/resume evidence.
- Readiness reporting is descriptive. It never changes an authorization, prepares a phase or invokes a trainer.

## Results

- Quality: PASS. Rights: approved for internal pretraining subject to retained attribution and exclusions; public distribution is not approved.
- Streaming: prior native next-batch recovery and fixed-validation replay passed; the full-model parent is unchanged.
- Training authorization: owner-authorized for the future continuation. Execution during this task: prohibited. Saved machine launch record: pending finalization in a later launch-preparation task.

## Failures and limitations

- Prior live throughput predates the added navigation exclusion; no new provider or GPU benchmark is claimed.
- Gradient residency does not measure Adam-update speed or sustained training throughput.
- Conversational competence, SFT readiness, mathematics training and publication remain separate gates.

## Supporting evidence

| Evidence | Artifact location | SHA256 |
| --- | --- | --- |
| protocol | `local:.agents/scratch/wikimedia-approved-reevaluation-v1/protocol.json` | `5ce85641bcf30c1fdc27482b777255e337865b851ff55e6d8af9b2917a9656cf` |
| capacity | `local:.agents/scratch/wikimedia-approved-reevaluation-v1/capacity.json` | `5f6b6851bc20bef34445f2ca9f905b73d230d40cc26a0a2c062361bd3b3b3e67` |
| compute | `local:.agents/scratch/wikimedia-approved-reevaluation-v1/compute-readiness.json` | `8aad6df82e83b4933fb1603cbb89eac4fd17fd7067053779913e1a514fc8d1dd` |
| rights | `local:.agents/scratch/wikimedia-review-v1/rights.json` | `69b58432f4f060c016a140513f38d0b51cc57c6a9d3c1b6bd2f3b1cc0ce15f99` |
| authorization | `local:.agents/scratch/wikimedia-approved-reevaluation-v1/training-authorization-pending.json` | `aff4efae071c9241748617afa41d0003e9927d1710cb876e894c2a0dbd396a2a` |
| Original historical document | `archive:research-2026-11-10.tar.gz!docs/TRAINING.md` | `98eaf69d31946d518d2facdb52db181ea07df299d5a81f24210b1c55e05731b9` |
| Original historical document | `archive:research-2026-11-10.tar.gz!docs/DATASETS.md` | `5ad748bd6656ab07fc9b134b845f2a0e3c2781ce63c806f6052027f8160530cb` |

`external:` locations are relative to the private artifact root; payloads are not distributed here.

## Required next actions

- Complete repository verification and review the cleanup PR.
- In a later authorized launch-preparation task, finalize source/config-bound machine authorization and revalidate all unchanged gates before preparing a fresh phase.
- Do not launch training as part of repository maintenance.
