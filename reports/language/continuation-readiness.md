<!-- Generated from the adjacent evidence record; edit the JSON and regenerate. -->
# 1B language continuation readiness

The old 1B proposal had source approval but unresolved machine launch preparation.

**Recorded status: BLOCKED · 2026-11-10**

> **Historical snapshot.** Decisions, budgets and next actions below describe 2026-11-10. Use the [current documentation](../../docs/index.md) for supported workflows.

> This assessment concerns the earlier 1B proposal. The current phase adds 500M targets; see [language continuation](../../docs/training.md#language-continuation).

## Finding

- The owner authorized the future 1B-token continuation in the cleanup request. This task explicitly prohibits starting it.
- The pinned English Wikipedia subset passes its approved general-English quality rubric. The saved machine authorization record remains pending and unchanged.

## Run and scope

| Field | Recorded value |
| --- | --- |
| Date | 2026-11-10 |
| Run | research-status-2026-11-10 |
| Model | native-v1 500M; parent 54fca007f272fb37 |
| Dataset | common-pile/wikimedia_filtered@0641bb84bd9b7162bcddf8be7836822161a9a342 |
| Assessment scope | Show source admission, preserved baseline, authorization and remaining launch preparation independently. |

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

## Measurements

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

## Interpretation

- Quality: PASS. Rights: approved for internal pretraining subject to retained attribution and exclusions; public distribution is not approved.
- Streaming: prior native next-batch recovery and fixed-validation replay passed; the full-model parent is unchanged.
- Training authorization: owner-authorized for the future continuation. Execution during this task: prohibited. Saved machine launch record: pending finalization in a later launch-preparation task.

## Limitations

- Prior live throughput predates the added navigation exclusion; no new provider or GPU benchmark is claimed.
- Gradient residency does not measure Adam-update speed or sustained training throughput.
- Conversational competence, SFT readiness, mathematics training and publication remain separate gates.

## Method

- Reuse the pinned source panel, internal-use rights record, protected exclusions, native-token capacity estimate and prior compute/resume evidence.
- Readiness reporting is descriptive. It never changes an authorization, prepares a phase or invokes a trainer.

## Actions recorded at the time

- Complete repository verification and review the cleanup PR.
- In a later authorized launch-preparation task, finalize source/config-bound machine authorization and revalidate all unchanged gates before preparing a fresh phase.
- Do not launch training as part of repository maintenance.

## Evidence

<details>
<summary>Artifact locations and original SHA-256 checksums</summary>

**protocol**

Location: `local:.agents/scratch/wikimedia-approved-reevaluation-v1/protocol.json`

SHA-256: `5ce85641bcf30c1fdc27482b777255e337865b851ff55e6d8af9b2917a9656cf`

**capacity**

Location: `local:.agents/scratch/wikimedia-approved-reevaluation-v1/capacity.json`

SHA-256: `5f6b6851bc20bef34445f2ca9f905b73d230d40cc26a0a2c062361bd3b3b3e67`

**compute**

Location: `local:.agents/scratch/wikimedia-approved-reevaluation-v1/compute-readiness.json`

SHA-256: `8aad6df82e83b4933fb1603cbb89eac4fd17fd7067053779913e1a514fc8d1dd`

**rights**

Location: `local:.agents/scratch/wikimedia-review-v1/rights.json`

SHA-256: `69b58432f4f060c016a140513f38d0b51cc57c6a9d3c1b6bd2f3b1cc0ce15f99`

**authorization**

Location: `local:.agents/scratch/wikimedia-approved-reevaluation-v1/training-authorization-pending.json`

SHA-256: `aff4efae071c9241748617afa41d0003e9927d1710cb876e894c2a0dbd396a2a`

**Original historical document**

Location: `archive:research-2026-11-10.tar.gz!docs/TRAINING.md`

SHA-256: `98eaf69d31946d518d2facdb52db181ea07df299d5a81f24210b1c55e05731b9`

**Original historical document**

Location: `archive:research-2026-11-10.tar.gz!docs/DATASETS.md`

SHA-256: `5ad748bd6656ab07fc9b134b845f2a0e3c2781ce63c806f6052027f8160530cb`

`repository:` identifies a checked-in path; `archive:` identifies a checksummed archive member. `external:` is relative to the private artifact root; `local:` identifies ignored local evidence. Private payloads are not distributed here.

</details>
