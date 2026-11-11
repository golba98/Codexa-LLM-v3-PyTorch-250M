<!-- Generated from the adjacent evidence record; edit the JSON and regenerate. -->
# Historical dataset-v2 decisions

Earlier audited token snapshots failed their quality gates; later Wikipedia review is separate.

**Recorded status: FAIL · 2026-11-10**

> **Historical snapshot.** Decisions, budgets and next actions below describe 2026-11-10. Use the [current documentation](../../docs/index.md) for supported workflows.

## Finding

- Historical audited token snapshots failed their quality gates. These results are preserved and do not overturn the later separately reviewed Wikimedia source decision.

## Run and scope

| Field | Recorded value |
| --- | --- |
| Date | 2026-11-10 |
| Run | research-status-2026-11-10 |
| Model | native-v1 500M; parent 54fca007f272fb37 |
| Dataset | dataset-v2 remediated-v4c-20261106 and earlier candidates |
| Assessment scope | Retain failed corpus decisions without presenting them as current Wikimedia readiness. |

## Measurements

| Metric | Value | Scope |
| --- | --- | --- |
| Retained v4c positions | 2,052,040,276 | Historical independently audited snapshot |
| FineMath severe defects | 11 / 40 | Historical primary panel |
| FineWeb-Edu usable records | 21 / 40 | Historical primary panel |
| Primary/extended records | 900 | Two independent ratings each |

## Interpretation

- Both extended panels also failed. A separate semantic-admission census is preprocessing evidence, not a fresh passing approval panel.

## Limitations

- Historical snapshot availability differs after previously authorized storage cleanup; retained manifests and hashes describe their capture time.
- Token count and completed assessments do not confer admission or training authorization.

## Method

- Keep native-token audits, split/protection checks and quality decisions separate.
- Full original source-specific results and remediation chronology are retained in the archive.

## Actions recorded at the time

- Retain original decisions and provenance.
- Use the current source readiness report for the selected language continuation; do not rebuild these candidates during cleanup.

## Evidence

<details>
<summary>Artifact locations and original SHA-256 checksums</summary>

**Original historical document**

Location: `archive:research-2026-11-10.tar.gz!docs/CODEXA_V3_DATASET_V2.md`

SHA-256: `a834404a13a8dae82f58fe015d2899277925b1b7f80bb2426c4ebb9f1cab3901`

**Original historical document**

Location: `archive:research-2026-11-10.tar.gz!docs/CODEXA_V3_TRAINING_READINESS.md`

SHA-256: `5ca35072c8bf7407345ea7e983e04ed7373faec9aa23bdeeb2b3005984e2ec0e`

**Original historical document**

Location: `archive:research-2026-11-10.tar.gz!reports/codexa-v3-comprehensive-status-audit.md`

SHA-256: `d8b07fd7f068ce5b1716c452bcec64ec43bab41b7bdcdf70eaa9dfcb07de16bc`

`repository:` identifies a checked-in path; `archive:` identifies a checksummed archive member. `external:` is relative to the private artifact root; `local:` identifies ignored local evidence. Private payloads are not distributed here.

</details>
