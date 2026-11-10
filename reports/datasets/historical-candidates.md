<!-- Generated from the adjacent evidence record; edit the JSON and regenerate. -->
# Historical dataset-v2 decisions

Retain failed corpus decisions without presenting them as current Wikimedia readiness.

| Date | Run | Model | Dataset | Status |
| --- | --- | --- | --- | --- |
| 2026-11-10 | research-status-2026-11-10 | native-v1 500M; parent 54fca007f272fb37 | dataset-v2 remediated-v4c-20261106 and earlier candidates | FAIL |

## Executive summary

- Historical audited token snapshots failed their quality gates. These results are preserved and do not overturn the later separately reviewed Wikimedia source decision.

## Key metrics

| Metric | Value | Scope |
| --- | --- | --- |
| Retained v4c positions | 2,052,040,276 | Historical independently audited snapshot |
| FineMath severe defects | 11 / 40 | Historical primary panel |
| FineWeb-Edu usable records | 21 / 40 | Historical primary panel |
| Primary/extended records | 900 | Two independent ratings each |

## Methodology

- Keep native-token audits, split/protection checks and quality decisions separate.
- Full original source-specific results and remediation chronology are retained in the archive.

## Results

- Both extended panels also failed. A separate semantic-admission census is preprocessing evidence, not a fresh passing approval panel.

## Failures and limitations

- Historical snapshot availability differs after previously authorized storage cleanup; retained manifests and hashes describe their capture time.
- Token count and completed assessments do not confer admission or training authorization.

## Supporting evidence

| Evidence | Artifact location | SHA256 |
| --- | --- | --- |
| Original historical document | `archive:research-2026-11-10.tar.gz!docs/CODEXA_V3_DATASET_V2.md` | `a834404a13a8dae82f58fe015d2899277925b1b7f80bb2426c4ebb9f1cab3901` |
| Original historical document | `archive:research-2026-11-10.tar.gz!docs/CODEXA_V3_TRAINING_READINESS.md` | `5ca35072c8bf7407345ea7e983e04ed7373faec9aa23bdeeb2b3005984e2ec0e` |
| Original historical document | `archive:research-2026-11-10.tar.gz!reports/codexa-v3-comprehensive-status-audit.md` | `d8b07fd7f068ce5b1716c452bcec64ec43bab41b7bdcdf70eaa9dfcb07de16bc` |

`external:` locations are relative to the private artifact root; payloads are not distributed here.

## Required next actions

- Retain original decisions and provenance.
- Use the current source readiness report for the selected language continuation; do not rebuild these candidates during cleanup.
