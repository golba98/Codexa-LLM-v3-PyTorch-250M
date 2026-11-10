<!-- Generated from the adjacent evidence record; edit the JSON and regenerate. -->
# Recorded engineering benchmarks

Index original measurements with their hardware and provenance limits.

| Date | Run | Model | Dataset | Status |
| --- | --- | --- | --- | --- |
| 2026-11-10 | research-status-2026-11-10 | native-v1 500M; parent 54fca007f272fb37 | Synthetic-token engineering workloads | PASS |

## Executive summary

- Original benchmark JSON is preserved byte-for-byte. PASS indicates retained engineering evidence, not trained model capability or current hardware readiness.

## Key metrics

| Metric | Value | Scope |
| --- | --- | --- |
| Retained JSON records | 9 | Original records; not new measurements |
| Primary parameter count | 248,687,488 | Associated primary configuration |

## Methodology

- Original synthetic-token benchmarks use real forward/backward/AdamW updates, synchronized CUDA timing and BF16 autocast where supported.
- Original hardware scope: RTX 4080, Python 3.13.14, PyTorch 2.11.0+cu130; inspect each JSON for its precise configuration.

## Results

- Context, dashboard, smoke and primary measurements remain individually inspectable in evidence/.

## Failures and limitations

- Exact executed dirty-source identities were not captured for early measurements; the foundation commit is only an associated snapshot.
- Peak allocated memory excludes driver/context and reserved memory. Short-run extrapolations exclude validation, checkpoint and sustained-load effects.
- No measurements were rerun or fabricated during repository organization.

## Supporting evidence

| Evidence | Artifact location | SHA256 |
| --- | --- | --- |
| bench-250m-acc16-cuda | `repository:reports/benchmarks/evidence/bench-250m-acc16-cuda.json` | `4fc79d1a8bee6356e2d8a01678f09630463043c8f78a15f1ca3b7eab846ac513` |
| bench-250m-cuda | `repository:reports/benchmarks/evidence/bench-250m-cuda.json` | `4dc34be4f226953079de20a66c48f914b93561def2ba4405725f690fcf91f3cd` |
| bench-50m-cuda | `repository:reports/benchmarks/evidence/bench-50m-cuda.json` | `a9ac5e417e27c2189cc0f787c722b59adb04d7c49a128bd2d7957df352528280` |
| bench-smoke-cpu | `repository:reports/benchmarks/evidence/bench-smoke-cpu.json` | `24f3b3b02778b22ed4ea394423a1985dcc536cdba5e5e83f70ad419646385b44` |
| bench-smoke-cuda | `repository:reports/benchmarks/evidence/bench-smoke-cuda.json` | `c73b493b7fe166a59b1feb1cbe90ae655d8ae45b39ec85c585422def1835b464` |
| m25-context | `repository:reports/benchmarks/evidence/m25-context.json` | `c2dd48a787295ac803978a7fef3346f1d82d9f5b01fc0c2e78c0cfe79f1c7e7e` |
| m25-dashboard-performance | `repository:reports/benchmarks/evidence/m25-dashboard-performance.json` | `68b21076452d95eb131b90910107e855634fd9924cf430fa1ce9dc140473be9d` |
| overhaul-250m | `repository:reports/benchmarks/evidence/overhaul-250m.json` | `c82c1d35a302d87d374b6a88246c92f0b4c85b67297eaeeb9687f1fc780fd53c` |
| verification | `repository:reports/benchmarks/evidence/verification.json` | `2e31cf4014ed8885101c57d68c91f26d627639fd78cd357effd557a6a8bcbd17` |

`external:` locations are relative to the private artifact root; payloads are not distributed here.

## Required next actions

- Use original JSON for configuration-specific values.
- Collect a new explicitly bounded measurement with source/config/environment provenance before claiming a performance change.
