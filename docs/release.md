# Release process

## Language-first artifact policy

The future 1B language continuation is owner-authorized; this repository cleanup prohibits executing it. The saved machine authorization remains pending and unchanged. Source quality and internal-use rights do not authorize publication. Preserve all checkpoints, tokenizer files, optimizer states, reviews and protected mathematics assets outside Git. See [current readiness](../reports/language/continuation-readiness.md).

The pinned English Wikipedia assessment is limited to internal pretraining with retained attribution and exclusions. Third-party exceptions, weight licensing and public redistribution require separate review. Neither dataset quality approval nor training authorization grants a publication license.

GitHub contains source; an eventual model repository contains SafeTensors, configuration, tokenizer, card, generation defaults, provenance, checksums and approved weight license. No pretrained public model is currently available and no upload command is implemented.

1. Run `python scripts/verify.py` in the intended pinned environment.
2. Test locally: `codexa release fixture --corpus examples/corpus --output outputs/exports/fixture-NEW`.
3. Run `codexa release preflight --artifact outputs/exports/fixture-NEW --verification-report outputs/logs/verification.json`.
4. For trained artifacts, export a trusted native checkpoint with `codexa export --checkpoint PATH --metadata reviewed.json --output outputs/exports/NAME`. Metadata must identify stage, version, origin, source commit, training tokens, datasets, evaluation and weight-license review.
5. Review every model-card field, dataset permissions, license terms, reproducibility, resource measurements and scans. Obtain explicit publication authorization separately. No fixture is publish-ready.

SafeTensors exports are schema version 1 and use Codexa's own loader; no Transformers, GGUF or llama.cpp compatibility is claimed. Manifests hash payloads, not themselves. Validation reports stay outside export directories. Target directories must not exist. Native checkpoints remain version 1 and contain recovery state; do not distribute them as public inference weights.

Public model blockers include demonstrated capability and approved weight licensing/provenance. Review security reporting settings and current hosted CI separately; neither is established by a local preflight. Source code is Apache-2.0 and original corpus CC0; neither approves a future weight license. Preflight checks technical evidence and review declarations; it cannot independently certify legal rights or model capabilities.

## Dataset-v2 artifacts

The [historical dataset decisions](../reports/datasets/historical-candidates.md) registry records source revisions, hashes, declarations and attribution, while retaining underlying-content limitations. These records do not approve publication of the combined corpus or resulting weights. Raw text, token shards, evaluation outputs and native continuation bundles remain on `/mnt/codexa`, outside Git. Original authored benchmark fixtures are CC0. No release or remote metadata change is authorized by this preparation phase.
