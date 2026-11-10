# Release process

## Language-first artifact policy

The current language-priority change authorizes direct-stream implementation and bounded source review toward 1B additional tokens; it authorizes neither training nor publication. Preserve historical failed corpus decisions. New protocol authorization does not confer source training rights, approve a corpus, establish language competence or authorize a weight license. Keep raw text, token shards, model/optimizer states, independent reviews and generated outputs on the dedicated volume outside Git. Continue using native v1 checkpoints for recovery and SafeTensors for separately authorized public exports.


The approved Wikimedia quality review and existing CC BY-SA 4.0 assessment admit only the pinned, attributable English Wikipedia subset for internal pretraining. Preserve provider/article/history/license and conversion provenance. This does not clear third-party exceptions, model-weight share-alike questions or public redistribution. The explicit 1B launch authorization remains pending; the original model is unchanged.

GitHub contains source; an eventual model repository contains SafeTensors, configuration, tokenizer, card, generation defaults, provenance, checksums and approved weight license. No pretrained model is currently available and no upload command is implemented.

1. Run `python scripts/verify.py` in the intended pinned environment.
2. Test locally: `codexa release fixture --corpus examples/corpus --output outputs/exports/fixture-NEW`.
3. Run `codexa release preflight --artifact outputs/exports/fixture-NEW --verification-report outputs/logs/verification.json`.
4. For trained artifacts, export a trusted native checkpoint with `codexa export --checkpoint PATH --metadata reviewed.json --output outputs/exports/NAME`. Metadata must identify stage, version, origin, source commit, training tokens, datasets, evaluation and weight-license review.
5. Review every model-card field, dataset permissions, license terms, reproducibility, resource measurements and scans. Obtain explicit publication authorization separately. No fixture is publish-ready.

SafeTensors exports are schema version 1 and use Codexa's own loader; no Transformers, GGUF or llama.cpp compatibility is claimed. Manifests hash payloads, not themselves. Validation reports stay outside export directories. Target directories must not exist. Native checkpoints remain version 1 and contain recovery state; do not distribute them as public inference weights.

Public blockers: trained/evaluated weights, approved weight licensing and provenance, enabled private security reporting, and actual hosted CI execution. Source code is Apache-2.0 and original corpus CC0; neither approves a future weight license. Preflight checks technical evidence and review declarations; it cannot independently certify legal rights or model capabilities.

## Dataset-v2 artifacts

The [dataset-v2 preparation](CODEXA_V3_DATASET_V2.md) registry records source revisions, hashes, declarations and attribution, while retaining underlying-content limitations. These records do not approve publication of the combined corpus or resulting weights. Raw text, token shards, evaluation outputs and native continuation bundles remain on `/mnt/codexa`, outside Git. Original authored benchmark fixtures are CC0. No release or remote metadata change is authorized by this preparation phase.
