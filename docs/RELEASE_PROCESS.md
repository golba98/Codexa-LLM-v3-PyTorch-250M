# Release process

GitHub contains source; an eventual model repository contains SafeTensors, configuration, tokenizer, card, generation defaults, provenance, checksums and approved weight license. No pretrained model is currently available and no upload command is implemented.

1. Run `python scripts/verify.py` in the intended pinned environment.
2. Test locally: `codexa release fixture --corpus examples/corpus --output outputs/exports/fixture-NEW`.
3. Run `codexa release preflight --artifact outputs/exports/fixture-NEW --verification-report outputs/logs/verification.json`.
4. For trained artifacts, export a trusted native checkpoint with `codexa export --checkpoint PATH --metadata reviewed.json --output outputs/exports/NAME`. Metadata must identify stage, version, origin, source commit, training tokens, datasets, evaluation and weight-license review.
5. Review every model-card field, dataset permissions, license terms, reproducibility, resource measurements and scans. Obtain explicit publication authorization separately. No fixture is publish-ready.

SafeTensors exports are schema version 1 and use Codexa's own loader; no Transformers, GGUF or llama.cpp compatibility is claimed. Manifests hash payloads, not themselves. Validation reports stay outside export directories. Target directories must not exist. Native checkpoints remain version 1 and contain recovery state; do not distribute them as public inference weights.

Public blockers: trained/evaluated weights, approved weight licensing and provenance, enabled private security reporting, and actual hosted CI execution. Source code is Apache-2.0 and original corpus CC0; neither approves a future weight license. Preflight checks technical evidence and review declarations; it cannot independently certify legal rights or model capabilities.
