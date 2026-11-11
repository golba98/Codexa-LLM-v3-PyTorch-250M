# Release and artifact formats

Use this guide to validate local exports and understand what can be shared. **No public trained weights are currently available**, and local preflight does not authorize publication.

[Documentation](index.md) · [Training and recovery](training.md) · [Third-party notices](../THIRD_PARTY_NOTICES.md)

## Choose the right artifact

| Artifact | Contains | Intended use |
| --- | --- | --- |
| Native v1 checkpoint | Weights, optimizer, scheduler, RNG, sampler and recovery metadata | Private training and exact resume |
| Private best weights | Selected weights and associated metadata | Local research; incomplete recovery state |
| SafeTensors export | Inference weights plus configuration, tokenizer and reviewed metadata | Local inference and eventual approved distribution |
| Fixture export | Tiny engineering model and original fixture assets | Export/preflight testing |

Public model artifacts use **SafeTensors**, not native recovery checkpoints. Codexa exports use schema version 1 and Codexa's own loader. No Transformers, GGUF or llama.cpp compatibility is claimed.

An eventual model repository needs configuration, tokenizer, model card, generation defaults, provenance, checksums and an approved weight license alongside its weights. GitHub holds source code. No upload command is implemented.

## Validate a fixture locally

In the pinned environment:

```sh
python scripts/verify.py
codexa release fixture --corpus examples/corpus --output outputs/exports/fixture-NEW
codexa release preflight --artifact outputs/exports/fixture-NEW --verification-report outputs/logs/verification.json
```

Choose a new destination for every export. Artifact directories must not exist in advance. Keep validation reports outside export directories; manifests hash payloads, not themselves.

Fixture success checks the export tooling. It does not make the fixture publish-ready.

## Export a trained checkpoint

With a trusted native checkpoint and reviewed metadata:

```sh
codexa export --checkpoint /path/to/checkpoint.pt --metadata reviewed.json --output outputs/exports/NAME
```

Metadata must identify stage, version, origin, source commit, training-token count, datasets, evaluation and weight-license review. Review the model card, permissions, reproducibility evidence, resource measurements and scans before requesting publication authorization.

Preflight verifies technical evidence and review declarations. It cannot independently establish legal rights or model competence. Review hosted CI and security-reporting settings separately.

## Rights and publication

| Material | Current boundary |
| --- | --- |
| Source code | Apache-2.0 |
| Original authored corpus and benchmark fixtures | CC0 |
| Selected English Wikipedia source | Reviewed for internal pretraining with attribution and exclusions |
| Combined datasets and trained weights | Require separate rights and publication review |

Source and fixture licenses do not approve a trained-weight license. Historical dataset registries record revisions, hashes, declarations and attribution with underlying-content limitations; they do not approve distribution of the combined corpus or weights.

Public weights still need demonstrated capability and approved licensing/provenance. Publishing, releasing or changing remote metadata requires explicit authorization.

## Preserve training and data artifacts

The current English continuation adds exactly 500M targets to the preserved parent. It uses two retained rolling recovery files, a final checkpoint and optional best weights. Obsolete rolling files may be pruned only by the owning run's recovery manager.

Keep native checkpoints, optimizer states, tokenizer files, raw text, token shards, private evaluations, continuation bundles and protected mathematics assets outside Git. Preserve the original parent and other runs. Documentation or export preparation does not authorize deleting these artifacts.

See [rolling recovery](training.md#rolling-recovery), [dataset provenance](datasets.md#provenance-and-evaluation-separation) and the [historical source decisions](../reports/datasets/historical-candidates.md).
