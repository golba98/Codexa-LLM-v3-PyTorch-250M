# Datasets and tokenizer

Language continuation and Grade 1–12 mathematics are separate data workstreams. This guide explains their source, preparation and evaluation boundaries.

[Documentation](index.md) · [Training](training.md) · [Dataset reports](../reports/README.md)

## Language source

The current continuation uses the English Wikipedia subset of **`common-pile/wikimedia_filtered`**, pinned to revision `0641bb84bd9b7162bcddf8be7836822161a9a342`.

| Review dimension | Recorded result |
| --- | --- |
| Usable documents | 94 / 100 |
| Usable windows | 293 / 300 |
| Rights scope | Internal pretraining with attribution and source-specific exclusions |
| Current continuation budget | Exactly 500M additional targets |

Original decisions and changed judgments remain retained. Approval for internal use does not grant permission to redistribute the source or trained weights.

The stream retains page/article IDs, provider and original URLs, history/contributor/license metadata, exclusions, source hashes, tokenizer identity and protected evaluation indexes. Provider access, quality, eligible-token capacity and launch authorization are independent checks.

The [training guide](training.md#provider-cache) describes the disposable compressed-byte cache. It reuses exact provider bytes and preserves native sampler state; it creates no tokenized snapshot or new corpus.

### Earlier source candidates

FineMath, FineWeb-Edu, OpenWebMath, DCLM, Pressbooks, TinyStories and Common Corpus have distinct rights, quality and chronology. Their configurations remain for reproducibility; a filename does not establish current admission.

Read the [historical candidate decisions](../reports/datasets/historical-candidates.md) for their outcomes and the [archive](../reports/archive/index.json) for original evidence. The old 1B [readiness assessment](../reports/language/continuation-readiness.md) also retains its original scope.

## Frozen tokenizer

The parent checkpoint uses a **16,384-token byte-pair encoding (BPE) tokenizer**. Continuation preserves its vocabulary, special IDs and identity; it does not train a replacement.

Candidate tokenizers must be fitted on training text only. Tokenizer files remain outside Git. Source configuration examples are in [configs/data](../configs/data/), with operator-supplied storage paths.

## Prepare a small fixture

These commands use the original engineering corpus and a 512-token fixture tokenizer:

```sh
codexa tokenizer train --corpus examples/corpus --vocab-size 512 --output outputs/tokenizers/demo/tokenizer.json
codexa data prepare --corpus examples/corpus --tokenizer outputs/tokenizers/demo/tokenizer.json --output outputs/datasets/demo
codexa data verify --data outputs/datasets/demo
```

Use fresh output destinations. This fixture is separate from the approved production stream.

Production preparation supports disk-backed curation, text/JSONL and optional Parquet input, provenance manifests, exact/approximate deduplication, parent-group splits, protected spans and immutable indexed uint16 shards. Legacy uint32 data remains supported. Direct streaming instead transfers and tokenizes pinned records on demand.

## Provenance and evaluation separation

| Record | Include |
| --- | --- |
| Origin | Source, revision, edition and acquisition context |
| Rights | Applicable permissions, source-specific exceptions and attribution |
| Integrity | Source and output hashes; independently checked token boundaries |
| Processing | Cleaning decisions, rejected/quarantined material and deduplication |
| Splits | Parent grouping, split policy, protected spans and evaluation exclusions |

Keep evaluation and review samples excluded from training. Checksums establish file identity; they do not validate split boundaries or token correctness. Near-duplicate detection is approximate, and semantic contamination needs separate review.

## Mathematics corpus

Grade 1–12 materials remain protected on external storage. They include source/pilot inventories, curriculum references, saved page/EPUB/PDF evidence, recoveries, synthetic candidates, rights records, review snapshots and protected assessments/exclusions.

The [mathematics report](../reports/datasets/mathematics-corpus.md) records the remaining admission gates. Important distinctions are:

- Numeric answer checks do not establish explanation correctness or curriculum coverage.
- Official CAPS/ATP references and retrieval seeds are separate from a reconciled curriculum denominator.
- Visual gaps, quarantines and review disagreements must remain explicit.
- The exposed earlier assessment pool is development-only.

Reusable helpers live in `codexa/data`. Extraction, inventory, recovery, screening and assembly scripts are grouped under `scripts/datasets/mathematics`. Some optional tools need preserved external workers, BeautifulSoup or system `pdftotext`; installation does not run acquisition tools automatically.

Corpus artifacts alone do not authorize mathematics training. Admission and a training budget require separate decisions.

## Storage and evidence

Keep checkpoints, optimizer states, complete corpora, tokenizer files and private reviews/evaluations on their existing external storage. Preserve ignored outputs too. Manifests describe specific captures; reorganizing documentation does not change their source hashes or historical receipts.

The [historical storage audit](../reports/datasets/storage-history.md) is evidence of an earlier capture, with no current deletion authorization. For public artifact boundaries, see the [release guide](release.md).
