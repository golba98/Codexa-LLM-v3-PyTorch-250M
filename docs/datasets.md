# Datasets and tokenizer

Language continuation and Grade 1–12 mathematics corpus development are separate workstreams. Large payloads, tokenizer files, reviews and protected assessments remain outside Git. Source configuration examples live in [configs/data](../configs/data/); paths must be supplied for the operator's storage.

## Current language source

The selected source is the English Wikipedia subset of `common-pile/wikimedia_filtered`, revision `0641bb84bd9b7162bcddf8be7836822161a9a342`. Its approved general-English panel passes at 94/100 usable documents and 293/300 windows, with the original decisions and changed judgments retained. Rights approval is restricted to internal pretraining with source-specific attribution and exclusions. This is a recorded assessment, not permission to redistribute the source or trained weights.

The production binding retains article/page identifiers, provider and original URLs, history/contributor/license metadata, exclusions, source hashes, tokenizer identity and protected evaluation indexes. Provider access, corpus quality, eligible native-token capacity and training authorization are independent checks. See [current readiness and evidence](../reports/language/continuation-readiness.md).

Earlier FineMath, FineWeb-Edu, OpenWebMath, DCLM, Pressbooks, TinyStories and Common Corpus candidates have different rights, quality and chronology. Their configurations are retained for reproducibility; filenames do not imply current source approval. Use [historical candidate decisions](../reports/datasets/historical-candidates.md) and the [original research archive](../reports/archive/index.json) for full results. Do not rebuild those corpora as a cleanup step.

## Tokenizer and preparation

The trained parent uses the preserved **16,384-token BPE**, vocabulary and special IDs. Do not fit a replacement for continuation. The commands below describe the small original engineering fixture, not the approved production corpus:

```sh
codexa tokenizer train --corpus examples/corpus --vocab-size 512 --output outputs/tokenizers/demo/tokenizer.json
codexa data prepare --corpus examples/corpus --tokenizer outputs/tokenizers/demo/tokenizer.json --output outputs/datasets/demo
codexa data verify --data outputs/datasets/demo
```

Use fresh destinations. Production tooling also supports disk-backed curation, original text/JSONL and optional Parquet input, provenance manifests, exact/approximate deduplication, parent-group splitting, protected spans and immutable indexed uint16 shards. Legacy uint32 data remains supported. The direct production stream instead transfers and tokenizes pinned records on demand; it does not require rebuilding a static token corpus.

Record source origin, revision, edition, source-specific rights, attribution, hashes, cleaning decisions, deduplication and split policy. Fit candidate tokenizers on training text only. Keep evaluation and review samples excluded, record rejected/quarantined material, and validate actual token boundaries independently of manifest checksums. Near-duplicate matching is approximate; semantic contamination and source-rights exceptions require separate review.

## Mathematics corpus

All existing materials under `/mnt/codexa/pytorch/codexa-v3/school-math/` are protected. This includes pilot/source inventories, Grade 1–12 curriculum references, saved pages/EPUB/PDF evidence, recoveries, synthetic candidates, review snapshots, rights records, protected assessments and exclusions. Repository cleanup does not modify those files.

Reusable helpers remain in `codexa/data`; extraction, inventory, recovery, screening and assembly scripts are grouped in `scripts/datasets/mathematics`. Some optional acquisition tools depend on preserved external workers, BeautifulSoup, or system `pdftotext`; those are explicitly local research prerequisites, not core inference dependencies. No tool is run automatically by installation.

Answer checks do not establish explanation correctness, curriculum coverage, source rights or training admission. Official CAPS/ATP references and retrieval seeds are distinct from a reconciled curriculum denominator. Preserve unresolved visual gaps, quarantines and review disagreements. The exposed earlier assessment pool remains development-only. See [mathematics status](../reports/datasets/mathematics-corpus.md) for remaining gates; no mathematics training is authorized by the corpus artifacts.

## Evidence and storage

Manifests and decision files describe specific captures. Original source hashes and historical receipts must not be rewritten when paths or reports change. Checkpoints, optimizer states, complete corpus payloads and private evaluations belong on their existing external storage; ignored local outputs are also preserved. [Historical storage findings](../reports/datasets/storage-history.md) are not a current deletion authorization.
