# Codexa v3 dataset v2

**Stricter remediation remains blocked on quality.** The independently audited successor `remediated-v4c-20261106` retains 2,052,040,276 training positions, but its complete `codexa-quality-4` reviews failed. FineMath had 11/40 severe defects and FineWeb-Edu only 21/40 records meeting all usability checks; both larger panels also failed. All 900 records received two independent agent assessments. The separate 2,542-document semantic-admission census has all 5,084 assessments, followed by 556 completed evidence-based reassessments. Of 5,400 required assessments in the disjoint 2,700-document expansion, 5,308 are recorded and 92 remain unreviewed following an automatic safety-review interruption. These are preprocessing evidence, not passing approval panels. Historical reviews below remain unchanged evidence. No successor has quality approval or permission from the gates to train; no token-count quota overrides quality.

Phase 2 acquired and prepared new language material using the existing tokenizer without training. Its original immutable snapshot is `snapshots/reviewed-2b-20261105`. The subsequent remediation and launch-gate evidence is recorded below; an audited snapshot is not automatically approved for training.

**Launch quality review failed on 2026-11-05.** The source-blinded assistant review found three severe extraction/boilerplate defects in the fixed 40-document FineMath panel, exceeding the agreed maximum of two. The review stopped decisively with 87 ratings and 413 samples unscored; no corpus-wide pass rate is asserted. The dataset remains byte-intact, but this snapshot is blocked for the proposed continuation. See [training readiness](CODEXA_V3_TRAINING_READINESS.md#continuation-launch-gate-failed-on-2026-11-05) for the frozen rubric, exact examples, evidence and limitations.

## Identity and counting

Artifacts live at `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814` (`CODEXA_V2_ROOT`). The volume is ext4 on `/dev/mapper/codexa-assets`, backed by the 512,110,190,592-byte HS-SSD-E100 SATA drive, mounted at `/mnt/codexa`, UUID `279352d0-0011-4733-8348-6815a4308196`.

The historical preparation target was **at least 2 billion newly acquired, retained training-token positions**. The 2026-11-06 instruction prioritizes quality over that count and authorizes only a 50M continuation, without a 5B expansion. The original working-pool cap was 2.2B; it is not a claim that 2.2B will survive. The old 884,183,601-position training corpus and the checkpoint's 500,006,912 consumed targets are separate counts. Deduplication excludes the entire old corpus, including its unused remainder. Neither repeated exposure nor held-out positions count toward the new training target. Approximate duplicate detection cannot establish semantic novelty.

The tokenizer is unchanged: 16,384 IDs, PAD=0, UNK=1, BOS=2, EOS=3; SHA256 `60a5e06447542b9462d458577f4a3056551c135e5aa1b8859d39624a3eee8771`. Every retained document reconstructs exactly from its encoded content. Packing is `[BOS, text, EOS]`, stored as little-endian uint16 with uint64 document offsets. Sequences still cross document boundaries under causal attention, as in the original objective.

## Sources and decisions

`configs/data/v2_sources.yaml` defines pinned candidates. The on-volume `source-registry.json` records exact file paths, byte counts, SHA256 values and attribution. Download receipts verify actual payloads. `reports/source-cards/manifest.json` checksums the pinned dataset cards; these declarations do not resolve every underlying content right or approve redistribution of a combined corpus or model.

| Source | Pinned revision | Declared rights and provenance | Decision |
| --- | --- | --- | --- |
| [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` | ODC-By-1.0; Common Crawl terms and underlying rights retained; original URLs preserved | Admit additional sample-10BT files, excluding the historical file, with integer educational score ≥4 and prose gates. |
| [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) | `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64` | CDLA-Sharing-1.0; GPT-4 synthetic stories by Eldan and Li | Admit GPT4 training file after verification. Official validation is protected and never tokenized for training. No overlapping older release/parquet copy is mixed in. |
| [English Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) | `b04c8d1ceb2f5cd4588862100d08de323dccfbaa` | Pinned card records CC-BY-SA-3.0/GFDL; preserve article ID, title and URL | Admit factual prose; remove reference appendices and reject stubs, disambiguation/list pages, table markup and incomplete endings. Many biographical articles remain; this is not a balanced school encyclopedia. |
| [Cosmopedia](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia), educational and stories | `0ae6ec63f91742bd2d1eaef4f02232c55d719385` | Apache-2.0 dataset declaration; Mixtral synthetic material with seed provenance | Inspected downloaded web, OpenStax-derived and children-story subsets are excluded from this mixture. Reasons below; the acquired evidence is retained. |
| [OpenWebMath](https://huggingface.co/datasets/open-web-math/open-web-math) | `fde8ef8de2300f5e778f56261843dab89f230815` | ODC-By-1.0; Common Crawl and underlying rights; URLs retained | Admit only a narrow readable educational-domain subset with prose/explanation requirements. Exclude the historical raw file. No claim of verified solutions or complete Grade 1–12 coverage. |
| [FineMath-4+](https://huggingface.co/datasets/HuggingFaceTB/finemath) | `e92b25a616738fe95dc186b64dfb19f9c8525594` | ODC-By-1.0; underlying content terms retained; original URLs preserved | Admit a restricted explanatory subset after inspecting actual samples. Score ≥4 alone did not remove navigation, code or extraction damage. |
| [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) | `9bb295ddab0e05d785b879661af7260fed5140fc` | ODC-By-1.0; Common Crawl and underlying rights | Candidate acquisition was paused with its partial preserved to prioritize admitted sources. No unverified partial is admitted. |

Cosmopedia screening found promotional web prose, forced science stories, truncated references and a damaged repeated string. One inspected OpenStax-derived geometric-mean unit falsely says swapping values can change the result. The exact counterexample and source ordinal are in `reports/cosmopedia-math-counterexample.json`: GM(1,4)=GM(4,1)=2; its four-number example also evaluates to 0.0554444 rather than its claimed 0.0608. This is evidence against that document, not a measured error rate for all Cosmopedia. The uncertainty and formulaic style make these subsets a poor fit for this phase's priorities.

The unfiltered OpenWebMath sample included software APIs, code, abstracts and fragmented forums. A stricter pilot retained only 106 of its first 10,000 records before overlap checks. The admitted policy requires six named educational domains, ≥80 English words, alphabetic fraction ≥0.5, an explanation cue, and no detected code/table markup or scrambled words. Dates and TeX rendering artifacts can remain. Downloading the whole collection merely to fill a quota is not justified by this yield.

## Mixture

The original hypothesis was 75% general/explanatory English, 15% narratives and 10% mathematics. Inspection led to an explicitly recorded collection hypothesis of **80% general (35% FineWeb-Edu, 45% Wikipedia), 15% TinyStories and 5% readable mathematics**. Excluding the inspected synthetic material and reducing noisy mathematics takes priority over filling percentages. `source-admission.json` records this decision and all restrictions; each admission revision is preserved under `reports/admission-*.json`.

These were collection ceilings; the final source-specific totals and characters/token are measured below. A future trainer samples windows from those actual arrays; it does not silently rebalance or repeat a short source.

The mathematics ceiling is split into 4.5% restricted FineMath-4+ and 0.5% readable OpenWebMath. The FineMath pilot retained 499/10,000 records under the final prose/domain gates before duplicate/evaluation checks. Recognized course-menu prefixes and Asymptote blocks are removed; search-index, revision-diff and broken website-template pages are quarantined. Advanced mathematical discussion, forum style and references to missing figures remain limitations. The assistant screened 100 deterministic final-policy excerpt pairs; this is not a full-document human correctness audit.

FineMath was added through an explicit append-only registry transaction. Existing entries/configuration were preserved; both registry versions and the recovery journal are retained under `reports/registries/`. Interrupted publication completes only from verified old/new snapshots. Unknown registry edits still fail preparation identity checks. This data-preparation extension does not bypass native training-resume fingerprints.

## Cleaning, overlap and evaluation protection

The versioned cleaner rejects corrupt Unicode, literal reserved tokens, oversized documents, heavy HTML, low-information fragments, excessive repeated sentences, detected OCR fragmentation, code-dominated prose and markup-dominated mathematics. It removes anchored navigation/contact lines without flattening useful paragraphs or mathematical notation. Source-admission gates add minimum prose, complete-ending, format and URL restrictions. These are transparent heuristics, not a learned grammar or factuality classifier. Names in legitimate public biographies are not automatically treated as unwanted personal information; the regex checks do not establish a PII-free corpus. A separate 100-excerpt review of one verified Wikipedia shard found **3 openings beginning with a bare “is”**, indicating missing subjects; see `reports/wiki-reviewed-sample.json`. A versioned Wikipedia gate now rejects that narrow opening pattern and the final snapshot reapplies it to all staging documents. This is a bounded sample observation, not a corpus-wide error rate.

A completed disk-backed reference index covers **760,819 historical documents**, including all old splits. Exact canonical-text hashes, 64-value MinHash signatures with 16 bands and verified Jaccard ≥0.85, and position-aware passage anchors check both old/new and new/new overlap. A passage rejection requires ≥128 identical word/punctuation tokens and ≥40 alphabetic words. Candidate limits quarantine expensive ambiguous cases. Non-discriminative anchors with >256 matches are skipped, so passage recall is approximate. Original text is compressed in the new index; old text is read from the untouched historical database.

Five-gram verification arrays are cached within 128 MiB/4,096 entries; full MinHash features are not recomputed for every candidate comparison. This optimization preserves the actual Jaccard calculation. Candidate-overflow rejections are reported separately from confirmed duplicates.

Official TinyStories validation documents are additionally indexed for exact, near-document and long-passage matching. A preliminary 13-gram protection scheme matched 678/1,000 stories in an **unverified download prefix**, often on formulaic openings. The revised scheme preserves old holdout/item protection and uses 64-token official-story spans plus the document/passage index. That exploratory prefix was not admitted as verified data. Both protection versions and the comparison are retained.

The benchmark's question wording and conservative lexical paraphrases/problem identities are screened separately. Ordinary prose teaching the same fact remains eligible; isolated answer words such as Paris are not banned. This is not an exhaustive semantic-contamination guarantee. Protected-span fingerprints and screening-code hashes are bound into immutable snapshot manifests. URL-group hashing assigns 99%/0.5%/0.5% train/validation/test before any model-based selection; near duplicates are removed globally, not separately within each split.

## Storage, integrity and reproducibility

Initial free space was approximately 139.68 GiB. The preparation root has an **84 GiB total ceiling** and a **40 GiB free-space floor**. After measuring 26.63 GiB root use, 19.42 GiB raw use and 113.05 GiB free, the raw ceiling was explicitly raised from 30 to 45 GiB to acquire more reviewed English and Wikipedia shards. The first projection was 82.25 GiB root use; see `reports/storage-plan-revision-45g.json`. At 51.72 GiB measured root use, 40.24 GiB raw use and 87.96 GiB free, completing the remaining pinned Wikipedia files projected 44.98 GiB raw use. A second recorded revision raised the raw ceiling to **47 GiB** for transfer headroom, with a bounded 83.48 GiB projected root use and 56.19 GiB projected free space; see `reports/storage-plan-revision-47g.json`. Every actual allocation still checks the UUID, root ceiling and free-space floor. Projections are budgets, not measured future usage.

The final `reports/storage-inventory-final.json` measured **69.369 GiB allocated** under the preparation root and **70.303 GiB free** on the verified volume. It counted hard-linked payloads once and agreed across two scans. Major components were 44.769 GiB pinned raw sources, 12.779 GiB disk-backed deduplication/protection indexes, 4.118 GiB working uint16 shards, 4.624 GiB immutable snapshots including the earlier stage, 2.781 GiB future-continuation bundle, and 0.251 GiB caches. Logs, reports, benchmarks and temporary files together used under 0.05 GiB. This measured state is under the 84 GiB root ceiling and 47 GiB raw ceiling, with 30.303 GiB of headroom above the free-space floor. No original dataset, checkpoint or unrelated file was deleted.

Every acquisition verifies filesystem UUID, free space, pinned size and SHA256. HTTP range recovery requires a valid 206/Content-Range response. Optional installed hf-xet transport is monitored; its retained payload is hard-linked rather than duplicated. No dependency pins changed. All HF, PyTorch, Python, CUDA/Triton, Ruff/mypy and temporary paths are directed under the preparation root through `env.sh`.

Preparation uses bounded CPU workers, SQLite transactions, a single-writer lock, per-file cursors and fsynced shards. Uncommitted database changes roll back; matching orphan shards can be recovered, while mismatched files cause failure. A finalization pass reconstructs every document, applies the final admission/protection policy, and writes a fresh immutable snapshot. It preserves earlier staging and snapshots. Failed finalization leaves an incomplete new directory for inspection; it cannot overwrite an old snapshot and must be restarted under a fresh name.

Reproduce or extend the existing preparation explicitly:

```sh
source /mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/env.sh
RUN=/mnt/codexa/pytorch/codexa-v3/pretraining/20261103-500m-233201
.venv/bin/python -B -m codexa.data.v2_catalog profile --root "$CODEXA_V2_ROOT"
# Select a fresh bounded batch after reviewing admission policy; no download occurs here.
.venv/bin/python -B -m codexa.data.v2_catalog select --root "$CODEXA_V2_ROOT" --limit-gib 47 --per-source 1 --batch "$CODEXA_V2_ROOT/next-batch.json"
.venv/bin/python -B -m codexa.data.v2_catalog acquire-batch --root "$CODEXA_V2_ROOT" --batch "$CODEXA_V2_ROOT/next-batch.json" --transport http
.venv/bin/python -B -m codexa.data.v2 prepare --root "$CODEXA_V2_ROOT" --historical "$RUN/curated/documents.sqlite" --tokenizer "$RUN/tokenizer/tokenizer.json" --target-tokens 2200000000 --workers 4
.venv/bin/python -B -m codexa.data.v2 verify --root "$CODEXA_V2_ROOT"
# Choose a new snapshot name each time.
.venv/bin/python -B -m codexa.data.v2_finalize --root "$CODEXA_V2_ROOT" --tokenizer "$RUN/tokenizer/tokenizer.json" --name reviewed-next
.venv/bin/python -B scripts/audit_v2_snapshot.py --root "$CODEXA_V2_ROOT" --snapshot "$CODEXA_V2_ROOT/snapshots/reviewed-next" --output "$CODEXA_V2_ROOT/reports/reviewed-next-token-audit.json"
.venv/bin/python -B scripts/inventory_v2_storage.py --root "$CODEXA_V2_ROOT" --output "$CODEXA_V2_ROOT/reports/storage-inventory-next.json"
```

Source discovery for a fresh root is separately callable as `python -m codexa.data.v2_catalog catalog --root <initialized-root> --config configs/data/v2_sources.yaml`. It retrieves metadata only and refuses to replace a different existing registry. Use the preserved registry for this preparation's exact file order. Increasing `--target-tokens` extends source cursors without replaying accepted documents; it does not create missing source material. Run one preparation/finalization writer at a time.

## Verified snapshot evidence

The selected final snapshot, `snapshots/reviewed-2b-20261105`, contains **2,108,050,326 new training positions**, **10,598,943 validation positions** and **10,464,092 test positions**. Its manifest SHA256 is `5cc849cc50680c86a50c40fe10aa6f56d0b0cfc77e1dcab5feb7ddba8c5dfddf`. Finalization reconstructed every one of the 3,349,425 working documents and reapplied the final admission and evaluation-protection rules. It rejected 59,257 documents, removing 62,750,477 training positions from the 2,170,800,803-position working pool. The retained training count exceeds the current 2B goal by 108,050,326 positions; the 2.2B working cap was never reached because the reviewed sources were exhausted or had reached their source quotas.

| Source in final snapshot | Training documents | Training positions | Share | Characters/token |
| --- | ---: | ---: | ---: | ---: |
| FineWeb-Edu | 702,073 | 736,800,702 | 34.952% | 4.343 |
| Wikipedia | 938,810 | 960,463,955 | 45.562% | 3.929 |
| TinyStories GPT-4 | 1,541,030 | 330,026,485 | 15.656% | 3.742 |
| Restricted FineMath-4+ | 74,573 | 80,367,192 | 3.812% | 3.004 |
| Restricted OpenWebMath | 592 | 391,992 | 0.019% | 3.060 |

The actual broad mixture is 80.514% general/factual English, 15.656% stories and 3.830% explanatory mathematics. Mathematics fell below its 5% collection ceiling because the inspected readable subsets yielded less than the quota; no noisy material was admitted to force the ratio. Characters/token is a fragmentation proxy, not a readability or factuality score. The final audit (`audit.json`) lists exclusions: 30,195 Wikipedia and 26,478 FineWeb-Edu documents failed the updated complete-prose gate; 1,770 Wikipedia documents began with the narrow missing-subject pattern. Small counts failed other admission or protected-question checks. The working preparation report separately records 11,111,493 source-record rejections before finalization, dominated by educational score, source-domain and stub/list filtering; these counts are not 11 million unique bad documents in the final snapshot.

The independent `reports/reviewed-2b-token-audit.json` rehashed every shard and inspected all uint16 IDs and document offsets. It found 3,257,078 train, 16,610 validation and 16,480 test documents; exactly 3,290,168 BOS and EOS IDs; zero PAD or UNK IDs; and 16,343 of 16,384 vocabulary IDs in use. This checks storage and tokenization compatibility, not whether every word is good English.

The first audited snapshot, `snapshots/stage-0001`, contains **107,211,957 training positions**, 593,408 validation positions and 541,758 test positions. Its manifest SHA256 is `3ee1cbcdc7cfef0f3cc3c7772f568ad07eb21c05ac607ecc5b7279ca58ea4233`. It is an early verified subset, not the final mixture. Both snapshots have their own audit and manifest.

| Source in stage-0001 | Training documents | Training positions | Characters/token |
| --- | ---: | ---: | ---: |
| FineWeb-Edu | 64,624 | 69,409,951 | 4.359 |
| Wikipedia | 46,667 | 37,410,014 | 3.895 |
| Readable OpenWebMath | 592 | 391,992 | 3.059 |

`reports/dataset-statistics.json` describes the larger working pool and includes rejection counts; it must not be substituted for the audited snapshot total. `reports/profiles/` contains before/after excerpts, seeded sampling and filter counts. Reviews inspected excerpts, not every full document; no ≥90% linguistic-quality percentage is asserted. Snapshot `audit.json` records final exclusions, source totals and full reconstruction checks. See [training readiness](CODEXA_V3_TRAINING_READINESS.md) for continuation and remaining quality gates.

## Rejection-only remediation and complete review

The original failed review remains at `reviews/continuation-50m-20261105`, including its 413 unscored items. R0085 contained decorative white TeX padding, R0195 merged exercise teasers and missing visual inputs, and R0202 repeated promotional navigation. `codexa/data/remediate.py` screens the complete retained corpus, quarantines whole records with source identities and reasons, and never rewrites a mathematical solution. Missing-figure detection was narrowed after inspecting false positives in explanatory articles; those development samples are preserved separately and excluded from approval panels.

Version 2 (`snapshots/remediated-v2-20261105`, manifest SHA256 `156bd2d07b95ddd45a136cb884d58aed178efcfeab90829f0ede7efb7a6f1fe3`) retains 2,077,136,357 training positions and quarantines 24,415 documents relative to the original snapshot. Its independent verification compares every retained document's IDs, source, split, order and complete token sequence against the parent, verifies the removed-record partition, and audits every uint16 ID and document boundary. No retained text was re-tokenized or altered. The exact byte-subsequence proof inherits the parent's tokenizer reconstruction and evaluation exclusions; it does not claim a new semantic-contamination audit.

A fresh source-blinded assistant review used seed 2026110502 and completed **all 500 ratings** before source identities were revealed. It excluded original review documents and matching 64-token passages. Actual native BOS/EOS markers were displayed to distinguish packed document boundaries from source corruption. This is assistant judgment, not independent human approval or mathematical fact checking.

| Panel | Usable | Severe | Decision |
| --- | ---: | ---: | --- |
| Random 1,024-token windows | 277/300 | 11/300 | Pass |
| FineMath-4+ | 35/40 | 1/40 | Pass |
| FineWeb-Edu | 36/40 | 0/40 | Pass |
| OpenWebMath | 33/40 | 2/40 | **Fail** |
| TinyStories GPT-4 | 40/40 | 0/40 | Pass |
| Wikipedia | 40/40 | 0/40 | Pass |

Unchanged acceptance requires at least 270/300 usable windows, at most 15 severe windows, at least 34/40 usable documents and at most two severe documents per source, and zero observed critical leaks/private identifiers. The frozen decision at `reviews/remediated-v2-fresh-20261105/decision.json` blocks training. Its ratings SHA256 is `92eb9cd929baacee3bc2d4e4d520ffe5905bc43706b03d6a9fa217633ed6ad7e`.

The seven OpenWebMath failures include three internally contradictory answers, absent molecular structure, repeated missing equations and two worked solutions obstructed by raw AsciiMath rendering commands. Readable language in other samples does not certify correct mathematics. Version 3 adds source-wide rejection of repeated raw white-padding commands, repeated missing equation blocks and explicitly absent molecular structures to both mathematics sources. Its seed 2026110503 and unchanged thresholds were registered before screening finished. Earlier snapshots, failed decisions and source records remain intact.

### Version 3: independently verified data, failed quality gate

The immutable `snapshots/remediated-v3-20261105` manifest is SHA256 `199183a06b130912d87712e4d857a4d706649909de59d4c55b665a89e83d2c85`. It contains **2,076,912,557 training**, **10,432,298 validation** and **10,335,976 test positions**, with 3,232,557 training documents and 3,265,425 documents across all splits. Version 3 quarantines another 328 whole records relative to version 2: 314 FineMath records and 14 OpenWebMath records. The two successive published candidates remove 24,743 documents and 31,137,769 training positions relative to the original reviewed-2b snapshot. Rejected text remains in the source database; quarantine ledgers retain document IDs, source revision/hash, URL or source ID, split and reason. No solution was rewritten.

| Source | Training documents | Training positions |
| --- | ---: | ---: |
| FineWeb-Edu | 701,601 | 735,296,863 |
| Wikipedia English | 918,780 | 933,682,880 |
| TinyStories GPT-4 | 1,541,030 | 330,026,485 |
| FineMath-4+ | 70,630 | 77,559,185 |
| OpenWebMath | 516 | 347,144 |

The loader samples the concatenated training stream by token-position windows; it does not apply a second source reweighting. Mathematics contributes 77,906,329 positions (about 3.75%), and stories about 15.89%. Do not describe this as a 10% mathematics mixture. Document boundaries remain BOS/EOS; native packed windows can cross those explicit boundaries. Validation/test membership and retained document order are unchanged. No prior training windows are deliberately replayed: the inherited historical-exclusion identity remains bound to the new manifest, subject to the original approximate deduplication limitations.

`independent-verification.json` compared all 3,265,425 retained documents' complete token sequences, IDs, source, split and order against version 2, verified the complete removed-record partition, and rehashed every shard and index. Its token audit found exactly 3,265,425 BOS and EOS IDs, zero PAD/UNK, and 16,343 used vocabulary IDs. Tokenizer SHA256 remains `60a5e06447542b9462d458577f4a3056551c135e5aa1b8859d39624a3eee8771`. These are compatibility/integrity measurements, not an approval of linguistic or factual quality. `reports/original-counterexamples-remediation-proof.json` separately proves that original R0085, R0195 and R0202, and the two other severe development counterexamples, were removed by general predicates rather than ID exceptions.

The fresh version-3 panel excluded documents and matching 64-token passages from earlier review/development panels. **All 500 records were read and rated before revealing source mappings.** Documents longer than 8,192 tokens received explicitly marked beginning/middle/end excerpt review. Ratings were frozen without replacements or corrections after unblinding.

| Panel | Usable | Severe | Decision |
| --- | ---: | ---: | --- |
| Random 1,024-token windows | 276/300 | 2/300 | Pass |
| FineMath-4+ | 35/40 | **3/40** | **Fail: maximum two severe** |
| FineWeb-Edu | **32/40** | 2/40 | **Fail: minimum 34 usable** |
| OpenWebMath | 37/40 | 2/40 | Pass |
| TinyStories GPT-4 | 40/40 | 0/40 | Pass |
| Wikipedia English | 40/40 | 0/40 | Pass |

Zero reviewed samples were labelled as critical protected-evaluation leaks/private identifiers. This is an observation in the panel, not exhaustive proof. A published AMC question appeared in one math sample; this project review does not certify independence from external AMC benchmarks. Historical article names and public organization addresses were not counted as private identifiers. Underlying copyright/noncommercial/reuse notices were observed in web text; local research preparation does not authorize redistribution or replace the recorded license limitations.

The frozen decision is `reviews/remediated-v3-fresh-20261105/decision.json`: **failed; training_allowed=false; unreviewed=0**. Ratings SHA256 is `2d5aa041359e4818b3df43d8eab0c1153bd2982420f266ba9085b7672d1d6f26`, protocol SHA256 `62bcc71acb3f864d1503ee92a598f9b02c19ef0bf611ade0091a3d019cba9df8`, rendering SHA256 `798558d32db134c090bf0823a360d2c1b1846cf5361cb19b0188d3d894d83511`. Reviewer identity is **source-blinded assistant, not independent human approval**. Small source panels vary between independent samples; version-2 versus version-3 counts are not a controlled estimate of quality improvement or regression.

The FineMath severe failures are R0025 (editor/timestamp clutter dominating a short solution), R0041 (flattened polynomial/fraction notation and absent workings across all three excerpts), and R0488 (a tagged-question catalogue containing truncated unrelated prompts). FineWeb-Edu failures R0083/R0177 are incomplete essay previews; R0136 is a definition list missing its terms; R0142 is a resource-search catalogue; R0172/R0267 promote absent teaching materials; R0179 is a contradictory/nonresponsive FAQ; R0253 has pervasive malformed English and missing factual context. They expose systemic gaps beyond white-TeX padding. Existing heuristics do not reliably detect these families. **The reviewed candidate is not approved for training.** Its failed records remain in the immutable candidate, with complete source evidence preserved; do not mistake a failure ledger for a rebuilt passing snapshot.

Further remediation must screen the entire relevant sources for interface/catalogue dominance, missing mathematical structure and incomplete prose, calibrate against valid worked examples, and produce another immutable snapshot plus a new complete unseen review. Semantic contradictions and poorly written prose require broader quality assessment than additional rendering regexes. Preserve the failed panels as development evidence and exclude their documents/passages from future approval panels; do not search seeds until one passes. Language usability labels do not certify correct facts: local scientific errors, nonidiomatic prose and fictional implausibilities were explicitly recorded even in usable samples.

Reproduce the rejection-only stage with fresh output names; these commands perform no model training. Frozen version-3 and version-4 code remains in the historical snapshots; the current module implements version 5. Semantic admission additionally requires an explicit completed census through `--semantic-admission`:

```sh
source /mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/env.sh
.venv/bin/python -B -m codexa.data.remediate --root "$CODEXA_V2_ROOT" --parent "$CODEXA_V2_ROOT/snapshots/remediated-v3-20261105" --output "$CODEXA_V2_ROOT/snapshots/remediated-current-REPEAT"
.venv/bin/python -B scripts/verify_remediated_snapshot.py --root "$CODEXA_V2_ROOT" --parent "$CODEXA_V2_ROOT/snapshots/remediated-v3-20261105" --snapshot "$CODEXA_V2_ROOT/snapshots/remediated-current-REPEAT" --output "$CODEXA_V2_ROOT/reports/remediated-current-REPEAT-proof.json"
.venv/bin/python -B scripts/review_continuation_corpus.py --help
```

The current storage ceiling prevents another conservative 5-GiB rebuild reservation without a revised bounded plan; do not run the reproduction blindly. `reports/storage-inventory-remediation.json` measures **81.377 GiB allocated** in the preparation root and **58.295 GiB free**, on the required UUID. Relative to the 69.369-GiB final preparation inventory, preserved remediation artifacts consume approximately **12.008 GiB more**; the earlier 8-GiB estimate was insufficient. Components now include 13.660 GiB snapshots, 5.561 GiB continuation bundles, 44.769 GiB raw sources, 12.779 GiB indexes and 4.118 GiB working tokens. Hard-linked blocks are counted once. The 84-GiB root ceiling, 47-GiB raw ceiling and 40-GiB free reserve remain enforced. No historical artifacts were deleted and no large downloads occurred during remediation.


## Version 4: complete stricter review failed

The successor `snapshots/remediated-v4c-20261106` has manifest SHA256 `a6ad911a943305a59b70447b02d13c2d436505962e0c117f3af41503e2d423a2`. It retains **2,052,040,276 training**, **10,335,047 validation** and **10,254,300 test positions**. Relative to v3, it quarantines **14,308 complete records**, including 14,185 training records and 24,872,281 training positions. Every rejected text remains in the preserved source database with source hash/revision, ordinal, split and reason in the quarantine ledger.

| Source | Training documents | Training positions |
| --- | ---: | ---: |
| FineWeb-Edu | 691,311 | 717,676,767 |
| Wikipedia English | 918,780 | 933,682,880 |
| TinyStories GPT-4 | 1,541,030 | 330,026,485 |
| FineMath-4+ | 66,776 | 70,349,059 |
| OpenWebMath | 475 | 305,085 |

The new checks address interface/timestamp residue, search catalogues, duplicated question indexes, absent definition/list labels, resource-only promotion, unfinished document endings, missing visual inputs, undefined numbered angles, duplicated or unbalanced math rendering and flattened powers. A small allowlisted AST parser constructs exact SymPy affine expressions without evaluating arbitrary corpus code; it identifies demonstrated inequivalent rewrites and answers that miss a stated point. Unsupported statements remain unchecked. Ordinary prose and mathematical notation are never reconstructed by guessing.

Calibration caught an overbroad multiline ellipsis rule before publication; the incomplete `remediated-v4-20261106` directory is explicitly invalid and preserved. The completed v4b screening stage remains separate evidence. The final v4c policy narrows incidental-download and arithmetic-context rejections, preserving legitimate sequences, optional practice links, modular arithmetic and corrected learner mistakes. Cache reuse requires unchanged screening predicates and symbolic helper hashes plus a verified AST narrowing; every cached rejection is screened again. This reuse does not relabel earlier quality reviews.

Independent verification compared all **3,251,117 retained documents** against v3 and proved an exact ordered subsequence with unchanged source/split assignments. Every shard/index was rehashed. The token audit found exactly 3,251,117 BOS and EOS IDs, zero PAD/UNK, and 16,341 vocabulary IDs in use. The frozen tokenizer hash remains unchanged. Exact subsequence inheritance preserves the previous deduplication and evaluation exclusions, including their documented approximation limits.

The `codexa-quality-4` protocol was recorded before sampling, using seeds 2026110601 and 2026110602. Primary and extended packets exclude earlier review/development document IDs and matching 64-token passages; the extended packet also excludes the primary packet. Every source document was supplied and assessed in full, including long documents. Two separate evaluation agents completed all 500 primary and 400 extended records with explicit notation, completeness, visual-input, correctness, education, coherence and contamination checks. The 1,800 original ratings and rationales remain separate and are conservatively combined. These are evaluation-agent judgments, not independent human approval.

| Primary panel | Usable under all applicable checks | Severe | Decision |
| --- | ---: | ---: | --- |
| Random 1,024-token windows | 265/300 | 38/300 | Fail |
| FineMath-4+ | 12/40 | 11/40 | Fail; 28 prohibited or unresolved records |
| FineWeb-Edu | 21/40 | 6/40 | Fail |
| OpenWebMath | 22/40 | 13/40 | Fail |
| TinyStories GPT-4 | 40/40 | 0/40 | Pass |
| Wikipedia English | 38/40 | 2/40 | Pass under existing numeric requirements |

The disjoint 200-document panels also failed: FineMath had 56/200 usable, 44 severe and 144 prohibited/unresolved records; FineWeb-Edu had 82/200 usable and 44 severe. Both decisions have zero unreviewed records and `training_allowed=false`. Structural integrity did not establish semantic quality. Failures include wrong mathematical transformations and conclusions, omitted required definitions/figures, unsupported instructional claims, extraction damage and incoherent or promotional material. Panel counts across versions are not a controlled quality comparison because both samples and the explicitly authorized rubric changed.

The primary decision is `reviews/remediated-v4c-fresh-20261106/decision.json`, with protocol SHA256 `fb30348ba09308f827959161b731002664ce88528d3acf8b20d31a7a149a6f2a`, ratings SHA256 `177f68f393ce5af31572b84ecf1db6a00bb2590a78667e83c682d58e2d47227b` and rendering SHA256 `63bc95dc63c0f618ee9d2d435ec24d38b1195815013f99692d910774d40450b4`. The extended decision is in its `extended` subdirectory, with protocol SHA256 `6290e682be046822ae3d909e62cc41e38152549af185d352e950718aa4da468c`, ratings SHA256 `af589473317e54ea4fe95e1d5bab36032fad769d273a8876245d1bdd53fbf145` and rendering SHA256 `2415138c8ad6ea4c5b3707cde172576a78a89deacb4c114af060da2be43e8bb4`. An additional mechanical audit re-encoded all 600 full documents and compared them to their actual retained BOS/content/EOS sequences. Neither those token proofs nor the passing other-source panels override a failed gate.

A separate read-only development audit rejected proposed aggressive upstream-score cutoffs without applying them. FineWeb-Edu score >=4.5 would retain only 5,187 training documents and 4,547,401 positions; FineMath >=4.75 would retain 4,127 documents and 5,870,034 positions, almost eliminating Math StackExchange material. These scores measure relevance rather than correctness and still retain some broken examples. The audit is preserved at `reports/high-quality-remediation-20261106/upstream-score-diversity-audit.json` on the dedicated volume. No source/genre removal was used to manufacture a passing result.

The post-build inventory measures **91,347,017,728 allocated bytes** in the preparation root and **128,470,224,896 free bytes** on the required volume. It counts hardlinked inodes once and agrees across two scans. The recorded ceiling is 104 GiB, with 47 GiB for raw data and the mandatory 40 GiB free reserve. Raw sources, historical decisions, tokenizer and baseline checkpoint remain preserved; no dataset download or real continuation occurred.

## Version 5: semantic admission in progress

Work resumed on 2026-11-09 after the requested pause. The completed development screen examined all 928,109 Wikipedia documents and flagged 7,611: 6,323 with repeated erased content slots, 145 with erased mathematical slots, 109 with contradictory birth fields, 79 with missing introduced content blocks, 954 with erased reaction relations and one contradictory sovereignty/locality claim. Explicit birth wording avoids confusing a parent's dates or later career dates with the subject's birth; first-name agreement avoids conflating different people. Additional valid Unicode and LaTeX reaction arrows are preserved. Replaying all 7,611 flagged records with that preservation change altered no decisions. A 20-record matched-context calibration is development evidence only: it establishes neither full-document usability nor an approval score. The future rebuild must rescreen all Wikipedia records with its captured policy source.

An additional evidence-resolution protocol was frozen while the original census remained incomplete, before selecting its targets or conducting its rereviews. After the entire original census is frozen, it selects every record whose only unresolved check is factual correctness, with no known defects or severe/critical flags. Two independent judges must reread each full document and establish the claims against primary evidence; unresolved cases remain unadmitted. `scripts/semantic_evidence.py` prepares that complete packet and composes actual new judgments with unchanged original judgments, preserving both evidence chains and requiring fresh token verification before finalization. This supports specialist educational diversity without defaulting uncertain claims to correct, waiving defects, resampling approval panels or changing acceptance thresholds.

The original census was completed and frozen on November 9: all 2,542 documents have two full assessments and zero unreviewed records. It admits 279 FineMath, 45 FineWeb-Edu and 60 OpenWebMath training documents, plus 40 held-out documents across these sources. This is insufficient FineWeb-Edu capacity for the disjoint 40- and 200-document approval panels. The decision SHA-256 is `81f9f2a69a0cedf3a7bf951d87f3de9407a8c93694d2f27dd22bdaa93ee952be`; its admission ledger is `2f1e8ebe1d13796c0bcc6537becacf94f4fa86c0557d70f211e1b8a306f45026`. All 70 supporting files are bound by inventory `eebeb819aaad8ca77701cdf123c93ee756a553653aaf3dec512914268c69e15b`.

The complete eligible evidence-resolution packet contains 278 documents: 261 FineWeb-Edu (244 training, 17 held out), 15 FineMath (13 training, two held out), and two training OpenWebMath records. Its packet-integrity SHA-256 is `500cdabadff730e03db3f09d09d28c9a4849e6f897f753022e5e61daafdd4da1`. All 278 full texts were independently re-encoded and matched against their original native BOS/content/EOS sequences in their original splits. Three fresh source-blinded agents completed all 556 reassessments under frozen balanced ownership. The evidence-resolution decision is `bf095acf700947937a75d44baf14907fb1daf9a69f8b20b9428371629aa09b7d`; its ledger is `a8171d7ecf10f4a064b9ce385f8e0798fe964410ec8cbe81b39c536b29616b3f`. Five author-written reference/rationale corrections were applied verbatim in separate effective copies after checking their hashes and unchanged decision fields. All original raw files, errata and assignments remain preserved in the 33-file supporting inventory `c01800edd9e049bc0b56e7c67587e59ab26fb97f5d71b6a486f62b0471dbd2c4`. Both new judges clear 12 additional FineWeb-Edu and two FineMath training records, plus one FineMath validation record. The resolved original census therefore admits 57 FineWeb-Edu, 281 FineMath and 60 OpenWebMath training documents, plus 41 held-out records. All 2,542 composite texts passed a fresh native-input comparison; the composed decision is `a5be2c18a910c9017aaeb9da9720f458625d11a7f4a3a6c409ef08e1ee17c735`. This remains curation evidence, without a new dataset snapshot or passing approval score.

During those rereviews, at least 80 distinct FineWeb-Edu training records failed to clear one reviewer, reducing the maximum possible training pool to 209, below the 240 required for the disjoint approval panels. All original reassessments subsequently finished. Before inspecting additional records, a separate expansion was preregistered with seed `2026110917`: 2,500 FineWeb-Edu and 100 FineMath training candidates, plus 25 validation and 25 test candidates from each source. It uses the same verified local parent and excludes every original census identity/passage and all six historical quality/development packets. There is no score, topic or genre cutoff, no download and no change to the approval thresholds or attempt-1 seeds.

All 2,700 expansion texts were matched to their native input sequences. Packet-integrity SHA-256 is `7b5ecfaa33b3d0a98246e9713d122c92b8a45d4bac74f492698d1140a841ab32`; the preinspection plan is `d1751aa10725d4d3ce18d41718e4d2cbeddb028a27dc11235932116f99c27da9`. Each candidate requires two full assessments, with primary evidence recorded for every new admission. The implementation joins only disjoint completed censuses of the same parent, preserves actual judgments and their evidence requirements, and requires a fresh combined token proof. These are curation records; none supplies a passing independent approval score.

The final fixed-assignment coverage observation is `semantic-admission/v5-extension-20261109/coverage-observation-20261109T220537Z.json` (SHA-256 `51357bf0a161549c3c3ed9f8bc4b245e8dd999c8b5e250e994de9dc7f12bb91a`). It records 5,308 of 5,400 assessments and leaves 92 judgments unreviewed after the automatic safety-review interruption. Assignment transfers for other previously unopened records are frozen through `judge-assignments-v14.json`; raw judgments and author corrections remain separate. This incomplete census has no admission decision and cannot authorize training.

The composition also preserves a separately checksummed supporting-evidence inventory, including raw shards, assignment history, transparent amendments, coverage records and exact calculations. These files are copied into the composite packet and bound by its integrity hashes without rewriting the original blinded packet. A changed supporting file makes composition fail before creating an output directory.

Preparation environments explicitly redirect pytest's cache as well as Python, model, compiler, Ruff and mypy caches onto the preparation volume. The cache-location check launches a bounded child pytest process, including a path containing spaces, and verifies the actual cache files. The repository cache written during the November 9 checks was preserved on `/mnt/codexa` when this missing setting was corrected.

`semantic-admission/v5-20261106/protocol.json` freezes a bounded candidate census before inspection: seed 2026110611, 1,000 training candidates per affected source and 50 candidates per held-out split where available. The source-blinded packet contains 2,542 complete FineMath, FineWeb-Edu and OpenWebMath records, excluding prior quality/development identities and passages. Every candidate requires two distinct judges, explicit check results, a specific rationale and topic/level/format annotations. Uncertainty remains explicit; no unreviewed candidate is considered passing. Curators may verify niche claims using primary references. This census selects a candidate corpus; its admission rates cannot substitute for independent approval scores.

The version-5 implementation admits affected-source records only when both assessments clear the frozen admission rules. Other affected records remain quarantined with rejected, unresolved or review-pending reasons and preserved source text. This will substantially change source composition; retained counts, educational diversity and sufficient unseen records for the 40+200 approval samples must be audited before any rebuild is approved. No topic, difficulty, forum format or upstream score is banned to improve a score. Broad checks for contradictory identity fields and erased mathematical slots also address observed Wikipedia extraction failures, while preserving qualified disputes and intentional fill-in exercises.

No version-5 snapshot or quality decision exists yet. The proposed positive semantic-census successor has an additional freshness blocker: every admitted affected-source document was inspected during curation, so a new reviewer cannot make an approval sample from those records previously unseen. Strict approval preparation refuses this snapshot type. A future approach must first produce enough systematically admitted, previously uninspected affected-source records and exclude all earlier approval and development records. Any successor also needs an independent token/subsequence audit and new primary/extended approval panels under the preregistered attempt 1 seeds 2026111601 and 2026111602. All original numeric gates remain mandatory. A failed or incomplete review cannot trigger the continuation launcher.
