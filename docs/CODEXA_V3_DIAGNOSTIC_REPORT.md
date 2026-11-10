# Codexa v3 diagnostic report

Date: 4 November 2026. Scope: the completed 248,687,488-parameter base model; **diagnosis only**. No training, fine-tuning, backward pass, optimizer update, supervisor restart, existing-model weight edit, or checkpoint rewrite was performed.

The report separates **confirmed observations**, **likely causes**, and **unverified hypotheses**. Forward tests can exclude specific defects and expose failures; they cannot establish the exact causal benefit of a different training corpus or schedule without a controlled training experiment. No such experiment was run.

## 1. Executive summary

Codexa has learned vocabulary, many local grammatical patterns, and some factual associations. It has **not learned reliable paragraph-level language generation or elementary factual completion**. Describing it as having learned no English would overlook its local abilities, but calling it conversational would substantially overstate the evidence.

The small candidate tests prefer correct grammar in 20/20 pairs and the expected fact in 16/20 four-choice sets (15/20 when the common terminal period is excluded), while free outputs remain unreliable. These are limited discrimination tests, not general competence scores.

The strongest explanation is an inadequately developed base model: 500,006,912 training tokens are only **2.01 tokens per parameter**, and held-out loss was still improving at completion. The run completed its chosen budget, rather than demonstrating language readiness. The corpus also contains extracted web scaffolding, repeated passages, technical material, tables, code, and damaged OCR. Those are real quality limitations, although the corpus contains substantial readable English and is not simply corrupt or predominantly mathematical.

Dialogue adds another problem: training was ordinary next-token pretraining, with no supervised assistant role, response targets, or conversational stopping convention. That explains some instruction and turn-taking failures. It does not explain away incorrect plain-text capital completions or loops in simple stories.

No training-invalidating attention, causal-mask, RoPE, residual, normalization, weight-tying, target-shift, tokenizer-identity, or KV-cache defect was found in the inspected path. Independent numerical references and actual checkpoint inference support that finding. This is bounded evidence, not a proof that all code paths are correct. Greedy decoding magnifies repetition; the tested sampling settings reduce it without restoring reliable language or factuality.

**Recommendation:** retain this checkpoint and the 249M architecture, improve and broaden the permitted language corpus, and plan a measured continuation with explicit compatibility and schedule handling. Reaching coherent base completion must precede conversational fine-tuning. Restarting is justified if a future audit finds a training-invalidating defect, a tokenizer replacement becomes necessary, or a controlled comparison demonstrates a clear advantage. Nothing in this investigation authorizes or starts that work.

## 2. Architecture and implementation audit

### Actual model and objective

The checkpoint contains 24 decoder blocks, width 896, 14 attention heads of dimension 64, SwiGLU intermediate width 2,432, a 16,384-token vocabulary, and a 2,048-token context. Input embeddings and the bias-free output projection share one matrix. Parameter counting and strict native reload give **248,687,488** unique parameters. The metadata field `context_target: 32768` is a roadmap target; this checkpoint was trained at **2,048**, with no RoPE scaling.

The audit read the actual files, rather than substituting a pretrained implementation:

| Component | Code inspected | Finding and independent check |
| --- | --- | --- |
| Q/K/V and attention | `codexa/architecture/attention.py:22–57` | Packed projection is split into Q, K and V with the correct head axes. Scale is the default `1/sqrt(64)`. A separate float64 implementation used linear projection, complex-number RoPE, explicit triangular masking and softmax. Maximum output difference: **1.98e-7**. |
| Causal masking | Same file, lines 49–55 | Full prefill uses causal SDPA. Cached chunks use a lower-right causal mask over the existing prefix plus current chunk. Altering the future half of an actual-checkpoint sequence changed earlier logits by **0** in both tested precisions. |
| RoPE | `codexa/architecture/rope.py:17–26` | Adjacent coordinate pairs, base 10,000, factor 1, correct cache position offset. Angles and rotations run in FP32. Independent float64 reference at offset 123: max error **1.09e-5**, consistent with FP32 angle arithmetic. |
| Residuals / RMSNorm | `codexa/architecture/transformer.py`; `normalization.py` | Pre-norm attention residual followed by pre-norm SwiGLU residual. RMS reduction is FP32 and weights start at one. RMSNorm reference error **4.44e-7**. No missing residual, softmax-on-logits loss, or inappropriate batch normalization was found. |
| Initialization / tying | `codexa/architecture/model.py:22–38` | Shared embedding initialized once, linear std 0.02, attention output and FF down projections scaled by `sqrt(2L)`. Native embedding/head tensors are equal; loaded modules share the same parameter object. No zero embedding rows. |
| Next-token targets | `codexa/data/production.py:621–660`; `packing.py`; `training/trainer.py:26–27,285–313` | A sequence-plus-one window becomes `x=t[:-1]`, `y=t[1:]`. Loss is mean FP32-logit cross entropy, divided by accumulation 16 before each historical backward pass. Independent manual log-sum-exp CE error **2.77e-9**. Actual packed documents also match frozen-tokenizer output. |
| Optimizer and accounting | `training/optimizer.py`; `scheduler.py`; `trainer.py` | Historical AdamW betas (0.9,0.95), eps 1e-8, matrix-only decay 0.1, norm clip 1. Logs account for 16 × 2,048 target tokens per update. Optimizer state was inspected as tensors only; no optimizer was constructed. |
| KV cache | `architecture/model.py:57–116`; `attention.py:30–35` | Cache owner/capacity/validity checks, append at the correct offset, prefix slicing, one length increment after all layers. Single-token tails and multi-token chunks were compared with full forward at lengths 1,2,7,31,127,511,1023,2047. Results below. |
| Sampling / termination | `inference/generate.py` | Evaluation mode is set and restored; cached and uncached paths use the last next-token logits; temperature, top-k and nucleus filtering are applied before sampling. EOS ID 3 terminates only when newly generated. Context/budget violations raise rather than silently truncate. No assistant-turn stop or repetition penalty is configured. These are generation-policy limitations, not evidence of a faulty next-token implementation. |

For the SDPA interpretation, the inspected behavior agrees with the installed-version [PyTorch 2.11 SDPA documentation](https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.scaled_dot_product_attention.html) and [lower-right causal bias documentation](https://docs.pytorch.org/docs/2.11/generated/torch.nn.attention.bias.causal_lower_right.html).

### Numerical and cache controls

All 16 FP32 cache logit comparisons met `atol=rtol=1e-4`; the largest absolute difference was **5.72e-6**, with maximum measured KL about **1.01e-7**. All 16 BF16 comparisons retained the same top-1 token at those fixed contexts, but differences reached **0.0625** and KL **4.32e-5**. The ordinary backend was also compared with explicit math SDPA at four context lengths per precision.

Twelve actual language prompts were also generated for 128 tokens with cached and uncached FP32/BF16 greedy paths. **FP32 token sequences matched in 12/12 cases**, while BF16 matched in 10/12. BF16 divergence first appeared at generated indices 12 and 17 (zero-based) for the girl-story and France prompts. The France error and repetitive behavior also occur in uncached FP32. These observations support normal precision/kernel sensitivity in the tested paths rather than a material cache failure; they do not establish universal BF16 token parity. Complete IDs and texts are preserved in `results/generation_cache_controls.json`.

The parameter tensors and stored optimizer tensors are finite. Embedding row norms range **0.542–1.571**, median **0.882**. One 64-token activation inspection found finite residual RMS values increasing from embedding RMS **0.0263** to final-block output RMS **6.57**. Attention entropy varied by layer rather than every head being uniformly saturated. These are useful sanity observations, not a calibrated test proving optimal activation scales or healthy gradients everywhere.

**Confirmed implementation bugs affecting this checkpoint: none found.** Remaining limitations include the 2,048-token context, no trained conversational protocol, and limited degeneracy controls. Untested context-extension, externally manipulated caches, and all possible input shapes remain outside this audit. The first CPU diagnostic harness failed during Dynamo registration because the safety monkeypatches were installed too early; its traceback is retained. Import registration was moved before distinct guarded callables were installed. This was a diagnostic-harness issue, not a model defect, and no optimizer/backward operation occurred.

## 3. Tokenizer assessment

The byte-level BPE has exactly 16,384 IDs: PAD=0, UNK=1, BOS=2, EOS=3. It uses the full byte alphabet, `add_prefix_space=False`, and no automatic BOS/EOS insertion in `encode`. Training explicitly packs `[BOS, encode(document), EOS]`; checkpoint inference explicitly prefixes `[BOS, encode(prompt)]`. It correctly does not append EOS to an unfinished inference prompt.

The embedded tokenizer is byte-identical to the frozen training tokenizer, SHA256 `60a5e06447542b9462d458577f4a3056551c135e5aa1b8859d39624a3eee8771`. Sixty actual packed documents across train/validation/test were compared token-for-token to this convention. Another 120 source-stratified texts round-tripped exactly. No test-set model scoring was performed; checking byte reconstruction does not expose test answers to training.

In the 2,000-document stratified corpus sample, **zero complete-text round-trip failures and zero UNK tokens** were observed. The full training stream contains **zero PAD and zero UNK**. BOS/EOS each occur 759,754 times in prepared data; consumed targets contain 429,823 BOS and 429,807 EOS. Only **18 of 16,384 IDs** are unused, including PAD/UNK; this is not a mostly dead vocabulary. Whole-corpus characters/token are **4.247** for FineWeb-Edu and **3.390** for OpenWebMath, including boundary-token overhead in the denominator. These are measured source-specific fertility summaries, not a comparison with an untested replacement tokenizer.

The tokenizer fitting record shows a 33,554,341-byte training-only sample, 7,025 documents (6,536 FineWeb-Edu, 489 OpenWebMath), with its sample hash preserved in `tokenizer/identity.json`. This is much smaller than the expanded corpus but is not itself proof of a defective vocabulary. A 16K vocabulary spends more tokens on some uncommon words and notation than a larger vocabulary might; no measured evidence warrants replacing it and invalidating checkpoint compatibility.

Literal strings such as `<bos>` and `<eos>` are recognized as control tokens even with automatic special insertion disabled. Default decoding skips them, so literal-control-string round trips are an intentional exception. The training corpus was configured to reject those strings. Single vocabulary pieces can decode to incomplete UTF-8 fragments; the relevant fidelity test is complete encoded text, not isolated byte-piece display. Punctuation, whitespace, CR/LF, tabs, accented text, Greek, Chinese, emoji and mathematical notation were probed explicitly.

**Assessment:** no ordinary-English reconstruction failure or train/inference tokenization mismatch was found. The vocabulary is a possible efficiency constraint, not a supported primary root cause.

## 4. Training dataset quality and diversity

### Provenance, split protection and real exposure

The expanded run used one local pinned Parquet file per source: FineWeb-Edu revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`, `sample/10BT/000_00000.parquet`; OpenWebMath revision `fde8ef8de2300f5e778f56261843dab89f230815`, `train-00000-of-00114-5a023365406cb9c4.parquet`. These are finite samples of larger collections. The receipt `text_bytes` fields describe the earlier pilot sampling budget; the expanded extraction actually used **3,464,805,971** FineWeb bytes and **421,552,125** OpenWebMath bytes before curation. They should not be mistaken for the smaller pilot payloads.

Local source records declare ODC-By-1.0, Common Crawl terms and retained underlying-content rights. This audit verifies recorded provenance and hashes; it does not grant new redistribution rights. No dataset was downloaded. The curated SQLite database SHA256 is `f9f9ec69be4368fdf0ea6ae743a5f22c7377c680d36e2b96a0d94fdf021da4e6`; its hash and all six input JSONL hashes were independently rechecked. All token/index shard hashes and document boundaries passed verification.

The training corpus contains **759,754 documents and 884,183,601 token positions**. Frozen validation has 541 documents / 606,854 tokens; test has 524 / 625,781. The completed run started afresh rather than adding the older 99.32M pilot to its count: `checks/pilot-compatibility.json` records changed data fingerprints and schedule as reasons. Thus the evaluated weights have 500.007M training tokens, not 599M cumulative training from that pilot.

| Source | Documents | Prepared tokens | Actual consumed target tokens | Consumed share |
| --- | ---: | ---: | ---: | ---: |
| fineweb-edu | 709,044 | 780,368,434 | 441,408,641 | 88.28% |
| openwebmath | 50,710 | 103,815,167 | 58,598,271 | 11.72% |

OpenWebMath is **11.72% of consumed tokens**, so an explanation based on mathematical-source dominance is contradicted by the actual mixture. FineWeb-Edu supplies 88.28%.

Replaying the exact seed-42 Feistel sampler reaches the checkpoint state `epoch=0, position=244144, size=431730`. All **244,144 consumed windows are distinct**, covering **56.55%** of available training windows. Their target positions sum to exactly **500,006,912**. This disproves a repeated-full-epoch explanation. It does not imply 500M semantically novel tokens: different documents can repeat content, and common words naturally recur.

At least one target position was consumed from **514,189 documents**; all token positions were targeted in **342,628**. These counts refer to loss targets, not a guarantee of complete document context. There remain **384,176,128 unused target positions** in the full-window training space, plus 560 trailing positions outside that space. Only **0.0869%** of window inputs start with BOS, although many more contain a document start internally.

All retained training IDs were mapped back to an original source URL. The corpus spans **349,523 distinct URL netloc strings**, with the top ten accounting for **2.759% of documents**. Netloc includes subdomains/ports and is not a count of independent publishers. Domain breadth is substantial; a few domains dominating the entire corpus is **not** supported. Subject/reading-level balance and semantically novel coverage remain different questions.

### Readability and contamination

The audit streamed every training document for SHA identity, source totals and explicit pattern counts. A deterministic bottom-hash sample of **2,000** documents, stratified by two sources and three length buckets, retains full text and measurements. This is a deliberately balanced diagnostic sample, not a token-weighted random sample of the mixture. Eighteen further fixed examples were inspected manually by the assistant: entire short texts, and beginning/middle/end snippets of longer ones. Those judgments are descriptive; they are not independent human quality labels or a corpus-wide quality percentage.

Actual retained examples include:

| Example ID prefix / source | Observation |
| --- | --- |
| `00005e384547` / FineWeb | Coherent opinion article about science education. |
| `00001a17dc14` / FineWeb | Readable household methodology, definitions and lists. |
| `000057b4f44d` / FineWeb | Damaged OCR: interleaved columns and fragments such as “Grimm's Grammar on the theory of of comparison.” Ordinary syntax is impaired. |
| `000200bdba38` / FineWeb | Readable medical definition followed by editorial and promotional metadata. |
| `00003ed80774` / FineWeb | Book review/retail listing, flattened headings and product details. |
| `000435434765` / OpenWebMath | Readable worked marble-probability solution; relevant to school mathematics. |
| `000a652b2c05` / OpenWebMath | Mathematical search-results table with little continuous prose. |
| `0007bdb11a76` / OpenWebMath | Bibliographic author index and navigation. |
| `00008903e02f` / OpenWebMath | TeX programming Q&A dominated by macros and comments. |
| `001ffd627a47` / OpenWebMath | Coherent graph-algorithm lesson that refers to missing figures. |

| Proxy (document-level presence) | FineWeb-Edu | OpenWebMath |
| --- | ---: | ---: |
| Metadata keyword | 45,824 (6.46%) | 4,816 (9.50%) |
| URL | 57,710 (8.14%) | 7,770 (15.32%) |
| HTML tag pattern | 512 (0.07%) | 290 (0.57%) |
| Math / LaTeX | 666 (0.09%) | 28,425 (56.05%) |
| Code / TeX source | 720 (0.10%) | 3,039 (5.99%) |


Metadata-keyword, URL and math/LaTeX matches are **mechanical proxies**, not synonymous with bad English. A copyright footer can coexist with an excellent article; valid equations and code are useful in their intended mixture. Conversely, OCR damage can evade every listed proxy. The corpus is suitable as one input to language pretraining after better quality controls; the observed extraction faults and lack of a deliberate simple-language curriculum make it a poor sole guarantee of small-model coherence. Mathematical-source membership is not proof of Grade 1–12 coverage, and general-source membership is not proof of elementary factual coverage.

### Deduplication and boundaries

Canonical whitespace-normalized document hashes are valid for all training rows, and the primary-key identity removes exact canonical duplicate documents. The curation manifest records 2,223 duplicate rejections, 17,673 holdout-overlap rejections, 581 corruption rejections, three extreme-repetition rejections and three reserved-token rejections. These counters are pipeline categories, not a new estimate of all defective material.

The actual quality filter rejects long documents with fewer than eight unique whitespace words; it does not meaningfully constrain repeated sentences inside otherwise varied text. Near-duplicate removal uses five-token shingles on documents with at least 128 word/punctuation tokens, 64 minhash values and 16 bands, then a Jaccard threshold of 0.85. It is approximate and document-level. It does not remove every repeated passage or short near-duplicate. Protected 13-token spans and frozen URL groups help split separation; complete paraphrase or semantic leakage has not been proven absent.

A new diagnostic index measured exact whitespace-normalized newline-delimited units of at least 40 characters across the full training corpus: **13,306,181 distinct units**, **13,591,335 occurrences**. **82,435 units** appear in multiple documents. Their occurrences occupy **1.276%** of all audited-unit characters; extra copies beyond the first occupy **1.098%**. This is a character-based exact-line measure, excluding short lines and arbitrary partial/semantic matches; it is not an exact corpus-wide unique-token count.

Examples include the zbMATH site heading in **494** documents and a login-to-comment notice in **488**. Repeated website material is confirmed, but the measured exact-line copy rate is modest and does not show that the corpus is mostly duplicates. It cannot by itself account quantitatively for the far higher repetition in free generation.

Documents are packed as BOS/text/EOS in canonical SHA order, then sampled in nonoverlapping 2,048-target windows. There is no padding loss, double shifting or missing EOS in the verified arrays. Attention can cross document boundaries by design, and a sampled window may begin inside a document. That can teach abrupt transitions and web-page scaffolding, but no controlled result establishes it as the dominant failure. A future document-aware packing comparison would be an ablation, not a repair already justified here.

![Source mixture, quality proxies and vocabulary usage](/mnt/codexa/pytorch/codexa-v3/diagnostics/20261104-120458/figures/corpus_diagnostics.png)

## 5. Training convergence and overfitting

All **15,259** metric rows and step events were parsed. Steps are contiguous; every token count equals `step × 32768`; metric loss/LR values agree with events; losses, validation losses and gradient norms are finite. Two historical sessions consist of the initial short acceptance segment, a pause at step 10, and continuation to completion. There is no evidence of omitted updates, a fresh accidental restart in the second session, or repeated-epoch sampling.

| Quantity | Measured value |
| --- | --- |
| Initial training loss | 9.90224 |
| Final update loss / last-60 mean | 3.14055 / 3.11748 |
| Final monitored validation slice | 3.005710 over 65,536 tokens |
| Best monitored validation | 3.005181 at step 15,000 |
| Complete validation windows | 3.048003; perplexity 21.0732 over 606,208 tokens |
| Validation near 403M tokens | 3.052895 on the monitored slice |
| Fitted final-100M validation slope | −0.05088 nats/token per additional 100M training tokens |
| LR | 1.5e-7 initially; peak 3e-4; final 3e-5 |
| Warmup | 2,000 updates / 65,536,000 tokens (13.1% of run) |
| Maximum pre-clip gradient norm | 33.6679 |
| Peak allocated training CUDA memory | 7,534,522,368 bytes / 7.017 GiB |
| Historical end-to-end rate | 23,054.9 tokens/s |
| Post-warmup update-only rate | 24,023.4 tokens/s |
| Supervisor / event-interval wall time | 21,687.666 / 21,680.446 seconds |

The LR recorded for every update matches the actual warmup/cosine formula exactly. The logged value is taken **before** the optimizer step, so this is not an apparent off-by-one caused by logging the next LR. LR warmup spends a sizable fraction of this small run reaching its peak; this is a possible efficiency choice to revisit, not a proven cause of the failure. It was not changed.

Gradient clipping is common early: all first 100 updates, 98.22% of updates 101–1,000, and 16.1% of 1,001–2,000. It falls to 0.0794% over 14,001–15,259. Occasional large spikes are real, but they do not coincide with a sustained loss blow-up. There is no evidence here of persistent numerical divergence or a silent nonfinite optimizer state.

A new forward-only BF16 evaluation measured:

| Slice | Windows / target tokens | Mean loss | Window-loss SD |
| --- | ---: | ---: | ---: |
| train_seen | 128 / 262,144 | 3.059353 | 0.3368 |
| train_unseen | 128 / 262,144 | 3.122532 | 0.3480 |
| validation_all | 296 / 606,208 | 3.048003 | 0.3638 |

Seen and unused training windows are sampled without replacement with seed 42 at the same sequence length. Seen windows have a **0.0632-nat** lower mean loss, a small difference relative to window variability (rough independent-window standard error of the difference ≈0.0428; document dependence and difficulty are not controlled). This is compatible with some exposure advantage and does not establish severe overfitting. Full validation independently reproduces the earlier 3.048003 result.

Source-stratified document prefixes (at most 512 targets, minimum 32) give:

| Source / split | Documents | Target tokens | Loss |
| --- | ---: | ---: | ---: |
| fineweb-edu / train | 96 | 42,260 | 3.1246 |
| openwebmath / train | 96 | 40,158 | 2.7274 |
| fineweb-edu / val | 96 | 42,921 | 3.1573 |
| openwebmath / val | 46 | 20,491 | 2.8200 |

These prefixes are the first eligible canonical-SHA records per source/split, not full-document or token-weighted mixture estimates. OpenWebMath prefixes have lower loss than general-source prefixes in this sample. Math-proxy validation prefixes measure 2.6057 versus 3.1567 without that proxy, but the groups differ in source and content. Metadata-keyword validation prefixes actually have lower loss (2.9107 vs 3.0640), demonstrating why low perplexity is not a quality label. All group sizes and per-document values are retained; no causal quality effect is inferred from these confounded slices.

The complete-validation loss is different from the 32-window monitoring slice because it evaluates more of the same frozen validation corpus; it is not a regression between two checkpoints. Comparing final training minibatches with a different fixed validation slice also does not produce a controlled generalization gap. The matched seen/unseen slices above address part of that uncertainty; domain and document difficulty still vary.

![Historical loss, LR, gradients and throughput](/mnt/codexa/pytorch/codexa-v3/diagnostics/20261104-120458/figures/training_diagnostics.png)

**Overfitting conclusion:** no broad overfitting signal is established. Validation improved markedly and remained on a declining trend; the tiny final difference from the best monitored point is insufficient evidence of meaningful regression. Selected training windows had not cycled. Some memorization and retained passage duplication remain possible, but “the model looped because it saw the entire corpus many times” is contradicted by the sampler replay.

Continuing through the **unseen remainder** of this corpus is scientifically defensible if separately authorized and evaluated, because there is still learning signal. Indefinitely replaying the same corpus is not justified by these measurements, and cannot supply billions of new unique positions. Quality repair, new diverse language data and explicit continuation scheduling should accompany the next phase. No training was executed to test that recommendation.

## 6. Language and conversation evaluation

### Protocol and limits

The checkpoint was strictly loaded and frozen, with `eval()`, `requires_grad_(False)` and inference/no-grad execution. Diagnostic guards reject backward/autograd, optimizer construction/steps, trainer train/resume/checkpoint entry points and `torch.save`. In-memory model state is hashed before/after each GPU stage; the original checkpoint is hashed independently. Seeds, token IDs and every actual output are retained.

The frozen fixture defines 80 base prompts: 20 sentence completions, 20 paragraph starts, 20 grammar prefixes and 20 elementary factual associations. Each has a 64-token completion under FP32 greedy, BF16 greedy and BF16 sampling with temperature 0.7, top-k 40, top-p 0.9, seeds 42/123/2026. The same 20 paragraph starts also receive 256-token budgets in all five settings. Twenty balanced base prompts additionally use Q/A and User/Assistant formats in greedy and seed-42 sampled BF16. Twelve dialogue scenarios run two turns using each **actual generated first reply** as history, in Q/A, User/Assistant and two-example few-shot Q/A formats, greedy and sampled. Total: **724 generated outputs**, plus separate cache and degeneration-trace controls.

These are small, authored diagnostic probes, not an independently held-out benchmark. Some prefixes are intentionally ambiguous; e.g. “The cat” can validly continue as “The catalogue.” “Humans breathe air through their” can validly continue “nose and mouth,” even though the candidate test lists lungs. We therefore do not present a naive keyword-hit score as factual accuracy. Grammar candidate ranking and factual multiple choice are separate tests from unrestricted generation. Seed variants are repeated observations of the same prompts, not independent new factual questions.

### Controlled quantitative results

A repeated token-4-gram fraction is `1 − unique_4grams / all_4grams`. The strict loop detector requires at least three adjacent identical blocks, period 1–16 tokens, totaling at least 24 tokens. It misses some obvious longer or interleaved sentence loops; its counts are a lower bound on degeneration, not a coherence score.

| Setting (20 matched paragraphs each) | 64-token repeated 4-grams | 256-token repeated 4-grams | Strict loops: 64 / 256 | EOS: 64 / 256 |
| --- | ---: | ---: | ---: | ---: |
| FP32 greedy | 67.87% | 92.25% | 9/20 / 11/20 | 0/20 / 0/20 |
| BF16 greedy | 66.39% | 91.90% | 9/20 / 12/20 | 0/20 / 0/20 |
| BF16 sample, seed 42 | 16.97% | 62.45% | 1/20 / 6/20 | 0/20 / 0/20 |
| BF16 sample, seed 123 | 13.11% | 61.96% | 0/20 / 10/20 | 0/20 / 0/20 |
| BF16 sample, seed 2026 | 10.90% | 51.34% | 0/20 / 8/20 | 0/20 / 2/20 |

A paired bootstrap over the **20 prompt identities**, averaging the three sampled seeds within each prompt and resampling 5,000 times with seed 42, estimates a short-to-long repetition increase of **44.92 percentage points**, descriptive 95% interval **40.92–49.03**. For FP32 greedy the increase is **24.38 points**, interval **20.15–28.61**. These intervals describe variation within the authored prompt set, not uncertainty over all English tasks. Sampling looks substantially better over the first 64 tokens than over 256.

Candidate likelihood gives **20/20 grammatical alternatives preferred by summed continuation NLL**, or **19/20 by mean token NLL**. The disagreement is “They walk to school.” versus “They walks to school.”: the ungrammatical form splits into more BPE tokens, illustrating length-normalization bias. All candidate concatenations were checked against encoding the complete prompt-plus-completion text.

For 20 four-choice elementary factual associations, the expected answer wins **16/20 by summed NLL** and **14/20 by mean NLL**. Summed-NLL failures are sky color (red preferred to blue by a very small margin), frozen water (sand preferred to ice), two plus two (three preferred to four), and baby cat (puppy preferred to kitten). The foils are often easy and the sample is small; 80% here is not general factual accuracy.

Importantly, the France probe ranks “ Paris.” above the three supplied foils (summed NLL **5.82**, versus **7.33** for “ London.”), while unconstrained greedy decoding begins “the city of Bologna.” The model therefore contains some usable associations but does not select or sustain them reliably over the full continuation space. Candidate discrimination, local grammar and free-running coherence are different capabilities.

As a same-tokenizer descriptive control, the first 256 text tokens of eligible stratified source samples have mean repeated-token-4-gram fractions **2.33%** (FineWeb, n=670) and **7.68%** (OpenWebMath, n=737). This is not a prompt/domain-matched benchmark, but it shows that the much higher generated repetition is not simply the typical token repetition level of these source excerpts.

Assistant review of all **20 full 256-token seed-42 sampled paragraph continuations** found **0/20** meeting the stated topic/entity/progression/no-dominant-loop rubric. This is a transparent, non-blinded qualitative screen, not an independent human-rated benchmark. Per-example reasons, including partially fluent passages, are preserved in `results/manual_generation_review.json`.

![Matched short and long paragraph generation](/mnt/codexa/pytorch/codexa-v3/diagnostics/20261104-120458/figures/generation_diagnostics.png)

### Actual continuations and local language ability

The following excerpts are literal model outputs with line breaks flattened; their factual claims are not endorsed. Full responses, exact prompt text, IDs, precision, seed, EOS status and timing are in `raw/language_outputs.jsonl`.

| Prompt / setting | Actual observed excerpt | Implication |
| --- | --- | --- |
| `The capital of France is` / FP32 greedy | “the city of Bologna. The city is the capital of the Roman Catholic Church.” | Incorrect association in ordinary completion, independent of chat formatting. |
| `The largest planet in our solar system is` / FP32 greedy | “the Sun. The Sun is the largest planet in our solar system.” | Basic factual error followed by self-reinforcement. |
| `Water freezes at zero degrees` / FP32 greedy | “Celsius, and the water is pumped out of the tank.” | A correct local association can coexist with unrelated/repetitive continuation. |
| `The baby was tired and began to` / FP32 greedy | “cry. The baby was crying for a long time.” | Plausible first completion, followed by repeated copies of the second sentence. |
| Lena bicycle paragraph / FP32 greedy | “was a little girl, but she was afraid of her father.” | Loses the antecedent “Her father” and changes the scene. |
| Cat-search paragraph / sampled seed 42, 256 tokens | “we found her. We found her at the cemetery in the Trenton-Wilson area. Her name was C. G. Gray.” | Sampling avoids the simplest greedy loop but drifts into an incompatible biography. |
| Maria tomato paragraph / sampled seed 42, 256 tokens | “When she was about six months old, she was told that she would be able to grow tomatoes in her garden.” | Locally parseable clauses are insufficient for sensible narrative continuity. |

Many generated clauses have recognizable English syntax. The failures are often semantic inconsistency, missing event progress, repetitive paraphrase and invented facts, rather than exclusively broken word order. Improving only a chat template would leave those failures intact.

### Prompt and conversation sensitivity

| Same 20 completion prompts | BF16 greedy: repeated 4-grams / strict loops | BF16 sampled seed 42: repeated 4-grams / strict loops |
| --- | ---: | ---: |
| plain | 60.41% / 11/20 | 24.75% / 2/20 |
| qa | 72.13% / 13/20 | 54.75% / 7/20 |
| chat | 75.57% / 15/20 | 36.80% / 6/20 |

The wrappers include a completion instruction, so their differences measure formatting plus task phrasing, not a pure isolated role-token intervention. Neither tested wrapper consistently improves stability.

| Actual second turns (12 scenarios per cell) | Greedy repeated 4-grams / strict loops | Sampled seed 42 repeated 4-grams / strict loops |
| --- | ---: | ---: |
| qa | 77.32% / 8/12 | 63.93% / 3/12 |
| chat | 78.14% / 11/12 | 64.48% / 3/12 |
| fewshot | 67.49% / 3/12 | 66.67% / 6/12 |

For the exact-word instruction, **0/6** first-turn outputs equal `blue` and **0/6** second-turn outputs equal `green` after trimming surrounding whitespace, across the three formats and two settings. That is one instruction scenario, not six independent tasks. The Alice-memory User/Assistant greedy reply repeatedly says “My name is Alice. What is my name?”; it contains the name but fails to answer as the assistant. Keyword occurrence would misleadingly count this as memory success. The sampled reply instead loops on being a teacher. The Japan follow-up repeats the question greedily or produces unrelated US-president text with sampling. Few-shot formatting does not establish reliable multi-turn behavior in the preserved outputs.

### EOS and free-running versus teacher-forced behavior

At natural ends of 32 fixed validation documents, mean EOS probability is **0.1026**, median **0.0442**; EOS is top-1 in 3/32 and top-10 in 27/32. At within-document prefixes of the same documents, mean is **0.00227**, median **6.03e-6**, top-1 in 0/32. End contexts longer than the limit use the trailing 2,047 text tokens with BOS; that truncation is recorded. EOS is therefore learned and reachable, rather than demonstrably the wrong token ID. A 64/256-token cap can precede a natural document ending, so failure to emit EOS in a short completion does not alone prove broken stopping.

Perplexity 21.07 evaluates next-token probabilities under **real preceding text**. Free generation instead conditions on the model's own errors, which can move it into a high-probability repetition attractor. `results/degeneration_traces.json` preserves entropy, top-1 probability/margin and EOS probability along eight actual greedy continuations. For the France trace, mean entropy falls from **3.71** nats in the first 16 generated positions to **0.391** in the last 16, while mean top-1 probability rises from **0.325** to **0.942**. This is an observed confident loop, not random corrupt logits. This helps explain why the validation number does not guarantee fluent paragraphs; it does not change the objective or excuse poor generation. Decoding-related degeneration is a known research observation, but the local sampling controls are the evidence for this model. See [Holtzman et al.](https://arxiv.org/abs/1904.09751).

### Follow-up: identical prefixes and supplied factual context

The continued read-only investigation adds **100 FP32 greedy 32-token outputs**, using every one of the existing 20 factual prompts in five conditions: ordinary BOS prefix, no BOS, trailing space, a repeated correct reference sentence, and a repeated first-foil sentence. The last condition deliberately inserts false test stimuli; these are neither factual assertions nor training data. All conditions and reference sentences were fixed before generation. There are now **824 language-suite outputs** in total (724 main + 100 follow-up), excluding separate cache/trace controls. Baseline first-32-token sequences match the prior FP32 outputs in every factual case.

**Cache differences isolated.** At each of the two BF16 divergence points, cached and uncached paths were evaluated on the exact same token prefix. FP32 max logit differences were at most **4.77e-6** across auto/math backends, retaining identical argmax. BF16 auto differences reached **0.0625**, exactly the size of the full-forward top-two margin. In the cached output this creates a tie between punctuation tokens: `!”` versus `.”` for the girl story, and period versus comma for France. Argmax then follows the tie rather than a different causal history. KL at these two specific contexts is about **0.000400 / 0.000406**; explicit math BF16 retains the same cached/uncached argmax in both. These direct controls strengthen the rounding/kernel explanation for these observed divergences. They do not establish every BF16 cache path as correct, but do show that these two output differences are not evidence of a systematic positional/masking failure.

**Context changes candidate preference, without ensuring a usable response.** The same four candidate answers were scored under each ordinary reference condition:

| Condition (20 facts) | Expected world fact preferred, answer plus period | Expected world fact preferred, answer tokens only | First supplied foil preferred, answer plus period |
| --- | ---: | ---: | ---: |
| Ordinary BOS | 16/20 | 15/20 | 1/20 |
| No BOS (diagnostic ablation) | 17/20 | Not measured | 1/20 |
| Repeated correct reference | 20/20 | 19/20 | 0/20 |
| Repeated false first-foil reference | 2/20 | Not measured | 18/20 |

Answer-only scoring removes the common terminal period from each candidate's summed NLL. These are restricted candidate preferences, not free factual accuracy. Supplying false references pushes the supplied foil to first place in 18/20 cases, demonstrating strong local copying/conditioning rather than reliable truth verification. The one-point no-BOS difference is too small and task-specific to justify changing the correct frozen BOS convention.

The trailing-space outputs are retained as an additional prompt-sensitivity control. Their originally scored candidate strings also begin with a space, so those NLLs describe double-space text and are deliberately **not compared** with ordinary candidate scores here. Retaining this limitation avoids attributing a segmentation/scoring difference to factual ability or a tokenizer bug.

**Why a correct restricted ranking can coexist with a wrong greedy answer.** Exact next-token probabilities and per-token candidate losses were decomposed on all 40 unprimed/correct-reference contexts, with full-forward first-token choices matching actual cached generation throughout:

- After `The capital of France is`, ` the` has probability **13.02%**, while ` Paris` is only **0.94%** and ranks **21st**. Paris beats the supplied city foils, but loses to general phrase continuations outside that candidate set.
- After explicitly providing `The capital of France is Paris.` and repeating the unfinished sentence, ` Paris` rises to **3.13%**, rank **3**, while ` the` remains **11.89%**. Greedy output becomes “the capital of France” repeatedly, rather than reliably using the supplied Paris answer.
- After supplying the correct east reference for the sunrise prompt, greedy next-token probabilities favor ` west` (**39.58%**) over ` east` (**37.88%**). Yet scoring the answer **plus its period** marginally favors east (summed NLL about **1.5377**, versus **1.5461** for west). Removing the period reveals the wrong word preference. This is a measured scoring effect, not evidence that 20/20 full-candidate preference equals perfect factual understanding.

Correct references improve several local free completions: the Jupiter and frozen-water cases now start with “Jupiter” and “ice.” They still loop. The year example starts with the correct “months” and then repeatedly claims “A year has six months.” The sunrise example contradicts its supplied east reference. Thus there is both recoverable local information and a failure to select, preserve and develop it autoregressively. A chat-format change alone would not establish competence. The effect of actual instruction tuning remains unmeasured; these base deficits must be re-evaluated after any future SFT rather than assumed to persist unchanged or disappear.

The new evidence sharpens, rather than reverses, the ranked findings: generic phrase/continuation preferences and unstable self-conditioning are directly measured; the training/data changes needed to correct them remain hypotheses requiring future controlled training. No optimizer operation was used to obtain these results.

### Why dialogue fails

Missing instruction training is strongly supported by the objective and data path: the model was never trained to distinguish assistant responses from arbitrary text, obey role labels, retrieve user-provided facts on demand, or stop at a conversational turn boundary. Q/A and role labels can instead look like web-page text to continue or repeat. Few-shot examples can influence output format without supplying a missing language model.

Weak base language is separately confirmed by plain completion failures, candidate tests and long outputs. Dataset extraction/coverage limitations plausibly amplify it; their exact causal share is unmeasured. Tokenizer mismatch and the tested cache/mask paths are not supported explanations. Generation settings change the severity and surface form of repetition but do not restore the missing factual reliability. Instruction tuning can teach response behavior, as demonstrated on other models in [Ouyang et al.](https://arxiv.org/abs/2203.02155); that paper does not establish that Codexa's current base would become competent after SFT.

## 7. Ranked root causes and confidence

| Rank | Explanation | Status / confidence | Supporting evidence and limit |
| --- | --- | --- | --- |
| 1 | Base pretraining is insufficient for the desired broad language behavior | **Likely primary cause; high confidence**, not a proven counterfactual | Only 2.01 training tokens/parameter; declining validation curve; local syntax better than discourse; pervasive raw-completion degeneration. More/better training has not been executed to establish an exact dose or guarantee a cure. |
| 2 | No supervised instruction/dialogue objective | **Confirmed absence; high confidence contributor to dialogue failures** | Standard all-token pretraining, no assistant-target supervision. Role and few-shot controls do not reliably establish dialogue. Cannot explain away base-completion defects. |
| 3 | Corpus quality and effective coverage limitations | **Confirmed properties; moderate confidence causal contribution** | OCR damage, web scaffolding, repeated passages, one shard per source, technical/non-prose examples; only 56.55% of windows consumed. No quality-cleaned training ablation, and much retained English is readable. |
| 4 | Greedy / constrained sampling exposes repetitive attractors | **Confirmed setting effect; high confidence contributor** | Matched prompts and lengths across precisions and three seeds. Sampling helps diversity but retains contradictions/loops. No claim that an untested decoding parameter would solve competence. |
| 5 | Small 249M capacity relative to broad web complexity | **Plausible secondary limitation; moderate confidence** | Errors across heterogeneous tasks and technical input. No size ablation; cannot conclude 249M is inherently too small for coherent English. |
| 6 | Packing across boundaries, short context and training schedule efficiency | **Unverified secondary hypotheses; low to moderate confidence** | Cross-document causal attention, 2K context, 13.1% warmup and late low LR are confirmed configurations. No experiment isolates their effect. Most simple failures fit well within 2K. |
| — | Wrong tokenizer, missing next-token shift, causal leak, corrupted weights, or material cache defect | **Not supported in tested paths** | Exact tokenizer/packing checks, finite state, independent references, future perturbation, cached/uncached FP32 controls. A bounded audit cannot rule out every possible implementation issue. |
| — | Many full passes over a tiny dataset caused repetition | **Contradicted for this run** | Exact sampler replay: epoch 0, 244,144 unique windows of 431,730. Repeated passages within different documents remain real. |

The confidence ranking concerns explanations, not a numerical apportionment of blame. This investigation cannot honestly claim that a fixed percentage of failures comes from each factor or that one untested change will resolve everything.

## 8. Improvements prioritized by expected impact

1. **Establish base-language gates and improve the permitted corpus.** Audit prose quality, remove or repair broken OCR/column extraction, boilerplate and repeated passages, preserve useful equations, and add diverse clean explanatory and narrative English. Measure mixture by tokens, domain and reading level. This directly targets observed weaknesses; no current data was rewritten.
2. **Plan more unique-data base pretraining with intermediate evaluations.** Keep the present architecture/tokenizer and checkpoint as the starting candidate. Evaluate coherent completion and factual associations every planned interval alongside source-specific held-out loss. A larger token counter alone is not acceptance.
3. **Treat continuation as a documented new phase.** The current exact-resume function rejects changed configurations/fingerprints and the original schedule is complete. Do not edit checkpoints or disable checks. Any future transition must explicitly validate architecture/tokenizer/tensor schema, account for cumulative tokens and data exposure, and declare optimizer/sampler/LR handling in a fresh run directory.
4. **Use controlled decoding for diagnosis and eventual serving.** Preserve greedy as a stress test; compare sampling across seeds, length limits, and proposed turn stops/repetition controls. Those can improve presentation but must not conceal a failing base-language gate. No production generation behavior was changed.
5. **Instruction-tune only a stronger base.** Then use permitted, deduplicated conversational targets with explicit roles and correct turn endings; evaluate short instructions, contextual retrieval, multi-turn consistency and base-language retention separately.
6. **Consider architecture/tokenizer changes only with evidence.** Keep 249M for now. A smaller model can make more tokens affordable; a larger one costs more and does not replace adequate data. Either size change invalidates direct native weight compatibility. A new vocabulary similarly requires an explicit compatibility strategy or fresh training.

The [Chinchilla scaling study](https://arxiv.org/abs/2203.15556) supports considering data and parameter count together; it does not prescribe a guaranteed competence threshold for this model. Billions of tokens are a practical investigation range, not a promise. [TinyStories](https://arxiv.org/abs/2305.07759) demonstrates that small models can produce coherent restricted-domain stories with suitable data; it does not establish broad factual competence, endorse an automatic download, or prove that the architecture must change.

## 9. Concrete next-phase proposal — not executed

### Options and compute

| Option | Benefit | Risk / decision | Approximate GPU budget at current model/context |
| --- | --- | --- | --- |
| Continue through current unseen corpus | Retains learned local grammar; 384.18M unused full-window targets remain | Does not repair the corpus or supply billions of novel tokens. Requires an explicit new phase because exact-resume schedule is complete. Useful bounded health check, not a full solution. | Remaining 384.18M ≈4.63h; additional 500M ≈6.02h would need new data or declared replay. |
| Continue on a better expanded mixture | Reuses current learning and can address observed distribution weaknesses | Carries existing biases; new-data distribution and LR/moment handling need evaluation. **Preferred starting candidate.** | To 2B cumulative: additional 1.5B ≈18.07h. To 5B cumulative: additional 4.5B ≈54.22h. |
| Restart from random initialization on better data | Clean objective/data history; necessary for an incompatible architecture/tokenizer or invalid old training | Discards useful learning; no confirmed defect currently requires it. Compare only if evidence warrants the extra experiment. | Fresh 2B ≈24.10h; fresh 5B ≈60.24h. |
| Expand to billions of unique positions | More breadth and fewer repeated exposures; addresses limited budget | Token count alone is insufficient; quality and split protection remain essential. Storage/preparation may be the first constraint. | About 12.05h per additional billion; acquisition/curation extra. |
| Improve mixture before more exposure | More coherent simple/explanatory language; deliberate math balance | Optimal fractions and causal benefit remain untested; replacing all technical material would discard useful capability. | Training rate similar only if model/context/batch remain comparable; preparation time unmeasured. |
| SFT after base gates | Teaches role, response and turn-ending behavior | Premature SFT can imitate answers while retaining factual/paragraph failures. Must test base retention. | Illustrative 10–50M processed tokens ≈7–36 min, plus evaluation; supervised-token counts differ from processed-token counts. |
| Keep 249M versus resize | Current model fits GPU and has compatible weights; no architecture blocker found | Smaller could buy more tokens per compute, larger increases cost/VRAM. Both require fresh architecture/weight compatibility decisions. | Keep: measured rate applies. Resize: **no measured rate**, so benchmark before estimating wall time. |

Time estimates use the **measured historical end-to-end rate 23,054.9 tokens/s**, which includes the old validation/checkpoint overhead. They exclude dataset preparation, new evaluation, scheduling interruptions and changed kernels/context/batch settings. They are estimates, not new throughput measurements. Additional 500M ≈ **6.02h**; 1B ≈ **12.05h**; 2B ≈ **24.10h**; 4.5B ≈ **54.22h**; 5B fresh ≈ **60.24h**. Instruction-tuning overhead depends on sequence packing, supervised-token fraction and evaluation; simple token-rate extrapolation is only a rough upper-level budget.

Storage is the actual **HS-SSD-E100 512G**, `/dev/sda` (512,110,190,592 bytes), mounted through `/dev/mapper/codexa-assets` as ext4 at **`/mnt/codexa`**, UUID `279352d0-0011-4733-8348-6815a4308196`. Final measured available space is about **139.7 GiB**, after diagnostic outputs. No old files were removed. The present curated database is **21.55 GiB**, or an empirical **24.37 GiB per prepared billion tokens**. Naively scaling that layout to 5B gives **121.8 GiB** for the database alone, plus **9.31 GiB** of uint16 token arrays, raw files, indexes, temporary staging and checkpoints. Adding even a 20-GiB reserve to database+arrays exceeds current free space. **A new 5B corpus cannot be assumed to fit this layout while retaining existing artifacts.** A reviewed bounded-staging/storage plan or more capacity is required before such preparation. These are extrapolations, not measured 5B artifact sizes.

### Proposed sequence

1. Freeze this report, checkpoint hash, source snapshot, raw outputs and untouched test split. Create new independent evaluation families before selecting training data, with document/URL/domain, exact-span and semantic-overlap review. The current tiny diagnostic fixture becomes a regression suite, not a secret future benchmark.
2. Prepare a reviewed **2–5B-token** corpus plan. A starting mixture proposal is **75% clean general/explanatory English, 15% narrative/simple-language material, 10% verified mathematics**, measured by tokens; these fractions are hypotheses to validate, not an optimized recipe. Preserve prose surrounding formulas. Add code or advanced mathematics only deliberately and track effects on general language. Check permissions, snapshots, hashes, deduplication, reading-level diversity and storage budget before any acquisition; no large download is authorized by this report.
3. Keep width/layers/heads/vocabulary/context unchanged for the first comparison. Specify a fresh continuation phase with 32,768 target tokens/update, checkpoints at most every 250 updates, fixed retention/volume checks and recovery verification. Preserve old native checkpoints. Because data/schedule changes fail exact resume, implement and test an explicit phase transition **only under future authorization**, with no integrity-check bypass. Decide whether to preserve Adam moments or reset them and record the reason; do not silently claim exact training continuation if resetting state.
4. Start with a **separately authorized 50M-token continuation health check**, about **36 minutes** at the historical rate, comparing against the frozen current checkpoint on identical evaluations. A trial schedule could use peak LR 1e-4, warmup 100 updates (~3.28M tokens), cosine to 1e-5; these are explicit candidate values, not an optimality claim or a change to defaults. Main-phase warmup and decay must be specified for its actual duration. A 50M fresh model is too immature to fairly decide whether a restart eventually wins. Consider a fresh-data restart comparison only if the data/compatibility audit supplies a credible trigger; budget a meaningful from-scratch pilot (e.g. 500M ≈6.02h), disclose the unequal prior training/compute, and evaluate the same improved mixture and held-out families. Do not infer long-run superiority from a tiny pilot. Keep this investigation's current recommendation conditional on those future results.
5. If continuation is healthy, stage the main budget in **250M-token increments**, evaluate the full validation set plus fixed general-language/source-specific gates after each, and retain comparable samples. Work toward 1B then 2B cumulative tokens, with expansion toward ~5B contingent on actual language improvement, storage and compute approval. Declare unique versus replayed tokens at each stage. A 2–5B unique-data plan exceeds the current finite corpus, even though its unseen remainder is useful for an early phase.
6. Stop for diagnosis on nonfinite state, unexpected train/validation divergence, factual/grammar regression, worsening long-output loops or failed recovery. Readiness gates below, rather than loss alone, decide whether SFT is appropriate. Only after passing them, propose a bounded instruction-tuning phase and obtain authorization for it separately.

This is a specification for future review. No configuration, dataset, production source, weights, scheduler, sampler or training process was changed to implement it.

## 10. Acceptance criteria before conversational fine-tuning

These are **proposed project gates**, not thresholds established by a cited paper or scores the model has already achieved. Freeze independent items and scoring rules before future training; report denominators, uncertainty and failure examples. Do not train on these evaluation answers.

| Gate | Required evidence |
| --- | --- |
| Implementation and artifact integrity | No unresolved training-invalidating masking, RoPE, tying, shift, tokenizer or material cache issue. Finite state, exact parameter count, source/data/tokenizer hashes and frozen checkpoint provenance. FP32 cached/uncached logits within declared tolerance on varied real contexts through 2K; token divergence explained at low margins rather than silently dismissed. |
| Text fidelity | Exact complete-text round trips on English/Unicode/whitespace/math fixtures; no unintended UNK/PAD in ordinary corpus text; identical explicit BOS/EOS conventions and document boundary checks. |
| Sentence completion and grammar | At least **90%** correct on ≥200 independent, balanced grammar minimal pairs, with sum/length-normalized scoring audited for token-length bias. At least **90%** of ≥100 ordinary sentence starts completed grammatically and sensibly by a fixed rubric. |
| Paragraph continuity | At least **80%** of ≥100 independent 128–256-token continuations preserve topic, entities and event/argument progression with readable grammar. Two reviewers should score a blinded subset; record disagreement. Merely repeating a grammatical sentence fails. |
| Stable generation | Severe repetitive loops in **≤5%** of paragraph outputs under the chosen sampled setting across three seeds. Also disclose greedy failure rates and full 4-gram repetition distributions; sampling success must not hide unstable decoding. |
| Elementary factual associations | At least **80%** on ≥100 independent clear factual associations and paraphrases, using candidate likelihood **and** manually verified free completion. Separate plausible alternative completions and contradictions; keyword presence alone is insufficient. No claim of broad factual or mathematical competence from this gate. |
| Held-out generalization | Full frozen validation plus new clean general-English/source-specific losses improve or remain within a predeclared nonregression tolerance (e.g. 0.05 nats) while generation gates improve. Track seen/unseen data, domains and duplicate exposure; do not optimize only the first 32 windows. |
| Boundary and length behavior | Correct decoder termination on forced-EOS fixtures, sensible natural completion on independently ended documents, explicit max-budget outcomes and no silent context truncation. Conversation-specific turn stopping is validated during the later SFT phase. |
| Context-faithful base completion | At least **90%** of ≥100 independently authored completion contexts preserve a clearly supplied fact/entity without contradictory continuation. Report truthful-world references and explicitly fictional counterfactual references separately. Score answer tokens independently of shared punctuation and inspect free responses. |
| Readiness decision | Pass the language/factual/stability gates together. Instruction obedience and multi-turn assistant reliability are separate post-SFT goals, not pretraining capabilities inferred from perplexity. |

The current model clearly fails the paragraph and stable-generation requirements. The small candidate tests below do not establish the required independent grammar/factual denominators. Therefore it is **not ready to be treated as a competent conversational base**.

## Evidence, reproducibility and verification

All bulky diagnostic artifacts are on the existing `/mnt/codexa` volume:

`/mnt/codexa/pytorch/codexa-v3/diagnostics/20261104-120458`

The requested reviewed report is stored under repository `docs/`, with a byte-identical evidence copy on that volume. Raw model/corpus outputs and weights are not added to Git. Key files relative to that diagnostic directory:

| Artifact | Purpose |
| --- | --- |
| `setup.json`, `env.json`, `env.sh`, `source_snapshot/` | Initial checkpoint/source hashes, exact environment and inspected source copy. |
| `scripts/common.py`, `run_diagnostic.py`, `cpu_checks.py` | Forward-only safety guards, sequential GPU budget enforcement and safe CPU test allowlist. |
| `scripts/numerics.py`, `results/numerics.json` | Independent reference implementations, actual-model masking/cache/backend checks, activation and EOS observations. |
| `scripts/provenance.py`, `results/provenance.json` | Original database/input hashes, finite tensors, embedded tokenizer identity, actual packed token comparisons. |
| `scripts/corpus.py`, `results/corpus.json`, `domain_diversity.json`, `passage_repetition.json` | Full local corpus/source/coverage/repetition/domain audit. Result JSON files live under `results/`. |
| `raw/document_coverage.npz`, `raw/token_frequencies.csv`, `raw/passage_counts.sqlite` | Replayed windows, per-token all/consumed counts and exact newline-unit repetition index. The auxiliary SQLite database is new diagnostic output, not the curated training database. |
| `raw/corpus_samples.jsonl`, `manual_source_samples.jsonl`, `results/manual_corpus_review.json`, `results/manual_generation_review.json` | Full stratified samples and explicitly scoped corpus/generation observations. Raw JSONL files live under `raw/`. |
| `scripts/logs.py`, `raw/training_steps.csv`, `raw/validation_steps.csv`, `results/training_analysis.json` | Complete historical log analysis, LR checks and graph inputs. |
| `scripts/language.py`, `raw/language_fixture.json`, `raw/language_outputs.jsonl`, `results/language.json` | Exact authored prompts, seeds/settings, all 724 generated outputs and descriptive summaries. |
| `scripts/loss_slices.py`, `results/window_losses.json`, `document_losses.json`, `generation_cache_controls.json`, `degeneration_traces.json` | Matched teacher-forced windows, source slices, complete parity controls and free-running probability traces. Result JSON files live under `results/`. |
| `scripts/analyze_results.py`, `results/evaluation_analysis.json` | Reproducible aggregate tables, confidence intervals and natural-text repetition control. |
| `scripts/followup.py`, `factual_tokens.py`, `followup_analysis.py`, `render_followup.py` | Identical-prefix divergence controls, 100 additional outputs, and per-token factual score decomposition. Scripts live under `scripts/`; evidence is in `raw/followup_*` and `results/followup*`, `identical_prefix_cache.json`, `factual_tokens.json`. |
| `history/before-followup/` | Preserved original report, integrity record, GPU ledger, evidence manifest and changed diagnostic-script versions before the continuation. |
| `scripts/plot_results.py`, `figures/*.png`, `figures/*.svg` | Standalone scientific figures and plotting code. |
| `raw/*.log`, `results/*.status.json`, `gpu_budget.json`, `final_verification.json`, `artifact_sha256.json` | Actual commands/statuses, retained harness failure, GPU time ledger and end-of-investigation integrity evidence. |

Model diagnostics use the pinned repository Python **3.13.14**, PyTorch **2.11.0+cu130**, NumPy **2.5.4** and tokenizers **0.23.3**. Plots use the existing system Matplotlib executable as a separate CSV/JSON consumer; no project dependencies or lockfiles were changed. Cache/temp paths are redirected under the diagnostic directory. GPU stages run sequentially, never through the training controller, within the agreed 45-minute diagnostic budget.

Reproduction from this repository (each GPU stage consumes part of the remaining budget; these commands perform inference, not training):

```sh
source /mnt/codexa/pytorch/codexa-v3/diagnostics/20261104-120458/env.sh
.venv/bin/python -B "$DIAG_ROOT/scripts/run_diagnostic.py" cpu_checks.py
.venv/bin/python -B "$DIAG_ROOT/scripts/run_diagnostic.py" numerics.py --gpu
.venv/bin/python -B "$DIAG_ROOT/scripts/run_diagnostic.py" language.py --gpu
.venv/bin/python -B "$DIAG_ROOT/scripts/run_diagnostic.py" loss_slices.py --gpu
```

The corpus script creates its own repetition-index database and intentionally does not overwrite an existing one; to reproduce the complete investigation use a fresh diagnostic root with the recorded setup/environment and unchanged inputs. Full pytest and `scripts/verify.py` were deliberately not rerun because they include backward/optimizer training tests, which conflict with this phase's strict prohibition. Instead, the guarded forward-only allowlist passed **15 tests**; repository Ruff check, Ruff format check and mypy also passed. These checks supplement the independent references and do not replace them.

Final verification: checkpoint SHA256 remains **`54fca007f272fb3784c759d18db44912091227aedc233b62c9731514b1835c73`**, with size, modification timestamp and inode unchanged. All **42** inspected `codexa/` source hashes match the initial snapshot. Every guarded stage records **zero backward calls, zero optimizer updates and zero forbidden-call attempts**, and every loaded model's in-memory hash is unchanged. GPU diagnostic process wall time totaled **1082.03 seconds (18.03 minutes)** of the 45-minute limit, conservatively including loading and CPU work inside those stages. All 8 bounded stages exited successfully. No Python CUDA process or training controller remained at final verification; the supervisor was never invoked.
