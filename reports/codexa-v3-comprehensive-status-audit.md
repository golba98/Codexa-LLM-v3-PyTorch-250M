# Codexa v3 comprehensive status audit

Audit date: 2026-11-09 UTC. Evidence gathered from approximately 21:42 UTC; review capture cutoff 21:47:13 UTC; review capture time is specified below. **READY** means supported by current applicable evidence; **BLOCKED** means a known unmet condition; **UNKNOWN** means insufficient current evidence. Historical results do not certify current artifacts. No training, review judgments, downloads, remediation or dataset rebuilds were initiated by this audit.

## Executive Summary

**Training is BLOCKED.** The authorized 500M-token run completed, and its native checkpoint remains valid. A comparable 50M-token continuation projects to **36.15 minutes** from measured end-to-end throughput. No trained 550M successor exists. Current census capture is **5,138/5,400 judgments (95.1481%), with 262 missing**, including 92 pending after a safety interruption.

The expanded review is still incomplete and its files advanced during inspection. It is a 2,700-document curation census requiring 5,400 independent judgments, not an approval panel. The earlier 5,064/336 report was accurate at its saved timestamp but is outdated; the captured recount below supersedes it.

Completing that census will not resolve the main blocker: all positively admitted affected-source documents have already been inspected, so strict approval preparation rejects them as fresh approval material. The latest completed approval panels failed. A current passing corpus, candidate-bound continuation bundle, source verification and CUDA smoke are still required.

Recommend a **hybrid** strategy: preserve useful local material and evidence, replace failing portions through a reproducible source-wide preparation policy, and reserve unseen approval material before inspection. Do not spend further review effort assuming it alone unlocks training. Blocker resolution, rebuild and approval durations remain UNKNOWN; there is no defensible calendar launch date.

## Verified Progress Dashboard

| Workstream | Completed evidence | Remaining / status |
|---|---|---|
| Dataset sourcing and selection | Pinned local sources and provenance registry; historical 2B preparation | Successor selection unresolved; BLOCKED |
| Dataset review | Original 2,542-document census and 278-document evidence-resolution packet completed historically; expansion counts below | Expanded census incomplete; approval remains separate |
| Cleaning and preparation | v1–v4c candidates, preserved quarantines and token proofs | No v5 snapshot; successor policy/freshness unresolved |
| Final quality validation | Complete v4c primary and extended panels | FAILED; new immutable candidate and passing fresh panels required |
| Training readiness | Valid parent, compatible capability queries, bounded CPU checks passing | Candidate-bound launch requirements missing |
| Next exact 50M continuation | Conversion/accounting infrastructure and untrained v3 bundle | 0/50M new training tokens; BLOCKED |
| Post-training evaluation | Historical parent baselines | No successor to evaluate; BLOCKED |

Branch: `feature/m3-data-foundation`. HEAD: `05058aa13b31fc4a552143ff7f5df6ef3ce3d388`, committed 2026-11-03T03:25:47+02:00, “Use physical PTY dimensions instead of inherited CI terminal size”. Initial inspection found 16 modified tracked files and 48 untracked paths; report creation adds another path. Existing changes include training recovery/accounting, data acquisition/remediation/semantic review, evaluation and documentation. The full observed list is reproduced in the appendix. None was reverted.

The last completed authorized development/research outcome is the 500,006,912-token budget and its checkpoint verification, followed by completed preparation/review artifacts. This does **not** close M8 language competence or M9 mathematics. The active gate is preparation and independent validation of a corpus for the conditional exact-50M continuation. Acceptance requires all unchanged corpus gates, native token/identity proofs, preserved baseline, exact accounting and candidate/current-code-bound launch evidence. Roadmap prose mixes historical stages and is not a reliable single current milestone label. No weighted overall percentage is justified: the remaining quality/freshness design work is unbounded.

## Dataset Inventory

Paths below use **R** = `/mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814` and **P** = `/mnt/codexa/pytorch/codexa-v3/pretraining/20261103-500m-233201`.

Latest completed final remediation snapshot: `R/snapshots/remediated-v4c-20261106`. Manifest SHA256: `a6ad911a943305a59b70447b02d13c2d436505962e0c117f3af41503e2d423a2`. Manifest-declared training positions sum exactly to 2,052,040,276; validation 10,335,047; test 10,254,300. These are native packed positions, not publisher token counts. uint16 token arrays alone require 4,145,259,246 bytes across these splits, excluding indexes and metadata. Historical proof compared 3,251,117 retained documents to the parent; full shard rehashing was not repeated in this bounded audit.

| Source | Pinned revision | v4c training documents | Training positions | Licence/provenance |
|---|---|---:|---:|---|
| HuggingFaceFW/fineweb-edu | 87f09149ef4734204d70ed1d046ddc9ca3f2b8f9 | 691,311 | 717,676,767 | ODC-By; Common Crawl and underlying terms; URLs retained |
| wikimedia/wikipedia, 20231101.en | b04c8d1ceb2f5cd4588862100d08de323dccfbaa | 918,780 | 933,682,880 | CC-BY-SA-3.0/GFDL recorded in pinned card; attribution required |
| roneneldan/TinyStories GPT-4 | f54c09fd23315a6f9c86f9dc80f725de7d8f9c64 | 1,541,030 | 330,026,485 | CDLA-Sharing-1.0; synthetic provenance |
| HuggingFaceTB/finemath, 4plus | e92b25a616738fe95dc186b64dfb19f9c8525594 | 66,776 | 70,349,059 | ODC-By; underlying content terms |
| open-web-math/open-web-math | fde8ef8de2300f5e778f56261843dab89f230815 | 475 | 305,085 | ODC-By; underlying content terms |

Source configs also list original Cosmopedia educational/stories revision `0ae6ec63f91742bd2d1eaef4f02232c55d719385` and FineWeb `9bb295ddab0e05d785b879661af7260fed5140fc`; these have **zero positions in v4c**, so config entries must not be counted as acquired/admitted training capacity. Registry: `R/source-registry.json`; admission records: `R/source-admission.json` and `R/reports/admission-*.json`; snapshot manifest binds the registry digest `d9bc0517b3616662a636a08b7bfb42a10edd90c1f92732e7a59a1103421d993e`.

Historical stages: initial v2 2,108,050,326 positions; v3 2,076,912,557; v4c 2,052,040,276. v4c quarantined 14,308 complete records relative to v3, including 14,185 training documents and 24,872,281 training positions. The incomplete v4 directory is invalid; v4b is an intermediate screening artifact. No v5 snapshot exists. Dataset replacement is **not implemented**: source configs and counterexample reports show investigation, not a replacement corpus manifest or passing approval.

## Review Progress

The original census, evidence resolution and expansion are distinct reviews. Original census: 2,542 documents ×2 =5,084 judgments, completed historically. Evidence resolution: 278 documents ×2 =556 additional assessments, completed historically. Its composed admission admits 281 FineMath, 57 FineWeb-Edu and 60 OpenWebMath training documents plus 41 held-out records. These are curation dispositions, not fresh approval results.

Expansion: `R/semantic-admission/v5-extension-20261109`; frozen packet SHA256 `7b5ecfaa33b3d0a98246e9713d122c92b8a45d4bac74f492698d1140a841ab32`; plan SHA256 `d1751aa10725d4d3ce18d41718e4d2cbeddb028a27dc11235932116f99c27da9`; latest observed ownership `judge-assignments-v11.json`. It requires 2,700 complete documents ×2 distinct assessments =5,400 judgments. A saved observation at 21:37:54.747033Z records 5,064/5,400=93.7778%, 336 missing. The final captured counts and validity analysis follow in the supplemental evidence below.

A missing second assessment leaves a document pending; it is not an accepted document or automatically a rejected document. Binary clearance disagreement is computed only on fully paired records. Authored errata must match original row hashes and preserve the original files. Partial rows can be reused only with matching packet, identity, ownership, full-document coverage and valid evidence. Complete coverage is necessary for finalization under the frozen protocol; it is currently absent. No judgment was inferred or supplied by this audit.

### Captured expansion results

**Stable capture: 2026-11-09T21:47:13.061570–21:47:13.139516Z.** No captured files changed during this read. **5,138 / 5,400 = 95.1481%; 262 missing.** Later activity does not retroactively change this observation. Saved rows passed schema, frozen ownership and required-evidence-structure checks; zero duplicate or unexpected IDs. This does not independently verify every cited factual claim. Applied 57 authored corrections with exact original-row matches. Interim full-pair dispositions: **252 both-clear, 2,186 unadmitted, 262 pending**. Fully paired documents: 2,438; clearance disagreement **358/2,438 = 14.68%**. Unadmitted includes explicit defects and unresolved checks; it is not a count of proven false documents.

| Source/split | Candidates | Expected judgments | Saved | Remaining | Both-clear | Unadmitted | Pending |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fineweb-edu/train | 2500 | 5000 | 4752 | 248 | 219 | 2033 | 248 |
| fineweb-edu/test | 25 | 50 | 47 | 3 | 3 | 19 | 3 |
| finemath-4plus/val | 25 | 50 | 48 | 2 | 6 | 17 | 2 |
| fineweb-edu/val | 25 | 50 | 49 | 1 | 2 | 22 | 1 |
| finemath-4plus/train | 100 | 200 | 193 | 7 | 17 | 76 | 7 |
| finemath-4plus/test | 25 | 50 | 49 | 1 | 5 | 19 | 1 |


| Historical reviewer identity | Expected | Saved | Remaining | Clear ratings | Severe ratings | Critical ratings |
| --- | --- | --- | --- | --- | --- | --- |
| /root | 2149 | 2149 | 0 | 408 | 582 | 7 |
| /root/evidence_reviewer_1 | 675 | 583 | 92 | 82 | 0 | 0 |
| /root/evidence_reviewer_2 | 945 | 830 | 115 | 77 | 3 | 0 |
| /root/evidence_reviewer_3 | 900 | 900 | 0 | 132 | 28 | 0 |
| /root/evidence_reviewer_4 | 450 | 450 | 0 | 128 | 283 | 4 |
| /root/long_record_reviewer | 1 | 1 | 0 | 0 | 1 | 0 |
| /root/evidence_reviewer_5 | 280 | 225 | 55 | 71 | 56 | 0 |

These are historical packet actor labels, not the identities or activity of this audit’s agents. Severity flags differ sharply by reviewer; those are recorded judgments, not calibrated interchangeable measurements.

| Batch file | Expected | Saved | Remaining |
| --- | --- | --- | --- |
| root.jsonl | 350 | 350 | 0 |
| evidence_reviewer_1.jsonl | 450 | 450 | 0 |
| evidence_reviewer_2-added.jsonl | 325 | 325 | 0 |
| root-added.jsonl | 450 | 450 | 0 |
| evidence_reviewer_2.jsonl | 620 | 505 | 115 |
| evidence_reviewer_1-added.jsonl | 225 | 133 | 92 |
| evidence_reviewer_3.jsonl | 900 | 900 | 0 |
| evidence_reviewer_4.jsonl | 450 | 450 | 0 |
| root-added-c.jsonl | 450 | 450 | 0 |
| root-added-e.jsonl | 199 | 199 | 0 |
| long_record_reviewer.jsonl | 1 | 1 | 0 |
| root-added-d.jsonl | 250 | 250 | 0 |
| root-added-b.jsonl | 450 | 450 | 0 |
| evidence_reviewer_5-added.jsonl | 80 | 25 | 55 |
| evidence_reviewer_5.jsonl | 200 | 200 | 0 |

Remaining fixed ranges: reviewer 1 T01034–T01125 (92); reviewer 2 T02306–T02420 (115); reviewer 5 T02446–T02500 (55). **T01899 is now complete**, so the earlier missing-long-record observation is outdated. Automatic safety review concerns the initial interrupted records; the whole 92-record remainder is pending because that reviewer stopped, not evidence that all 92 records triggered safety review.

Rejection/uncertainty reasons below are **judgment-check occurrences**, may overlap, and must not be summed into documents. There are 953 severe ratings and 11 critical ratings across 9 distinct documents. Critical flags include unsupported medical claims, missing essential guidance, sensitive/personal identifiers and one missing-image/contact-caption finding; the last illustrates rubric/severity variability and is preserved without relabeling. No personal identifiers are reproduced.

| Check finding | Occurrences |
| --- | --- |
| correctness=uncertain | 1415 |
| completeness=defect | 1233 |
| contamination=defect | 708 |
| correctness=defect | 2216 |
| visuals=defect | 382 |
| coherence=defect | 841 |
| notation=defect | 363 |
| visuals=uncertain | 18 |
| education=defect | 559 |
| notation=uncertain | 2 |
| education=uncertain | 22 |
| coherence=uncertain | 3 |
| completeness=uncertain | 1 |
| contamination=uncertain | 1 |

Live activity: files demonstrably advanced from 5,117 to 5,123 to 5,138 while inspected. Thus the original review has continuing saved-output activity, but accessible process names do not establish individual historical agent liveness. No trainer or review CLI process was found. Historical assignment files describe stopped/reassigned tasks; they are not live task registries. The two audit inspection agents completed; **zero audit subagents remain active**. Number of active original reviewers: **UNKNOWN**, not zero. No reliable current session handle was available for those prior reviewers.

## Blocker Register

Seven independent launch work packages are counted below; subfindings are not inflated into additional blockers.

| ID / severity | Evidence and impact | Exact condition for readiness / dependencies |
|---|---|---|
| B1 Critical: approval freshness | Strict `scripts/review_continuation_corpus.py` rejects `semantic_admission_decision_sha256` successors: every admitted affected-source record was inspected during curation | A source-wide preparation policy yielding enough previously uninspected eligible documents, preserving frozen exclusions/seeds. New reviewers alone do not fix document freshness. No waiver or relabeling |
| B2 High: incomplete census and safety interruption | v6 ownership records reviewer 1 twice interrupted by automatic biological-content review at T01034–T01039; T01034–T01125 remain 92 missing judgments. v10 assigns long T01899 separately | Actual valid independent assessments through a permitted recovery path; untouched work can follow administrative transfer precedent. Retry of flagged material is not presumed safe or authorized. Preserve valid partial judgments. Audit does not bypass safety |
| B3 Critical: failed semantic quality | v4c complete primary/extended decisions have training_allowed=false | New immutable candidate passes every primary/extended gate, fresh samples at preregistered attempt-1 seeds 2026111601/2026111602; no threshold changes |
| B4 High: successor preparation/integrity absent | No v5 snapshot or matching fresh subsequence/token proof | Freeze reviewed source policy; build successor within storage envelope; prove native bytes, tokenizer, boundaries, splits, dedup/exclusions, mixture/diversity and unseen sample capacity |
| B5 High: continuation bundle bound to failed v3 | Existing exact50M bundle is untrained and candidate-specific | Prepare a fresh bundle for the passing immutable candidate; preserve all parent weights/AdamW moments and exact50M accounting; no baseline overwrite |
| B6 High: current launch verification/smoke absent | Old source verification digest differs; no current candidate-bound source-verification.json or cuda-smoke.json found | Stable source digest, applicable verification artifact and authorized disposable CUDA smoke matching source, candidate, parent and settings. Audit CPU checks do not substitute |
| B7 High: licence/contamination limitations | Registry records underlying rights; historical decontamination is approximate; new candidate has no reviewed provenance/exclusion proof | Document source-specific permissions/attribution and candidate evaluation separation, retain limitations and reject prohibited overlap before launch; public weight licence requires separate review |

**What is stale?** Approval freshness affects all positively admitted affected-source curation documents, not merely one batch; the full current affected-source admission pool cannot provide unseen approval material. Historical original composed pool alone contains 398 affected-source training admissions, all inspected. Expansion adds further inspected admissions; final captured counts below quantify them. This does not invalidate their curation judgments. It prevents reuse as unseen approval evidence. No dataset manifest/hash disagreement was established by this audit. The smallest defensible fix is to change the successor preparation approach so admission is systematic across uninspected material, then reserve untouched approval material; finishing the census alone cannot accomplish that.

A separate stale-evidence issue affects source verification: old checks certify an older digest. The fresh bounded audit receipt certifies its captured digest only. Concurrent source changes mean launcher freshness must be rechecked at launch.

No new corruption, incompatible tokenizer or nonfinite parent weights was detected. This is bounded evidence, not a complete fresh rehash of all corpora. Disk/hardware capability is currently adequate; actual full-shape CUDA execution is UNKNOWN until the required smoke. No failing test was observed in the selected checks. Full training/recovery test coverage was deliberately excluded.

## Dataset Strategy Comparison

Availability checked against primary publisher cards on audit date. Publisher claims are not independent local quality measurements. No replacement dataset was downloaded or tokenized.

| Candidate | Available material and declared licence | Suitability and risks |
|---|---|---|
| Cosmopedia v2 | Over 39M Mixtral-generated educational texts/stories; SmolLM-Corpus card ODC-By | Actual pretraining candidate for small models; web-seed provenance and hallucinations require review. Mistral token_length is not Codexa tokens. Local original-v1 counterexample proves a wrong geometric-mean explanation, not a v2 error rate. [Publisher](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) |
| OpenMathInstruct-2 | 14M Llama3.1-405B-generated problem/solution pairs, CC-BY-4.0 | Instruction tuning; GSM8K/MATH train-derived and augmented families create evaluation-separation risk. Consider later mathematics mixture, not replacement for general prose. [Publisher](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2) |
| NuminaMath 1.5 | 896,215 rows, card reports 531MB, Apache-2.0 | Post-training mathematics with provenance/validity fields; competition geometry may require images. Audit source rights, correctness and benchmark families before use. [Publisher](https://huggingface.co/datasets/project-numina/NuminaMath-1.5) |
| FineWeb-Edu / deduplicated subset | Main card advertises 1.3T tokens; SmolLM dedup subset 220B; ODC-By | Pretraining prose; educational relevance is not correctness or extraction completeness. Existing score>=4 local material failed. Higher score alone is not a validated remedy. Publisher tokens use a different tokenizer. [Main card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), [dedup subset](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) |

The cited cards establish available row/token claims, not locally usable token totals. **Verified Codexa tokens available from replacement candidates: UNKNOWN.** Numina's move from AI-MO to project-numina is a current redirect, not missing availability. Other local Wikipedia/TinyStories material offers known capacity, but source-specific passing historical panels do not certify a new mixture. Original openly licensed textbook material could support educational diversity; no specific uninspected local textbook corpus was established, so it is not credited with tokens or a completion advantage.

| Strategy | Work / review effort | Quality / integrity / complexity | Time until training |
|---|---|---|---|
| A Finish current remediation | Complete missing judgments, resolve freshness, develop source-wide remediation, rebuild, fresh 500+400 two-judge panels | Highest exposure to demonstrated source defects; sunk reviews are reusable development evidence, not approval. Large retained corpus exists but is unapproved | UNKNOWN; completing the census alone is insufficient |
| B Replace problematic sources | Pin/acquire within authorized bounds, licences, tokenizer counts, dedup/exclusions, systematic preparation, untouched validation | Potentially better text; introduces acquisition and provenance risks; instruction corpora cannot replace broad prose directly. High integration work | UNKNOWN; no measured local acquisition/preparation rate |
| C Hybrid — recommended | Preserve suitable existing material; replace failing subsets with a bounded pretraining source; apply source-wide policy and fresh approval | Reuses storage/provenance/tokenizer infrastructure, limits new ingest, preserves breadth. New mixture must still pass all frozen gates. Moderate integration work | UNKNOWN, but avoids treating more doomed curation as the critical path |

The recommendation is an engineering judgment, not a measured guarantee that hybrid wins in wall time. Keep all historical artifacts. A smaller preliminary source review may inform selection, but cannot replace or shrink the frozen approval protocol without explicit future authorization. For illustration, zero defects in an independent random sample of 100 only gives an approximate 95% upper defect-rate bound of 3%; that is not proof of correctness or a licence clearance.

## Model and Training Readiness

**Overall: BLOCKED.** Latest verified trained checkpoint: `P/final/checkpoint.pt`, native v1; SHA256 `54fca007f272fb3784c759d18db44912091227aedc233b62c9731514b1835c73`, 2,985,607,403 bytes. Fresh read-only full hash matched in 5.739 seconds. CPU mmap inspection confirmed finite model tensors, 170 optimizer states, 15,259 updates, 500,006,912 tokens, 244,144 microsteps; scheduler last_epoch=15,259, step_count=15,260; saved sampler position=244,144. Hash and schema evidence do not replace a fresh resumed-update parity test.

Architecture: custom decoder, 24 layers, width 896, 14 heads, intermediate 2,432, context 2,048, tied input/output embedding. Unique parameters **248,687,488**; counting tied state-dict entries twice is wrong. Frozen byte BPE vocab **16,384**, PAD/UNK/BOS/EOS=0/1/2/3. Tokenizer SHA256 `60a5e06447542b9462d458577f4a3056551c135e5aa1b8859d39624a3eee8771`.

| Requirement | Status | Evidence / ready condition |
|---|---|---|
| Parent model, saved optimizer/scheduler/sampler | READY for preparation | Current CPU inspection and matching hash; native exact-resume path retained |
| Tokenizer/configuration compatibility | READY for inspected artifacts | Fixed identity/dimensions; candidate must inherit these |
| CPU verification | READY at receipt digest | Ruff, format, mypy and 135 forward-only tests passed; source must remain identical |
| GPU capability | READY by capability queries | RTX4080, BF16, Torch2.11.0+cu130, CUDA13, driver615.78.08 |
| Full-shape candidate-bound CUDA smoke | UNKNOWN / required | No current smoke artifact; no GPU compute run by audit |
| Data and approval | BLOCKED | B1–B4, B7 |
| Launch bundle/current verification | BLOCKED | B5–B6 |
| Storage | READY at observation | Correct UUID; about119.52GiB available, greater than40GiB reserve+18GiB run envelope; recheck before use |

Historical run: BF16, sequence2048, microbatch1, accumulation16, nominal32,768 target tokens/update. Existing `R/continuation/remediated-v3-exact50m` retains optimizer state but explicitly starts a new warmup/cosine schedule; changed data receives a new sampler order. It is continuation, not exact replay of the original data/schedule. Config: LR1e-4→1e-5,100 warmup,1,526 updates including final partial update, target550,006,912 lifetime tokens. Verification records zero backward/optimizer calls. **The next50M does not require random reinitialization**, but the old failed-candidate bundle cannot launch against new data. No550M checkpoint means no post-continuation quality result exists.

## Historical Performance Measurements

`P/training/metrics.jsonl` contains each step1–15,259 exactly once. `P/training/events.jsonl` has16,316 events, first2026-11-04T01:24:37.494381Z, last07:25:57.940192Z. Supervisor and event intervals differ slightly; use the recorded supervisor denominator for end-to-end throughput.

| Measurement | Value | Scope |
|---|---:|---|
| 500M supervisor duration |21,687.666s =6h1m27.666s|`P/final/postrun-report.json` |
| End-to-end throughput |23,054.897 tokens/s|Includes historical run overhead |
| Sustained optimizer throughput |24,023.446 tokens/s|Not total wall-clock throughput |
| Final training loss / last60 mean |3.140551 /3.117484|Historical distribution only |
| Monitored validation loss |3.005710|Fixed monitored slices |
| Full historical validation |606,208 targets; loss3.048003; perplexity21.073212|Different scope from monitored slices |
| Full validation forward time |11.551s|Not complete language/math/generation suite |
| Broader baseline budget usage |3,617.736s|`R/benchmarks/baseline-54fca007f272fb37/budget.json`; accounted work, not established contiguous elapsed time |

Historical memory measurements are not recomputed; current ~1,958MiB GPU desktop use is not training peak memory. No current benchmark throughput or peak allocated/reserved training memory was measured. Historical rebuild and quality-panel wall-clock durations are UNKNOWN unless a matched start/end pair is supplied in supplemental findings. Modification times alone do not establish execution duration. Review aggregate timing calculations follow below.

### Measured review timing


| UTC observation | Saved judgments |
| --- | --- |
| 2026-11-09T20:37:13.578474+00:00 | 4387 |
| 2026-11-09T21:03:31.637134+00:00 | 4642 |
| 2026-11-09T21:37:54.747033+00:00 | 5064 |
| 2026-11-09T21:47:13.139516+00:00 | 5138 |


Aggregate intervals, including observed idle/coordination time: 255 judgments in 1,578.059 seconds = **581.73/hour**; 422 in 2,063.110 seconds = **736.36/hour**; 74 in 558.392 seconds = **477.08/hour**. Combined: 751 in 4,199.561 seconds = **643.78/hour**. These are actual saved-coverage rates for the changing mix of reviewers, not per-agent rates. Individual judgment timestamps and sufficiently comparable recent reviewer-specific rates are not available; prefix mtimes alone are not used as a substitute.

At the latest measured aggregate rate, ordinary remaining work is `(262−92)/477.0838×60 = **21.38 minutes**`; hypothetical all-judgment projection `262/477.0838×60 = **32.95 minutes**` is invalid as a finish promise because safety recovery has no measured duration. Optimistic ordinary-work reference uses the fastest observed interval: `170/736.36×60 = **13.85 minutes**`. A slower-work upper bound cannot be established. The tail may have fewer simultaneous reviewers or harder documents, so even 21.38 minutes is conditional. No agent-count multiplier was applied.

## Completion Forecast

There is no defensible numeric total until freshness and safety recovery have valid resolution paths. This is a specific dependency failure, not a range of hours/days. Define U=unmeasured blocker resolution, R=rebuild plus integrity proof, Q=fresh approval panels, V=candidate-bound readiness checks/smoke, E=complete paired evaluation; none can be assigned zero merely because its duration is unknown.

50M projection: `50,000,000 /23,054.897 =2,168.737s =36.146min`. Optimizer-only lower reference: `50,000,000 /24,023.446 =2,081.300s =34.688min`. Same GPU, precision and packing are necessary assumptions; changed data loading and checkpoint/evaluation cadence can alter performance.

| Stage | Optimistic: no new failures | Expected: normal work/rework | Pessimistic: another quality failure |
|---|---|---|---|
| Remaining review |13.85min ordinary work; safety UNKNOWN|21.38min ordinary work plus safety UNKNOWN|UNKNOWN if additional review invalidity/recovery occurs |
| Blocker resolution |UNKNOWN (U)|UNKNOWN (U)|UNKNOWN (U plus additional intervention) |
| Dataset rebuild/proof |UNKNOWN (R)|UNKNOWN (R)|UNKNOWN, potentially repeated R |
| Quality panels |UNKNOWN (Q)|UNKNOWN (Q)|UNKNOWN, at least another failed/completed attempt and authorized remediation |
| Readiness verification |UNKNOWN (V)|UNKNOWN (V)|UNKNOWN, possibly repeated V |
| 50M training |34.69min optimizer-only reference, overhead extra|36.15min historical end-to-end projection|UNKNOWN if throughput/config/hardware changes;36.15min remains reference only |
| Post-training evaluation |11.551s for historical validation-sized subtask; complete E UNKNOWN|Complete E UNKNOWN|Complete E UNKNOWN |
| Total |UNKNOWN; measurable training reference34.69min plus overhead, U+R+Q+V+E|UNKNOWN;36.15min + review work + U+R+Q+V+E, adjusted for safe overlap|UNKNOWN; no evidence supports an upper bound |

Training start time is UNKNOWN even if all rate-extrapolatable judgments finish promptly. Complete final evaluation time is also UNKNOWN;11.551seconds is not a substitute for the requested whole suite. Confidence: high in historical training arithmetic and current blocking conditions; medium/low in review extrapolation; no supported confidence interval for calendar completion.

## Critical Path

1. Resolve successor freshness design and source suitability without changing frozen approval rules.
2. Establish provenance, contamination exclusions and adequate untouched panel capacity; freeze source-wide policy.
3. Build a new immutable candidate and prove native token/identity/subsequence integrity and source mixture.
4. Run fresh independent primary500 and extended400 record panels, two actual ratings per record, at frozen seeds. All gates must pass.
5. Prepare a candidate-bound preserved-parent exact50M bundle; obtain current source verification and authorized disposable CUDA smoke; recheck storage and baseline identity.
6. Execute the separately conditional50M run, then paired held-out loss, arithmetic and actual generation evaluation.

Completing the current census can inform step1 but cannot substitute for steps1–4. Read-only provenance research, checkpoint inspection and CPU checks can safely overlap; elapsed forecasts must use the maximum parallel branch duration, not blindly sum it. GPU jobs require exclusive coordination. No automatic next step is triggered by this report.

## Recommended Execution Plan

1. **Highest priority: settle a technically valid freshness-preserving successor policy.** Acceptance: a documented way to produce enough uninspected eligible records for every frozen source panel, without using curated admissions as unseen approval or changing gates. No more large review is necessary to establish the existing contradiction.
2. Use hybrid preparation as the default proposal: retain useful unchanged local material, replace failing portions with a bounded educational pretraining source. Acceptance: pinned source, rights/attribution ledger, actual native tokens, diversity and evaluation-family exclusion proof. Obtain authorization before new acquisition or restructuring.
3. Preserve current review evidence and finish only through an explicitly permitted recovery path if its curation value justifies the effort. Acceptance:5,400 valid owned judgments, every correction hash-checked, full-document coverage, complete primary evidence on clearance, no safety bypass. Full census completion still does not authorize training.
4. After separate implementation authorization, prepare one immutable candidate and apply the unchanged approval protocol. Acceptance: complete passing primary/extended decisions plus token proofs; failed panels remain failures.
5. Refresh bundle/source/smoke evidence only after candidate acceptance. Acceptance: all launcher predicates true at one source/candidate identity and sufficient storage; then the existing conditional50M authorization can be evaluated against its exact scope.
6. Evaluate parent and successor on identical protected suites; report loss, correctness and generated text separately. Acceptance: reproducible paired results with runtime and contamination limitations, without competence claims from token count alone.

## Evidence Appendix

Primary repository sources: `AGENTS.md`, `docs/ROADMAP.md`, `docs/TRAINING.md`, `docs/CODEXA_V3_DATASET_V2.md`, `docs/CODEXA_V3_TRAINING_READINESS.md`, `configs/data/v2_sources.yaml`, `configs/data/v2_finemath.yaml`, `scripts/review_continuation_corpus.py`, `scripts/semantic_admission.py`, `scripts/continuation_healthcheck.py`, `codexa/data/semantic.py`, `codexa/data/quality_protocol.py`.

Current bounded verification command:

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 .venv/bin/python -B scripts/verify.py --forward-only --output .agents/scratch/audit-verification.json
```

Result: Ruff passed;120 files formatted; mypy passed66 source files;135 tests passed in2.03s. Receipt source digest `c78e0d4ae84b48db090dc1994357e0ff4a0d83e6c1f274c6fd01a835ca77e385`, dirty HEAD05058aa. Training/backward/update tests were deliberately excluded by the script's forward-only guards. Logs and receipt: `.agents/scratch/audit-verification.log` and `.agents/scratch/audit-verification.json`. Older passing final-source receipt digest `e4838e8a…` does not certify this code. A prior observation during planning saw `10eb17cf…`; these changing digests require a final launch-time check.

Other read-only commands: `git status --short`, `git branch --show-current`, `git log -1 --format=…`, `git diff --stat`, targeted `rg`, manifest JSON parsing, process snapshots, `nvidia-smi`, volume UUID/free-space queries, CPU checkpoint mmap and SHA256. Audit agents performed independent read-only review/training inspection; no review agent was instructed to rate documents.

Historical quality evidence: `R/reviews/remediated-v4c-fresh-20261106/decision.json` and `extended/decision.json`. Primary protocol digest `fb30348ba09308f827959161b731002664ce88528d3acf8b20d31a7a149a6f2a`; extended `6290e682be046822ae3d909e62cc41e38152549af185d352e950718aa4da468c`. v4c primary: windows265/300 usable,38 severe; FineMath12/40 usable,11 severe; FineWeb21/40 usable,6 severe; OpenWebMath22/40 usable,13 severe; TinyStories40/40 usable,0 severe; Wikipedia38/40 usable,2 severe. Extended: FineMath56/200 usable,44 severe; FineWeb82/200 usable,44 severe. All1,800 independent ratings completed. Primary numeric minima:270/300 usable windows and34/40 per source; severe maxima15/300 and2/40 respectively, with additional mandatory checks. These failures cannot be erased by passing structural proofs.

Original census decision `81f9f2a69a0cedf3a7bf951d87f3de9407a8c93694d2f27dd22bdaa93ee952be`; composed evidence-resolution decision `a5be2c18a910c9017aaeb9da9720f458625d11a7f4a3a6c409ef08e1ee17c735`. Local synthetic-data warning: `R/reports/cosmopedia-math-counterexample.json`, ordinal9871, original-v1 source hash `05ed6afca68a93f1ec1f062165a49c08dd6d29b2b728165fce7ab96a7f702e24`; it is one inspected counterexample, not v2 measurement.

### Review file capture hashes

| File | SHA256 at capture |
| --- | --- |
| root.jsonl | abaa2ea6e2b841fb043d6cf1ba63ca186447b88a0b9782efa0b1b6c6833dfec4 |
| evidence_reviewer_1.jsonl | 9cbfbf284cb539157122fad0d67ad28ab4d983215cabd3122b264967295bee2d |
| evidence_reviewer_2-added.jsonl | 32a76e03baadc219461145fb56ef63da0819ec21d4206bed90689009d8b5e288 |
| root-added.jsonl | 3c2c743b834d316827bda15ad498dac7bb0241668a2141b1e2131e8c12355662 |
| evidence_reviewer_2.jsonl | 25fc4646981a1c834dc5b42bbde76d158b852ddda7a729fd4b465dccf1a08102 |
| evidence_reviewer_1-added.jsonl | f9172ef4ac3131d35eadf709ff703c4e81a3968577cf5ed7cac8b391c7082520 |
| evidence_reviewer_3.jsonl | 2a18e96657572670fdc0de995220bf01146a631a8cc806e69d014477accceea2 |
| evidence_reviewer_4.jsonl | 025766912c1445ef971d5a11e906ef89ab43b490af618fe0cdbd4ad9ed2243e3 |
| root-added-c.jsonl | 0e6197cc4554c4844857838beb89c877fdfb19d4cb503a9d223b941c75aa1f33 |
| root-added-e.jsonl | 51a358c6d18d98aa10ee465db0eb2a43e363adc82c9629ddf7f3d5c9fe4e4862 |
| long_record_reviewer.jsonl | b1f88b1778878a26ade9d5df9facbfca17443a50796ba2752da4ea0c35304c79 |
| root-added-d.jsonl | edf9400dafa8539e579510e345b04c0bd7a59a5224bdd632327a0b339d664bfb |
| root-added-b.jsonl | 97c536d293186728290048f3f96b0aa41fcf76c43f20e5b5e465494901213b0a |
| evidence_reviewer_5-added.jsonl | 277018c951b2bce16e81a5550db2f606589656cda8e6083a6c6ab9a8a8edd2ec |
| evidence_reviewer_5.jsonl | bd35ea45e7843a3349e785f45def7d4c03807b95a433045a2dfc5178693d7662 |


### Observed Git working tree

```text
 M CHANGELOG.md
 M README.md
 M codexa/cli.py
 M codexa/data/commands.py
 M codexa/data/production.py
 M codexa/training/checkpoint.py
 M codexa/training/terminal/render.py
 M codexa/training/trainer.py
 M docs/ARCHITECTURE.md
 M docs/DATASETS.md
 M docs/RELEASE.md
 M docs/ROADMAP.md
 M docs/TRAINING.md
 M scripts/terminal_acceptance.py
 M scripts/verify.py
 M tests/test_data.py
?? codexa/continuation.py
?? codexa/data/math_integrity.py
?? codexa/data/quality_findings.py
?? codexa/data/quality_protocol.py
?? codexa/data/remediate.py
?? codexa/data/representation.py
?? codexa/data/sampling.py
?? codexa/data/semantic.py
?? codexa/data/v2.py
?? codexa/data/v2_acquire.py
?? codexa/data/v2_catalog.py
?? codexa/data/v2_finalize.py
?? codexa/data/v2_index.py
?? codexa/data/v2_policy.py
?? codexa/data/v2_profile.py
?? codexa/data/v2_quality.py
?? codexa/data/v2_registry.py
?? codexa/data/v2_storage.py
?? codexa/evaluation/benchmark.py
?? codexa/evaluation/fixtures.py
?? codexa/evaluation/loss.py
?? codexa/evaluation/overlap.py
?? codexa/evaluation/protection.py
?? codexa/evaluation/scoring.py
?? codexa/safety.py
?? configs/data/v2_finemath.yaml
?? configs/data/v2_sources.yaml
?? docs/CODEXA_V3_DATASET_V2.md
?? docs/CODEXA_V3_DIAGNOSTIC_REPORT.md
?? docs/CODEXA_V3_STORAGE_AUDIT.md
?? docs/CODEXA_V3_TRAINING_READINESS.md
?? scripts/audit_v2_snapshot.py
?? scripts/combine_quality_reviews.py
?? scripts/continuation_healthcheck.py
?? scripts/inventory_v2_storage.py
?? scripts/plot_language_baseline.py
?? scripts/review_continuation_corpus.py
?? scripts/semantic_admission.py
?? scripts/semantic_evidence.py
?? scripts/verify_remediated_snapshot.py
?? scripts/verify_review_packet.py
?? tests/fixtures/
?? tests/test_corpus_review.py
?? tests/test_exact_phase.py
?? tests/test_quality_protocol.py
?? tests/test_remediation.py
?? tests/test_sampling.py
?? tests/test_semantic_admission.py
?? tests/test_v2.py
```

### Detailed training/source inspection evidence

The following is the read-only inspection record. Registry enumerations identify catalogued source files and sizes; **they do not prove every listed file was acquired or remains on disk**. Do not sum them as actual storage. Physical current corpus rehashes were deliberately bounded.

```text
AUDIT evidence captured 2026-11-09. Read-only source/artifact inspection; only this ignored scratch finding is written.

TRAINING: Latest trained parent /mnt/codexa/pytorch/codexa-v3/pretraining/20261103-500m-233201/final/checkpoint.pt
SHA256 verified this audit 54fca007f272fb3784c759d18db44912091227aedc233b62c9731514b1835c73; 2985607403 bytes. CPU mmap: native v1, step15259, tokens500006912, microsteps244144, 170 optimizer states, all model tensors finite; scheduler last_epoch15259/_step_count15260; sampler position244144. Unique parameter count historical proof248687488, tied state-dict entries must not be double-counted. Architecture24x896,14heads,intermediate2432,vocab16384,context2048. Tokenizer SHA60a5e06447542b9462d458577f4a3056551c135e5aa1b8859d39624a3eee8771, special IDs0/1/2/3.

HISTORICAL performance proof: pretraining/20261103-500m-233201/final/postrun-report.json; events.jsonl independently parsed16316events, first2026-11-04T01:24:37.494381+00:00,last07:25:57.940192+00:00; metrics15259 contiguous. supervisor21687.6662347s=6h1m27.666s;500006912tokens;endtoend23054.8970363tok/s;postwarmup optimizer24023.4461302tok/s. 50M/endtoend=2168.736643s=36.145611min;50M/optimizer=2081.300065s=34.688334min. Same RTX4080/torch2.11+cu130/BF16/microbatch1/accum16/seq2048. Current continuation LR differs1e-4 vs parent3e-4; no measured continuation throughput. Historical final loss3.14055066;60stepmean3.11748388;fixed validation3.00570979;full validation606208tokens loss3.048002665 perplexity21.0732121,evaluation11.551204s. Broader baseline budget.json records3617.735793s used,5400s limit; budget-accounted time, not necessarily wall time. Paired full evaluation future duration UNKNOWN.

CURRENT: capability query Python3.13.14,Torch2.11.0+cu130,CUDA13.0,CUDA available,BF16 supported,RTX4080 driver615.78.08. GPU desktop snapshot1percent1958MiB. No matching trainer/review_continuation/continuation_healthcheck/semantic_admission/real_pipeline process at inspection. /mnt/codexa UUID279352d0-0011-4733-8348-6815a4308196,free128341311488bytes119.52GiB (time-sensitive), launcher requires58GiB.

BLOCKERS: complete current candidate quality4 primary+extended passes absent; latest v4c failed. No continuation/*/cuda-smoke.json and no source-verification.json under preparation root. Exact50M bundle continuation/remediated-v3-exact50m/initial.pt is untrained, tied to failed oldv3, requires replacement for eventual candidate; verification zero backward/updates,preserved parent+optimizer,target550006912. Current source digest at previous probe10eb17cf250f8d73e4acffa9b54dea6a811ad6bded8023664e17dfd8a041a997; old source-verification-freshness-gate digest4a29adbd152e9da5100e55da83c28a410433a39fd4f769381db212ff414ac9dd stale. New candidate-bound source/token/integrity/review proof +bundle+CUDA smoke required; no restart from random required.

SOURCE registry exact acquisition inventory (NOT all retained):
{
  "fineweb-edu": {
    "files": 13,
    "bytes": 26365374301,
    "revision": "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
    "license": "ODC-By-1.0; Common Crawl terms; underlying rights retained",
    "repo": "HuggingFaceFW/fineweb-edu"
  },
  "tinystories-gpt4": {
    "files": 2,
    "bytes": 2250255763,
    "revision": "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64",
    "license": "CDLA-Sharing-1.0",
    "repo": "roneneldan/TinyStories"
  },
  "wikipedia-en": {
    "files": 41,
    "bytes": 11630929031,
    "revision": "b04c8d1ceb2f5cd4588862100d08de323dccfbaa",
    "license": "CC-BY-SA-3.0/GFDL as recorded in the pinned dataset card; underlying terms apply",
    "repo": "wikimedia/wikipedia"
  },
  "cosmopedia-educational": {
    "files": 120,
    "bytes": 33005247139,
    "revision": "0ae6ec63f91742bd2d1eaef4f02232c55d719385",
    "license": "Apache-2.0 dataset declaration; synthetic seed provenance retained",
    "repo": "HuggingFaceTB/cosmopedia"
  },
  "cosmopedia-stories": {
    "files": 43,
    "bytes": 11902294709,
    "revision": "0ae6ec63f91742bd2d1eaef4f02232c55d719385",
    "license": "Apache-2.0 dataset declaration; synthetic seed provenance retained",
    "repo": "HuggingFaceTB/cosmopedia"
  },
  "openwebmath": {
    "files": 113,
    "bytes": 27194773112,
    "revision": "fde8ef8de2300f5e778f56261843dab89f230815",
    "license": "ODC-By-1.0; Common Crawl terms; underlying rights retained",
    "repo": "open-web-math/open-web-math"
  },
  "fineweb": {
    "files": 15,
    "bytes": 30639384917,
    "revision": "9bb295ddab0e05d785b879661af7260fed5140fc",
    "license": "ODC-By-1.0; Common Crawl terms; underlying rights retained",
    "repo": "HuggingFaceFW/fineweb"
  },
  "finemath-4plus": {
    "files": 64,
    "bytes": 18365184633,
    "revision": "e92b25a616738fe95dc186b64dfb19f9c8525594",
    "license": "ODC-By-1.0; Common Crawl and underlying content terms retained",
    "repo": "HuggingFaceTB/finemath"
  }
}

MANIFEST /mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/snapshots/remediated-v1b-20261105/manifest.json SHA256 7d1a9e38589988113d795019e4aa3e081a032551fd79025a38c1180926f2eab3
{"splits": {"test": {"tokens": 10288379, "documents": 16345, "shards": 1}, "train": {"tokens": 2066850759, "documents": 3227813, "shards": 32}, "val": {"tokens": 10366927, "documents": 16460, "shards": 1}}, "source_tokens": {"finemath-4plus": 76224834, "fineweb-edu": 728015034, "openwebmath": 352811, "tinystories-gpt4": 330026183, "wikipedia-en": 932231897}}

MANIFEST /mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/snapshots/remediated-v2-20261105/manifest.json SHA256 156bd2d07b95ddd45a136cb884d58aed178efcfeab90829f0ede7efb7a6f1fe3
{"splits": {"test": {"tokens": 10335976, "documents": 16373, "shards": 1}, "train": {"tokens": 2077136357, "documents": 3232885, "shards": 32}, "val": {"tokens": 10432298, "documents": 16495, "shards": 1}}, "source_tokens": {"finemath-4plus": 77773576, "fineweb-edu": 735296863, "openwebmath": 356553, "tinystories-gpt4": 330026485, "wikipedia-en": 933682880}}

MANIFEST /mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/snapshots/remediated-v3-20261105/manifest.json SHA256 199183a06b130912d87712e4d857a4d706649909de59d4c55b665a89e83d2c85
{"splits": {"test": {"tokens": 10335976, "documents": 16373, "shards": 1}, "train": {"tokens": 2076912557, "documents": 3232557, "shards": 32}, "val": {"tokens": 10432298, "documents": 16495, "shards": 1}}, "source_tokens": {"finemath-4plus": 77559185, "fineweb-edu": 735296863, "openwebmath": 347144, "tinystories-gpt4": 330026485, "wikipedia-en": 933682880}}

MANIFEST /mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/snapshots/remediated-v4b-20261106/manifest.json SHA256 e249c51d3579fe32461c5619537c1908a0461b722933818b85b17d87f42c4787
{"splits": {"test": {"tokens": 10245856, "documents": 16304, "shards": 1}, "train": {"tokens": 2051108449, "documents": 3217781, "shards": 32}, "val": {"tokens": 10331425, "documents": 16432, "shards": 1}}, "source_tokens": {"finemath-4plus": 69778353, "fineweb-edu": 717315646, "openwebmath": 305085, "tinystories-gpt4": 330026485, "wikipedia-en": 933682880}}

MANIFEST /mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/snapshots/remediated-v4c-20261106/manifest.json SHA256 a6ad911a943305a59b70447b02d13c2d436505962e0c117f3af41503e2d423a2
{"splits": {"test": {"tokens": 10254300, "documents": 16310, "shards": 1}, "train": {"tokens": 2052040276, "documents": 3218372, "shards": 32}, "val": {"tokens": 10335047, "documents": 16435, "shards": 1}}, "source_tokens": {"finemath-4plus": 70349059, "fineweb-edu": 717676767, "openwebmath": 305085, "tinystories-gpt4": 330026485, "wikipedia-en": 933682880}}

MANIFEST /mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/snapshots/reviewed-2b-20261105/manifest.json SHA256 5cc849cc50680c86a50c40fe10aa6f56d0b0cfc77e1dcab5feb7ddba8c5dfddf
{"splits": {"test": {"tokens": 10464092, "documents": 16480, "shards": 1}, "train": {"tokens": 2108050326, "documents": 3257078, "shards": 32}, "val": {"tokens": 10598943, "documents": 16610, "shards": 1}}, "source_tokens": {"finemath-4plus": 80367192, "fineweb-edu": 736800702, "openwebmath": 391992, "tinystories-gpt4": 330026485, "wikipedia-en": 960463955}}

MANIFEST /mnt/codexa/pytorch/codexa-v3/dataset-v2/20261104-133814/snapshots/stage-0001/manifest.json SHA256 3ee1cbcdc7cfef0f3cc3c7772f568ad07eb21c05ac607ecc5b7279ca58ea4233
{"splits": {"test": {"tokens": 541758, "documents": 564, "shards": 1}, "train": {"tokens": 107211957, "documents": 111883, "shards": 2}, "val": {"tokens": 593408, "documents": 603, "shards": 1}}, "source_tokens": {"fineweb-edu": 69409951, "openwebmath": 391992, "wikipedia-en": 37410014}}

REBUILD/PANEL duration: inspected finalize-reviewed-2b.log, remediated-v3-20261105.log,v4c-build.log and primary protocol files; no reliable start/end/duration fields. File mtimes are not duration evidence. Durations UNKNOWN absent further transcript evidence.

```
