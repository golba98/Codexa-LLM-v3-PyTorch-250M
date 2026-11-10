# Training roadmap

## Current priority: direct published English data

The direct-stream source-admission milestone is complete for the pinned English Wikipedia subset of `common-pile/wikimedia_filtered`, revision `0641bb84bd9b7162bcddf8be7836822161a9a342`. The explicitly authorized general-English reevaluation passes with 94/100 documents and 293/300 windows, retaining the unchanged severe/critical ceilings and both independent judgments. Internal-use CC BY-SA conditions, exclusions, conservative capacity above 1B and existing compute/exact-resume checks pass. The production gate stops only at the project owner's missing explicit 1B training-launch authorization; no training is authorized or started. See [source evidence and remaining decision](DATASETS.md#current-1b-token-admission-and-readiness). Do not reopen broad dataset comparisons or reconstruct a corpus. Preserve all original checkpoints and mathematics assets; specialization, SFT and publication retain their separate gates. Historical phases below describe the wider research roadmap, not instructions to rebuild the current language data or tokenizer.

## Phase A — validate architecture and training

Establish causal correctness, exact accounting, CPU/CUDA execution, true resumption, tokenizer fidelity, and tiny-sample memorization. Benchmark full-context primary training steps before setting a batch budget. Smoke success demonstrates software behavior, not pretraining or mathematical competence.

## Phase B — tokenizer and datasets

Collect permitted local text with source/license records. Remove corrupt documents, duplicates, personal information where appropriate, and evaluation contamination. Freeze document-level splits before fitting a 16,384-token byte BPE. Audit fertility on prose, code, numbers, and mathematical notation. Extend corpus ingestion to a bounded-memory manifest/shard pipeline before multi-billion-token preparation. Dataset quality, balance, and clean evaluation matter more than raw volume.

## Phase C — general language pretraining

The completed 500M-token run has not established coherent base-language generation. The [diagnostic report](CODEXA_V3_DIAGNOSTIC_REPORT.md) records the evidence, ranks likely causes and proposes measurable grammar, factuality, paragraph-coherence and stability gates before specialization or conversational fine-tuning. Its continuation/restart comparison and compute/storage budgets are future proposals only; no further training was executed or authorized by the investigation.

Initially plan approximately five billion target tokens. For 248,687,488 parameters, the rough dense-transformer training estimate `6NT` is 7.46e18 FLOPs, excluding detailed attention and system overhead. Derive wall time from measured sustained throughput, not advertised hardware TFLOPs. Compare held-out loss, text quality, stability, and throughput across shorter pilots before approving a long run. Consider longer runs only after benchmarking. Keep checkpoint retention, disk capacity, and recovery plans explicit.

## Phase D — mathematical continued pretraining

Only after the base pipeline is validated, introduce permitted mathematical prose and verified synthetic material. Mix general text and earlier material into training. Catastrophic forgetting occurs when updates for a new distribution degrade previously learned capabilities; track fixed general-language and earlier-topic evaluations, retain checkpoints, and adjust mixture proportions based on observed regressions.

## Phase E — structured curriculum

| Grades | Future topics |
|---|---|
| 1–3 | Counting, number recognition, addition, subtraction, simple word problems |
| 4–6 | Multiplication, division, fractions, decimals, percentages, introductory geometry |
| 7–9 | Algebra, equations, ratios, exponents, graphs, geometry |
| 10–12 | Functions, advanced algebra, trigonometry, analytical geometry, statistics, appropriate introductory calculus concepts |

The current Grade 1 prototype establishes deterministic generation, exact arithmetic/SymPy checks, worked steps, difficulty, and stable partitions. Future generators should use topic-specific modules and verification, rather than one generalized prompt engine. Align educational scope with a chosen syllabus and expert review before claiming curriculum coverage.

Partition canonical mathematical identities before rendering language. Extend holdouts to operand ranges, templates, compositions, and problem families. Identical-expression separation is only a first defense: mathematically equivalent or paraphrased problems may still leak. Track both answer correctness and step validity, retain frozen evaluation sets for every learned topic, and reevaluate them after subsequent stages. Progress only when the held-out capability threshold defined for that topic is reached.

## Phase F — evaluation and refinement

Evaluate accuracy by grade/topic/difficulty alongside general language quality. Separate memorization from generalization using unseen ranges, wording, and compositions. Compare independent seeds and preserve failed cases. A 250M model has limited capacity and may struggle with long reasoning, reliable multi-step arithmetic, factual breadth, and unfamiliar notation. Fluent worked text can still be wrong. Consider larger models only when evidence identifies capacity rather than data or pipeline defects as the bottleneck.

## Numbered engineering and research gates

| Milestone | Objective and implementation | Dependencies | Validation / required evidence / exit | Risks |
|---|---|---|---|---|
| M1 | Repository, agents, packaging, CI and local export/preflight | Foundation | Passing local CI-equivalent checks, preservation inventory, fixture roundtrip; hosted CI explicitly pending until authorized push | Unsupported release claims, accidental data loss |
| M2 | Architecture verification | M1 | Counts, causal/reference attention, RoPE, norms, tying, initialization and finite gradients; fixed-seed regression evidence | Numerical drift |
| M3 | Deterministic provenance-aware tokenizer/data pipeline with bounded-memory ingestion | M1–M2 | Freeze permitted corpus and holdouts; hashes, duplicate/contamination audits, tokenizer roundtrips and memory measurements | Licensing, leakage, corpus quality |
| M4 | Harden training recovery, metrics and precision | M2–M3 | Interrupted versus uninterrupted equality, optimizer/sampler restoration, fault injection and sustained finite pilot losses | Silent recovery divergence |
| M5 | Optimize RTX 4080 performance from measurements | M4 | Repeat throughput/memory measurements across sequence, accumulation and optimizer settings; no accepted unexplained regression | OOM, noisy short benchmarks |
| M6 | Bounded smoke and intermediate pretraining pilots | M3–M5 and approved pilot budgets | Loss curves, held-out evaluation, checkpoint recovery and documented data/compute; explicit go/no-go report | Smoke memorization mistaken for learning |
| M7 | Prepare and conduct actual 250M base training | M6 and explicit compute authorization | Reviewed corpus mixture, measured budget, recovery/monitoring plan, training logs and verified final checkpoint | Expensive failures, inadequate data |
| M8 | Evaluate base language model | M7 | Independent held-out loss, generation and generalization assessment with limitations and contamination review | Biased benchmarks, unsupported capabilities |
| M9 | Progressive Grade 1–12 mathematics | M8 | Verified generators, unseen family/range/template holdouts, topic metrics and earlier-skill retention after every stage | Incorrect steps, forgetting, memorization |
| M10 | Public trained model release | M8–M9 as applicable; approved weight license and publication authorization | Independent safe loading, completed card, provenance, checksums, current tests/security review and supported runtime instructions | Redistribution rights, secret leaks, unsafe claims |
| M11 | Post-release research and scaling | M10 | Reproducible comparisons, refined datasets, evidence-based capacity decisions | Benchmark overfitting, compute cost |

No completion dates are promised. Existing smoke evidence partially supports M2/M4/M5 but does not close sustained research gates. M7 and mathematics training must never start automatically.

## M3 readiness sequence

A: Validate bounded ingestion, provenance, splitting and candidate freezing. B: Execute bounded CPU/CUDA smoke training with the new loader and recovery. C: Run a 100M-token pilot only after explicit authorization and dataset/tokenizer/contamination/resource review. D: Review pilot evidence before authorizing the full approximately 5B-token base run. E: Evaluate held-out language performance and consider continued pretraining. F: Develop Grade 1–12 specialization while reevaluating general language and earlier topics. Neither a verified pipeline nor a frozen candidate authorizes stages C–F. See DATASETS.md for measured sample scope, theoretical resource budgets and remaining gates.

## Bounded real-data pilot evidence

The educational-web/mathematics sample now supplies 121.19M training tokens and a reproducible frozen 16,384-token BPE. A separately authorized 99.32M-token full-model pilot completed, with exact CUDA recovery, actual Rich terminal acceptance, decreasing fixed-slice validation loss and measured sustained throughput/memory. See DATASETS.md and TRAINING.md for scope and numbers. This supplies initial M3–M6 evidence; it does not authorize M7 or establish general language/mathematical competence. Before a larger run, review source-specific rights, independent quality/benchmark decontamination, substantially more unique training text, scaled index/storage costs and checkpoint cadence. No 1B/5B run or publication starts automatically.

The separately authorized 500M-token run completed on 2026-11-04: a fresh full-model CUDA/BF16 run reached 500,006,912 tokens in 15,259 updates. Expansion of the same pinned snapshots supplied 884.18M verified training tokens while preserving evaluation artifacts and tokenizer. Independent final-checkpoint verification, exact fixed-slice validation replay and full validation-window evaluation passed; see TRAINING.md for measured loss, throughput, memory and scope. This completes the authorized token budget and supplies further M3–M8 evidence. Broader language/mathematics benchmarks, source-rights review and external benchmark-contamination limitations remain open. A 1B/5B run or publication requires separate authorization; no further training starts automatically.

## Language preparation before further pretraining

[Dataset v2](CODEXA_V3_DATASET_V2.md) now contains 2,108,050,326 new, audited training positions from a bounded, frozen-tokenizer preparation. [Training readiness](CODEXA_V3_TRAINING_READINESS.md) records the unchanged checkpoint baseline, unresolved quality/benchmark limitations and a verified, untrained 50M-token continuation health-check bundle. The user deferred a 5B corpus expansion. Further training and later conversational fine-tuning require separate authorization. Reaching a token budget is not a language-readiness gate.


The current remediation candidate must clear every unchanged corpus gate before the authorized exact-50M continuation. Version 2 completed all 500 ratings and failed OpenWebMath. Systematic extraction fixes produced a byte-verified 2.077B-position version 3; its separate complete 500-item review still fails FineMath's severe-defect limit and FineWeb-Edu's usable-document minimum. Further source-wide quality remediation and a bounded storage plan are needed before another immutable candidate and unseen review. Preserve the 500M baseline, defer the 5B expansion and use paired held-out loss, arithmetic and actual text generation after a successful continuation. No continuation was launched. See [readiness evidence](CODEXA_V3_TRAINING_READINESS.md#remediation-status-and-exact-continuation-accounting).
