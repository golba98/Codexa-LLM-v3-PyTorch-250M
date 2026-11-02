# Training roadmap

## Phase A — validate architecture and training

Establish causal correctness, exact accounting, CPU/CUDA execution, true resumption, tokenizer fidelity, and tiny-sample memorization. Benchmark full-context primary training steps before setting a batch budget. Smoke success demonstrates software behavior, not pretraining or mathematical competence.

## Phase B — tokenizer and datasets

Collect permitted local text with source/license records. Remove corrupt documents, duplicates, personal information where appropriate, and evaluation contamination. Freeze document-level splits before fitting a 16,384-token byte BPE. Audit fertility on prose, code, numbers, and mathematical notation. Extend corpus ingestion to a bounded-memory manifest/shard pipeline before multi-billion-token preparation. Dataset quality, balance, and clean evaluation matter more than raw volume.

## Phase C — general language pretraining

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
