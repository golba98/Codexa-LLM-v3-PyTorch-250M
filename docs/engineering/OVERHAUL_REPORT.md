# Overhaul handoff

## Audit and preservation
Baseline b8588ed had 23 passing tests and functional custom model/training/tokenizer/inference. Confirmed gaps and baseline commands are in REPOSITORY_AUDIT.md. MIGRATION_MAP.md and migration-hashes.json preserve all original runtime files. No checkpoint, data or historical benchmark was discarded. Primary architecture remains exactly 248,687,488 parameters; configurations and native checkpoint version 1 are unchanged.

## Structure and Git policy
Root: AGENTS.md, project metadata/locks, license and contributor/security/changelog policies. Source remains codexa/{architecture,tokenizer,data,training,inference,mathematics,evaluation,release}; configs and original examples retain their paths. .github holds CPU/manual-GPU workflows and templates. .agents/skills holds six validated reusable workflows. release holds model-card and metadata proposals; docs/engineering holds deliberate public records. Ignored outputs separates datasets, tokenizers, predictions, benchmarks, experiments, logs, packages and inference exports. Native run logs/checkpoints remain colocated to preserve the existing API. Existing empty utils package is retained for import compatibility.

Ignore rules cover Python/build/editor/runtime/secret files and nested local agent instructions, with root AGENTS.md, reviewed skills, docs and locks visible. Policy tests exercise git check-ignore. No broad Markdown exclusion is used. No weights, arrays, environments or private scratch are staged.

## Agents and presentation
Root instructions require reproducibility, provenance, bounded resource use, coordinated GPU work, separate worktrees and ownership, integration testing, and explicit authorization for remote actions. Six skill validators passed. README documents actual status, executable workflows and limits; no hosted passing badge or available trained model is claimed. Proposed name/description/topics are prepared, not applied. Code is Apache-2.0, original fixtures CC0; weight and external-data licensing remain separate. Dependency metadata inventory is not a legal certification.

## Release foundation
SafeTensors exports preserve tied weights and use a versioned configuration, tokenizer, generation defaults, metadata, model card, licensing statement and checksummed manifest. Loading validates files, schema, names/shapes, vocabulary and finite weights before inference. Preflight compares independent-process logits/greedy outputs and checks a current verification report and stage-specific evidence/reviews. Fixture validation explicitly reports publish_ready false. No publication command exists. Native recovery state remains separate. Scans detect limited known patterns and are not comprehensive secret detection or proof of legal rights.

## Validation
`pytest -q`: 33 passed, including CUDA, original resume/signal/fault tests and ten release/policy cases. `ruff check .`, `ruff format --check .`, `mypy codexa` (31 files), six skill validators, hash-preservation checks and exact CPU baseline digest passed. `scripts/verify.py` is the local CI-equivalent entry point and writes ignored verification evidence.

CPU and CUDA `scripts/smoke.sh` each completed 200 updates, stopping/resuming at 50, followed by generation. CPU final-step loss 0.0420020707 and validation loss 0.6915101148 exactly match the original run. CUDA final-step loss 0.0420667529, validation 0.6940225884; comparisons across FP32/BF16 are not bitwise requirements. Full-held-out evaluation of the preserved CPU checkpoint scored loss 0.70108353 over 640 tokens (different from the trainer's four-batch prefix).

Primary bounded post-overhaul benchmark: 20,800.99 tokens/s, 98.45684 ms, 6,566,576,128 allocated bytes. Fresh pre-overhaul baseline: 21,485.39 tokens/s, 95.32057 ms, 6,561,824,768 bytes. Throughput difference about -3.2%; memory difference below 0.1%. This short comparison does not establish sustained performance. Historical results remain unchanged. No large training run occurred.

Package wheel/sdist build succeeds; sdist now includes conftest, configs and corpus. Installed-wheel import/CLI and fixture roundtrips are checked outside the source checkout. Local environment is Python 3.13.14 / PyTorch 2.11.0+cu130 / RTX 4080. Python 3.11/3.12 CPU jobs and hosted workflows are configured but have not executed here; no GPU runner has been configured remotely.

## Remaining blockers and next milestone
No actual pretrained model, production dataset/16K tokenizer, approved weight license, broad evaluation or publication authorization exists. Private GitHub vulnerability reporting is disabled and needs authorized setup. Hosted CI requires a later authorized push. Public base-model evidence and licensing gates cannot be satisfied by the random fixture. Large-data ingestion remains in memory; this overhaul does not solve streaming, contamination beyond exact duplicates, or general capability evaluation.

Next: M3 bounded-memory tokenizer/data preparation, source/license records and frozen contamination-resistant holdouts. The numbered M1–M11 roadmap supplies dependencies, risks and acceptance criteria. Nothing was pushed or published; remote name, metadata and settings were not changed.
