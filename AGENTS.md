# Codexa v3

Build a custom PyTorch decoder-only model from random initialization, targeting 248,687,488 parameters, then general language and Grade 1–12 mathematics research. Never alter other Codexa generations.

## Correctness and reproducibility

Keep architecture in codexa/architecture; training, tokenization, data, inference and release have separate modules. Use Python 3.11–3.13, type hints and small explicit interfaces. Do not silently change primary dimensions, initialization, objective, special IDs or configuration defaults. Never substitute pretrained implementations or weights, fabricate benchmark results, infer competence from random inference, or introduce evaluation leakage. Preserve native v1 checkpoints and exact CPU resume. Document intentional compatibility changes.

## Verification and dependencies

Run pytest, Ruff check/format check, mypy and scripts/verify.py after relevant source changes. Use bounded CPU tests first; GPU jobs must be coordinated, never concurrently launched by independent agents. Record memory and throughput separately from estimates. Do not launch expensive training without authorization or overwrite valuable outputs. Use fresh run directories. Preserve dependency pins; update locks explicitly and review the diff. Record dataset origin, license, hashes, deduplication and split policy. Never download large datasets automatically.

## Safety and collaboration

Preserve unrelated changes. Use separate branches/worktrees for concurrent editing. Assign ownership: architecture/correctness; training/performance; data/mathematics; release/docs. Coordinate edits to CLI/configs and run integration checks before combining work. Keep scratch/transcripts under .agents/scratch (ignored); maintain only reviewed docs under docs and reusable workflows under .agents/skills. Never add nested agent instructions without a deliberate ignore exception.

Never push, deploy, publish, create a public release or change remote metadata without explicit authorization. Do not put credentials, private data, weights or exports into Git. Public exports use SafeTensors, not native training checkpoints. Code Apache-2.0, original fixtures CC0; future dataset and weight licenses require separate review. Do not claim Transformers/runtime compatibility without tests. Report failures, missing evidence and unfinished work accurately. Keep README commands executable and update architecture, provenance and release docs when behavior changes.
