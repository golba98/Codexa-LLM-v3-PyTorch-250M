# Changelog

## Unreleased

Terminal rendering reads the actual output PTY dimensions on every refresh, including when inherited COLUMNS/LINES describe another terminal. Regression tests exercise conflicting environment sizes.

CI: preserve real terminal startup frames and capture completed VT frames independently of PTY read chunking; expose child-process failure diagnostics. Keep all Python matrix jobs running to completion without duplicate feature-branch push runs. NumPy RNG checkpoints explicitly retain their legacy tuple contract.


M3: consolidated five public guides and a checksummed history archive; extended ignores; disk-backed curation, protected splits, exact/approximate deduplication, candidate BPE fitting/freezing, uint16 indexed shards, bounded-memory permutation sampling and compatible training/evaluation integration. Small real sample, CPU/CUDA and recovery checks only; no base pretraining or model release.


Repository preservation, agent workflows, pinned development profiles, CI, safe inference exports, release preflight and public engineering documentation. No trained model release.

## 0.1.0 foundation

Custom Transformer, tokenizer, training/resume, inference, arithmetic prototype and measured smoke/primary benchmarks.
