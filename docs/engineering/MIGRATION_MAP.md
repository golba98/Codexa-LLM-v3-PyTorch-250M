# Migration map

Runtime categories are untracked and moved intact; checkpoints and metrics remain colocated for native API compatibility. Tracked source, configs, corpus, and historical reports remain at their existing paths. Curated benchmarks stay in docs/benchmarks; identical disposable copies are retained under outputs. No existing data was deleted.

| Original | Destination | Purpose / preservation |
|---|---|---|
| artifacts/data | outputs/datasets/smoke | Runtime evidence; exact SHA256 verification |
| artifacts/tokenizer.json | outputs/tokenizers/smoke/tokenizer.json | Runtime evidence; exact SHA256 verification |
| artifacts/arithmetic-test.jsonl | outputs/datasets/arithmetic-test.jsonl | Runtime evidence; exact SHA256 verification |
| artifacts/predictions.json | outputs/predictions/arithmetic-test.json | Runtime evidence; exact SHA256 verification |
| runs/smoke | outputs/experiments/smoke | Runtime evidence; exact SHA256 verification |
| runs/cuda-smoke | outputs/experiments/cuda-smoke | Runtime evidence; exact SHA256 verification |
| artifacts/bench-smoke-cpu.json | outputs/benchmarks/historical/bench-smoke-cpu.json | Runtime evidence; exact SHA256 verification |
| artifacts/bench-smoke-cuda.json | outputs/benchmarks/historical/bench-smoke-cuda.json | Runtime evidence; exact SHA256 verification |
| artifacts/bench-50m-cuda.json | outputs/benchmarks/historical/bench-50m-cuda.json | Runtime evidence; exact SHA256 verification |
| artifacts/bench-250m-cuda.json | outputs/benchmarks/historical/bench-250m-cuda.json | Runtime evidence; exact SHA256 verification |
| artifacts/bench-250m-acc16-cuda.json | outputs/benchmarks/historical/bench-250m-acc16-cuda.json | Runtime evidence; exact SHA256 verification |

Full preservation inventory: [migration-hashes.json](migration-hashes.json).
