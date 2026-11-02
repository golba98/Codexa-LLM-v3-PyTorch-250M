# Benchmark evidence

Existing JSON and the historical implementation report remain unchanged. Their method is synthetic tokens with real forward/backward/AdamW updates, synchronized CUDA timing, FP32 state and BF16 autocast where supported. Original hardware: RTX 4080, Python 3.13.14, PyTorch 2.11.0+cu130; CLI CPU threads four. Exact executed dirty-source identities were not recorded. The foundation commit is an associated code snapshot, not an invented original run identity.

Peak allocated memory excludes context, driver and reserved memory. Short-run extrapolations exclude validation, checkpoints and sustained load effects. No benchmark means pretraining occurred. Runtime duplicates are preserved under outputs/benchmarks/historical. New measurements must include source/configuration/environment provenance and distinguish estimates from measurements. See ../engineering/REPOSITORY_AUDIT.md for fresh baseline results.
