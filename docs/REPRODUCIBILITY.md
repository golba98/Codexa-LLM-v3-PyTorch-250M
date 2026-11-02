# Reproducibility

Reference GPU profile: Python 3.13, Linux x86_64, PyTorch 2.11.0+cu130, RTX 4080. requirements-lock.txt preserves that environment. requirements-cpu-lock.txt is the bounded CPU development profile for Python 3.11–3.13. Refresh explicitly with scripts/lock_dependencies.py; review all differences. Do not upgrade dependencies merely to obtain newer versions.

Run `python scripts/verify.py`; its report records commit, tracked-source digest, interpreter, environment and exact commands. CPU checks require deterministic seeds and equal outputs in the same environment. CUDA comparisons use tolerances and record driver/precision; bitwise identity across kernels/hardware is not promised.

Native checkpoints restore optimizer, scheduler, RNGs and sampler as well as weights. Preserve configuration and data hashes. Public inference exports omit recovery state. Historical benchmarks have incomplete original source-tree metadata; preserve those records rather than invent provenance. New benchmark wrappers record configuration/source/environment alongside measurements. See engineering/REPOSITORY_AUDIT.md and benchmarks/README.md.
