# Repository audit

Baseline: b8588ed, clean main, inspected before migration. Python 3.13.14, PyTorch 2.11.0+cu130, RTX 4080 16GB, driver 615.78.08. No ancestor or repository agent instructions existed.

Baseline commands: `pytest -q` (23 passed, including CUDA), `ruff check .`, `ruff format --check .`, `mypy codexa` (25 modules), `uv pip check` (compatible), `uv build` (wheel and sdist built). No inherited test failures. Fresh CPU and CUDA training stopped at step three and resumed to six, with inference working. Primary parameter count: 248,687,488. CPU seed 42, one thread, input arange(32).reshape(2,16): logits SHA256 be43fd6ccdff542397b182c8c81ea359ad99d21d04ed19bb2a23b20e85147f6d, loss 6.247674942016602.

Architecture, BPE, packing, training, checkpoints, inference and arithmetic prototype work. Checkpoint version 1 contains optimizer/scheduler/RNG/sampler recovery state. The local CPU checkpoint is at 200 smoke updates, CUDA checkpoint at 20. Neither is a pretrained 250M model. Tokenizer training repeated identically and matched the saved tokenizer.

Confirmed deficiencies: minimal ignores; no agent workflows, CI, license, export, release preflight or comprehensive packaging metadata. Sdist lacks conftest, configs and corpus. Existing runtime commands can replace files. Dataset ingestion holds text in RAM and validation examines a bounded prefix. Empty evaluation/utils packages provide no functionality.

All five runtime benchmark copies match tracked curated JSON exactly. Existing tracked files contain no runtime checkpoints or token arrays. Original artifacts occupy approximately 96KB and runs 3.3MB; .venv is local and ignored. Existing lock is a Linux/Python 3.13/CUDA profile, not a portable CPU lock.

Remote inspected read-only: public golba98/Codexa-LLM-v3-PyTorch-250M, main; no description, topics or license. Private vulnerability reporting is disabled. Remote configuration must not change in this task.

Fresh baseline primary benchmark: context 2048, micro-batch one, accumulation one, one warmup and three measured AdamW steps: 21,485.39 tokens/s, 95.32057 ms/update, 6,561,824,768 peak allocated CUDA bytes. Historical benchmarks remain evidence, with original source-tree identity incompletely recorded; do not retrospectively invent that identity.
