# Implementation report

Implemented and verified locally on 2026-11-02. The current directory was empty; Git was initialized there without touching Codexa v1/v2. Branch: `main`. Origin: `https://github.com/golba98/Codexa-LLM-v3-PyTorch-250M.git`.

## Architecture and interfaces

The repository implements token embeddings, adjacent-pair RoPE, causal MHA using PyTorch SDPA, RMSNorm, SwiGLU, pre-normalized residual blocks, and a tied output head. All projections are bias-free. All weights start randomly. Primary dimensions: 24 layers, width 896, 14 heads × 64 dimensions, FFN width 2432, vocabulary 16384, context 2048. The actual instantiated parameter count is **248,687,488**, matching the requested count exactly. The primary GPU benchmark allocated this real model and performed forward/backward/optimizer updates. Smoke and intermediate counts are 133,440 and 49,295,872.

`Codexa.forward` maps `[batch, sequence]` token IDs to `[batch, sequence, vocabulary]` logits. Typed YAML configurations feed tokenizer/data/training workflows; the `codexa` CLI provides tokenizer training, packing, pretraining/resume, generation, benchmarks, arithmetic generation, and categorized evaluation. Full equations and the component accounting are in ARCHITECTURE.md.

## Verification performed

- `pytest -q`: **23 passed**, including the CUDA/BF16 test; final run completed in 1.22 seconds.
- `ruff check .` and `ruff format --check .`: passed.
- `mypy codexa`: passed across 25 source files with installed PyTorch types.
- `git diff --check`: passed.
- Byte BPE trained to exactly 512 tokens on the training documents only; Unicode, mathematical symbols, whitespace, and save/reload round trips passed.
- Original corpus: 11 training documents / 7,525 tokens; one held-out document / 697 tokens. At sequence 64, unused trailing counts are 36 and 56 respectively.
- CPU corpus training: 200 optimizer updates, stopped/checkpointed at 50 and resumed to 200. Total: 25,600 target tokens. First-step loss **6.24101**, final-step loss **0.04200**, final held-out loss **0.69151**. Different batches contribute to first/final training-step loss; these are not a fixed-batch comparison.
- Independent fixed tiny sample: 100 updates, loss **6.161198 → 0.012485**, a reduction of about **99.80%**. This is the explicit overfit demonstration.
- CUDA BF16 corpus training: 10 steps, checkpoint/resume to step 20; finite training and held-out loss. This validates the GPU training path separately from synthetic benchmarks.
- Greedy and sampled inference CLIs successfully loaded the actual smoke checkpoint. Output repeated corpus text, which demonstrates memorization only.
- Resumed CPU weights, Adam state, scheduler, sampler, token counts, and subsequent Python/NumPy/PyTorch random draws matched an uninterrupted run exactly, with dropout and accumulation enabled.
- Checkpoint failed-write simulation retained the previous file; SIGINT during backward completed one optimizer step before saving and restored the prior signal handler.
- Arithmetic CLI generated 20 held-out problems and evaluated a deliberately supplied answer-key prediction file at 20/20. This checks the scoring harness, **not model mathematics accuracy**. Tests independently checked arithmetic solutions, deterministic generation, disjoint identities, missing answers, and finite pool exhaustion.

## Measured performance

Python 3.13.14, PyTorch 2.11.0+cu130, CUDA BF16 autocast on NVIDIA RTX 4080 16GB; parameters/gradients/Adam state FP32. CLI CPU thread count: four. Benchmarks use synthetic token IDs and real AdamW updates. All micro-batches contain one sequence.

| Benchmark | Parameters | Sequence | Accumulation | Tokens/s | Update ms | Peak allocated CUDA |
|---|---:|---:|---:|---:|---:|---:|
| 250m-acc16-cuda | 248,687,488 | 2048 | 16 | 26,357 | 1,243.22 | 7.026 GiB |
| 250m-cuda | 248,687,488 | 2048 | 1 | 21,460 | 95.43 | 6.116 GiB |
| 50m-cuda | 49,295,872 | 1024 | 1 | 57,521 | 17.80 | 1.223 GiB |
| smoke-cpu | 133,440 | 64 | 1 | 29,317 | 2.18 | N/A |
| smoke-cuda | 133,440 | 64 | 1 | 15,865 | 4.03 | 0.018 GiB |

Each standard smoke/intermediate benchmark used two warmup updates and five measured updates. Primary accumulation-one used one warmup and three measured updates; accumulation-16 used one warmup and two measured updates. These are short measurements, not sustained research-run throughput. Peak allocated CUDA memory excludes driver/context memory and allocator reservation. Results are retained as JSON in `docs/benchmarks`.

The primary accumulation-16 run measured 32,768 target tokens per optimizer update, roughly 26,357 tokens/s, and 7.026 GiB peak allocated memory. Thus this specific configuration fits on this machine; larger micro-batches or other settings require fresh measurement. Five billion tokens would be about 52.7 hours at that short benchmark's throughput: an extrapolation excluding validation, checkpoints, data overhead, thermal variation, and failures. The independent dense-model compute estimate is 7.4606e18 FLOPs. No five-billion-token run was performed.

## Reproduction commands

Activate `.venv`, then run:

```sh
pytest -q
ruff check .
ruff format --check .
mypy codexa
sh scripts/smoke.sh
codexa benchmark --config configs/250m.yaml --device cuda --sequence 2048 --accumulation 16 --steps 2 --warmup 1
codexa generate --checkpoint runs/smoke/latest.pt --prompt 'Morning light' --max-tokens 20 --top-k 10 --top-p 0.9 --seed 42
```

The smoke script trains the tokenizer, packs data, trains to step 50, resumes to 200, and generates text. The README provides separate commands for every stage. The environment lock records exact dependencies; the general installation remains compatible with other supported PyTorch environments. Initial package downloads were slow; the tested environment reused a cached official PyTorch CUDA build through uv 0.11.0. No model checkpoint or pretrained tokenizer was imported.

## Limitations and next milestone

This is a working training foundation, not a pretrained base model. The sample corpus is small, repetitive, and shares wording across document splits; its held-out loss is unsuitable for assessing generalization. The 250M model received only synthetic benchmark updates, with those temporary weights discarded. No real 16K production tokenizer or large dataset is supplied.

The implementation uses a single process, no data workers, no KV cache, no activation checkpointing, no distributed training, no compilation, and no automatic checkpoint retention. Corpus ingestion holds document text in memory even though token output is streamed and training arrays are memory-mapped. Validation uses a bounded configurable prefix of held-out sequences. Exact resume requires unchanged configuration, data, device type, and precision; CUDA bitwise reproducibility across hardware/software is not guaranteed. Absolute throughput should be remeasured under sustained training conditions.

Mathematics is limited to a Grade 1 synthetic prototype. The Grade 1–12 roadmap is documented, but educational suitability and broader step verification require future work. No mathematics-focused training was performed.

Recommended next milestone: production tokenizer research and a permitted, deduplicated, versioned corpus with bounded-memory preprocessing; benchmark short 50M pilots, freeze independent evaluation sets, then budget primary general-language pretraining. Preserve the validation gates before mathematics training.

## Files created

Generated environments, tokenizer/data artifacts, and smoke checkpoints remain local and ignored. The tracked foundation files are:

```text
.gitignore
README.md
codexa/__init__.py
codexa/architecture/__init__.py
codexa/architecture/attention.py
codexa/architecture/model.py
codexa/architecture/normalization.py
codexa/architecture/rope.py
codexa/architecture/transformer.py
codexa/benchmark.py
codexa/cli.py
codexa/config.py
codexa/data/__init__.py
codexa/data/packing.py
codexa/evaluation/__init__.py
codexa/inference/__init__.py
codexa/inference/generate.py
codexa/mathematics/__init__.py
codexa/mathematics/arithmetic.py
codexa/tokenizer/__init__.py
codexa/tokenizer/bpe.py
codexa/training/__init__.py
codexa/training/checkpoint.py
codexa/training/optimizer.py
codexa/training/scheduler.py
codexa/training/trainer.py
codexa/utils/__init__.py
configs/250m.yaml
configs/50m.yaml
configs/smoke.yaml
docs/ARCHITECTURE.md
docs/IMPLEMENTATION_REPORT.md
docs/TRAINING_ROADMAP.md
docs/benchmarks/bench-250m-acc16-cuda.json
docs/benchmarks/bench-250m-cuda.json
docs/benchmarks/bench-50m-cuda.json
docs/benchmarks/bench-smoke-cpu.json
docs/benchmarks/bench-smoke-cuda.json
docs/benchmarks/verification.json
examples/corpus/00.txt
examples/corpus/01.txt
examples/corpus/02.txt
examples/corpus/03.txt
examples/corpus/04.txt
examples/corpus/05.txt
examples/corpus/06.txt
examples/corpus/07.txt
examples/corpus/08.txt
examples/corpus/09.txt
examples/corpus/10.txt
examples/corpus/11.txt
examples/corpus/PROVENANCE.md
pyproject.toml
requirements-lock.txt
scripts/smoke.sh
tests/conftest.py
tests/test_architecture.py
tests/test_pipeline.py
```
