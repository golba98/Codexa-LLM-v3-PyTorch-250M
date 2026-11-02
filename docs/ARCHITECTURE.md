# Architecture

Codexa implements its own decoder-only Transformer. Inputs `[B,T]` become token embeddings `[B,T,D]`; no learned positional embeddings or linear biases are used.

For each block:

```
u = x + Attention(RMSNorm(x))
y = u + Down(SiLU(Gate(RMSNorm(u))) * Up(RMSNorm(u)))
```

RMSNorm computes `x / sqrt(mean(x²) + epsilon) * scale`, using FP32 reductions, epsilon 1e-6, and one learned scale per feature. Final RMSNorm precedes the tied output matrix. No additional block dropout is used; configurable attention dropout defaults to zero.

Attention projects packed Q/K/V, reshapes each to `[B,H,T,D/H]`, rotates Q and K, and computes `softmax(Q Kᵀ / sqrt(D/H) + causal_mask) V`. SDPA dispatches to an available optimized backend; no particular fused kernel is guaranteed. The causal mask permits only positions `j <= i`. Evaluation always passes dropout probability zero.

RoPE rotates adjacent pairs `(a,b)` into `(a cos(theta)-b sin(theta), a sin(theta)+b cos(theta))`, with `theta = position / 10000^(2k/head_dim)`. Positions start at zero for each window. Head dimension must be even. Frequencies and trigonometry use FP32; rotations return the input dtype. RoPE buffers are nonpersistent and carry no trainable parameters.

Embeddings and projection weights start with independent normal draws, standard deviation 0.02. Attention output and SwiGLU down projections use `0.02 / sqrt(2L)`. Norm scales start at one. Tied embedding/head weights are initialized once and included once in AdamW. There are no pretrained components or calls to existing Transformer modules.

## Parameter accounting

For vocabulary V, width D, FFN width F, and L blocks:

`P = VD + L(4D² + 3DF + 2D) + D`

| Component | Primary parameters |
|---|---:|
| Shared token embedding/output | 14,680,064 |
| Attention projections, 24 blocks | 77,070,336 |
| SwiGLU projections, 24 blocks | 156,893,184 |
| 49 RMSNorm scales | 43,904 |
| Total | **248,687,488** |

Primary dimensions are L=24, D=896, H=14, head dimension=64, F=2432, V=16384, context=2048. The intermediate model has 49,295,872 parameters; smoke has 133,440. Meta-device model construction counts actual parameters without allocating primary weights during unit tests. Benchmarks allocate real models.

## Objective and inference

For packed tokens `s[0:T+1]`, inputs are `s[0:T]` and targets `s[1:T+1]`. The loss is mean negative log likelihood over every target position; no second shift occurs inside the loss. Packing deliberately permits context across EOS/BOS boundaries. Accumulated equal-sized micro-batch losses divide by accumulation count before backward.

Inference repeatedly evaluates the current context, then selects the final-position logits. Temperature precedes top-k and nucleus filtering. Nucleus filtering retains the token crossing the probability threshold. Greedy decoding uses argmax. EOS ends generation. M2.5 replaces sliding truncation with explicit budget enforcement and adds per-request KV caching, absolute positions, opt-in linear scaling and block activation checkpointing. See M25_CONTEXT_AND_TELEMETRY.md.
