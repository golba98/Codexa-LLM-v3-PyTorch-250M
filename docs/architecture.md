# Architecture

Codexa's primary model is a **248,687,488-parameter decoder-only Transformer** implemented in [codexa/architecture](../codexa/architecture/). It uses custom PyTorch modules and begins training from random initialization.

[Documentation](index.md) · [Training](training.md) · [Measured benchmarks](../reports/benchmarks/recorded-performance.md)

## Primary configuration

| Dimension | Value |
| --- | ---: |
| Decoder layers, L | 24 |
| Hidden width, D | 896 |
| Attention heads, H | 14 |
| Head dimension | 64 |
| Feed-forward width, F | 2,432 |
| Vocabulary, V | 16,384 |
| Training context | 2,048 |

The [primary](../configs/250m.yaml), [intermediate](../configs/50m.yaml) and [smoke](../configs/smoke.yaml) profiles contain 248,687,488, 49,295,872 and 133,440 parameters respectively. Context profiles are in [configs/context](../configs/context/).

## Decoder block

Token IDs with shape `[B, T]` become embeddings with shape `[B, T, D]`. Each block applies attention and a gated feed-forward network with residual connections:

```text
u = x + Attention(RMSNorm(x))
y = u + Down(SiLU(Gate(RMSNorm(u))) * Up(RMSNorm(u)))
```

There are no learned positional embeddings, linear biases or additional block dropout. Final RMSNorm precedes the output projection, whose weights are shared with the input embedding.

### Normalization and feed-forward network

RMSNorm computes `x / sqrt(mean(x²) + epsilon) * scale`. It uses FP32 reductions, epsilon 1e-6 and one learned scale per feature.

The SwiGLU network projects the normalized input through separate gate and up matrices, multiplies `SiLU(gate)` by the up projection, and projects back to the hidden width.

### Causal attention

A packed projection produces Q, K and V, reshaped to `[B, H, T, D/H]`. Attention computes:

```text
softmax(Q Kᵀ / sqrt(D/H) + causal_mask) V
```

The mask permits only positions `j <= i`. Configurable attention dropout defaults to zero and is always zero during evaluation. PyTorch scaled dot-product attention selects an available backend; a particular fused kernel is not guaranteed.

### Rotary positions

RoPE rotates adjacent feature pairs:

```text
(a, b) → (a cos(theta) - b sin(theta), a sin(theta) + b cos(theta))
theta = position / 10000^(2k/head_dim)
```

Training-window positions start at zero; cached inference uses absolute positions. Head dimension must be even. Frequencies and trigonometry use FP32, and rotated values return to the input dtype. RoPE buffers are nonpersistent and have no trainable parameters.

## Initialization

| Weights | Initialization |
| --- | --- |
| Embeddings and ordinary projections | Independent normal draws, standard deviation 0.02 |
| Attention output and SwiGLU down projections | Standard deviation `0.02 / sqrt(2L)` |
| RMSNorm scales | Ones |

The tied embedding/output matrix is initialized once and included once in AdamW. No pretrained weights or existing Transformer implementations are substituted.

## Parameter accounting

For vocabulary V, hidden width D, feed-forward width F and L blocks:

`P = VD + L(4D² + 3DF + 2D) + D`

| Component | Parameters |
| --- | ---: |
| Shared token embedding/output | 14,680,064 |
| Attention projections across 24 blocks | 77,070,336 |
| SwiGLU projections across 24 blocks | 156,893,184 |
| 49 RMSNorm scales | 43,904 |
| **Total** | **248,687,488** |

Unit tests count actual parameters on the meta device without allocating primary weights. Benchmarks allocate real models; see the [benchmark report](../reports/benchmarks/recorded-performance.md) for measurement scope.

## Objective and generation

For packed tokens `s[0:T+1]`, inputs are `s[0:T]` and targets are `s[1:T+1]`. Training minimizes mean negative log likelihood over target positions. The loss does not shift targets a second time. Packing allows context across EOS/BOS boundaries; equal-sized microbatch losses are divided by the accumulation count before backward.

Generation selects from final-position logits. Temperature is applied before top-k and nucleus filtering; nucleus filtering retains the token that crosses the probability threshold. Greedy decoding uses argmax, and EOS ends generation.

Inference enforces explicit context budgets and supports a per-request key/value cache. Experimental linear RoPE scaling and block activation checkpointing are opt-in. Configured profiles up to 32K do not establish learned long-context capability.

## Continuation compatibility

Dataset-v2 and the current language continuation preserve the primary architecture, 16,384-token vocabulary, special IDs, 2,048-token training context and next-token objective. Continuation reuses the existing weights and frozen tokenizer. Model and tokenizer identities are validated before recovery or phase preparation.

See [native recovery](training.md#resume-a-run) for compatibility boundaries and [the roadmap](roadmap.md) for the research sequence.
