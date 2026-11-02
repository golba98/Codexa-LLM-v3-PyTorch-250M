# Evaluation

`codexa evaluate --export DIR --data outputs/datasets/smoke` computes token-weighted held-out cross-entropy and perplexity over all complete validation sequences. `--checkpoint` supports native checkpoints instead; `--max-batches` explicitly bounds evaluation. Output identifies data fingerprints, token counts, sequence and model size. A small loss on the repeated smoke corpus is not evidence of general language competence.

`codexa math evaluate` scores supplied integer answers by grade/topic; an answer-key test validates the harness, not model skill. Keep canonical identities and future template/operand-range holdouts separate from training. Reevaluate earlier topics and general language after each curriculum stage to track forgetting. Broad capability, bias and safety evaluation remain future work.
