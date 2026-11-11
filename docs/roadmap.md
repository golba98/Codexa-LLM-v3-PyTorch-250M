# Research roadmap

The immediate goal is coherent general language. Conversational post-training and Grade 1–12 mathematics follow their own evaluation and data-admission decisions.

[Documentation](index.md) · [Research reports](../reports/README.md)

## Where the project stands

| Area | Recorded state | Evidence or guide |
| --- | --- | --- |
| Implementation | Custom Transformer, tokenizer pipeline, streaming and native recovery are implemented | [Architecture](architecture.md) |
| Base training | 500,006,912 target tokens completed | [Base run](../reports/training/base-500m.md) |
| Language generation | Baseline coherence and repetition remain significant limitations | [Evaluation](../reports/evaluation/language-baseline.md) |
| Continuation | Exactly 500M additional English targets; confirmed running on 11 November 2026 | [Training](training.md#language-continuation) |
| Mathematics | Materials and tooling exist; final corpus admission is pending | [Mathematics status](../reports/datasets/mathematics-corpus.md) |
| Public weights | Capability evidence and licensing review remain outstanding | [Release](release.md) |

## Language learning

The current continuation uses the reviewed, pinned English Wikipedia stream. It preserves the parent checkpoint, tokenizer, architecture, evaluation exclusions and native recovery. The fixed budget ends at **1,000,006,912 cumulative targets**; evaluation and follow-on experiments do not start automatically.

After the continuation, compare saved generations with the preserved baseline using identical protected prompts and decoding settings, with independent checkpoint-blinded scoring. Keep grammar, coherence, relevance, completeness and task success separate.

### Before conversational post-training

The base-language gate requires **100 fresh protected development prose completions**, at least **80 grammatical, coherent and complete responses**, and no more than **10 severe loops**. Existing assistant-task outputs do not substitute for that panel.

Before supervised fine-tuning (SFT), select a separate token budget, review dialogue rights and formatting, and define an assistant-only loss mask. A larger pretraining token count alone does not satisfy the language gate.

## Grade 1–12 mathematics

Develop the mathematics corpus independently of the language run. Remaining work includes:

- Reconcile the CAPS/ATP curriculum objective denominator before claiming coverage.
- Resolve visual and extraction gaps, quarantines and review disagreements.
- Preserve source grounding, exact editions, rights, mappings and protected splits.
- Verify synthetic explanations and intermediate steps independently of numeric answers.

Specialization requires a separately admitted corpus and explicit training authorization. Evaluate both general language and earlier mathematics topics afterward to measure forgetting. The [dataset guide](datasets.md#mathematics-corpus) explains the materials and review boundaries.

## Publication

Public model release still depends on reliable generation, broader independent evaluation, provenance and approved weight licensing. Architecture support for a context length or runtime is insufficient evidence of learned capability or compatibility.

There are no public trained weights or claimed Transformers/runtime integrations. Follow the [release guide](release.md) when the research and licensing evidence is ready.

## Historical plans

Earlier 50M/100M candidates, the superseded 1B proposal and 5B projections remain in the [research archive](../reports/archive/index.json). They describe their original decision state. Estimates remain separate from measured sustained performance.
