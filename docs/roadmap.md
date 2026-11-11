# Research roadmap

## Current state

The custom architecture, tokenizer, native recovery and streaming implementations exist. An internal 500,006,912-token run completed; its language generation remains weak. The selected pinned English Wikipedia source has passed the approved corpus review. The owner replaced the earlier 1B proposal with exactly 500M additional language tokens, using rolling recovery and the existing terminal viewer after mandatory preflight. See [readiness](../reports/language/continuation-readiness.md).

## Language workstream

1. Preserve the completed repository cleanup, trained baseline and approved source evidence.
2. Bind the explicit 500M authorization, verify optimizer/recovery on the RTX 4080 once, and launch a fresh streaming phase with two retained full recovery points.
3. Retain tokenizer, architecture, protected evaluations and exact recovery; stop after the fixed budget. Codex exits after confirming launch. Generation evaluation and subsequent work await the owner’s return.
4. Before conversational SFT, satisfy the separately specified base-language gate: 100 fresh protected development prose completions, at least 80 grammatical/coherent/complete responses and at most ten severe loops. Existing assistant-task outputs are not that fresh panel.
5. Choose a concrete SFT budget and review dialogue rights, formatting and assistant-only masking before introducing a new objective. Larger language budgets and publication require their own decisions.

Historical 50M/100M candidates and 5B projections are retained in the [archive](../reports/archive/index.json). They are not simultaneous current training instructions. Estimates must stay separate from measured sustained performance.

## Mathematics workstream

Develop Grade 1–12 materials independently of the language run. Preserve source grounding, exact editions, rights, curriculum mappings, split protections and independent review. Resolve visual/extraction gaps and reconcile the CAPS/ATP objective denominator before claiming coverage. Synthetic answers need independent verification of steps and explanations as well as numeric results.

Only a separately approved corpus and explicit mathematics training authorization can initiate specialization. Reevaluate general language and earlier mathematics topics to measure forgetting; a larger token count alone does not establish competence. See [mathematics status](../reports/datasets/mathematics-corpus.md).

## Release and remaining limits

Current blockers include reliable coherent generation, broader independent capability evaluation, mathematics admission and weight-publication licensing. The 32K profile is architectural support, not proof of trained 32K reasoning. No pretrained public model, Transformers runtime compatibility or production assistant capability is claimed. Publication follows the [release guide](release.md), with separate permission for weights and datasets.
