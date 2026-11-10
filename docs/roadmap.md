# Research roadmap

## Current state

The custom architecture, tokenizer, native recovery and streaming implementations exist. An internal 500,006,912-token run completed; its language generation remains weak. The selected pinned English Wikipedia source has passed the approved corpus review. The owner authorized a future 1B additional-token continuation, but cleanup execution is prohibited and the saved machine authorization still requires later finalization. See [readiness](../reports/language/continuation-readiness.md).

## Language workstream

1. Complete repository cleanup, CPU verification and public review while preserving the trained baseline and all evidence.
2. In a later launch-preparation task, finalize the bound authorization, revalidate source/rights/capacity/compute/recovery gates, and prepare a fresh native phase. No training starts from this roadmap.
3. For the authorized continuation, retain the frozen tokenizer, architecture, protected evaluations and exact native recovery; evaluate actual saved generations against the unchanged baseline.
4. Before conversational SFT, satisfy the separately specified base-language gate: 100 fresh protected development prose completions, at least 80 grammatical/coherent/complete responses and at most ten severe loops. Existing assistant-task outputs are not that fresh panel.
5. Choose a concrete SFT budget and review dialogue rights, formatting and assistant-only masking before introducing a new objective. Larger language budgets and publication require their own decisions.

Historical 50M/100M candidates and 5B projections are retained in the [archive](../reports/archive/index.json). They are not simultaneous current training instructions. Estimates must stay separate from measured sustained performance.

## Mathematics workstream

Develop Grade 1–12 materials independently of the language run. Preserve source grounding, exact editions, rights, curriculum mappings, split protections and independent review. Resolve visual/extraction gaps and reconcile the CAPS/ATP objective denominator before claiming coverage. Synthetic answers need independent verification of steps and explanations as well as numeric results.

Only a separately approved corpus and explicit mathematics training authorization can initiate specialization. Reevaluate general language and earlier mathematics topics to measure forgetting; a larger token count alone does not establish competence. See [mathematics status](../reports/datasets/mathematics-corpus.md).

## Release and remaining limits

Current blockers include reliable coherent generation, broader independent capability evaluation, mathematics admission and weight-publication licensing. The 32K profile is architectural support, not proof of trained 32K reasoning. No pretrained public model, Transformers runtime compatibility or production assistant capability is claimed. Publication follows the [release guide](release.md), with separate permission for weights and datasets.
