# Datasets

The only supplied corpus is twelve original fictional smoke documents under examples/corpus, licensed CC0-1.0. It is repetitive and unsuitable for language-generalization measurement. No production corpus or pretrained tokenizer is supplied.

Input is UTF-8 text documents or JSONL text records. Identical SHA256 document groups are deduplicated before deterministic splitting. The sorted-hash 10% holdout changes if documents are added; freeze the corpus. Fit BPE exclusively on training text. Prepared manifests bind token arrays to tokenizer fingerprints. BOS/document/EOS packing permits causal attention across document boundaries. Tails are unused, with count `(tokens-1) % sequence`.

Ingestion currently retains text in RAM. Large-scale work requires bounded-memory manifests/shards, source/license records, near-duplicate and contamination checks, frozen evaluation families, and review of redistribution rights. Arithmetic holdouts canonicalize addition operands and partition identities independently of random ordering; this does not establish generalization across equivalent word problems.
