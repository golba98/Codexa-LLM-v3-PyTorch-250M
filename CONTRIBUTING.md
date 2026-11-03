# Contributing

Create a branch or worktree; read AGENTS.md. Install a pinned environment as described in README. Run `python scripts/verify.py` before submitting. Preserve parameter counts, native checkpoints and corpus splits. Add meaningful regression tests and document numerical or compatibility changes. Contributions to code are under Apache-2.0; do not contribute third-party data without provenance and distribution permission.

Use one worktree per concurrent editor and agree ownership of architecture, training, data/math, or release/docs. Coordinate CLI/config changes and GPU jobs. Integrate only after the combined suite passes. Write task notes under ignored .agents/scratch, not scattered Markdown files. Changelog entries go under Unreleased; describe behavior and validation, not conversation history. Do not publish automatically.
