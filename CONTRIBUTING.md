# Contributing

Read the [documentation contribution policy](README.md#documentation-contributions) and the [documentation index](docs/index.md). Use a dedicated branch; preserve existing work, native checkpoints, parameter counts, tokenizer identity, corpus splits and research evidence. Code contributions use Apache-2.0; original fixtures are CC0. Third-party data requires provenance and applicable permission.

## Environment and checks

Use the pinned environment in [README](README.md). Run from the repository root:

```sh
python scripts/verify.py
sh scripts/training/smoke.sh
python -m scripts.maintenance.render_reports --check
uv build
```

Verification covers Ruff lint/format, mypy, repository policy and CPU tests. The smoke command updates only a tiny fixture model in a fresh ignored directory. GPU work requires coordination; production language and mathematics training never follow automatically from passing tests.

Optional Parquet support uses `uv pip install -e '.[data]'`. This explicitly changes the local environment; review the lock profile before combining optional research dependencies. The core lock files stay pinned. Plotting additionally needs Matplotlib; selected local mathematics tools need BeautifulSoup and/or system `pdftotext` and retained worker files on the data volume. These are not required for core model inference and are not downloaded automatically.

## Tool locations

| Purpose | Location / example |
| --- | --- |
| Verification | `python scripts/verify.py` |
| Smoke and terminal acceptance | `scripts/training/` |
| Historical 50M/100M continuation checks | `python -m scripts.training.continuation_healthcheck --help` |
| Baseline analysis and plots | `scripts/evaluation/` |
| Review, token audits and semantic admission | `scripts/datasets/` |
| Mathematics extraction and recovery | `scripts/datasets/mathematics/` |
| Memory/dashboard benchmarks | `scripts/benchmarks/` |
| Storage inventory, report renderer, lock maintenance | `scripts/maintenance/` |

Developer scripts formerly at the scripts root have moved into these groups. Invoke scripts that import sibling tools with `python -m scripts.<group>.<module>` from the repository root. Model/configuration paths and the `codexa` CLI remain stable. Historical command strings inside immutable evidence retain their original spelling; resolve them through this table and the matching filename.

Original CC0 evaluation source fixtures now ship in `codexa/evaluation/assets` instead of `tests/fixtures`, so installed wheels retain the same fixture content. This changes the source-file location only; existing frozen evaluation payloads and checksums are not rewritten.

## Documentation and reports

The README introduces the project and quick start. Keep implementation and workflows in the five canonical guides linked from [the documentation index](docs/index.md); put dated research findings in [reports](reports/README.md). Use short introductions, task-oriented headings and descriptive links. Caption fixture screenshots with their model size so their metrics cannot be mistaken for primary-model results.

Report inputs live at `reports/<workstream>/evidence/*-report.json`. Regenerate them and verify consistency:

```sh
python -m scripts.maintenance.render_reports
python -m scripts.maintenance.render_reports --check
```

CI checks that Markdown matches its inputs. Every schema-v1 record identifies model/dataset versions, run/date, status, scoped metrics, methods, limitations, evidence hashes and next actions. Two optional presentation fields retain compatibility with existing records:

| Field | Default | Meaning |
| --- | --- | --- |
| `takeaway` | First summary entry | A concise finding for the report lead and index |
| `historical` | `false` | Label the report as a dated historical decision and group it separately |

Historical labels do not change recorded outcomes. Keep original budgets and decisions intact; connect superseded proposals to current guides. Reports display evidence locations and checksums in an expandable section. Regeneration never runs experiments.

Baseline analysis and storage inventory accept `--report NEW.md` to summarize original JSON; historical continuation preflight also supports `--report` while retaining its receipt.

Do not hand-edit generated Markdown, regrade historical failures, fabricate missing metadata or rewrite immutable evidence. Use `external:` locators relative to the private artifact root and `local:` for ignored repository evidence. They are identifiers, not claims that payloads ship in Git. A reviewed source's quality PASS, its rights scope, authorization and whether execution started are distinct decisions.

Keep working inventories/transcripts under ignored `.agents/scratch`; update CHANGELOG for public behavior and compatibility changes. Preserve original history in checksummed archives, not new session or handoff documents. Never add nested agent instructions without a deliberate ignore exception.

## Review and publication

Use separate worktrees for concurrent editors and agree ownership of architecture, training, data/mathematics and release/docs. Coordinate CLI/config edits and run integration checks before combining them. Add meaningful regression tests for functional changes. Review dependency-lock changes explicitly.

Before publication, inspect source, archives and package contents for secrets, private reviews, corpus payloads, weights and exports. Report previously published sensitive material for separate remediation; never rewrite history automatically. Push, release, deploy and remote metadata changes require explicit authorization. See [release policy](docs/release.md) and [security](SECURITY.md).
