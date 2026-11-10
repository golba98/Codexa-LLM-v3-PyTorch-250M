"""Bounded local CI equivalent; writes only ignored verification output."""

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

from codexa.repository import check_repository, source_identity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/logs/verification.json"))
    parser.add_argument(
        "--forward-only",
        action="store_true",
        help="Run reviewed non-training tests under backward/update guards",
    )
    args = parser.parse_args()
    check_repository()
    commands = [
        [sys.executable, "-m", "ruff", "check", "."],
        [sys.executable, "-m", "ruff", "format", "--check", "."],
        [sys.executable, "-m", "mypy", "codexa"],
        [sys.executable, "-m", "scripts.maintenance.render_reports", "--check"],
        [sys.executable, "-m", "pytest", "-q"],
    ]
    if args.forward_only:
        import os

        os.environ["CODEXA_NO_TRAINING"] = "1"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        nodes = [
            "tests/test_reporting.py",
            "tests/test_streaming.py",
            "tests/test_language_protocol.py",
            "tests/test_language_data.py",
            "tests/test_common_corpus.py",
            "tests/test_language_sources.py",
            "tests/test_prior_candidate_holdouts.py",
            "tests/test_language_evaluation.py",
            "tests/test_language_readiness.py",
            "tests/test_remediation.py",
            "tests/test_corpus_review.py",
            "tests/test_quality_protocol.py",
            "tests/test_semantic_admission.py",
            "tests/test_v2.py",
            "tests/test_sampling.py",
            "tests/test_architecture.py::test_counts",
            "tests/test_architecture.py::test_forward_tying_causal",
            "tests/test_architecture.py::test_attention_reference",
            "tests/test_architecture.py::test_rope",
            "tests/test_architecture.py::test_norm_swiglu",
            "tests/test_architecture.py::test_invalid",
            "tests/test_pipeline.py::test_tokenizer_packing",
            "tests/test_pipeline.py::test_atomic_failure",
            "tests/test_pipeline.py::test_generation",
            "tests/test_pipeline.py::test_math",
            "tests/test_pipeline.py::test_sampler",
            "tests/test_context_dashboard.py::test_cache_and_positions",
            "tests/test_context_dashboard.py::test_sampled_parity_and_cache_owner",
            "tests/test_data.py::test_curation_resume_and_integrity",
            "tests/test_data.py::test_duplicate_holdout",
            "tests/test_data.py::test_curation_io_tuning_preserves_resume",
            "tests/test_data.py::test_completed_curation_does_not_add_indexes",
            "tests/test_data.py::test_permutation",
            "tests/test_data.py::test_formats",
            "tests/test_data.py::test_near_and_embedded_holdouts",
            "tests/test_data.py::test_hf_staging",
            "tests/test_data.py::test_interrupted_preparation",
            "tests/test_data.py::test_invalid_unicode_records",
            "tests/test_data.py::test_parent_group_isolation",
            "tests/test_data.py::test_checksums_do_not_replace_boundary_validation",
        ]
        code = (
            "from codexa.safety import forward_only; forward_only(); import pytest; raise SystemExit(pytest.main("
            + repr(["-q", *nodes])
            + "))"
        )
        commands[-1] = [sys.executable, "-c", code]
    for cmd in commands:
        subprocess.run(cmd, check=True)
    report = {
        "status": "passed",
        "scope": "forward-only; training tests deliberately excluded"
        if args.forward_only
        else "full",
        "source": source_identity(),
        "python": platform.python_version(),
        "commands": commands,
    }
    path = args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
