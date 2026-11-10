"""Strict gates bind content assessment, independent ratings and full coverage."""

import copy

import pytest

from codexa.data.math_integrity import affine_coefficients, affine_findings
from codexa.data.quality_findings import findings
from codexa.data.quality_protocol import CHECKS, adjudicate, specification
from codexa.data.remediate import prove_quality_narrowing
from scripts.datasets.combine_quality_reviews import combined


def packet(kind="primary"):
    protocol = specification(kind)
    mapping = [
        {"id": f"w{i}", "kind": "window", "source": "mixture"} for i in range(protocol["windows"])
    ] + [
        {"id": f"{source}{i}", "kind": "document", "source": source, "coverage": "full document"}
        for source in protocol["sources"]
        for i in range(protocol["documents_per_source"])
    ]
    ratings = [
        {
            "id": r["id"],
            "usable": True,
            "severe": False,
            "critical": False,
            "coverage_complete": True,
            "checks": dict.fromkeys(CHECKS, "pass"),
            "reason": "Complete source read; coherent explanation and self-contained inputs.",
        }
        for r in mapping
    ]
    return protocol, mapping, ratings


@pytest.mark.parametrize("kind", ["primary", "extended"])
def test_complete_strict_packet_and_missing_ratings(kind):
    protocol, mapping, ratings = packet(kind)
    assert adjudicate(mapping, ratings, protocol)["training_allowed"]
    assert not adjudicate(mapping, ratings[:-1], protocol)["training_allowed"]


@pytest.mark.parametrize(
    "source,field,value",
    [
        ("finemath-4plus", "correctness", "defect"),
        ("finemath-4plus", "correctness", "uncertain"),
        ("finemath-4plus", "visuals", "defect"),
        ("finemath-4plus", "notation", "defect"),
        ("fineweb-edu", "education", "not_applicable"),
        ("fineweb-edu", "completeness", "defect"),
    ],
)
def test_readability_cannot_override_content_failure(source, field, value):
    protocol, mapping, ratings = packet()
    row = next(r for r in ratings if r["id"] == source + "0")
    row["checks"][field] = value
    assert not adjudicate(mapping, ratings, protocol)["training_allowed"]


def test_threshold_changes_coverage_and_protocol_tampering_fail_closed():
    protocol, mapping, ratings = packet()
    next(r for r in ratings if r["id"] == "finemath-4plus0")["severe"] = True
    assert not adjudicate(mapping, ratings, protocol)["training_allowed"]
    ratings[0]["coverage_complete"] = False
    assert not adjudicate(mapping, ratings, protocol)["training_allowed"]
    protocol["fineweb_usable_minimum"] = 34
    with pytest.raises(ValueError, match="frozen version"):
        adjudicate(mapping, ratings, protocol)


def test_conservative_independent_combination_preserves_concerns():
    protocol, mapping, ratings = packet()
    second = copy.deepcopy(ratings)
    second[0]["usable"] = False
    second[0]["checks"]["coherence"] = "uncertain"
    rows = combined([ratings, second])
    assert rows[0]["usable"] is False
    assert rows[0]["checks"]["coherence"] == "uncertain"
    assert adjudicate(mapping, rows, protocol)["reviewed"] == 500
    with pytest.raises(ValueError):
        combined([ratings])


@pytest.mark.parametrize(
    "text,reason",
    [
        ("Then teach the underlying concepts\nExplain in detail...", "unfilled_answer_editor"),
        (
            "A useful lesson.\n• 2 minutes ago\n• 4 minutes ago\n• 8 minutes ago",
            "interface_timestamp_contamination",
        ),
        ("Questions tagged [geometry]\n12 views\nOther prompts...", "search_catalogue"),
        ("An explanation of the history of a theorem ends here...", "truncated_document"),
        ("The polynomial x2 + x3 + x4 + y2 + y3 has lost its powers.", "flattened_power_notation"),
        (r"We know $\angle 1 + \angle 2 = \angle 3$.", "undefined_numbered_angles"),
        (
            "Compute -4/3 - 1 = -1/3. Thus this is the intercept.",
            "explicit_arithmetic_contradiction",
        ),
    ],
)
def test_integrity_defect_families(text, reason):
    assert reason in {f["reason"] for f in findings(text, "finemath-4plus")}


def test_web_extraction_damage_and_paywall_do_not_hide_content():
    assert "orphaned_html_closing_tags" in {
        f["reason"]
        for f in findings("Leaves use sunlight to grow.</section></article>", "fineweb-edu")
    }
    assert "paywalled_content_missing" in {
        f["reason"] for f in findings("Subscribe to read the full article.", "fineweb-edu")
    }
    assert not findings(
        "HTML tags include </section> and </article> in code examples.", "fineweb-edu"
    )


@pytest.mark.parametrize(
    "text",
    [
        "The sequence is 1, 2, 3, ...\nThe next term is 4.",
        "The positive integers are 1, 2, 3, ...",
        "A title...\nA complete explanation follows with its conclusion.",
        "A square of side 3 has area 9 because area is side times side. Download the worksheet for extra practice.",
        "In arithmetic modulo 5, we have 4 + 3 = 2. The residue is two.",
        "A learner wrote 1 + 1 = 3.\n\nThat is incorrect. The correct sum is 2.",
        r"The polynomial $x^{2}+3x+2$ factors as $(x+1)(x+2)$.",
        "A learner wrote -4/3 - 1 = -1/3, which is incorrect. The correct value is -7/3.",
        "Decide whether 3 + 2 = 6. This statement is false, since 3 + 2 = 5.",
        "# Worked examples\n## Example 1\n2 + 3 = 5.\n## Example 2\n4 - 2 = 2.",
        "FAQ\nWhy does ice float?\nIce is less dense than liquid water.\nWhat is density?\nMass divided by volume.",
        "Let angle 1 denote ABC and angle 2 denote BCD. Then angle 1 + angle 2 = angle 3.",
        r"A matrix has entries $a_{ij}$ and arbitrary dimension $n$. No assumption of commutativity is made.",
    ],
)
def test_valid_educational_diversity_survives(text):
    assert findings(text, "finemath-4plus") == []


def test_affine_rewrite_and_stated_point_are_checked_exactly():
    damaged = r"Find the slope. Multiply to get $15 - 3 x = y$, or $y = 3 x - 15$."
    assert affine_findings(damaged)[0][0] == "inequivalent_affine_rewrite"
    valid = damaged.replace("y = 3 x - 15", "y = 15 - 3 x")
    assert affine_findings(valid) == []
    line = r"# Find the line that passes through (2,-1)." + "\n" + r"$y=\frac{2}{3}x-\frac{1}{3}$"
    assert affine_findings(line)[0][0] == "line_answer_misses_given_point"
    assert affine_findings(line.replace(r"\frac{1}{3}", r"\frac{7}{3}")) == []
    premise = (
        "# Find the line that passes through (1, 2).\n"
        "The given parallel line is $y = 2x + 3$. The answer is $y = 2x$."
    )
    assert affine_findings(premise) == []
    assert affine_coefficients("2x-3y=5") == (2, -3, -5)
    with pytest.raises(ValueError):
        affine_coefficients("__import__('os').system('echo unsafe')=0")
    with pytest.raises(ValueError):
        affine_coefficients("1/(x-x)=y")
    corrected = r"A learner's incorrect rewrite is $15 - 3 x = y$, or $y = 3 x - 15$. The correct slope is negative."
    assert affine_findings(corrected) == []


def test_cached_quality_acceptances_require_proven_narrowing():
    old = "pattern = '(?:download|buy|purchase)'\nunchanged = True\n"
    new = old.replace("(?:download|buy|purchase)", "(?:buy|purchase)")
    prove_quality_narrowing(old, new)
    with pytest.raises(ValueError):
        prove_quality_narrowing(old, new.replace("True", "False"))
    with pytest.raises(ValueError):
        prove_quality_narrowing(
            old, old.replace("(?:download|buy|purchase)", "(?:download|buy|purchase|click)")
        )


def test_comparison_outputs_do_not_reuse_historical_ledgers(tmp_path):
    from codexa.evaluation.benchmark import evaluation_directory

    historical = tmp_path / "benchmarks/baseline-original"
    historical.mkdir(parents=True)
    ledger = historical / "budget.json"
    ledger.write_text('{"used_seconds": 3617}')
    output = tmp_path / "benchmarks/comparison-new/parent"
    assert evaluation_directory(tmp_path, "original", output) == output
    assert ledger.read_text() == '{"used_seconds": 3617}'
    with pytest.raises(ValueError):
        evaluation_directory(tmp_path, "original", tmp_path.parent / "outside")


def test_continuation_terminal_exposes_phase_and_lifetime_counters():
    from io import StringIO

    from rich.console import Console

    from codexa.training.terminal.render import frame
    from codexa.training.terminal.state import TerminalState

    state = TerminalState()
    state.metadata = {
        "parameters": 248687488,
        "training": {"steps": 1526},
        "parent_tokens": 500006912,
        "parent_updates": 15259,
    }
    state.step, state.tokens = 2, 500072448
    output = StringIO()
    Console(file=output, width=140, height=42, color_system=None).print(
        frame(state.snapshot(), 140, 42)
    )
    rendered = output.getvalue()
    assert "Continuation tokens 65,536" in rendered
    assert "Lifetime step 15,261" in rendered
    assert "500,072,448" in rendered


def test_smoke_preflight_isolated_from_training_autograd(tmp_path, monkeypatch):
    import subprocess

    from scripts.training import continuation_healthcheck as health

    captured = {}

    def child(args, **kwargs):
        captured.update(arguments=args, options=kwargs)
        return subprocess.CompletedProcess(args, 0, stdout='{"status":"passed"}')

    monkeypatch.setattr(health.subprocess, "run", child)
    assert health.isolated_preflight(tmp_path, tmp_path, tmp_path, tmp_path) == {"status": "passed"}
    assert captured["arguments"][2] == "-c"
    assert captured["options"]["env"]["CODEXA_NO_TRAINING"] == "1"
    assert captured["options"]["env"]["CUDA_VISIBLE_DEVICES"] == ""


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_blind_document_must_match_actual_training_bytes(tmp_path, split):
    import json

    import numpy as np

    from scripts.datasets.verify_review_packet import compare_documents

    np.asarray([2, 4, 5, 3], dtype="<u2").tofile(tmp_path / "train.bin")
    np.asarray([0], dtype="<u8").tofile(tmp_path / "train.idx")
    (tmp_path / "documents.jsonl").write_text(
        json.dumps({"id": "doc", "split": split, "tokens": 4}) + "\n"
    )
    manifest = {"splits": {split: [{"file": "train.bin", "index": "train.idx"}]}}

    class Encoder:
        def encode(self, text):
            return [4, 5] if text == "actual" else [4, 6]

    assert compare_documents(tmp_path, manifest, {"doc": "actual"}, Encoder(), split) == 1
    with pytest.raises(ValueError, match="actual retained training tokens"):
        compare_documents(tmp_path, manifest, {"doc": "altered"}, Encoder(), split)
