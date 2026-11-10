"""Observed English-source defects must not silently recur in version 3."""

import gzip
import hashlib
import json

import pytest

from codexa.data.language import source_rows
from codexa.data.language_sources import (
    ADAPTERS,
    DCLM_ADAPTER,
    POLICY,
    PRESSBOOKS_ADAPTER,
    canonical_license,
    clean_source,
    cleaned_v3,
    covered,
    grouping,
)

TEXT = "Birds build nests in many different places. Some make their homes among the branches of trees. Others find a sheltered place close to the ground. A nest keeps their eggs together and offers protection from the weather."


def spec(adapter=DCLM_ADAPTER):
    repo, revision = ADAPTERS[adapter]
    return {
        "name": "dclm-edu" if adapter == DCLM_ADAPTER else "pressbooks-filtered",
        "adapter": adapter,
        "repo": repo,
        "revision": revision,
        "category": "general",
        "filter": {"require_terminal": True},
        "min_edu_score": 3,
    }


def row():
    return {
        "id": "original-id",
        "url": "https://example.org/nature/chapter",
        "text": TEXT,
        "edu_int_score": 4,
        "edu_score": 3.8,
        "language": "en",
        "language_score": 0.99,
    }


@pytest.mark.parametrize(
    "raw,spdx,status",
    [
        ("CC-By-4.0", "CC-BY-4.0", "explicit"),
        ("CC-By", None, "missing_version"),
        ("CC-By-SA", None, "missing_version"),
        ("CC0", "CC0-1.0", "alias"),
        ("Public Domain", None, "basis_required"),
        (
            "Creative Commons - Attribution - https://creativecommons.org/licenses/by/4.0/",
            "CC-BY-4.0",
            "explicit",
        ),
        ("https://creativecommons.org.evil.invalid/licenses/by/4.0/", None, "unresolved"),
    ],
)
def test_license_classification_preserves_raw_without_inventing_grants(raw, spdx, status):
    value = canonical_license(raw)
    assert value["raw"] == raw and value["spdx"] == spdx and value["status"] == status


def test_dclm_uses_actual_score_field_and_keeps_terminal_gate():
    assert cleaned_v3(row(), spec())[1] is None
    assert (
        cleaned_v3(row() | {"int_score": 5, "edu_int_score": 2}, spec())[1] == "educational_score"
    )
    assert (
        cleaned_v3(row() | {"edu_int_score": 3}, spec() | {"min_edu_score": 4})[1]
        == "educational_score"
    )
    assert (
        cleaned_v3(row() | {"text": TEXT + " An unfinished sentence"}, spec())[1]
        == "admission_incomplete_prose"
    )
    for footer in ["Add flashcard Cite Random", "Copyright © 2017, Original publisher"]:
        text, error = cleaned_v3(row() | {"text": TEXT + "\n\n" + footer}, spec())
        assert error is None and text == TEXT


def test_contact_line_quarantines_prose_instead_of_deleting_sentence():
    text = (
        TEXT
        + "\nSend mail to author@example.org. Because our buildings are closed, use the side entrance."
    )
    assert clean_source(row() | {"text": text}, spec())[2] == "language_mixed_contact_line"
    clean, decisions, error = clean_source(
        row() | {"text": TEXT + "\nEmail: author@example.org"}, spec()
    )
    assert error is None and clean == TEXT and decisions == ["standalone_contact"]


def test_local_dclm_and_gzip_preserve_source_metadata(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    path = tmp_path / "source.parquet"
    pq.write_table(
        pa.Table.from_pylist([row() | {"id": str(i)} for i in range(100)]), path, row_group_size=20
    )
    source = spec() | {
        "local_path": str(path),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "_development_limit": 4,
    }
    result = list(source_rows(source, tmp_path / "unused", 0))
    assert [i for i, _ in result] == [13, 38, 63, 88]
    assert all(
        r["edu_int_score"] == 4 and r["edu_score"] == 3.8 and r["language"] == "en"
        for _, r in result
    )
    path = tmp_path / "source.json.gz"
    document = {
        "id": "chapter",
        "text": TEXT,
        "metadata": {
            "book_url": "https://example.org/book/",
            "license": "https://creativecommons.org/licenses/by/4.0/",
            "title": "Birds",
            "subject": "Nature",
        },
    }
    with gzip.open(path, "wt") as f:
        f.write(json.dumps(document) + "\n")
    source = spec(PRESSBOOKS_ADAPTER) | {
        "local_path": str(path),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    assert list(source_rows(source, tmp_path / "unused", 0)) == [(1, document)]
    assert grouping(document, source) == grouping(document | {"id": "chapter-2"}, source)
    assert cleaned_v3(document, source)[1] is None
    assert (
        cleaned_v3(document | {"metadata": document["metadata"] | {"title": "Algebra"}}, source)[1]
        == "language_math_specialization"
    )


def test_original_grant_bound_to_book_license_and_proof(tmp_path):
    source = spec(PRESSBOOKS_ADAPTER)
    document = {
        "text": TEXT,
        "metadata": {"book_url": "https://example.org/book/", "license": "CC-By", "title": "Birds"},
    }
    proof = tmp_path / "original.txt"
    proof.write_text("Original CC BY 4.0 grant to fictional fixture book.")
    evidence = {
        "status": "approved",
        "repo": source["repo"],
        "revision": source["revision"],
        "adapter": source["adapter"],
        "policy": POLICY,
        "reviewer": "fixture reviewer",
        "coverage": [
            {
                "book_url": "https://example.org/book/",
                "license": "CC-BY-4.0",
                "scope": "original_book",
                "verification": "original_license_verified",
                "attribution": "Fixture author",
                "basis_url": "https://example.org/book/copyright",
                "proof": {
                    "path": str(proof),
                    "sha256": hashlib.sha256(proof.read_bytes()).hexdigest(),
                },
            }
        ],
    }
    assert not covered(document, source, {"status": "approved"})
    assert covered(document, source, evidence)
    assert not covered(
        document | {"metadata": document["metadata"] | {"license": "CC0"}}, source, evidence
    )
    assert not covered(
        document
        | {"metadata": document["metadata"] | {"book_url": "https://example.org/another-book"}},
        source,
        evidence,
    )
    proof.write_text("Changed evidence")
    assert not covered(document, source, evidence)


def test_copyright_discussion_is_not_a_copyright_footer():
    text = "Copyright protects original works. " + TEXT
    cleaned, error = cleaned_v3(row() | {"text": text}, spec())
    assert error is None and cleaned == text


def test_gzip_development_sample_is_distributed(tmp_path):
    path = tmp_path / "source.json.gz"
    with gzip.open(path, "wt") as stream:
        for number in range(10):
            stream.write(json.dumps({"id": str(number), "text": TEXT}) + "\n")
    source = spec(PRESSBOOKS_ADAPTER) | {
        "local_path": str(path),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "rows": 10,
        "_development_limit": 2,
    }
    assert [ordinal for ordinal, _ in source_rows(source, tmp_path / "unused", 0)] == [3, 8]


def test_v3_candidate_targets_preserve_capacity_and_frozen_defaults():
    from codexa.data.language import SPLIT_TARGETS, V3_VERSION, admitted_split_targets

    assert admitted_split_targets({"version": V3_VERSION}) == SPLIT_TARGETS
    targets = {"train": 52_000_000, "val": 1_000_000, "test": 1_000_000}
    assert admitted_split_targets({"version": V3_VERSION, "split_targets": targets}) == targets
    minimum = {"train": 50_003_969, "val": 1_000_000, "test": 1_000_000}
    assert admitted_split_targets({"version": V3_VERSION, "split_targets": minimum}) == minimum
    for changed in [
        minimum | {"train": 50_003_968},
        minimum | {"val": 999_999},
        minimum | {"test": True},
        minimum | {"train": 50_003_969.0},
        {"train": 52_000_000, "val": 1_000_000},
    ]:
        with pytest.raises(ValueError, match="capacity"):
            admitted_split_targets({"version": V3_VERSION, "split_targets": changed})
    for version in ["codexa-language-data-1", "codexa-language-data-2"]:
        with pytest.raises(ValueError, match="Historical"):
            admitted_split_targets({"version": version, "split_targets": SPLIT_TARGETS})


def test_internal_family_policy_preserves_unknown_version_and_compiles_fresh(tmp_path):
    from unittest.mock import patch

    from codexa.data.language_sources import INTERNAL_POLICY, compile_checker
    from codexa.data.production import digest

    source = spec(PRESSBOOKS_ADAPTER) | {"sha256": "a" * 64, "rights_policy": INTERNAL_POLICY}
    metadata = {
        "book_url": "https://example.org/book/",
        "title": "Birds",
        "author": "Fixture Author",
        "license": "https://creativecommons.org/licenses/by/4.0/",
    }
    document = {"text": TEXT, "metadata": metadata}
    proof = tmp_path / "book.json"
    proof.write_text(
        json.dumps(
            {
                "version": "pressbooks-archived-publisher-declaration-1",
                "source_repository": source["repo"],
                "source_revision": source["revision"],
                "source_file_sha256": source["sha256"],
                "source_ordinal": 1,
                "book_url": metadata["book_url"],
                "publisher_directory_metadata": metadata,
                "reconstructed_declared_family": "CC-BY",
                "effective_version": None,
            }
        )
    )
    policy = tmp_path / "policy.json"
    policy.write_text(
        json.dumps(
            {
                "version": "codexa-pressbooks-original-rights-policy-1",
                "status": "conditionally_approved_local_research_policy",
                "repo": source["repo"],
                "revision": source["revision"],
                "reviewer": "fictional fixture-only reviewer",
            }
        )
    )
    collector = tmp_path / "collector.txt"
    collector.write_text(
        "Original fixture collector maps CC BY family to nominal4.0; no real-source approval."
    )
    grant = {
        "license": "CC-BY",
        "version_status": "unresolved",
        "scope": "internal_research_training",
        "verification": "archived_publisher_declaration_verified",
        "book_url": metadata["book_url"],
        "attribution": "Fixture Author; Birds",
        "basis_url": "https://example.org/book",
        "proof": {"path": str(proof), "sha256": digest(proof)},
    }
    evidence = {
        "status": "approved",
        "repo": source["repo"],
        "revision": source["revision"],
        "adapter": source["adapter"],
        "policy": POLICY,
        "reviewer": "fictional fixture-only reviewer",
        "rights_policy": INTERNAL_POLICY,
        "redistribution_allowed": False,
        "policy_proof": {"path": str(policy), "sha256": digest(policy)},
        "collector_proof": {"path": str(collector), "sha256": digest(collector)},
        "coverage": [grant],
    }
    with patch("codexa.data.language_sources.digest", wraps=digest) as hashes:
        check = compile_checker(source, evidence)
        calls = hashes.call_count
        for _ in range(10):
            assert check(document) == grant
        assert hashes.call_count == calls
    assert canonical_license(metadata["license"])["spdx"] == "CC-BY-4.0"
    assert (
        check(document)["license"] == "CC-BY" and check(document)["version_status"] == "unresolved"
    )
    assert not covered(document, source, evidence | {"redistribution_allowed": True})
    assert not covered(document, source | {"rights_policy": "other"}, evidence)
    for licence in ["https://creativecommons.org/licenses/by-sa/4.0/", "CC0", "CC-BY-NC-4.0"]:
        assert check(document | {"metadata": metadata | {"license": licence}}) is None
    assert check(document | {"metadata": metadata | {"author": "Another author"}}) is None
    assert check(document | {"text": TEXT + " All rights reserved."}) is None
    proof.write_text("Changed book proof")
    assert compile_checker(source, evidence)(document) is None


def test_original_api_version_reconciliation_requires_collector_proof(tmp_path):
    from codexa.data.production import digest

    source = spec(PRESSBOOKS_ADAPTER)
    document = {
        "text": TEXT,
        "metadata": {
            "book_url": "https://example.org/book/front-matter/intro/",
            "license": "https://creativecommons.org/licenses/by/4.0/",
        },
    }
    proof = tmp_path / "api.json"
    proof.write_text(
        json.dumps(
            {
                "name": "Fixture Book",
                "author": [{"name": "Fixture author"}],
                "license": {"url": "https://creativecommons.org/licenses/by/3.0/"},
                "_links": {
                    "self": [{"href": "https://example.org/book/wp-json/pressbooks/v2/metadata"}]
                },
            }
        )
    )
    collector = tmp_path / "collector.txt"
    collector.write_text("Fictional collector mapping proof.")
    grant = {
        "book_url": document["metadata"]["book_url"],
        "license": "CC-BY-3.0",
        "scope": "original_book",
        "verification": "original_license_verified",
        "attribution": "Fixture author",
        "basis_url": "https://example.org/book/wp-json/pressbooks/v2/metadata",
        "proof": {"path": str(proof), "sha256": digest(proof)},
    }
    evidence = {
        "status": "approved",
        "repo": source["repo"],
        "revision": source["revision"],
        "adapter": source["adapter"],
        "policy": POLICY,
        "reviewer": "fixture reviewer",
        "coverage": [grant],
    }
    assert not covered(document, source, evidence)
    grant["collector_normalized_version"] = True
    assert not covered(document, source, evidence)
    grant["collector_proof"] = {"path": str(collector), "sha256": digest(collector)}
    assert covered(document, source, evidence)
    grant["chapter_url_prefixes"] = ["https://example.org/book/chapter/"]
    grant["original_work_only"] = True
    chapter = document | {"id": "https://example.org/book/chapter/birds/"}
    assert covered(chapter, source, evidence)
    assert not covered(
        document | {"id": "https://example.org/book/front-matter/editor/"}, source, evidence
    )
    assert not covered(
        document | {"id": "https://example.org/book/chapter/series-introduction/"}, source, evidence
    )
    grant.pop("chapter_url_prefixes")
    grant.pop("original_work_only")
    assert not covered(
        document | {"metadata": document["metadata"] | {"license": "CC0"}}, source, evidence
    )
    collector.write_text("Changed")
    assert not covered(document, source, evidence)


def test_observed_language_filters_and_safe_footnote_cleanup():
    press = spec(PRESSBOOKS_ADAPTER)
    document = {
        "text": TEXT,
        "metadata": {"book_url": "https://example.org/book", "title": "Learning", "license": "CC0"},
    }
    for text in ["Answer Key\n" + TEXT, "Responses to transitions\n" + TEXT]:
        assert cleaned_v3(document | {"text": text}, press)[1] == "language_answer_key"
    for title in ["Open Educational Resources Directory", "Medical Terminology"]:
        assert (
            cleaned_v3(document | {"metadata": document["metadata"] | {"title": title}}, press)[1]
            == "language_reference_or_directory"
        )
    text, decisions, error = clean_source(document | {"text": TEXT + " ↵"}, press)
    assert error is None and text == TEXT and "footnote_navigation_glyph" in decisions
    assert (
        cleaned_v3(document | {"text": TEXT + " No finished ending ↵"}, press)[1]
        == "admission_incomplete_prose"
    )
    attributions = TEXT + "\nAttributions\nImage by Fixture Photographer is public domain"
    assert cleaned_v3(document | {"text": attributions}, press) == (TEXT, None)
    assert (
        cleaned_v3(document | {"text": attributions + "\nCC BY-NC 4.0"}, press)[1]
        == "language_contrary_rights_notice"
    )
    assert (
        cleaned_v3(row() | {"text": "Group Theory\n" + TEXT}, spec())[1]
        == "language_math_specialization"
    )


def test_syndicated_notice_does_not_hide_under_book_waiver():
    source = spec(PRESSBOOKS_ADAPTER)
    document = {
        "text": TEXT + " This article is republished from Original Publisher.",
        "metadata": {"book_url": "https://example.org/book", "license": "CC0", "title": "Birds"},
    }
    assert cleaned_v3(document, source)[1] == "language_contrary_rights_notice"


def notice_fixture(tmp_path):
    from codexa.data.production import digest

    source = spec(PRESSBOOKS_ADAPTER) | {"sha256": "a" * 64}
    metadata = {
        "book_url": "https://example.org/book",
        "author": "Fixture Author",
        "title": "Birds",
        "license": "CC-BY-4.0",
    }
    document = {"id": "https://example.org/book/chapter/birds", "text": TEXT, "metadata": metadata}
    proof = tmp_path / "owner-grant.txt"
    proof.write_text("Original fictional fixture owner grants CC-BY4; no real-source clearance.")
    grant = {
        "license": "CC-BY-4.0",
        "scope": "original_book",
        "verification": "original_license_verified",
        "attribution": "Fixture Author; Birds",
        "basis_url": metadata["book_url"],
        "book_url": metadata["book_url"],
        "proof": {"path": str(proof), "sha256": digest(proof)},
    }
    evidence = {
        "status": "approved",
        "repo": source["repo"],
        "revision": source["revision"],
        "adapter": source["adapter"],
        "policy": POLICY,
        "reviewer": "Fictional fixture reviewer",
        "coverage": [grant],
    }
    return source, document, evidence, proof


@pytest.mark.parametrize(
    "notice",
    [
        "https://creativecommons.org/licenses/by-sa/4.0/",
        "All rights &#114;eserved.",
        "All rights &amp;#114;eserved.",
        "Creative Commons Attribution-NonCommercial 4.0",
        "Creative Commons Attribution-ShareAlike 4.0",
    ],
)
def test_restricted_notice_forms_reject_in_cleaning_and_compiled_rights(tmp_path, notice):
    from codexa.data.language_sources import compile_checker

    source, document, evidence, _ = notice_fixture(tmp_path)
    check = compile_checker(source, evidence)
    assert check(document)
    changed = document | {"text": TEXT + "\n" + notice}
    assert clean_source(changed, source)[2] == "language_contrary_rights_notice"
    assert check(changed) is None


def test_only_unambiguous_image_credit_lines_are_removed(tmp_path):
    from codexa.data.language_sources import compile_checker

    source, document, evidence, _ = notice_fixture(tmp_path)
    check = compile_checker(source, evidence)
    credit = (
        "Image: “Old version of Wikipedia logo” by Fixture Artist is licensed under CC BY-SA 3.0."
    )
    changed = document | {"text": TEXT + "\nAttributions\n" + credit}
    text, decisions, error = clean_source(changed, source)
    assert error is None and text == TEXT and "standalone_image_credits_excluded" in decisions
    assert check(changed)
    continuation = "A further paragraph follows the image credit and must survive."
    assert (
        continuation
        in clean_source(changed | {"text": changed["text"] + "\n" + continuation}, source)[0]
    )
    for notice in [
        credit + " This chapter uses that licence.",
        "CC BY-SA 4.0",
        "This chapter is adapted under CC BY-NC-SA 4.0.",
        "Figure 1. This entire course is licensed CC BY-NC 4.0.",
        "Image: copyright applies to all lessons in this course. All rights reserved.",
        "Image: A tree by Alice is licensed under CC BY-SA 4.0. This entire work is copyright Bob.",
    ]:
        bad = document | {"text": TEXT + "\nAttributions\n" + notice}
        assert clean_source(bad, source)[2] == "language_contrary_rights_notice"
        assert check(bad) is None


def test_archived_cc0_attribution_is_bound_and_linked_proofs_rechecked(tmp_path):
    from codexa.data.language_sources import compile_checker
    from codexa.data.production import digest

    source, document, evidence, proof = notice_fixture(tmp_path)
    document["metadata"]["license"] = "CC0"
    grant = evidence["coverage"][0]
    grant["license"] = "CC0-1.0"
    proof.write_text(
        json.dumps(
            {
                "version": "pressbooks-archived-publisher-declaration-1",
                "source_repository": source["repo"],
                "source_revision": source["revision"],
                "source_file_sha256": source["sha256"],
                "source_ordinal": 1,
                "book_url": grant["book_url"],
                "publisher_directory_metadata": document["metadata"],
                "reconstructed_declared_family": "CC0",
                "effective_version": "CC0-1.0",
            }
        )
    )
    grant["proof"]["sha256"] = digest(proof)
    check = compile_checker(source, evidence)
    assert check(document)
    for key in ["author", "title"]:
        assert check(document | {"metadata": document["metadata"] | {key: "Changed"}}) is None
    child = tmp_path / "linked-owner-proof.txt"
    child.write_text("Original fictional owner declaration.")
    wrapper = json.loads(proof.read_text())
    wrapper["rights_review"] = {"path": str(child), "sha256": digest(child)}
    proof.write_text(json.dumps(wrapper))
    grant["proof"]["sha256"] = digest(proof)
    assert compile_checker(source, evidence)(document)
    child.write_text("Altered underlying rights evidence.")
    assert compile_checker(source, evidence)(document) is None


def test_balanced_book_splits_are_explicit_and_historical_behavior_unchanged():
    from codexa.data.language import V3_VERSION, admitted_split_targets, source_split

    decision = {
        "version": V3_VERSION,
        "sources": [{"name": "pressbooks-filtered"}],
        "split_policy": "book-hash-balanced-1",
        "split_assignments": {"book-a": "val", "book-b": "test", "book-c": "train"},
    }
    admitted_split_targets(decision)
    assert source_split("book-a", decision) == "val"
    assert source_split("book-b", decision) == "test"
    assert source_split("book-c", decision) == "train"
    with pytest.raises(ValueError, match="absent"):
        source_split("unlisted-book", decision)
    for version in ["codexa-language-data-1", "codexa-language-data-2"]:
        with pytest.raises(ValueError, match="Historical"):
            admitted_split_targets(decision | {"version": version})
    for assignments in [{}, {"book-a": "validation"}, {"": "train"}]:
        with pytest.raises(ValueError, match="assignments"):
            admitted_split_targets(decision | {"split_assignments": assignments})


def test_optional_prose_screen_defaults_and_sentence_fraction():
    from codexa.data.language_sources import prose_admission_error

    questions = "Why do birds build nests in trees? Where do young birds wait for their parents? How can a sheltered branch protect the eggs?"
    assert prose_admission_error(questions, {"text": questions}, {}) is None
    options = {"minimum_prose_sentences": 3, "minimum_prose_fraction": 0.6}
    assert prose_admission_error(TEXT, {"text": TEXT}, options) is None
    assert (
        prose_admission_error(questions, {"text": questions}, options)
        == "language_insufficient_prose_sentences"
    )
    headings = "\n".join(
        ["Separate disconnected topic heading without connected body" for _ in range(12)]
    )
    combined = headings + "\n" + TEXT
    assert (
        prose_admission_error(combined, {"text": combined}, options)
        == "language_insufficient_prose_fraction"
    )
    for invalid in [
        {"minimum_prose_sentences": True},
        {"minimum_prose_fraction": 1.1},
        {"minimum_prose_fraction": float("nan")},
        {"reject_instructional_scaffolding": "yes"},
    ]:
        assert (
            prose_admission_error(TEXT, {"text": TEXT}, invalid) == "language_prose_filter_policy"
        )


def test_optional_scaffolding_filter_rejects_observed_templates_and_keeps_context():
    from codexa.data.language_sources import prose_admission_error

    options = {
        "minimum_prose_sentences": 3,
        "minimum_prose_fraction": 0.6,
        "reject_instructional_scaffolding": True,
    }

    def check(text):
        return prose_admission_error(text, {"text": text}, options)

    assert check(TEXT + "\nThe contact is <EMAIL_ADDRESS>.") == "language_redacted_scaffolding"
    assert check(TEXT + "\nThe procedure is [REDACTED].") == "language_redacted_scaffolding"
    assert (
        check(TEXT + "\nRead the following paper and answer the questions.")
        == "language_required_external_resource"
    )
    assert (
        check(TEXT + "\nWatch the provided video, then complete the exercise.")
        == "language_required_external_resource"
    )
    diagnostic = (
        "DSM-5\nDiagnostic Criteria\nA. Persistent symptoms have been present during the previous year.\nB. These symptoms cause significant distress during daily activities.\n"
        + TEXT
    )
    assert check(diagnostic) == "language_diagnostic_template"
    sufficient = "Mira left her DSM guide beside the window before joining her friends outside. The children were waiting by the old wall with a basket of apples. She placed the basket beneath a tree and told them about her afternoon. They talked until the evening light began to fade behind the houses."
    assert check(sufficient) is None
    outcomes = "Learning Outcomes\nBy the end of this chapter you will be able to:\n- Describe the different functions of roots and branches in a tree.\n- Compare the different kinds of nests that birds build in sheltered places."
    assert check(outcomes) == "language_learning_outcome_scaffolding"
    assert (
        check(TEXT + "\nOptional resources can offer additional examples for interested readers.")
        is None
    )
    assert check(TEXT + "\nLearning Outcomes\n- Describe nests.\n- Compare branches.") is None


def test_optional_scaffolding_filter_detects_missing_announced_quote_without_cropping():
    from codexa.data.language_sources import prose_admission_error

    options = {"reject_instructional_scaffolding": True}
    missing = (
        TEXT
        + "\nThe artist placed the ashes into an urn with a bronze plaque inscribed,\nThe urn and plaque, together with photographs, form the finished work."
    )
    assert (
        prose_admission_error(missing, {"text": missing}, options)
        == "language_missing_announced_quotation"
    )
    supplied = (
        TEXT
        + "\nThe artist placed the ashes into an urn with a bronze plaque inscribed,\n“Remember the work that was made here.”\nThe urn and plaque form the finished work."
    )
    assert prose_admission_error(supplied, {"text": supplied}, options) is None
    ordinary = TEXT + " The plaque was inscribed with the year in which the house was built."
    assert prose_admission_error(ordinary, {"text": ordinary}, options) is None


def test_v3_explicit_prose_options_dispatch_without_affecting_defaults():
    source = spec(PRESSBOOKS_ADAPTER)
    text = "Why do birds choose a sheltered place for a nest? How does a strong branch help protect their eggs? What happens when the weather changes during spring?"
    document = {
        "text": text,
        "metadata": {"book_url": "https://example.org/book", "license": "CC0", "title": "Birds"},
    }
    assert cleaned_v3(document, source)[1] is None
    configured = source | {
        "filter": source["filter"] | {"minimum_prose_sentences": 3, "minimum_prose_fraction": 0.6}
    }
    assert cleaned_v3(document, configured)[1] == "language_insufficient_prose_sentences"


def _policy2_fixture(tmp_path, monkeypatch):
    import codexa.data.language_sources as sources
    from codexa.data.production import digest

    source = spec(PRESSBOOKS_ADAPTER) | {
        "sha256": "a" * 64,
        "rights_policy": sources.INTERNAL_POLICY_V2,
    }
    metadata = {
        "book_url": "https://example.org/book/",
        "title": "Birds",
        "author": "Fixture Author",
        "license": "https://creativecommons.org/licenses/by-sa/4.0/",
    }
    document = {"text": TEXT, "metadata": metadata}
    binding = {key: source[key] for key in ("repo", "revision", "sha256")}

    def write(name, data):
        path = tmp_path / name
        path.write_text(json.dumps(data))
        return {"path": str(path), "sha256": digest(path)}

    proposal = {
        "version": "codexa-pressbooks-rights-policy-2-proposal",
        "status": "PROPOSED_PENDING_EXPLICIT_USER_AUTHORIZATION",
        "approved": False,
        "source_binding": {
            "repository": source["repo"],
            "revision": source["revision"],
            "source_file_sha256": source["sha256"],
        },
        "proposed_authorization_text": "Internal fictional fixture only; no redistribution.",
    }
    proposal_proof = write("proposal.json", proposal)
    monkeypatch.setattr(sources, "POLICY_V2_PROPOSAL_SHA256", proposal_proof["sha256"])
    authorization = {
        "version": "codexa-pressbooks-rights-policy-2-authorization",
        "status": "authorized",
        "approved": True,
        "rights_policy": sources.INTERNAL_POLICY_V2,
        "user_decision": "Explicit fixture-only authorization",
        "source_binding": binding,
        "proposal_proof": proposal_proof,
        "proposed_authorization_text": proposal["proposed_authorization_text"],
    }
    authorization_proof = write("authorization.json", authorization)
    policy = {
        "version": "codexa-pressbooks-original-rights-policy-2",
        "status": "conditionally_approved_local_research_policy",
        "approved": True,
        "rights_policy": sources.INTERNAL_POLICY_V2,
        "redistribution_allowed": False,
        "repo": source["repo"],
        "revision": source["revision"],
        "reviewer": "fixture reviewer",
        "source_binding": binding,
        "authorization_proof": authorization_proof,
        "proposal_proof": proposal_proof,
    }
    policy_proof = write("policy.json", policy)
    collector = tmp_path / "collector.txt"
    collector.write_text("Fictional reviewed collector maps CC-BY-SA family to nominal enum4.0.")
    book_proof = write(
        "book.json",
        {
            "version": "pressbooks-archived-publisher-declaration-1",
            "source_repository": source["repo"],
            "source_revision": source["revision"],
            "source_file_sha256": source["sha256"],
            "source_ordinal": 1,
            "book_url": metadata["book_url"],
            "publisher_directory_metadata": metadata,
            "reconstructed_declared_family": "CC-BY-SA",
            "effective_version": None,
        },
    )
    grant = {
        "license": "CC-BY-SA",
        "version_status": "unresolved",
        "scope": "internal_research_training",
        "verification": "archived_publisher_declaration_verified",
        "book_url": metadata["book_url"],
        "attribution": "Fixture Author; Birds",
        "basis_url": "https://example.org/book",
        "proof": book_proof,
    }
    evidence = {
        "status": "approved",
        "repo": source["repo"],
        "revision": source["revision"],
        "adapter": source["adapter"],
        "policy": POLICY,
        "reviewer": "fixture reviewer",
        "rights_policy": sources.INTERNAL_POLICY_V2,
        "redistribution_allowed": False,
        "policy_proof": policy_proof,
        "collector_proof": {"path": str(collector), "sha256": digest(collector)},
        "coverage": [grant],
    }
    return source, document, evidence, grant, policy, authorization


def test_policy2_scoped_sharealike_requires_explicit_authorization(tmp_path, monkeypatch):
    from codexa.data.language_sources import INTERNAL_POLICY, compile_checker

    source, document, evidence, grant, _, _ = _policy2_fixture(tmp_path, monkeypatch)
    check = compile_checker(source, evidence)
    assert check(document) == grant
    assert (
        check(document)["license"] == "CC-BY-SA"
        and check(document)["version_status"] == "unresolved"
    )
    assert cleaned_v3(document, source)[1] is None
    assert (
        cleaned_v3(document, source | {"rights_policy": INTERNAL_POLICY})[1]
        == "language_license_not_selected"
    )
    assert (
        compile_checker(
            source | {"rights_policy": INTERNAL_POLICY},
            evidence | {"rights_policy": INTERNAL_POLICY},
        )(document)
        is None
    )
    assert check(document | {"metadata": document["metadata"] | {"author": "Other author"}}) is None
    assert check(document | {"metadata": document["metadata"] | {"title": "Other book"}}) is None
    assert (
        compile_checker(
            source, evidence | {"excluded_book_urls": [document["metadata"]["book_url"]]}
        )(document)
        is None
    )
    assert check(document | {"text": TEXT + "\nThis chapter is licensed CC BY-SA 3.0."}) == grant
    for notice in [
        "This chapter is licensed CC BY-NC 4.0.",
        "This article is republished from Another Publisher.",
        "All rights reserved.",
        "https://creativecommons.org/licenses/by-sa/3.0/us/",
        "CC BY-SA 3.0 with additional restrictions.",
    ]:
        bad = document | {"text": TEXT + "\n" + notice}
        assert check(bad) is None
        assert cleaned_v3(bad, source)[1] == "language_contrary_rights_notice"
    ported = document | {
        "metadata": document["metadata"]
        | {"license": "https://creativecommons.org/licenses/by-sa/3.0/us/"}
    }
    assert check(ported) is None
    assert cleaned_v3(ported, source)[1] == "language_license_not_selected"
    by_book = document | {
        "metadata": document["metadata"]
        | {"license": "https://creativecommons.org/licenses/by/4.0/"},
        "text": TEXT + "\nThis chapter is licensed CC BY-SA 4.0.",
    }
    assert cleaned_v3(by_book, source)[1] == "language_contrary_rights_notice"
    assert check(by_book) is None
    # Exact SPDX SA is not accepted through the old unconditional SPDX allowlist.
    evidence["coverage"] = [
        grant
        | {
            "license": "CC-BY-SA-4.0",
            "scope": "original_book",
            "verification": "original_license_verified",
        }
    ]
    assert compile_checker(source, evidence)(document) is None


@pytest.mark.parametrize(
    "defect",
    [
        "missing_auth",
        "proposal_as_auth",
        "auth_hash",
        "auth_unapproved",
        "auth_proposed",
        "auth_source",
        "policy_unapproved",
        "proposal_hash",
        "book_source",
    ],
)
def test_policy2_rejects_missing_proposed_or_forged_bindings(tmp_path, monkeypatch, defect):
    from codexa.data.language_sources import compile_checker
    from codexa.data.production import digest

    source, document, evidence, grant, policy, authorization = _policy2_fixture(
        tmp_path, monkeypatch
    )
    assert compile_checker(source, evidence)(document)
    if defect == "missing_auth":
        policy.pop("authorization_proof")
    elif defect == "proposal_as_auth":
        policy["authorization_proof"] = policy["proposal_proof"]
    elif defect == "auth_hash":
        policy["authorization_proof"]["sha256"] = "0" * 64
    elif defect in {"auth_unapproved", "auth_proposed", "auth_source"}:
        if defect == "auth_unapproved":
            authorization["approved"] = False
        if defect == "auth_proposed":
            authorization["status"] = "proposed"
        if defect == "auth_source":
            authorization["source_binding"] = authorization["source_binding"] | {"sha256": "b" * 64}
        path = tmp_path / "authorization.json"
        path.write_text(json.dumps(authorization))
        policy["authorization_proof"]["sha256"] = digest(path)
    elif defect == "policy_unapproved":
        policy["approved"] = False
    elif defect == "proposal_hash":
        monkeypatch.setattr("codexa.data.language_sources.POLICY_V2_PROPOSAL_SHA256", "0" * 64)
    elif defect == "book_source":
        path = tmp_path / "book.json"
        data = json.loads(path.read_text())
        data["source_file_sha256"] = "b" * 64
        path.write_text(json.dumps(data))
        grant["proof"]["sha256"] = digest(path)
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy))
    evidence["policy_proof"]["sha256"] = digest(path)
    assert compile_checker(source, evidence)(document) is None


@pytest.mark.parametrize(
    "notice",
    [
        "This chapter is licensed CC BY-SA-NC 4.0.",
        "This chapter is licensed CC BY-SA-ND 4.0.",
        "This chapter is licensed CC BY-SA 4.0 NC.",
        "This chapter is licensed Attribution ShareAlike NonCommercial.",
        "This chapter is licensed CC BY-SA 4.0, non-commercial use only.",
        "This chapter is licensed CC BY-SA 4.0, no derivatives permitted.",
        "This chapter is licensed CC BY-SA 4.0, no derivative works are allowed.",
        "This chapter is licensed CC BY-SA 4.0, not for commercial use.",
        "This chapter is licensed CC BY-SA 4.9.",
        "This chapter is licensed CC BY-SA 5.",
        "This chapter is licensed CC BY-SA 1.1.",
        "This chapter is licensed Attribution ShareAlike 4.9.",
        "https://creativecommons.org/licenses/by-sa/3.0/scotland/",
        "https://creativecommons.org/licenses/by-sa/3.0/igo/",
        "https://creativecommons.org/licenses/by-sa/3.0/custom-additional-conditions/",
        "https://creativecommons.org/licenses/by-sa/4.9/",
        "https://creativecommons.org/licenses/by-sa/4.0/legalcode/custom/",
    ],
)
def test_sa_exemption_cannot_erase_custom_restrictions_or_unknown_variants(
    tmp_path, monkeypatch, notice
):
    from codexa.data.language_sources import compile_checker

    source, document, evidence, _, _, _ = _policy2_fixture(tmp_path, monkeypatch)
    check = compile_checker(source, evidence)
    assert check(document)
    bad = document | {"text": TEXT + "\n" + notice}
    assert check(bad) is None
    assert cleaned_v3(bad, source)[1] == "language_contrary_rights_notice"


@pytest.mark.parametrize(
    "suffix", ["us", "scotland", "igo", "custom-jurisdiction", "legalcode/custom"]
)
def test_sa_metadata_url_rejects_all_noncanonical_suffixes(tmp_path, monkeypatch, suffix):
    from codexa.data.language_sources import compile_checker

    source, document, evidence, _, _, _ = _policy2_fixture(tmp_path, monkeypatch)
    changed = document | {
        "metadata": document["metadata"]
        | {"license": f"https://creativecommons.org/licenses/by-sa/3.0/{suffix}/"}
    }
    assert cleaned_v3(changed, source)[1] == "language_license_not_selected"
    assert compile_checker(source, evidence)(changed) is None


@pytest.mark.parametrize(
    "version,suffix",
    [
        ("1.0", ""),
        ("2.0", "legalcode"),
        ("2.5", "deed.en"),
        ("3.0", "deed.fr"),
        ("4.0", "legalcode.en"),
        ("4.0", "deed.zh_CN"),
    ],
)
def test_canonical_sa_notice_suffixes_remain_eligible(tmp_path, monkeypatch, version, suffix):
    from codexa.data.language_sources import compile_checker

    source, document, evidence, grant, _, _ = _policy2_fixture(tmp_path, monkeypatch)
    notice = f"https://creativecommons.org/licenses/by-sa/{version}/{suffix}"
    candidate = document | {"text": TEXT + "\n" + notice}
    assert compile_checker(source, evidence)(candidate) == grant


@pytest.mark.parametrize(
    "qualifier",
    [
        "United States",
        "US",
        "Australia",
        "Scotland",
        "IGO",
        "(United States)",
        "(Scotland)",
        "Custom Jurisdiction",
        "International Australia",
        ", US",
    ],
)
def test_named_sa_ports_do_not_inherit_unported_internal_grants(tmp_path, monkeypatch, qualifier):
    from codexa.data.language_sources import compile_checker

    source, document, evidence, _, _, _ = _policy2_fixture(tmp_path, monkeypatch)
    bad = document | {"text": TEXT + f"\nThis chapter is licensed CC BY-SA 3.0 {qualifier}."}
    assert compile_checker(source, evidence)(bad) is None
    assert cleaned_v3(bad, source)[1] == "language_contrary_rights_notice"


@pytest.mark.parametrize(
    "notice",
    [
        "This chapter is licensed CC BY-SA 3.0 Unported.",
        "This chapter is licensed CC BY-SA 4.0 International.",
        "This chapter is licensed CC BY-SA 4.0 International Public License.",
        "This chapter is licensed CC BY-SA 3.0. Australia is discussed in the next chapter.",
        "This chapter is licensed CC BY-SA 4.0 International (https://creativecommons.org/licenses/by-sa/4.0/).",
    ],
)
def test_reviewed_sa_qualifiers_and_separate_following_prose_remain_allowed(
    tmp_path, monkeypatch, notice
):
    from codexa.data.language_sources import compile_checker

    source, document, evidence, grant, _, _ = _policy2_fixture(tmp_path, monkeypatch)
    candidate = document | {"text": TEXT + "\n" + notice}
    assert compile_checker(source, evidence)(candidate) == grant


@pytest.mark.parametrize(
    "notice",
    [
        "This chapter is licensed CC BY-SA Australia.",
        "This chapter is licensed Attribution ShareAlike United States.",
        "This chapter is licensed CC BY-SA (IGO).",
    ],
)
def test_unversioned_sa_port_names_require_separate_review(tmp_path, monkeypatch, notice):
    from codexa.data.language_sources import compile_checker

    source, document, evidence, _, _, _ = _policy2_fixture(tmp_path, monkeypatch)
    bad = document | {"text": TEXT + "\n" + notice}
    assert compile_checker(source, evidence)(bad) is None
    assert cleaned_v3(bad, source)[1] == "language_contrary_rights_notice"


def test_bare_generic_sa_family_notice_does_not_invent_a_version(tmp_path, monkeypatch):
    from codexa.data.language_sources import compile_checker

    source, document, evidence, grant, _, _ = _policy2_fixture(tmp_path, monkeypatch)
    candidate = document | {"text": TEXT + "\nThis chapter is licensed CC BY-SA."}
    assert compile_checker(source, evidence)(candidate) == grant
    assert grant["version_status"] == "unresolved"


def _narrative_rights_fixture(tmp_path):
    from codexa.data.language_sources import (
        TINYSTORIES_BYTES,
        TINYSTORIES_LICENSE,
        TINYSTORIES_PATH,
        TINYSTORIES_REPO,
        TINYSTORIES_REVISION,
        TINYSTORIES_RIGHTS_POLICY,
        TINYSTORIES_SHA256,
    )
    from codexa.data.production import digest

    binding = {
        "repo": TINYSTORIES_REPO,
        "revision": TINYSTORIES_REVISION,
        "path": TINYSTORIES_PATH,
        "sha256": TINYSTORIES_SHA256,
        "bytes": TINYSTORIES_BYTES,
    }
    spec = binding | {"rights_policy": TINYSTORIES_RIGHTS_POLICY, "license": TINYSTORIES_LICENSE}

    def proof(name, data):
        path = tmp_path / name
        path.write_text(data)
        return {"path": str(path), "sha256": digest(path)}

    rights_review = proof(
        "review.json",
        json.dumps(
            {"fixture": "Independent original creator-rights analysis; no real source approval."}
        ),
    )
    decision = {
        "version": "codexa-tinystories-creator-rights-decision-1",
        "status": "approved_for_internal_research",
        "reviewer": "fictional fixture reviewer",
        "source_binding": binding,
        "license": TINYSTORIES_LICENSE,
        "creator_text_grant_established": True,
        "internal_research_training_permitted": True,
        "redistribution_allowed": False,
        "universal_noninfringement_certified": False,
        "rights_review_proof": rights_review,
    }
    proofs = {
        name: proof(name + ".txt", "Original fictional test evidence for " + name)
        for name in ["dataset_card", "generation_paper", "legalcode", "creator_declaration"]
    }
    proofs["independent_review"] = proof("decision.json", json.dumps(decision))
    evidence = binding | {
        "version": "codexa-tinystories-creator-rights-evidence-1",
        "status": "approved",
        "reviewer": "fictional fixture reviewer",
        "rights_policy": TINYSTORIES_RIGHTS_POLICY,
        "license": TINYSTORIES_LICENSE,
        "scope": "internal_research_training",
        "redistribution_allowed": False,
        "attribution": "Ronen Eldan and Yuanzhi Li; GPT-4-generated synthetic stories; roneneldan/TinyStories",
        "proofs": proofs,
    }
    return spec, evidence, decision


def test_creator_rights_are_separate_from_protocol_authorization_and_v3_adapters(tmp_path):
    from codexa.data.language_sources import ADAPTERS, TINYSTORIES_REPO, validate_narrative_rights

    source, evidence, _ = _narrative_rights_fixture(tmp_path)
    assert validate_narrative_rights(source, evidence)
    assert all(repo != TINYSTORIES_REPO for repo, _ in ADAPTERS.values())
    assert not validate_narrative_rights(source, {"status": "authorized", "approved": True})
    assert not validate_narrative_rights(
        source | {"rights_policy": "codexa-publisher-internal-research-2"}, evidence
    )
    assert not validate_narrative_rights(source, evidence | {"redistribution_allowed": True})
    assert not validate_narrative_rights(source, evidence | {"attribution": "Dataset publisher"})
    assert not validate_narrative_rights(source, evidence | {"status": "pending"})
    for key, value in [
        ("repo", "someone/else"),
        ("revision", "0" * 40),
        ("path", "TinyStoriesV2-GPT4-valid.txt"),
        ("sha256", "0" * 64),
        ("bytes", 1),
    ]:
        assert not validate_narrative_rights(source | {key: value}, evidence)
        assert not validate_narrative_rights(source, evidence | {key: value})


@pytest.mark.parametrize(
    "role",
    ["dataset_card", "generation_paper", "legalcode", "creator_declaration", "independent_review"],
)
def test_creator_rights_require_each_unchanged_primary_and_review_proof(tmp_path, role):
    from codexa.data.language_sources import validate_narrative_rights

    source, evidence, _ = _narrative_rights_fixture(tmp_path)
    assert validate_narrative_rights(source, evidence)
    assert not validate_narrative_rights(
        source, evidence | {"proofs": {k: v for k, v in evidence["proofs"].items() if k != role}}
    )
    path = __import__("pathlib").Path(evidence["proofs"][role]["path"])
    path.write_text("Changed proof")
    assert not validate_narrative_rights(source, evidence)


@pytest.mark.parametrize(
    "change",
    [
        {"status": "proposed"},
        {"creator_text_grant_established": False},
        {"internal_research_training_permitted": False},
        {"redistribution_allowed": True},
        {"universal_noninfringement_certified": True},
        {"source_binding": {}},
    ],
)
def test_creator_rights_review_content_and_nested_hashes_must_agree(tmp_path, change):
    from pathlib import Path

    from codexa.data.language_sources import validate_narrative_rights
    from codexa.data.production import digest

    source, evidence, decision = _narrative_rights_fixture(tmp_path)
    decision.update(change)
    path = Path(evidence["proofs"]["independent_review"]["path"])
    path.write_text(json.dumps(decision))
    evidence["proofs"]["independent_review"]["sha256"] = digest(path)
    assert not validate_narrative_rights(source, evidence)
