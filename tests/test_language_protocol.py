"""Language approval cannot inherit a historical pass or silently change thresholds."""

import copy
import json

import pytest

from codexa.data.language_protocol import (
    V2_VERSION,
    V3_VERSION,
    V4_VERSION,
    VERSION,
    adjudicate,
    specification,
    specification_hash,
    validate_authorization,
)
from codexa.data.production import digest
from codexa.data.quality_protocol import CHECKS
from scripts.datasets.combine_quality_reviews import combined
from scripts.datasets.review_continuation_corpus import adjudicate as dispatch


def packet():
    protocol = specification(["wikipedia-en", "tinystories-gpt4"])
    mapping = [{"id": f"w{i}", "kind": "window", "source": "mixture"} for i in range(300)]
    mapping += [
        {
            "id": f"{source}-{i}",
            "document_id": f"{source}-{i}",
            "kind": "document",
            "source": source,
            "coverage": "full document",
        }
        for source in protocol["sources"]
        for i in range(100)
    ]
    ratings = [
        {
            "id": row["id"],
            "usable": True,
            "severe": False,
            "critical": False,
            "coverage_complete": True,
            "checks": dict.fromkeys(CHECKS, "pass"),
            "reason": "Complete document read; connected prose with necessary context.",
        }
        for row in mapping
    ]
    return protocol, mapping, ratings


def test_language_complete_and_missing_review():
    protocol, mapping, ratings = packet()
    assert dispatch(mapping, ratings, protocol)["training_allowed"]
    assert not adjudicate(mapping, ratings[:-1], protocol)["training_allowed"]
    assert adjudicate(mapping, ratings, protocol)["protocol_version"] == VERSION


def test_language_threshold_boundaries_and_uncertainty():
    protocol, mapping, ratings = packet()
    docs = [r for r in ratings if not r["id"].startswith("w")]
    for row in docs[:10]:
        row["checks"]["correctness"] = "uncertain"
    assert adjudicate(mapping, ratings, protocol)["training_allowed"]
    docs[10]["usable"] = False
    assert not adjudicate(mapping, ratings, protocol)["training_allowed"]


@pytest.mark.parametrize("field", ["critical", "severe", "coverage_complete"])
def test_language_critical_severe_and_coverage_fail(field):
    protocol, mapping, ratings = packet()
    docs = [r for r in ratings if r["id"].startswith("tinystories")]
    for row in docs[:3]:
        row[field] = field != "coverage_complete"
    assert not adjudicate(mapping, ratings, protocol)["training_allowed"]


def test_language_cannot_relabel_old_source_or_weaken_spec():
    with pytest.raises(ValueError):
        specification(["fineweb-edu"])
    protocol, mapping, ratings = packet()
    protocol["usable_per_source_minimum"] = 85
    with pytest.raises(ValueError, match="frozen"):
        adjudicate(mapping, ratings, protocol)


def test_language_duplicate_document_and_independent_concern():
    protocol, mapping, ratings = packet()
    other = copy.deepcopy(ratings)
    other[0]["critical"] = True
    assert not dispatch(mapping, combined([ratings, other]), protocol)["training_allowed"]
    mapping[-1]["document_id"] = mapping[-2]["document_id"]
    with pytest.raises(ValueError, match="Duplicate"):
        adjudicate(mapping, ratings, protocol)


def test_language_authorization_binds_sources_budget_and_spec(tmp_path):
    protocol, _, _ = packet()
    path = tmp_path / "authorization.json"
    record = {
        "version": VERSION,
        "status": "authorized",
        "sources": protocol["sources"],
        "specification_sha256": specification_hash(protocol),
        "additional_target_tokens": 50_000_000,
        "user_decision": "Authorize successor; implement the plan.",
        "historical_roots": ["/historical-evidence"],
        "replaces": protocol["replaces"],
    }
    path.write_text(json.dumps(record))
    protocol.update(authorization_path=str(path), authorization_sha256=digest(path))
    assert validate_authorization(path, protocol)["status"] == "authorized"
    record["additional_target_tokens"] += 1
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="authorization"):
        validate_authorization(path, protocol)


def test_development_excerpt_excludes_full_origin_identity(tmp_path):
    import sqlite3

    from scripts.datasets.review_continuation_corpus import development_exclusions

    profiles = tmp_path / "reports/profiles"
    profiles.mkdir(parents=True)
    (profiles / "sample.json").write_text(json.dumps({"source_sha256": "source-a"}))
    (profiles / "sample.jsonl").write_text(
        json.dumps({"ordinal": 7, "after": "Only an excerpt was displayed."}) + "\n"
    )
    database = tmp_path / "documents.sqlite"
    with sqlite3.connect(database) as db:
        db.execute("CREATE TABLE docs(hash BLOB, metadata TEXT, role TEXT)")
        db.execute(
            "INSERT INTO docs VALUES(?,?,?)",
            (b"abc", json.dumps({"source_sha256": "source-a", "ordinal": 7}), "new"),
        )
    identities, passages, files = development_exclusions([tmp_path], database)
    assert identities == {b"abc".hex()}
    assert passages == [("Only an excerpt was displayed.", 64)]
    assert len(files) == 2


def test_version_one_specification_hash_remains_frozen():
    original = specification(["cosmopedia-v2", "tinystories-gpt4", "wikipedia-en"])
    assert specification_hash(original) == (
        "c260c9e066616aaf32940be0962edd43a883022a694839a399ba495b65613d29"
    )


def test_version_two_specification_hash_remains_frozen():
    from codexa.data.language_protocol import V2_VERSION

    original = specification(["common-corpus"], version=V2_VERSION)
    assert specification_hash(original) == (
        "54cd33da604603e9b48330e07ac9ab41c69ff0f0c4d30c578fd9233814af5617"
    )


def test_version_two_retains_all_quality_thresholds_and_freezes_new_limits():
    from codexa.data.language_protocol import V2_VERSION

    old = specification(["common-corpus"])
    new = specification(["common-corpus"], version=V2_VERSION)
    for key in (
        "windows",
        "window_tokens",
        "documents_per_source",
        "maximum_full_document_tokens",
        "usable_window_minimum",
        "usable_per_source_minimum",
        "severe_window_maximum",
        "severe_per_source_maximum",
        "critical_maximum",
        "complete_required",
        "early_stop",
        "independent_reviewers_required",
        "required_checks",
        "rubric",
        "exclusion_policy",
        "representativeness_policy",
        "disagreement_policy",
    ):
        assert new[key] == old[key]
    assert new["development_network_maximum_bytes"] == 1024**3
    assert new["acquisition_maximum_bytes"] == 16 * 1024**3
    assert new["base_language_readiness"] == {
        "development_completions": 100,
        "grammatical_coherent_complete_minimum": 80,
        "severe_loop_maximum": 10,
    }
    assert specification_hash(new) != specification_hash(old)


@pytest.mark.parametrize(
    "field,value",
    [
        ("acquisition_maximum_bytes", 17 * 1024**3),
        ("development_network_maximum_bytes", 2 * 1024**3),
        ("usable_per_source_minimum", 89),
        ("base_language_readiness", {"development_completions": 50}),
    ],
)
def test_version_two_cannot_silently_change_limits(field, value):
    from codexa.data.language_protocol import V2_VERSION, validate_specification

    protocol = specification(["common-corpus"], version=V2_VERSION)
    protocol[field] = value
    with pytest.raises(ValueError, match="frozen"):
        validate_specification(protocol)


def test_version_two_authorization_cannot_borrow_version_one_or_other_sources(tmp_path):
    from codexa.data.language_protocol import V2_VERSION

    protocol = specification(["common-corpus"], version=V2_VERSION)
    path = tmp_path / "authorization.json"
    record = {
        "version": V2_VERSION,
        "status": "authorized",
        "sources": protocol["sources"],
        "specification_sha256": specification_hash(protocol),
        "additional_target_tokens": 50_000_000,
        "user_decision": "Authorize version 2; implement the plan.",
        "historical_roots": ["/historical-evidence"],
        "replaces": protocol["replaces"],
    }

    def bind():
        path.write_text(json.dumps(record))
        protocol.update(authorization_path=str(path), authorization_sha256=digest(path))

    bind()
    assert validate_authorization(path, protocol)["version"] == V2_VERSION
    for field, changed in (
        ("version", VERSION),
        ("sources", ["another-corpus"]),
        ("additional_target_tokens", 50_000_001),
        ("specification_sha256", "0" * 64),
    ):
        previous = record[field]
        record[field] = changed
        bind()
        with pytest.raises(ValueError, match="authorization"):
            validate_authorization(path, protocol)
        record[field] = previous


def test_version_two_adjudication_dispatch_preserves_failures():
    from codexa.data.language_protocol import V2_VERSION

    old, mapping, ratings = packet()
    protocol = specification(old["sources"], version=V2_VERSION)
    assert dispatch(mapping, ratings, protocol)["protocol_version"] == V2_VERSION
    for row in ratings[:31]:
        row["checks"]["coherence"] = "uncertain"
    assert not dispatch(mapping, ratings, protocol)["training_allowed"]


@pytest.mark.parametrize("source", ["dclm-edu", "pressbooks-filtered"])
def test_version_three_only_changes_authorized_source_binding(source):
    from codexa.data.language_protocol import V2_VERSION, V3_SOURCES, V3_VERSION

    old = specification(["common-corpus"], version=V2_VERSION)
    protocol = specification([source], version=V3_VERSION)
    changed = {"version", "sources", "seed", "replaces"}
    assert {key: value for key, value in old.items() if key not in changed} == {
        key: value for key, value in protocol.items() if key not in changed | {"source_bindings"}
    }
    assert protocol["seed"] == 2026111003
    assert protocol["source_bindings"] == {source: V3_SOURCES[source]}
    assert protocol["source_bindings"][source] is not V3_SOURCES[source]


@pytest.mark.parametrize(
    "sources", [[], ["common-corpus"], ["dclm-edu", "pressbooks-filtered"], ["smol-smoltalk"]]
)
def test_version_three_rejects_unregistered_sources_and_mixtures(sources):
    from codexa.data.language_protocol import V3_VERSION

    with pytest.raises(ValueError):
        specification(sources, version=V3_VERSION)


@pytest.mark.parametrize("field", ["repo", "revision"])
def test_version_three_rejects_substituted_source_binding(field):
    from codexa.data.language_protocol import V3_VERSION, validate_specification

    protocol = specification(["dclm-edu"], version=V3_VERSION)
    protocol["source_bindings"]["dclm-edu"][field] = "unapproved"
    with pytest.raises(ValueError, match="frozen"):
        validate_specification(protocol)


def test_version_three_authorization_binds_one_source_without_borrowing_approvals(tmp_path):
    from codexa.data.language_protocol import V2_VERSION, V3_VERSION

    path = tmp_path / "authorization.json"
    protocol = specification(["dclm-edu"], version=V3_VERSION)
    record = {
        "version": V3_VERSION,
        "status": "authorized",
        "sources": protocol["sources"],
        "specification_sha256": specification_hash(protocol),
        "additional_target_tokens": 50_000_000,
        "user_decision": "Authorize both source paths separately; implement the plan.",
        "historical_roots": ["/historical-evidence"],
        "replaces": protocol["replaces"],
    }

    def bind():
        path.write_text(json.dumps(record))
        protocol.update(authorization_path=str(path), authorization_sha256=digest(path))

    bind()
    assert validate_authorization(path, protocol)["sources"] == ["dclm-edu"]
    for field, changed in (
        ("version", V2_VERSION),
        ("sources", ["pressbooks-filtered"]),
        ("additional_target_tokens", 50_000_001),
    ):
        previous = record[field]
        record[field] = changed
        bind()
        with pytest.raises(ValueError, match="authorization"):
            validate_authorization(path, protocol)
        record[field] = previous

    bind()
    alternate = specification(["pressbooks-filtered"], version=V3_VERSION)
    alternate.update(authorization_path=str(path), authorization_sha256=digest(path))
    with pytest.raises(ValueError, match="authorization"):
        validate_authorization(path, alternate)


@pytest.mark.parametrize(
    "version,source",
    [("codexa-language-quality-3", "dclm-edu"), ("codexa-language-quality-4", "tinystories-gpt4")],
)
def test_successor_adjudication_dispatch_preserves_uncertainty_and_completeness(version, source):

    protocol = specification([source], version=version)
    mapping = [{"id": f"w{i}", "kind": "window", "source": "mixture"} for i in range(300)]
    mapping += [
        {
            "id": f"d{i}",
            "document_id": f"d{i}",
            "kind": "document",
            "source": source,
            "coverage": "full document",
        }
        for i in range(100)
    ]
    ratings = [
        {
            "id": row["id"],
            "usable": True,
            "severe": False,
            "critical": False,
            "coverage_complete": True,
            "checks": dict.fromkeys(CHECKS, "pass"),
            "reason": "Original fixture for complete review gate coverage.",
        }
        for row in mapping
    ]
    assert dispatch(mapping, ratings, protocol)["training_allowed"]
    assert not dispatch(mapping, ratings[:-1], protocol)["training_allowed"]
    for row in ratings[-11:]:
        row["checks"]["correctness"] = "uncertain"
    assert not dispatch(mapping, ratings, protocol)["training_allowed"]


@pytest.mark.parametrize(
    "source,expected",
    [
        ("dclm-edu", "ecb8fefee351b1a07daa74cf9efa8abe32b6d2daa5776b0ce2856602b5bc3c6d"),
        (
            "pressbooks-filtered",
            "970462b4f001c6f1bff669ac91799ee7c4521eb9089c2e954c3e4c9f3759e19f",
        ),
    ],
)
def test_version_three_specification_hash_remains_frozen(source, expected):
    from codexa.data.language_protocol import V3_VERSION

    assert specification_hash(specification([source], version=V3_VERSION)) == expected


def test_version_four_only_changes_pinned_source_and_version_metadata():
    from codexa.data.language_protocol import V3_VERSION, V4_SOURCES, V4_VERSION

    old = specification(["pressbooks-filtered"], version=V3_VERSION)
    new = specification(["tinystories-gpt4"], version=V4_VERSION)
    changed = {"version", "sources", "seed", "replaces", "source_bindings"}
    assert {k: v for k, v in old.items() if k not in changed} == {
        k: v for k, v in new.items() if k not in changed
    }
    assert new["seed"] == 2026111004
    assert new["source_bindings"] == V4_SOURCES
    assert new["source_bindings"]["tinystories-gpt4"] is not V4_SOURCES["tinystories-gpt4"]
    assert specification_hash(new) == (
        "26dc1c54233072eed5a62baaa44936156df2934b85f848cb7c9039851675072b"
    )


@pytest.mark.parametrize(
    "sources",
    [[], ["pressbooks-filtered"], ["tinystories-gpt4", "dclm-edu"], ["smol-smoltalk"]],
)
def test_version_four_rejects_other_sources_and_mixtures(sources):
    from codexa.data.language_protocol import V4_VERSION

    with pytest.raises(ValueError):
        specification(sources, version=V4_VERSION)


@pytest.mark.parametrize("field", ["repo", "revision", "path", "sha256", "bytes"])
def test_version_four_requires_exact_preserved_payload_binding(field):
    from codexa.data.language_protocol import V4_VERSION, validate_specification

    protocol = specification(["tinystories-gpt4"], version=V4_VERSION)
    protocol["source_bindings"]["tinystories-gpt4"][field] = "substituted"
    with pytest.raises(ValueError, match="frozen"):
        validate_specification(protocol)


def test_version_four_cannot_launch_with_pending_or_borrowed_authorization(tmp_path):
    from codexa.data.language_protocol import V3_VERSION, V4_VERSION

    protocol = specification(["tinystories-gpt4"], version=V4_VERSION)
    path = tmp_path / "authorization.json"
    record = {
        "version": V4_VERSION,
        "status": "authorized",
        "sources": protocol["sources"],
        "specification_sha256": specification_hash(protocol),
        "additional_target_tokens": 50_000_000,
        "user_decision": "Explicit authorization fixture for the TinyStories-only diagnostic.",
        "historical_roots": ["/historical-evidence"],
        "replaces": protocol["replaces"],
    }

    def bind():
        path.write_text(json.dumps(record))
        protocol.update(authorization_path=str(path), authorization_sha256=digest(path))

    bind()
    assert validate_authorization(path, protocol)["version"] == V4_VERSION
    for field, changed in (
        ("status", "PROPOSED_PENDING_EXPLICIT_USER_AUTHORIZATION"),
        ("version", V3_VERSION),
        ("sources", ["pressbooks-filtered"]),
        ("specification_sha256", "0" * 64),
        ("additional_target_tokens", 50_000_001),
    ):
        original = record[field]
        record[field] = changed
        bind()
        with pytest.raises(ValueError, match="authorization"):
            validate_authorization(path, protocol)
        record[field] = original


@pytest.mark.parametrize("version", [VERSION, V2_VERSION, V3_VERSION, V4_VERSION])
def test_100m_requires_v4_and_explicit_superseding_budget_record(tmp_path, version):
    source = {V3_VERSION: "dclm-edu", V4_VERSION: "tinystories-gpt4"}.get(version, "wikipedia-en")
    protocol = specification([source], version=version)
    path = tmp_path / "authorization.json"
    record = {
        "version": version,
        "status": "authorized",
        "sources": protocol["sources"],
        "specification_sha256": specification_hash(protocol),
        "additional_target_tokens": 100_000_000,
        "user_decision": "Start one 100M continuation instead of the pending 50M run.",
        "historical_roots": ["/preserved-evidence"],
        "replaces": protocol["replaces"],
        "supersedes_additional_target_tokens": 50_000_000,
        "cumulative_target_tokens": 600_006_912,
    }

    def bind():
        path.write_text(json.dumps(record))
        protocol.update(authorization_path=str(path), authorization_sha256=digest(path))

    bind()
    if version != V4_VERSION:
        with pytest.raises(ValueError, match="authorization"):
            validate_authorization(path, protocol)
        return
    assert validate_authorization(path, protocol)["additional_target_tokens"] == 100_000_000
    for field, changed in (
        ("supersedes_additional_target_tokens", None),
        ("cumulative_target_tokens", 650_006_912),
        ("additional_target_tokens", 150_000_000),
        ("user_decision", ""),
    ):
        original = record[field]
        record[field] = changed
        bind()
        with pytest.raises(ValueError, match="authorization"):
            validate_authorization(path, protocol)
        record[field] = original


def general_packet():
    from codexa.data.language_protocol import V7_VERSION

    protocol = specification(["wikimedia-wikipedia"], version=V7_VERSION)
    mapping = [
        {
            "id": f"d{i}",
            "kind": "document",
            "source": "wikimedia-wikipedia",
            "document_id": f"original-cc0-{i}",
            "coverage": "full document",
        }
        for i in range(100)
    ] + [{"id": f"w{i}", "kind": "window", "source": "wikimedia-wikipedia"} for i in range(300)]
    ratings = [
        {
            "id": x["id"],
            "kind": x["kind"],
            "band": "U1",
            "decision": "usable",
            "scope": "eligible",
            "coverage_complete": True,
            "severe": False,
            "critical": False,
            "confirmed_source_defect": True,
            "defect_categories": ["localized_original_fixture_typo"],
            "observed_evidence": ["Original CC0 fixture"],
            "reason": "A localized original fixture typo leaves the central explanation independently intelligible.",
            "residual_utility": "The complete central English explanation remains useful.",
        }
        for x in mapping
    ]
    return protocol, mapping, ratings


def test_general_bands_keep_fixed_denominators_and_thresholds():
    protocol, mapping, ratings = general_packet()
    for x in ratings[:10] + ratings[100:130]:
        x.update(band="X", decision="out_of_scope", scope="out_of_scope")
    report = adjudicate(mapping, ratings, protocol)
    assert report["status"] == "passed"
    assert report["panels"]["wikimedia-wikipedia"]["usable"] == 90
    assert report["panels"]["windows"]["usable"] == 270
    ratings[10].update(band="H", decision="hold")
    assert adjudicate(mapping, ratings, protocol)["status"] == "failed"
    ratings[10].update(band="U1", decision="usable")
    ratings[130].update(band="R2", decision="reject", severe=True)
    assert adjudicate(mapping, ratings, protocol)["status"] == "failed"


def test_general_combination_preserves_local_defects_and_critical_exclusions():
    from codexa.data.language_protocol import V7_VERSION, combine_reviews

    protocol, mapping, ratings = general_packet()
    second = copy.deepcopy(ratings)
    second[0].update(band="R2", decision="reject", severe=True)
    second[1].update(band="X", decision="out_of_scope", scope="out_of_scope")
    result = combine_reviews([ratings, second], version=V7_VERSION)
    assert result[0]["band"] == "R2" and result[0]["severe"]
    assert result[1]["band"] == "H"
    assert result[2]["band"] == "U1" and result[2]["confirmed_source_defect"]
    assert adjudicate(mapping, result, protocol)["status"] == "passed"
    ratings[1].update(band="C", decision="reject", critical=True)
    result = combine_reviews([ratings, second], version=V7_VERSION)
    assert result[1]["band"] == "C" and result[1]["critical"]
    assert adjudicate(mapping, result, protocol)["status"] == "failed"


def test_general_adjudication_requires_consistent_evidence_and_explicit_launch():
    from codexa.data.language_protocol import V7_VERSION, authorized_budget

    protocol, mapping, ratings = general_packet()
    ratings[0]["severe"] = True
    with pytest.raises(ValueError, match="severe"):
        adjudicate(mapping, ratings, protocol)
    ratings[0]["severe"] = False
    ratings[0]["observed_evidence"] = []
    with pytest.raises(ValueError, match="evidence"):
        adjudicate(mapping, ratings, protocol)
    assert not authorized_budget(
        {
            "additional_target_tokens": 1_000_000_000,
            "cumulative_target_tokens": 1_500_006_912,
            "user_decision": "Quality rubric approval",
            "training_launch_authorized": False,
        },
        V7_VERSION,
    )
