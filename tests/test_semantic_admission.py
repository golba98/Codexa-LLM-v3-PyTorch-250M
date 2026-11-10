"""Positive admission requires complete, distinct, intact semantic assessments."""

import hashlib
import json
import sqlite3
import zlib

import numpy as np
import pytest

from codexa.data.production import atomic, digest
from codexa.data.quality_protocol import CHECKS
from codexa.data.remediate import rebuild
from codexa.data.semantic import assessed, cleared, load_admission
from codexa.data.v2_storage import Storage
from scripts.datasets.semantic_admission import finalize, merge_shards
from scripts.datasets.verify_remediated_snapshot import check


def rating(ident="S1", judge="a"):
    return {
        "id": ident,
        "judge_id": judge,
        "usable": True,
        "severe": False,
        "critical": False,
        "coverage_complete": True,
        "checks": dict.fromkeys(CHECKS, "pass"),
        "reason": "Complete self-contained derivation checked with the stated assumptions.",
        "topic": "algebra",
        "level": "secondary",
        "format": "worked_solution",
    }


def census(path, parent_sha, records, texts=None):
    path.mkdir(parents=True, exist_ok=True)
    atomic(path / "protocol.json", {"parent_manifest_sha256": parent_sha})
    atomic(path / "sampling.json", {"candidates": len(records)})
    for name, rows in [
        ("mapping.jsonl", records),
        (
            "blind.jsonl",
            [
                {"id": r["id"], "text": (texts or {}).get(r["id"], "A complete explanation.")}
                for r in records
            ],
        ),
    ]:
        (path / name).write_text("".join(json.dumps(row) + "\n" for row in rows))
    atomic(
        path / "packet-integrity.json",
        {
            name: digest(path / name)
            for name in ("protocol.json", "mapping.jsonl", "blind.jsonl", "sampling.json")
        },
    )
    atomic(
        path / "token-verification.json",
        {
            "status": "passed",
            "manifest_sha256": parent_sha,
            "documents": len(records),
            "packet_integrity_sha256": digest(path / "packet-integrity.json"),
        },
    )
    for judge in ("a", "b"):
        (path / ("curator-" + judge + ".jsonl")).write_text(
            "".join(json.dumps(rating(r["id"], judge)) + "\n" for r in records)
        )
    finalize(path)


def refresh_census_fixture(path):
    """Refresh synthetic fixture hashes only; production requires native-token verification."""
    integrity = json.loads((path / "packet-integrity.json").read_text())
    atomic(path / "packet-integrity.json", {name: digest(path / name) for name in integrity})
    proof = json.loads((path / "token-verification.json").read_text())
    proof["packet_integrity_sha256"] = digest(path / "packet-integrity.json")
    atomic(path / "token-verification.json", proof)
    (path / "admission.jsonl").unlink()
    (path / "decision.json").unlink()
    finalize(path)


def test_extension_is_seeded_stratified_and_excludes_census_passages(tmp_path, monkeypatch):
    from scripts.datasets.semantic_admission import prepare

    monkeypatch.setattr(Storage, "check", lambda *a, **kw: None)
    Storage(tmp_path).initialize()
    parent = tmp_path / "snapshots/parent"
    parent.mkdir(parents=True)
    manifest = {"splits": {s: [] for s in ("train", "val", "test")}, "tokenizer_hash": "frozen"}
    atomic(parent / "manifest.json", manifest)
    monkeypatch.setattr("scripts.datasets.semantic_admission.verify", lambda _: manifest)
    records, texts = [], {}
    for source, train_count in (("fineweb-edu", 6), ("finemath-4plus", 3)):
        for split in ("train", "val", "test"):
            for i in range(train_count if split == "train" else 2):
                ident = hashlib.sha256(f"{source}:{split}:{i}".encode()).hexdigest()
                text = " ".join(f"{source}_{split}_{i}_{j}" for j in range(180))
                records.append({"id": ident, "source": source, "split": split, "tokens": 4})
                texts[ident] = text
    (parent / "documents.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    original = tmp_path / "semantic-admission/original"
    first, historical, shared = records[:3]
    census(
        original,
        digest(parent / "manifest.json"),
        [
            {
                "id": "S1",
                "document_id": first["id"],
                **{k: first[k] for k in ("source", "split", "tokens")},
            }
        ],
    )
    (original / "blind.jsonl").write_text(
        json.dumps({"id": "S1", "text": texts[first["id"]]}) + "\n"
    )
    refresh_census_fixture(original)
    texts[shared["id"]] = " ".join(texts[first["id"]].split()[:70]) + " " + texts[shared["id"]]
    db = sqlite3.connect(tmp_path / "indexes/documents.sqlite")
    db.execute("CREATE TABLE docs(hash BLOB PRIMARY KEY,payload BLOB)")
    db.executemany(
        "INSERT INTO docs VALUES(?,?)",
        [(bytes.fromhex(ident), zlib.compress(text.encode())) for ident, text in texts.items()],
    )
    db.commit()
    db.close()
    review = tmp_path / "historical"
    review.mkdir()
    (review / "mapping.jsonl").write_text(
        json.dumps({"id": "R1", "kind": "document", "document_id": historical["id"]}) + "\n"
    )
    (review / "blind.jsonl").write_text(
        json.dumps({"id": "R1", "text": texts[historical["id"]]}) + "\n"
    )
    atomic(
        review / "packet-integrity.json",
        {n: digest(review / n) for n in ("mapping.jsonl", "blind.jsonl")},
    )
    plan = tmp_path / "plan.json"
    atomic(
        plan,
        {
            "version": "semantic-admission-extension-1",
            "parent_manifest_sha256": digest(parent / "manifest.json"),
            "original_census_decision_sha256": digest(original / "decision.json"),
            "source_train_budgets": {"fineweb-edu": 2, "finemath-4plus": 1},
            "seed": 19,
            "heldout_candidate_budget_per_source_split": 1,
            "sample_prefix": "T",
            "require_primary_evidence": True,
        },
    )
    outputs = [tmp_path / "semantic-admission/one", tmp_path / "semantic-admission/two"]
    for output in outputs:
        prepare(tmp_path, parent, output, [review], plan, [original])
        selected = [json.loads(x) for x in (output / "mapping.jsonl").read_text().splitlines()]
        assert len(selected) == 7 and all(r["id"].startswith("T") for r in selected)
        assert not {first["id"], historical["id"], shared["id"]} & {
            r["document_id"] for r in selected
        }
        assert sum(r["source"] == "fineweb-edu" and r["split"] == "train" for r in selected) == 2
        assert (output / "sampling-plan.json").read_bytes() == plan.read_bytes()
    assert digest(outputs[0] / "packet-integrity.json") == digest(
        outputs[1] / "packet-integrity.json"
    )
    with pytest.raises(ValueError, match="original census exclusion"):
        prepare(tmp_path, parent, tmp_path / "missing-exclusion", [], plan)


def test_union_preserves_disjoint_judgments_and_primary_evidence_requirements(
    tmp_path, monkeypatch
):
    from scripts.datasets.semantic_evidence import union

    monkeypatch.setattr(Storage, "check", lambda *a, **kw: None)
    Storage(tmp_path).initialize()
    components = [tmp_path / "semantic-admission/old", tmp_path / "semantic-admission/new"]
    for i, component in enumerate(components):
        census(
            component,
            "parent",
            [
                {
                    "id": f"C{i}",
                    "document_id": f"doc{i}",
                    "source": "fineweb-edu",
                    "split": "train",
                    "tokens": 4,
                }
            ],
        )
    evidence = [
        {
            "claim": "Fixture claim",
            "source_url": "https://example.org/fixture",
            "finding": "Synthetic support only",
            "limitation": "No real review",
        }
    ]
    newer = components[1]
    atomic(
        newer / "protocol.json",
        {"parent_manifest_sha256": "parent", "require_primary_evidence": True},
    )
    with pytest.raises(ValueError, match="explicit primary evidence"):
        refresh_census_fixture(newer)
    for judge in "ab":
        (newer / f"curator-{judge}.jsonl").write_text(
            json.dumps(rating("C1", judge) | {"evidence": evidence}) + "\n"
        )
    finalize(newer)
    output = tmp_path / "semantic-admission/union"
    union(tmp_path, components, output)
    protocol = json.loads((output / "protocol.json").read_text())
    assert protocol["primary_evidence_required_ids"] == ["C1"]
    expected = [json.loads((p / "curator-a.jsonl").read_text()) for p in components]
    assert [
        json.loads(x) for x in (output / "curator-a.jsonl").read_text().splitlines()
    ] == expected
    with pytest.raises(FileNotFoundError):
        finalize(output)
    atomic(
        output / "token-verification.json",
        {
            "status": "passed",
            "manifest_sha256": "parent",
            "documents": 2,
            "packet_integrity_sha256": digest(output / "packet-integrity.json"),
        },
    )
    finalize(output)
    assert len(load_admission(output, "parent")) == 2
    with pytest.raises(ValueError, match="overlap"):
        union(tmp_path, [components[0], components[0]], tmp_path / "semantic-admission/overlap")
    other = tmp_path / "semantic-admission/other"
    census(
        other,
        "other-parent",
        [
            {
                "id": "other",
                "document_id": "other-doc",
                "source": "fineweb-edu",
                "split": "train",
                "tokens": 4,
            }
        ],
    )
    with pytest.raises(ValueError, match="different parents"):
        union(tmp_path, [components[0], other], tmp_path / "semantic-admission/mismatch")


def test_unchecked_uncertain_or_uncovered_records_are_not_admitted():
    row = rating()
    assert cleared(row, "finemath-4plus")
    row["checks"]["correctness"] = "uncertain"
    assert not cleared(row, "finemath-4plus")
    row["checks"]["correctness"] = "not_applicable"
    assert not cleared(row, "finemath-4plus")
    with pytest.raises(ValueError, match="entire preregistered"):
        assessed([], {"S1"})
    row["coverage_complete"] = False
    with pytest.raises(ValueError, match="Complete each"):
        assessed([row], {"S1"})


def test_identity_and_empty_math_slots_preserve_valid_prose():
    from codexa.data.representation import rejection

    bad = "Alex Smith (January 1, 1901–June 2, 1980) was a researcher.\n\nAlex Smith was born on January 1, 1911."
    assert rejection(bad) == "contradictory_birth_fields"
    assert rejection(bad.replace("1911", "1901")) is None
    assert rejection(bad.replace("Alex Smith was born", "Jane Smith was born")) is None
    assert (
        rejection("John Smith (1901–1980) was a researcher.\n\nJoan Smith was born in 1911.")
        is None
    )
    assert (
        rejection(
            "Alex Smith (1901–1980) was a researcher.\n\nAlex Smith was born in Boston to Jane Smith (1870–1950)."
        )
        is None
    )
    assert (
        rejection(
            "Alex Smith (1901–1980) was a researcher.\n\nAlex Smith was born in Boston, where he started work in 1920."
        )
        is None
    )
    assert (
        rejection("A functor from  to groups. If we take a ring, for example, , then this follows.")
        == "erased_mathematical_slots"
    )
    assert (
        rejection(
            "A functor from rings to groups. If we take a ring, for example, Z, then this follows."
        )
        is None
    )


def test_missing_content_checks_preserve_complete_technical_material():
    from codexa.data.representation import rejection

    assert (
        rejection("If  is a vector in  and  is a unit vector. Where  is the angle.")
        == "repeated_erased_content_slots"
    )
    assert (
        rejection("Proven gas reserves of . The width of , and height of .")
        == "repeated_erased_content_slots"
    )
    assert (
        rejection(
            "In  appeared a map. Its wording was incorrect:\n\ninstead of the original text:\n\n[[Image:map.png]]"
        )
        == "erased_required_content_block"
    )
    assert (
        rejection(
            "This enzyme catalyzes the chemical reaction\n\nNAD+ + substrate  product + byproduct"
        )
        == "erased_reaction_relation"
    )
    assert (
        rejection(
            "Exampleville is a sovereign nation near Oak Township, Cedar County, in the U.S. state of Indiana."
        )
        == "contradictory_sovereign_locality_claim"
    )
    valid = [
        "If x is a vector in R^3 and k is a unit vector, the formula is:\n\nv*cos(t)+(k cross v)*sin(t)+k*(k dot v)*(1-cos(t)).",
        "The enzyme catalyzes the chemical reaction\n\nNAD+ + substrate ⇌ product + byproduct",
        "The enzyme catalyzes the chemical reaction\n\nThe substrate is converted to the product.",
        "This self-proclaimed micronation is a sovereign nation in Example County, in the U.S. state of Nevada, according to its founders.",
        "The phonemes are transcribed as /a/ and /b/, pronounced as [a] and [b].",
        "The treaty wording was incorrect:\n\nIt assigned the island to the wrong country. The original text states:\n\nThe boundary follows the channel.",
        "Gas reserves of 42 billion cubic metres, width of 20 metres and height of 5 metres.",
    ]
    assert all(rejection(text) is None for text in valid)
    for arrow in (
        "⇄",
        "⇆",
        "⟷",
        r"\longrightarrow",
        r"\xrightarrow{enzyme}",
        r"\leftrightharpoons",
    ):
        assert (
            rejection(f"The enzyme catalyzes the chemical reaction\n\nsubstrate {arrow} product")
            is None
        )
    assert (
        rejection(
            "Fill in the blanks: a functor from  to groups. A ring, for example, , then complete the diagram."
        )
        is None
    )


def test_admission_rejects_stale_parent_and_modified_judgments(tmp_path):
    records = [
        {
            "id": "S1",
            "document_id": "one",
            "source": "finemath-4plus",
            "split": "train",
            "tokens": 4,
        }
    ]
    census(tmp_path, "parent", records)
    assert load_admission(tmp_path, "parent")["one"]["status"] == "admitted"
    with pytest.raises(ValueError, match="another parent"):
        load_admission(tmp_path, "other")
    with (tmp_path / "curator-b.jsonl").open("a") as out:
        out.write(json.dumps(rating()) + "\n")
    with pytest.raises(ValueError, match="judgments changed"):
        load_admission(tmp_path, "parent")


def test_two_assessments_by_same_judge_cannot_authorize_admission(tmp_path):
    records = [
        {
            "id": "S1",
            "document_id": "one",
            "source": "finemath-4plus",
            "split": "train",
            "tokens": 4,
        }
    ]
    census(tmp_path, "parent", records)
    # Retain a self-consistent ledger/hash set while simulating an invalid
    # provenance claim; the loader must independently enforce judge separation.
    data = (tmp_path / "curator-a.jsonl").read_text()
    (tmp_path / "curator-b.jsonl").write_text(data)
    decision = json.loads((tmp_path / "decision.json").read_text())
    decision["reviewer_hashes"]["curator-b.jsonl"] = digest(tmp_path / "curator-b.jsonl")
    atomic(tmp_path / "decision.json", decision)
    with pytest.raises(ValueError, match="different independent"):
        load_admission(tmp_path, "parent")


def test_changed_census_input_proof_cannot_authorize_admission(tmp_path):
    records = [
        {"id": "S1", "document_id": "one", "source": "fineweb-edu", "split": "val", "tokens": 4}
    ]
    census(tmp_path, "parent", records)
    proof = tmp_path / "token-verification.json"
    data = json.loads(proof.read_text())
    data["manifest_sha256"] = "other"
    atomic(proof, data)
    with pytest.raises(ValueError, match="text-to-token proof"):
        load_admission(tmp_path, "parent")


def test_shard_merge_requires_complete_actual_owned_judgments(tmp_path):
    (tmp_path / "mapping.jsonl").write_text('{"id":"S1"}\n{"id":"S2"}\n')
    assignments = tmp_path / "assignments.json"
    atomic(
        assignments,
        {
            "shards": [
                {
                    "range": ["S1", "S2"],
                    "a": "alice",
                    "b": "bob",
                    "a_file": "a.jsonl",
                    "b_file": "b.jsonl",
                }
            ]
        },
    )
    (tmp_path / "a.jsonl").write_text(
        "".join(json.dumps(rating(ident, "alice")) + "\n" for ident in ("S2", "S1"))
    )
    (tmp_path / "b.jsonl").write_text(json.dumps(rating("S1", "bob")) + "\n")
    with pytest.raises(ValueError, match="entire preregistered"):
        merge_shards(tmp_path, assignments)
    assert not (tmp_path / "curator-a.jsonl").exists()
    (tmp_path / "b.jsonl").write_text(
        "".join(json.dumps(rating(ident, "alice")) + "\n" for ident in ("S1", "S2"))
    )
    with pytest.raises(ValueError, match="declared independent owner"):
        merge_shards(tmp_path, assignments)
    (tmp_path / "b.jsonl").write_text(
        "".join(json.dumps(rating(ident, "bob")) + "\n" for ident in ("S1", "S2"))
    )
    merge_shards(tmp_path, assignments)
    assert [
        r["id"] for r in map(json.loads, (tmp_path / "curator-a.jsonl").read_text().splitlines())
    ] == ["S1", "S2"]
    with pytest.raises(FileExistsError):
        merge_shards(tmp_path, assignments)


def test_semantic_rebuild_quarantines_unreviewed_records_and_preserves_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(Storage, "check", lambda *a, **kw: None)
    Storage(tmp_path).initialize()
    parent = tmp_path / "snapshots/parent"
    parent.mkdir(parents=True)
    db = sqlite3.connect(tmp_path / "indexes/documents.sqlite")
    db.execute(
        "CREATE TABLE docs(id INTEGER PRIMARY KEY,hash BLOB,payload BLOB,metadata TEXT,role TEXT)"
    )
    listing, splits, texts = [], {}, {}
    for i, (split, source) in enumerate(
        [
            ("train", "finemath-4plus"),
            ("train", "finemath-4plus"),
            ("train", "finemath-4plus"),
            ("val", "wikipedia-en"),
        ]
    ):
        text = (
            "The graph shows the scores. What is the largest score?"
            if i == 0
            else f"A useful complete explanation number {i}."
        )
        ident = hashlib.sha256(text.encode()).hexdigest()
        data, index = f"tokens-{i}.bin", f"tokens-{i}.idx"
        np.asarray([2, 4, 5, 3], dtype="<u2").tofile(parent / data)
        np.asarray([0], dtype="<u8").tofile(parent / index)
        splits.setdefault(split, []).append(
            {
                "file": data,
                "index": index,
                "tokens": 4,
                "documents": 1,
                "sha256": digest(parent / data),
                "index_sha256": digest(parent / index),
            }
        )
        listing.append({"id": ident, "source": source, "split": split, "tokens": 4})
        texts[ident] = text
        db.execute(
            "INSERT INTO docs(hash,payload,metadata,role) VALUES(?,?,?,?)",
            (
                bytes.fromhex(ident),
                zlib.compress(text.encode()),
                json.dumps({"characters": len(text), "ordinal": i}),
                "new",
            ),
        )
    db.commit()
    db.close()
    (parent / "documents.jsonl").write_text("".join(json.dumps(row) + "\n" for row in listing))
    atomic(
        parent / "manifest.json",
        {
            "version": 2,
            "stage": "tokenized",
            "dtype": "<u2",
            "vocab_size": 260,
            "tokenizer_hash": "frozen",
            "identity": {},
            "splits": splits,
            "preparation_complete": True,
            "documents_sha256": digest(parent / "documents.jsonl"),
        },
    )
    atomic(parent / "audit.json", {"manifest_sha256": digest(parent / "manifest.json")})
    admission = tmp_path / "semantic-admission/census"
    census(
        admission,
        digest(parent / "manifest.json"),
        [
            {
                "id": "S1",
                "document_id": listing[0]["id"],
                "source": listing[0]["source"],
                "split": "train",
                "tokens": 4,
            },
            {
                "id": "S2",
                "document_id": listing[2]["id"],
                "source": listing[2]["source"],
                "split": "train",
                "tokens": 4,
            },
        ],
        {"S1": texts[listing[0]["id"]], "S2": texts[listing[2]["id"]]},
    )
    output = tmp_path / "snapshots/semantic"
    result = rebuild(tmp_path, parent, output, semantic_admission=admission)
    assert result["split_tokens"] == {"train": 4, "val": 4}
    rows = [json.loads(x) for x in (output / "quarantine.jsonl").read_text().splitlines()]
    assert [(row["id"], row["reason"]) for row in rows] == [
        (listing[0]["id"], "missing_essential_figure"),
        (listing[1]["id"], "semantic_review_pending"),
    ]
    assert rows[0]["semantic_disposition"]["status"] == "admitted"
    assert check(tmp_path, parent, output)["retained_documents_bitwise_compared"] == 2
    db = sqlite3.connect(tmp_path / "indexes/documents.sqlite")
    altered = texts[listing[2]["id"]].replace("useful complete", "useful  complete")
    db.execute(
        "UPDATE docs SET payload=? WHERE hash=?",
        (zlib.compress(altered.encode()), bytes.fromhex(listing[2]["id"])),
    )
    db.commit()
    db.close()
    with pytest.raises(ValueError, match="differs from independently verified review text"):
        rebuild(tmp_path, parent, tmp_path / "snapshots/changed-text", semantic_admission=admission)


def test_evidence_resolution_preserves_defects_and_requires_new_proof(tmp_path, monkeypatch):
    from scripts.datasets.semantic_evidence import compose, prepare

    monkeypatch.setattr(Storage, "check", lambda *a, **kw: None)
    Storage(tmp_path).initialize()
    base = tmp_path / "semantic-admission/original"
    records = [
        {
            "id": f"S{i}",
            "document_id": f"doc{i}",
            "source": "fineweb-edu",
            "split": "train",
            "tokens": 4,
        }
        for i in (1, 2, 3)
    ]
    census(base, "parent", records)

    def refresh_fixture(directory):
        # Synthetic fixture proofs only; production requires actual token comparison.
        integrity = json.loads((directory / "packet-integrity.json").read_text())
        atomic(directory / "packet-integrity.json", {n: digest(directory / n) for n in integrity})
        proof = json.loads((directory / "token-verification.json").read_text())
        proof["packet_integrity_sha256"] = digest(directory / "packet-integrity.json")
        atomic(directory / "token-verification.json", proof)
        (directory / "admission.jsonl").unlink()
        (directory / "decision.json").unlink()
        finalize(directory)

    rows = [rating(f"S{i}") for i in (1, 2, 3)]
    rows[1]["usable"] = False
    rows[1]["checks"]["correctness"] = "uncertain"
    rows[2]["usable"] = False
    rows[2]["checks"]["correctness"] = "uncertain"
    rows[2]["checks"]["notation"] = "defect"
    (base / "curator-a.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    refresh_fixture(base)
    supporting = base / "raw-amendment.jsonl"
    supporting.write_text('{"reason": "Preserve the original fixture label"}\n')
    atomic(base / "supporting-evidence.json", {"files": {supporting.name: digest(supporting)}})
    original_hashes = {p.name: digest(p) for p in base.iterdir()}
    policy = tmp_path / "policy.json"
    atomic(policy, {"original_packet_sha256": digest(base / "packet-integrity.json")})
    resolution = tmp_path / "semantic-admission/resolve"
    prepare(tmp_path, base, resolution, policy)
    assert [
        r["id"] for r in map(json.loads, (resolution / "mapping.jsonl").read_text().splitlines())
    ] == ["S2"]
    assert not (resolution / "curator-a.jsonl").exists()
    # Simulate two actual new assessments of the selected fixture document.
    for judge in "ab":
        (resolution / f"curator-{judge}.jsonl").write_text(json.dumps(rating("S2", judge)) + "\n")
    atomic(
        resolution / "token-verification.json",
        {
            "status": "passed",
            "manifest_sha256": "parent",
            "documents": 1,
            "packet_integrity_sha256": digest(resolution / "packet-integrity.json"),
        },
    )
    finalize(resolution)
    output = tmp_path / "semantic-admission/composed"
    with pytest.raises(ValueError, match="explicit primary evidence"):
        compose(tmp_path, base, resolution, output)
    assert not output.exists()
    for judge in "ab":
        row = rating("S2", judge)
        row["evidence"] = [
            {
                "claim": "Fixture claim",
                "source_url": "https://example.org/source",
                "finding": "Fixture support",
                "limitation": "Synthetic fixture, no real review",
            }
        ]
        (resolution / f"curator-{judge}.jsonl").write_text(json.dumps(row) + "\n")
    refresh_fixture(resolution)
    supporting_original = supporting.read_bytes()
    supporting.write_text("Changed raw evidence\n")
    with pytest.raises(ValueError, match="Supporting review evidence changed"):
        compose(tmp_path, base, resolution, output)
    assert not output.exists()
    supporting.write_bytes(supporting_original)
    compose(tmp_path, base, resolution, output)
    assert {p.name: digest(p) for p in base.iterdir()} == original_hashes
    with pytest.raises(FileNotFoundError):
        finalize(output)
    atomic(
        output / "token-verification.json",
        {
            "status": "passed",
            "manifest_sha256": "parent",
            "documents": 3,
            "packet_integrity_sha256": digest(output / "packet-integrity.json"),
        },
    )
    finalize(output)
    dispositions = load_admission(output, "parent")
    assert dispositions["doc1"]["status"] == "admitted"
    assert dispositions["doc2"]["status"] == "admitted"
    assert dispositions["doc3"]["status"] == "unresolved"
    assert (output / "original-curator-a.jsonl").read_bytes() == (
        base / "curator-a.jsonl"
    ).read_bytes()
    assert (output / "original-raw-amendment.jsonl").read_bytes() == supporting_original
    assert "original-supporting-evidence.json" in json.loads(
        (output / "packet-integrity.json").read_text()
    )
    # Even an apparently valid proof cannot authorize changing the reviewed text.
    (resolution / "blind.jsonl").write_text(
        json.dumps({"id": "S2", "text": "Different fixture"}) + "\n"
    )
    refresh_fixture(resolution)
    with pytest.raises(ValueError, match="entire unchanged eligible packet"):
        compose(tmp_path, base, resolution, tmp_path / "semantic-admission/tampered")
