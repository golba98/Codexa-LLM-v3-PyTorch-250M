"""Review decisions must fail closed, including partial reviews."""

import pytest

from scripts.datasets.review_continuation_corpus import adjudicate


def test_strict_review_refuses_preinspected_semantic_census(tmp_path, monkeypatch):
    import json

    from codexa.data.v2_storage import Storage
    from scripts.datasets.review_continuation_corpus import prepare

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "policy.json").write_text(
        json.dumps({"semantic_admission_decision_sha256": "a" * 64})
    )
    monkeypatch.setattr(Storage, "load", lambda *a: Storage(tmp_path))
    monkeypatch.setattr(Storage, "check", lambda *a: None)
    output = tmp_path / "review"
    with pytest.raises(ValueError, match="previously inspected"):
        prepare(
            tmp_path, snapshot, tmp_path / "tokenizer", output, 2026111601, strict_kind="primary"
        )
    assert not output.exists()


def test_census_exclusion_requires_intact_complete_packet(tmp_path):
    import json

    from codexa.data.production import digest
    from scripts.datasets.review_continuation_corpus import inspection_exclusions

    (tmp_path / "mapping.jsonl").write_text(
        json.dumps({"id": "T00001", "document_id": "doc-1"}) + "\n"
    )
    (tmp_path / "blind.jsonl").write_text(
        json.dumps({"id": "T00001", "text": "Previously inspected."}) + "\n"
    )
    (tmp_path / "packet-integrity.json").write_text(
        json.dumps({name: digest(tmp_path / name) for name in ("mapping.jsonl", "blind.jsonl")})
    )
    ids, texts, packets = inspection_exclusions([tmp_path])
    assert ids == {"doc-1"}
    assert texts == [("Previously inspected.", 64)]
    assert len(packets) == 1
    (tmp_path / "blind.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="packet changed"):
        inspection_exclusions([tmp_path])


def test_review_window_excludes_any_inspected_document_overlap():
    from scripts.datasets.review_continuation_corpus import overlaps_inspected_window

    intervals = [(5, 7), (15, 20)]
    starts = [5, 15]
    assert not overlaps_inspected_window(0, 5, intervals, starts)
    assert overlaps_inspected_window(0, 6, intervals, starts)
    assert overlaps_inspected_window(6, 1, intervals, starts)
    assert not overlaps_inspected_window(7, 8, intervals, starts)
    assert overlaps_inspected_window(14, 2, intervals, starts)
    assert not overlaps_inspected_window(20, 2, intervals, starts)


def test_strict_review_requires_all_known_curation_packets(tmp_path):
    from scripts.datasets.review_continuation_corpus import require_known_census_exclusions

    packet = tmp_path / "semantic-admission" / "review"
    packet.mkdir(parents=True)
    (packet / "packet-integrity.json").write_text("{}")
    with pytest.raises(ValueError, match="Every known inspected census"):
        require_known_census_exclusions(tmp_path, [])
    require_known_census_exclusions(tmp_path, [packet])


def packet():
    return [{"id": str(i), "kind": "window", "source": "mixture"} for i in range(300)] + [
        {"id": str(300 + i), "kind": "document", "source": "source-a"} for i in range(40)
    ]


def labels(rows):
    return [
        {
            "id": r["id"],
            "usable": True,
            "severe": False,
            "critical": False,
            "reason": "Reviewed readable connected prose.",
        }
        for r in rows
    ]


def test_review_requires_complete_ratings():
    rows = packet()
    assert adjudicate(rows, labels(rows))["training_allowed"]
    partial = adjudicate(rows, labels(rows[:10]))
    assert partial["status"] == "incomplete"
    assert not partial["training_allowed"]


def test_new_protocol_refuses_partial_finalization(tmp_path, monkeypatch):
    import json
    import sys

    from codexa.data.production import digest
    from codexa.data.v2_storage import Storage
    from scripts.datasets.review_continuation_corpus import main

    monkeypatch.setattr(Storage, "load", lambda *a: Storage(tmp_path))
    monkeypatch.setattr(Storage, "check", lambda *a: None)
    (tmp_path / "protocol.json").write_text(json.dumps({"complete_required": True}))
    (tmp_path / "mapping.jsonl").write_text("".join(json.dumps(r) + "\n" for r in packet()))
    (tmp_path / "blind.jsonl").write_text("")
    (tmp_path / "ratings.jsonl").write_text(json.dumps(labels(packet())[0]) + "\n")
    (tmp_path / "packet-integrity.json").write_text(
        json.dumps(
            {
                name: digest(tmp_path / name)
                for name in ("protocol.json", "mapping.jsonl", "blind.jsonl")
            }
        )
    )
    monkeypatch.setattr(
        sys, "argv", ["review", "finalize", "--root", str(tmp_path), "--output", str(tmp_path)]
    )
    with pytest.raises(ValueError, match="Complete every"):
        main()
    assert not (tmp_path / "decision.json").exists()


def test_review_decisive_failure_with_unseen_samples():
    rows = packet()
    ratings = labels(rows[300:307])
    for r in ratings:
        r.update(usable=False, reason="Disconnected extracted fragments.")
    result = adjudicate(rows, ratings)
    assert result["status"] == "failed"
    assert result["unreviewed"] == 333
    assert not result["training_allowed"]


def test_review_rejects_severe_critical_duplicate_and_missing_reasons():
    rows = packet()
    ratings = labels(rows[300:303])
    for r in ratings:
        r["severe"] = True
    assert adjudicate(rows, ratings)["status"] == "failed"
    ratings = labels(rows[:1])
    ratings[0]["critical"] = True
    assert adjudicate(rows, ratings)["status"] == "failed"
    with pytest.raises(ValueError):
        adjudicate(rows, ratings + ratings)
    ratings[0]["reason"] = ""
    with pytest.raises(ValueError):
        adjudicate(rows, ratings)
