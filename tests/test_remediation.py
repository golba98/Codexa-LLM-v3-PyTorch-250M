"""Regression and preservation checks for rejection-only corpus remediation."""

import hashlib
import json
import sqlite3
import zlib

import numpy as np
import pytest

from codexa.data.production import atomic, digest
from codexa.data.remediate import prove_narrowing, rebuild, rejection
from codexa.data.v2_storage import Storage
from scripts.verify_remediated_snapshot import check


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        (
            r"Solve. \textcolor{w h i t e}{XXXX} x=1. \textcolor{white}{000} Done.",
            "decorative_tex_padding",
        ),
        (
            "### First puzzle\nA question.\n### Another puzzle\nAnother question.\n# Different solution\nAn answer.",
            "merged_exercise_teasers",
        ),
        (
            "Answer A. Try our free classes today! Explanation follows.",
            "promotional_solution_contamination",
        ),
        ("The graph shows the scores. What is the largest score?", "missing_essential_figure"),
        (
            (
                "The graph shows a shaded triangle. Find the shaded area from the diagram. "
                "In another worked example, the slope of $y = 2x + 1$ is 2."
            ),
            "missing_essential_figure",
        ),
        ("The complete list is as follows:", "incomplete_explanation"),
        ("The area of , and the temperature of .", "missing_measurements"),
    ],
)
def test_damage_families(text, reason):
    assert rejection(text) == reason


@pytest.mark.parametrize(
    "text",
    [
        r"Subtract x on both sides. $\cancel{x}+2=\cancel{x}+3$. This is impossible.",
        r"A highlighted answer is $\textcolor{red}{x=2}$. Substitute to check.",
        "# Areas\n## Example 1\nA square has side 2 and area 4.\n## Example 2\nA square has side 3 and area 9.",
        "The triangle has sides 3, 4 and 5. Its area is 6. I will insert a graph later.",
        "The graph shows the line y=2x+1. Calculate its slope. The slope is 2.",
        "As shown in Figure below, the bone grows from its ends. The text explains this process.",
    ],
)
def test_valid_mathematics_survives(text):
    assert rejection(text) is None


def test_raw_math_extension_rejects_damage_without_reconstructing_it():
    assert (
        rejection("q = color(white)(a/a) mcT color(white)(l)", "openwebmath")
        == "unrendered_asciimath_padding"
    )
    missing = "The field is given by:\n\nWhere B is the field.\n" * 3
    assert rejection(missing, "openwebmath") == "repeated_missing_equations"
    assert (
        rejection("Given in the above molecule: bonds which are shown below.", "finemath-4plus")
        == "missing_molecular_structure"
    )
    valid = r"The field is given by:" + "\n\n" + r"$B=\mu I/(2r)$" + "\n\nWhere I is the current.\n"
    assert rejection(valid * 3, "openwebmath") is None
    assert (
        rejection(
            r"A highlight $\textcolor{white}{x}$ is followed by a complete explanation.",
            "finemath-4plus",
        )
        is None
    )


def test_cached_acceptances_require_a_provably_narrower_predicate():
    old = 'def rejection(text):\n if figure(text):\n  return "missing_essential_figure"\n return None\n'
    narrow = old.replace("if figure(text):", "if figure(text) and essential(text):")
    prove_narrowing(old, narrow)
    with pytest.raises(ValueError):
        prove_narrowing(old, old.replace("if figure(text):", "if figure(text) or essential(text):"))
    with pytest.raises(ValueError):
        prove_narrowing(old, narrow.replace("return None", 'return "another_rejection"'))


def test_rebuild_preserves_splits_bytes_history_and_quarantine(tmp_path, monkeypatch):
    monkeypatch.setattr(Storage, "check", lambda *a, **k: None)
    storage = Storage(tmp_path)
    storage.initialize()
    parent = tmp_path / "snapshots/parent"
    parent.mkdir(parents=True)
    db = sqlite3.connect(tmp_path / "indexes/documents.sqlite")
    db.execute(
        "CREATE TABLE docs(id INTEGER PRIMARY KEY,hash BLOB,payload BLOB,metadata TEXT,role TEXT)"
    )
    splits, listing = {}, []
    for split in ("train", "val", "test"):
        texts = ["A clear explanation gives a useful result."]
        if split == "train":
            texts.append("The graph shows missing data. What is its value?")
        entries = []
        for i, text in enumerate(texts):
            text = split + " " + text
            ident = hashlib.sha256(text.encode()).hexdigest()
            data = parent / f"{split}-{i}.bin"
            index = data.with_suffix(".idx")
            data.write_bytes(np.array([2, 8, 9, 3], dtype="<u2").tobytes())
            index.write_bytes(np.array([0], dtype="<u8").tobytes())
            entries.append(
                {
                    "file": data.name,
                    "index": index.name,
                    "tokens": 4,
                    "documents": 1,
                    "sha256": digest(data),
                    "index_sha256": digest(index),
                }
            )
            listing.append({"id": ident, "source": "math", "split": split, "tokens": 4})
            db.execute(
                "INSERT INTO docs(hash,payload,metadata,role) VALUES(?,?,?,?)",
                (
                    bytes.fromhex(ident),
                    zlib.compress(text.encode()),
                    json.dumps({"ordinal": i}),
                    "new",
                ),
            )
        splits[split] = entries
    db.commit()
    db.close()
    (parent / "documents.jsonl").write_text("".join(json.dumps(r) + "\n" for r in listing))
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
    old_hash = digest(parent / "manifest.json")
    output = tmp_path / "snapshots/new"
    audit = rebuild(tmp_path, parent, output)
    assert audit["split_tokens"] == {"train": 4, "val": 4, "test": 4}
    assert len((output / "quarantine.jsonl").read_text().splitlines()) == 1
    assert digest(parent / "manifest.json") == old_hash
    assert (output / "val-0.bin").stat().st_ino != (parent / "val-0.bin").stat().st_ino
    assert (output / "train-0.bin").read_bytes() == (parent / "train-0.bin").read_bytes()
    with pytest.raises(FileExistsError):
        rebuild(tmp_path, parent, output)
    assert check(tmp_path, parent, output)["retained_documents_bitwise_compared"] == 3
    policy = output / "policy.json"
    original_policy = policy.read_bytes()
    policy.write_bytes(original_policy + b" ")
    with pytest.raises(ValueError, match="policy differs from manifest"):
        check(tmp_path, parent, output)
    policy.write_bytes(original_policy)
    quarantine = output / "quarantine.jsonl"
    original_quarantine = quarantine.read_bytes()
    changed = json.loads(original_quarantine)
    changed["split"] = "test"
    quarantine.write_text(json.dumps(changed) + "\n")
    manifest = json.loads((output / "manifest.json").read_text())
    original_manifest = manifest.copy()
    original_manifest["identity"] = manifest["identity"].copy()
    manifest["identity"]["quarantine_sha256"] = digest(quarantine)
    atomic(output / "manifest.json", manifest)
    with pytest.raises(ValueError, match="Quarantine identity differs"):
        check(tmp_path, parent, output)
    quarantine.write_bytes(original_quarantine)
    atomic(output / "manifest.json", original_manifest)
    # Simulate a rehashed but rewritten retained shard.
    altered = output / "altered.partial"
    altered.write_bytes(np.array([2, 10, 9, 3], dtype="<u2").tobytes())
    altered.replace(output / "train-0.bin")
    manifest = json.loads((output / "manifest.json").read_text())
    manifest["splits"]["train"][0]["sha256"] = digest(output / "train-0.bin")
    atomic(output / "manifest.json", manifest)
    with pytest.raises(ValueError, match="token bytes changed"):
        check(tmp_path, parent, output)
