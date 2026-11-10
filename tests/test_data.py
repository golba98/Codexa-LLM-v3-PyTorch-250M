import json
import sqlite3
from pathlib import Path

import pytest
import torch

from codexa.config import Config
from codexa.data.production import (
    PermutationSampler,
    ShardedData,
    curate,
    digest,
    documents,
    records,
    tokenize,
    verify,
)
from codexa.training.trainer import Trainer


def fixture(tmp_path):
    sources = []
    for split in ["train", "val", "test"]:
        path = tmp_path / (split + ".jsonl")
        path.write_text(
            "\n".join(
                json.dumps(
                    {
                        "text": f"{split} document {i}. Mathematics preserves x² ≤ π and ordinary prose. "
                        * 8
                    }
                )
                for i in range(4)
            )
            + "\n{bad\n"
        )
        sources.append(
            {
                "name": split,
                "paths": [str(path)],
                "split": split,
                "license": "CC0-1.0",
                "attribution": "original test fixture",
                "snapshot": "v1",
            }
        )
    return {"sources": sources, "near_duplicates": False, "protect_spans": False}


def test_curation_resume_and_integrity(tmp_path):
    config = fixture(tmp_path)
    root = curate(config, str(tmp_path / "curated"))
    m = json.loads((root / "manifest.json").read_text())
    assert m["counts"] == {"train": 4, "val": 4, "test": 4}
    assert m["errors"]["encoding_or_json"] == 3
    assert len(list(documents(root))) == 4
    curate(config, str(root), True)
    assert json.loads((root / "manifest.json").read_text())["counts"] == m["counts"]
    Path(config["sources"][0]["paths"][0]).write_text("changed")
    with pytest.raises(ValueError):
        curate(config, str(root), True)


def test_duplicate_holdout(tmp_path):
    config = fixture(tmp_path)
    held = json.loads(Path(config["sources"][1]["paths"][0]).read_text().splitlines()[0])
    with Path(config["sources"][0]["paths"][0]).open("a") as f:
        f.write(json.dumps(held) + "\n")
    root = curate(config, str(tmp_path / "curated"))
    assert json.loads((root / "manifest.json").read_text())["errors"]["duplicate"] == 1
    db = sqlite3.connect(root / "documents.sqlite")
    assert db.execute("select count(*) from docs").fetchone()[0] == 12
    db.close()


def test_curation_io_tuning_preserves_resume(tmp_path):
    config = fixture(tmp_path)
    full = curate(config, str(tmp_path / "full"))

    def interrupt(_):
        raise RuntimeError("committed interruption")

    partial = tmp_path / "partial"
    with pytest.raises(RuntimeError, match="committed interruption"):
        curate(config, str(partial), progress=interrupt, commit_every=2, cache_mib=16)
    curate(config, str(partial), resume=True, commit_every=1000, cache_mib=256, wal=True)
    for table in ("docs", "errors", "split_groups"):
        with sqlite3.connect(full / "documents.sqlite") as a:
            expected = sorted(a.execute(f"SELECT * FROM {table}").fetchall())
        with sqlite3.connect(partial / "documents.sqlite") as b:
            assert sorted(b.execute(f"SELECT * FROM {table}").fetchall()) == expected
    with sqlite3.connect(partial / "documents.sqlite") as db:
        plan = db.execute(
            "EXPLAIN QUERY PLAN DELETE FROM bands WHERE id=?", ("fixture",)
        ).fetchall()
        assert any("SEARCH bands" in row[-1] for row in plan)
    for key in ("identity", "config", "counts", "errors"):
        assert (
            json.loads((full / "manifest.json").read_text())[key]
            == json.loads((partial / "manifest.json").read_text())[key]
        )
    with pytest.raises(ValueError, match="Positive curation"):
        curate(config, str(tmp_path / "invalid"), commit_every=0)


def test_completed_curation_does_not_add_indexes(tmp_path):
    config = fixture(tmp_path)
    root = curate(config, str(tmp_path / "completed"))
    with sqlite3.connect(root / "documents.sqlite") as db:
        db.execute("DROP INDEX bands_id")
    manifest = json.loads((root / "manifest.json").read_text())
    checksum = digest(root / "documents.sqlite")
    manifest["database_sha256"] = checksum
    (root / "manifest.json").write_text(json.dumps(manifest))
    manifest_checksum = digest(root / "manifest.json")
    curate(config, str(root), resume=True, wal=True, cache_mib=256)
    assert digest(root / "documents.sqlite") == checksum
    assert digest(root / "manifest.json") == manifest_checksum


@pytest.mark.parametrize("size", [1, 2, 3, 17, 128, 1001])
def test_permutation(size):
    sampler = PermutationSampler(size, 42)
    assert sorted(sampler.next(size)) == list(range(size))
    state = sampler.state_dict()
    expected = sampler.next(20)
    resumed = PermutationSampler(size, 42)
    resumed.load_state_dict(state)
    assert resumed.next(20) == expected


def test_shards_and_resume(prepared, tmp_path):
    _, tok, _ = prepared
    dataset = curate(fixture(tmp_path), str(tmp_path / "curated"))
    output = tokenize(dataset, tok, str(tmp_path / "tokens"), shard_tokens=150)
    reader = ShardedData(output, "train", 64)
    x, y = reader.batch([0, 1], torch.device("cpu"))
    assert torch.equal(x[:, 1:], y[:, :-1])
    assert x[0, 0] == 2
    original = digest(output / "manifest.json")
    tokenize(dataset, tok, str(output), True, shard_tokens=150)
    assert digest(output / "manifest.json") == original
    c = Config()
    a = Trainer(c, str(output), tok, str(tmp_path / "full"), "cpu")
    a.train(until=6, ui="none", handle_signals=False)
    b = Trainer(c, str(output), tok, str(tmp_path / "partial"), "cpu")
    b.train(until=3, ui="none", handle_signals=False)
    r = Trainer(c, str(output), tok, str(tmp_path / "resume"), "cpu")
    r.resume(str(tmp_path / "partial/latest.pt"))
    r.train(until=6, ui="none", handle_signals=False)
    for k, v in a.model.state_dict().items():
        assert torch.equal(v, r.model.state_dict()[k])
    p = output / json.loads((output / "manifest.json").read_text())["splits"]["train"][0]["file"]
    p.write_bytes(b"bad")
    with pytest.raises(ValueError):
        verify(output)


def test_formats(tmp_path):
    p = tmp_path / "text.txt"
    p.write_bytes("Hello π\n\nAnother paragraph\n".encode())
    assert len(list(records(p, "text"))) == 2
    pq = pytest.importorskip("pyarrow.parquet")
    pa = pytest.importorskip("pyarrow")
    p = tmp_path / "data.parquet"
    pq.write_table(pa.table({"text": ["one", "two"]}), p)
    assert [r["text"] for _, r in records(p, "parquet")] == ["one", "two"]


@pytest.mark.parametrize(
    "io_options",
    [
        {},
        {"commit_every": 1000, "cache_mib": 256, "wal": True, "protected_cache_entries": 2000},
        {"protected_cache_entries": 1},
    ],
)
def test_near_and_embedded_holdouts(tmp_path, io_options):
    config = fixture(tmp_path)
    text = " ".join("word" + str(i) for i in range(180))
    val = Path(config["sources"][1]["paths"][0])
    train = Path(config["sources"][0]["paths"][0])
    with val.open("a") as f:
        f.write(json.dumps({"text": text}) + "\n")
    with train.open("a") as f:
        f.write(json.dumps({"text": text.replace("word175", "different")}) + "\n")
    config["near_duplicates"] = True
    root = curate(config, str(tmp_path / "near"), **io_options)
    assert json.loads((root / "manifest.json").read_text())["errors"]["duplicate"] >= 1
    # Short embedded protected span in a much longer training document.
    config["near_duplicates"] = False
    config["protect_spans"] = True
    phrase = " ".join("protected" + str(i) for i in range(20))
    with val.open("a") as f:
        f.write(json.dumps({"text": phrase}) + "\n")
    with train.open("a") as f:
        f.write(json.dumps({"text": text + " " + phrase}) + "\n")
    # Shared fixture boilerplate is intentionally contaminated too: strict gate may empty train.
    protected = curate(config, str(tmp_path / "protected"), **io_options)
    assert json.loads((protected / "manifest.json").read_text())["errors"]["holdout_overlap"] >= 1
    assert not any(phrase in t for t in documents(protected))


def test_hf_staging(tmp_path):
    from codexa.data.production import stage_huggingface

    p = stage_huggingface(
        ({"text": "row " + str(i)} for i in range(100)),
        tmp_path / "hf.jsonl",
        {"name": "original", "snapshot": "1", "license": "CC0", "attribution": "test"},
        max_bytes=100,
    )
    assert p.stat().st_size <= 100
    assert list(records(p, "jsonl"))


def test_archive_and_ignore():
    import hashlib
    import tarfile

    from codexa.repository import check_repository

    with tarfile.open("docs/archive/history.tar.gz", "r:gz") as tar:
        for item in json.loads(Path("docs/archive/index.json").read_text()):
            assert (
                hashlib.sha256(tar.extractfile(item["path"]).read()).hexdigest() == item["sha256"]
            )
    check_repository()


def test_interrupted_preparation(prepared, tmp_path, monkeypatch):
    from codexa.data import production

    _, tok, _ = prepared
    cfg = fixture(tmp_path)

    # Interrupt after an actual committed transaction, then resume from byte cursors.
    def stop(_):
        raise RuntimeError("interruption")

    with pytest.raises(RuntimeError):
        curate(cfg, str(tmp_path / "interrupted"), progress=stop)
    root = curate(cfg, str(tmp_path / "interrupted"), resume=True)
    completed = digest(root / "manifest.json")
    curate(cfg, str(root), resume=True)
    assert digest(root / "manifest.json") == completed
    atomic = production.atomic

    def failed_write(path, value):
        atomic(path, value)
        if path.name == "progress.json":
            raise RuntimeError("interruption")

    monkeypatch.setattr(production, "atomic", failed_write)
    with pytest.raises(RuntimeError):
        tokenize(root, tok, str(tmp_path / "tokenized"), shard_tokens=150)
    monkeypatch.setattr(production, "atomic", atomic)
    tokenize(root, tok, str(tmp_path / "tokenized"), resume=True, shard_tokens=150)
    verify(tmp_path / "tokenized")


def test_streaming_memory_bound(tmp_path):
    import subprocess
    import sys

    if not Path("/proc/self/status").exists():
        pytest.skip("Linux RSS measurement")
    values = []
    for count in [1000, 100000]:
        out = tmp_path / str(count)
        subprocess.run(
            [
                sys.executable,
                "scripts/check_data_memory.py",
                "--documents",
                str(count),
                "--output",
                str(out),
            ],
            check=True,
            capture_output=True,
            timeout=60,
        )
        values.append(json.loads((out / "measurement.json").read_text())["peak_rss_kib"])
    assert values[1] - values[0] < 16384


def test_invalid_unicode_records(tmp_path):
    config = fixture(tmp_path)
    path = Path(config["sources"][0]["paths"][0])
    with path.open("a") as f:
        f.write(json.dumps({"text": "\ud800"}) + "\n")
        f.write(json.dumps({"text": ""}) + "\n")
    root = curate(config, str(tmp_path / "unicode"))
    errors = json.loads((root / "manifest.json").read_text())["errors"]
    assert errors["encoding"] == 1 and errors["empty"] == 1


def test_parent_group_isolation(tmp_path):
    cfg = fixture(tmp_path)
    for source in cfg["sources"][:2]:
        with Path(source["paths"][0]).open("a") as f:
            f.write(
                json.dumps(
                    {
                        "text": source["split"]
                        + " completely separate prose about arithmetic and natural language.",
                        "group_id": "same-parent",
                    }
                )
                + "\n"
            )
    root = curate(cfg, str(tmp_path / "groups"))
    assert json.loads((root / "manifest.json").read_text())["errors"]["group_overlap"] == 1


def test_checksums_do_not_replace_boundary_validation(prepared, tmp_path):
    import numpy as np

    _, tok, _ = prepared
    dataset = curate(fixture(tmp_path), str(tmp_path / "curated"))
    output = tokenize(dataset, tok, str(tmp_path / "tokens"))
    path = output / "manifest.json"
    manifest = json.loads(path.read_text())
    entry = manifest["splits"]["train"][0]
    index = output / entry["index"]
    offsets = np.fromfile(index, dtype="<u8")
    offsets[0] = 1
    offsets.tofile(index)
    entry["index_sha256"] = digest(index)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="boundaries"):
        verify(output)
