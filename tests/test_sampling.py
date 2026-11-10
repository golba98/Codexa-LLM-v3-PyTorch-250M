import io
from pathlib import Path

import pytest

from codexa.data.production import digest
from codexa.data.sampling import sample
from codexa.training.checkpoint import check_volume


def test_bounded_snapshot(tmp_path, monkeypatch):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    source = tmp_path / "source.parquet"
    pq.write_table(
        pa.table(
            {
                "text": ["ordinary educational prose " * 4] * 10,
                "url": ["https://example.org/article"] * 10,
            }
        ),
        source,
    )
    payload = source.read_bytes()

    def fetch(*args, **kwargs):
        response = io.BytesIO(payload)
        response.headers = {"Content-Length": str(len(payload))}
        return response

    monkeypatch.setattr("urllib.request.urlopen", fetch)
    config = {
        "sources": [
            {
                "name": "fixture",
                "url": "https://example.org/pinned",
                "sha256": digest(source),
                "snapshot": "fixture-v1",
                "license": "CC0-1.0",
                "attribution": "original fixture",
                "text_bytes": 500,
            }
        ]
    }
    for name in ("first", "second"):
        sample(config, str(tmp_path / name), len(payload))
    assert (tmp_path / "first/fixture.jsonl").read_bytes() == (
        tmp_path / "second/fixture.jsonl"
    ).read_bytes()
    with pytest.raises(ValueError, match="budget"):
        sample(config, str(tmp_path / "bounded"), 10)
    config["sources"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="checksum"):
        sample(config, str(tmp_path / "corrupt"), len(payload))


def test_checkpoint_volume_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setattr("subprocess.check_output", lambda *a, **kw: "wrong-uuid\n")
    with pytest.raises(OSError, match="volume"):
        check_volume(Path(tmp_path), "expected-uuid")
    check_volume(Path(tmp_path), None)


def test_stratified_fit_reads_training_only(tmp_path, monkeypatch):
    import json
    import sqlite3

    import yaml

    from codexa.data.commands import main
    from codexa.data.packing import read_documents
    from codexa.tokenizer.bpe import CodexaTokenizer

    dataset = tmp_path / "curated"
    dataset.mkdir()
    dbpath = dataset / "documents.sqlite"
    db = sqlite3.connect(dbpath)
    db.execute("CREATE TABLE docs(id TEXT,source TEXT,split TEXT,text TEXT)")
    training = list(read_documents("examples/corpus"))
    for i, text in enumerate(training):
        db.execute("INSERT INTO docs VALUES(?,?,?,?)", (str(i), "english", "train", text))
    db.execute("INSERT INTO docs VALUES('val','english','val','PRIVATE EVALUATION TARGET')")
    db.execute("INSERT INTO docs VALUES('math','excluded','train','EXCLUDED SOURCE')")
    db.commit()
    db.close()
    (dataset / "manifest.json").write_text(json.dumps({"database_sha256": digest(dbpath)}))
    fit = tmp_path / "fit.yaml"
    fit.write_text(yaml.safe_dump({"source_bytes": {"english": 1048576}}))
    observed = []
    original = CodexaTokenizer.train

    def capture(texts, size):
        observed.extend(texts)
        return original(observed, size)

    monkeypatch.setattr(CodexaTokenizer, "train", capture)
    main(
        [
            "tokenizer",
            "train",
            "--dataset",
            str(dataset),
            "--config",
            str(fit),
            "--vocab-size",
            "512",
            "--output",
            str(tmp_path / "candidate.json"),
        ]
    )
    assert sorted(observed) == sorted(training)
