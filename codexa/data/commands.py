"""Production data/tokenizer commands, dispatched without importing PyTorch."""

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

from .production import atomic, curate, digest, documents, tokenize, verify


def main(argv):
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import yaml

    from codexa.tokenizer.bpe import CodexaTokenizer

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("domain", choices=["data", "tokenizer"])
    p.add_argument("action")
    p.add_argument("--config")
    p.add_argument("--dataset")
    p.add_argument("--data")
    p.add_argument("--tokenizer")
    p.add_argument("--output")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--vocab-size", type=int, default=16384)
    p.add_argument("--max-bytes", type=int, default=33554432)
    a = p.parse_args(argv)
    if a.domain == "data" and a.action == "sample" and a.config:
        from .sampling import sample as sample_snapshots

        print(sample_snapshots(yaml.safe_load(Path(a.config).read_text()), a.output, a.max_bytes))
    elif a.domain == "data" and a.action == "sample":
        import re
        import urllib.request

        import pyarrow.parquet as pq

        root = Path(a.output)
        root.mkdir(parents=True, exist_ok=False)
        revision = "b08601e04326c79dfdd32d625aee71d232d685c3"
        total = 0
        sources = []
        budget = min(a.max_bytes, 100 * 1048576)
        if budget <= 0:
            raise ValueError("Positive sample budget required")
        for split in ("train", "validation", "test"):
            url = f"https://huggingface.co/datasets/Salesforce/wikitext/resolve/{revision}/wikitext-2-raw-v1/{split}-00000-of-00001.parquet"
            path = root / (split + ".parquet")
            with urllib.request.urlopen(url, timeout=30) as response, path.open("xb") as f:
                while chunk := response.read(min(1048576, budget - total + 1)):
                    total += len(chunk)
                    if total > budget:
                        raise ValueError(
                            "Sample download budget exceeded; partial outputs retained"
                        )
                    f.write(chunk)
            output = root / (split + ".jsonl")
            article: list[str] = []
            title = ""
            with output.open("w") as text_stream:
                for batch in pq.ParquetFile(path).iter_batches(batch_size=32):
                    for text in batch.column("text").to_pylist():
                        if re.match(r"^ = [^=].* = \n$", text):
                            if article:
                                text_stream.write(
                                    json.dumps({"text": "".join(article), "group_id": title}) + "\n"
                                )
                            article = []
                            title = text.strip()
                        article.append(text)
                if article:
                    text_stream.write(
                        json.dumps({"text": "".join(article), "group_id": title}) + "\n"
                    )
            sources.append(
                {
                    "name": "wikitext2-" + split,
                    "paths": [str(output)],
                    "format": "jsonl",
                    "split": "val" if split == "validation" else split,
                    "snapshot": revision,
                    "license": "CC-BY-SA-3.0/GFDL metadata; CC-BY-SA-4.0 card prose; production review pending",
                    "attribution": "WikiText authors and Wikipedia contributors",
                }
            )
        (root / "config.yaml").write_text(
            yaml.safe_dump({"sources": sources, "near_duplicates": True, "protect_spans": True})
        )
        atomic(
            root / "source.json",
            {
                "dataset": "Salesforce/wikitext",
                "revision": revision,
                "downloaded_bytes": total,
                "budget": budget,
                "files": {p.name: digest(p) for p in root.glob("*.parquet")},
            },
        )
        print(root / "config.yaml")
    elif a.domain == "data" and a.action == "curate":
        from codexa.training.callbacks.telemetry import EventBus
        from codexa.training.terminal import choose_terminal

        view, _ = choose_terminal()
        meta = {"phase": "Dataset preparation", "dataset": Path(a.output).name}
        bus = EventBus(Path(a.output + ".telemetry"), meta, (view,) if view else ())
        bus.emit("started", 0, 0, metadata=meta)
        last_count = [0]

        def progress(count):
            last_count[0] = count
            bus.emit("step_completed", count, 0, prepared_documents=count)

        try:
            result = curate(
                yaml.safe_load(Path(a.config).read_text()),
                a.output,
                a.resume,
                progress,
            )
            bus.emit("completed", last_count[0], 0)
            if view is None:
                print(result)
        finally:
            bus.close()
            if view:
                view.close()
    elif a.domain == "data" and a.action == "prepare":
        print(tokenize(a.dataset, a.tokenizer, a.output, a.resume))
    elif a.domain == "data" and a.action == "verify":
        print(json.dumps(verify(a.data), indent=2))
    elif a.domain == "tokenizer" and a.action == "train":
        import hashlib
        import importlib.metadata

        sample_hash = hashlib.sha256()
        sample_counts = [0, 0]
        source_counts: dict[str, int] = {}
        fitting = yaml.safe_load(Path(a.config).read_text()) if a.config else None
        if a.max_bytes <= 0:
            raise ValueError("Positive fitting budget required")

        def sample():
            used = 0
            rows: Iterable[tuple[str, str]]
            if fitting:
                import sqlite3

                manifest = json.loads((Path(a.dataset) / "manifest.json").read_text())
                if digest(Path(a.dataset) / "documents.sqlite") != manifest["database_sha256"]:
                    raise ValueError("Curated database changed")
                db = sqlite3.connect(
                    f"file:{Path(a.dataset) / 'documents.sqlite'}?mode=ro", uri=True
                )
                rows = db.execute("SELECT source,text FROM docs WHERE split='train' ORDER BY id")
            else:
                db = None
                rows = (("all", text) for text in documents(a.dataset))
            per_source: dict[str, int] = {}
            for source, text in rows:
                size = len(text.encode())
                if used + size > a.max_bytes:
                    if fitting:
                        continue
                    break
                if fitting and per_source.get(source, 0) + size > fitting["source_bytes"].get(
                    source, 0
                ):
                    continue
                per_source[source] = per_source.get(source, 0) + size
                source_counts[source] = source_counts.get(source, 0) + 1
                used += size
                sample_counts[0] += 1
                sample_counts[1] += size
                sample_hash.update(size.to_bytes(8, "little") + text.encode())
                yield text
            if db is not None:
                db.close()

        tok = CodexaTokenizer.train(sample(), a.vocab_size)
        tok.save(a.output)
        atomic(
            Path(a.output + ".provenance.json"),
            {
                "dataset": digest(Path(a.dataset) / "manifest.json"),
                "max_bytes": a.max_bytes,
                "vocabulary": a.vocab_size,
                "stage": "candidate",
                "sample_documents": sample_counts[0],
                "sample_bytes": sample_counts[1],
                "source_documents": source_counts,
                "sample_sha256": sample_hash.hexdigest(),
                "tokenizers_version": importlib.metadata.version("tokenizers"),
                "peak_process_rss_kib": int(
                    next(
                        line.split()[1]
                        for line in Path("/proc/self/status").read_text().splitlines()
                        if line.startswith("VmHWM:")
                    )
                )
                if Path("/proc/self/status").exists()
                else None,
            },
        )
    elif a.domain == "tokenizer" and a.action == "evaluate":
        tok = CodexaTokenizer.load(a.tokenizer)
        chars = tokens = count = 0
        for text in documents(a.dataset, "val"):
            ids = tok.encode(text)
            if tok.decode(ids) != text:
                raise ValueError("Tokenizer round trip failed")
            chars += len(text)
            tokens += len(ids)
            count += 1
        expressions = ["12 + 34 = 46", "0.125 × 8 = 1", "x² + y² = z²", "∫ x dx = x²/2 + C"]
        print(
            json.dumps(
                {
                    "documents": count,
                    "tokens_per_character": tokens / chars,
                    "tokens_per_document": tokens / count,
                    "math_fragments": {s: len(tok.encode(s)) for s in expressions},
                }
            )
        )
    elif a.domain == "tokenizer" and a.action == "freeze":
        import shutil
        import tempfile

        root = Path(a.output)
        if root.exists():
            raise FileExistsError(root)
        tok = CodexaTokenizer.load(a.tokenizer)
        if tok.backend.get_vocab_size() != a.vocab_size:
            raise ValueError("Vocabulary mismatch")
        provenance = Path(a.tokenizer + ".provenance.json")
        if not provenance.exists():
            raise ValueError("Tokenizer provenance required")
        root.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root.parent, prefix=".freeze-") as temporary:
            tmp = Path(temporary) / "artifact"
            tmp.mkdir()
            shutil.copyfile(a.tokenizer, tmp / "tokenizer.json")
            atomic(
                tmp / "identity.json",
                {
                    "version": 1,
                    "stage": "candidate",
                    "sha256": digest(tmp / "tokenizer.json"),
                    "provenance": json.loads(provenance.read_text()),
                },
            )
            tmp.rename(root)
    else:
        p.error("Unsupported production command")


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
