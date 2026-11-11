"""Cached source bytes and cold duplicate replay preserve the original stream."""

import gzip
import io
import json
import re

import pytest

from codexa.data import provider_cache
from codexa.data.provider_cache import CachedProvider
from codexa.data.stream_admission import OnlineDuplicates
from codexa.data.streaming import CompleteResponse


@pytest.fixture
def provider(monkeypatch):
    rows = [{"text": " ".join(f"fixture{n}word{i}" for i in range(180))} for n in range(30)]
    payload = gzip.compress(b"".join(json.dumps(r).encode() + b"\n" for r in rows))
    shard = {
        "url": "https://huggingface.co/datasets/fixture/source/resolve/" + "a" * 40 + "/source.gz",
        "etag": "b" * 64,
    }
    calls = []

    class Response(io.BytesIO):
        status = 206

        def __init__(self, start, end):
            super().__init__(payload[start : end + 1])
            self.headers = {
                "Content-Length": str(end - start + 1),
                "Content-Range": f"bytes {start}-{end}/{len(payload)}",
                "ETag": shard["etag"],
            }

    def ranges(request, timeout):
        start, end = map(
            int, re.fullmatch(r"bytes=(\d+)-(\d+)", request.get_header("Range")).groups()
        )
        calls.append((start, end))
        return Response(start, end)

    monkeypatch.setattr(
        provider_cache,
        "open_provider",
        lambda _: CompleteResponse(io.BytesIO(payload), len(payload)),
    )
    monkeypatch.setattr(provider_cache, "urlopen", ranges)
    return shard, payload, rows, calls


def test_exact_bytes_cross_blocks_and_reopen_offline(tmp_path, provider, monkeypatch):
    shard, payload, _, calls = provider
    cache = CachedProvider(tmp_path, block_bytes=256, maximum_bytes=1024**2, reserve_bytes=0)
    stream = cache(shard)
    result = stream.read(1) + stream.read(277) + stream.read(0) + stream.read()
    assert result == payload and stream.read(3) == b""
    stream.close()
    with pytest.raises(ValueError, match="closed"):
        stream.read(1)
    count = len(calls)
    monkeypatch.setattr(
        provider_cache, "urlopen", lambda *a, **kw: (_ for _ in ()).throw(OSError("offline"))
    )
    reopened = CachedProvider(tmp_path, block_bytes=256, maximum_bytes=1024**2, reserve_bytes=0)
    assert reopened(shard).read() == payload
    assert len(calls) == count


def test_repeated_cold_duplicate_lookups_fetch_each_block_once(tmp_path, provider):
    shard, _, rows, calls = provider
    cache = CachedProvider(tmp_path, block_bytes=256, maximum_bytes=1024**2, reserve_bytes=0)
    original = OnlineDuplicates([shard], opener=cache)
    original.begin()
    assert original.add(rows[17]["text"], 0, 18) is None
    original.commit()
    saved = original.state_dict()
    for _ in range(8):
        ledger = OnlineDuplicates([shard], opener=cache)
        ledger.load_state_dict(saved)
        assert ledger.text(1) == rows[17]["text"]
        assert ledger.state_dict() == saved
    assert len(calls) == len(set(calls))
    assert cache.cache_hits > 0


def test_corrupt_cache_is_refetched_from_pinned_source(tmp_path, provider):
    shard, payload, _, calls = provider
    cache = CachedProvider(tmp_path, block_bytes=256, maximum_bytes=1024**2, reserve_bytes=0)
    assert cache(shard).read(30) == payload[:30]
    path = next(tmp_path.glob("*.block"))
    path.write_bytes(b"corrupt cache")
    cache.memory.clear()
    assert cache(shard).read(30) == payload[:30]
    assert len(calls) == 2


@pytest.mark.parametrize(
    "field,value",
    [("ETag", "changed"), ("Content-Range", "bytes 1-2/3"), ("Content-Length", "999")],
)
def test_bad_range_identity_rejected(tmp_path, provider, monkeypatch, field, value):
    shard, _, _, _ = provider
    original = provider_cache.urlopen

    def wrong(*args, **kwargs):
        response = original(*args, **kwargs)
        response.headers[field] = value
        return response

    monkeypatch.setattr(provider_cache, "urlopen", wrong)
    cache = CachedProvider(tmp_path, block_bytes=256, reserve_bytes=0)
    with pytest.raises(ValueError, match="identity or length"):
        cache(shard).read(30)
    assert not list(tmp_path.glob("*.block"))


def test_eviction_is_bounded_and_protects_unrelated_files(tmp_path, provider):
    shard, payload, _, _ = provider
    cache = CachedProvider(tmp_path, block_bytes=256, maximum_bytes=800, reserve_bytes=0)
    stream = cache(shard)
    unrelated = tmp_path / "original.pt"
    unrelated.write_bytes(b"protected checkpoint")
    assert stream.read() == payload
    assert sum(p.stat().st_size for p in tmp_path.glob("*.block")) <= 800
    assert unrelated.read_bytes() == b"protected checkpoint"


def test_low_space_preserves_bytes_without_publishing_blocks(tmp_path, provider, monkeypatch):
    shard, payload, _, _ = provider
    cache = CachedProvider(tmp_path, block_bytes=256, reserve_bytes=1024**4)
    assert cache(shard).read() == payload
    assert not list(tmp_path.glob("*.block"))


def test_cache_rejects_existing_unowned_directory(tmp_path, provider):
    shard = provider[0]
    (tmp_path / "checkpoint.pt").write_bytes(b"protected")
    with pytest.raises(ValueError, match="empty directory"):
        CachedProvider(tmp_path)(shard)


def test_cached_stream_preserves_next_tokens_after_native_restore(tmp_path, provider, prepared):
    from codexa.data.streaming import StreamingData
    from codexa.tokenizer.bpe import CodexaTokenizer

    shard, payload, _, _ = provider
    tokenizer = CodexaTokenizer.load(prepared[1])
    common = ([shard], tokenizer, 16, 42, "train", lambda row: True, lambda text: False)
    original = StreamingData(
        *common, policy_identity="fixture", opener=lambda _: io.BytesIO(payload)
    )
    original.next(9)
    state = original.state_dict()
    expected = original.next(45)
    cache = CachedProvider(
        tmp_path / "cache", block_bytes=256, maximum_bytes=1024**2, reserve_bytes=0
    )
    restored = StreamingData(*common, policy_identity="fixture", opener=cache)
    restored.load_state_dict(state)
    assert restored.next(45) == expected
    assert restored.state_dict() == original.state_dict()
