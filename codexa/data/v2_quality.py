"""Conservative, versioned prose filters and compact deterministic similarity features."""

import hashlib
import html
import re
from collections import Counter
from functools import lru_cache
from typing import Any

import numpy as np

VERSION = "codexa-prose-v2.4"
WORD = re.compile(r"\w+|[^\w\s]", re.UNICODE)
CONTACT = re.compile(
    r"(?i)(?:\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b|"
    r"\b(?:phone|tel|fax|mobile|ssn|social security|password|api[_ -]?key)\s*[:=])"
)
BOILER = re.compile(
    r"(?i)^(?:accept (?:all )?cookies|cookie (?:policy|settings)|privacy policy|"
    r"all rights reserved|you must be logged in|sign in to|log in to|"
    r"subscribe (?:to|for)|share (?:this\s*$|on\s+(?:facebook|twitter|linkedin|pinterest)\s*$)|skip to (?:main|content)|"
    r"previous (?:article|post)|next (?:article|post)|advertisement\s*$)"
)


def source_text(row: dict[str, Any], name: str) -> Any:
    """Remove Wikipedia appendices, retaining attribution separately in metadata."""
    text = row.get("text")
    if name == "wikipedia-en" and isinstance(text, str):
        text = re.split(
            r"(?im)^\s*(?:References|External links|Further reading|Bibliography|See also)\s*$",
            text,
            maxsplit=1,
        )[0].strip()
    return text


def source_error(row: dict[str, Any], spec: dict[str, Any]) -> str | None:
    if row.get("error"):
        return str(row["error"])
    if spec.get("min_edu_score") and row.get("int_score", -1) < spec["min_edu_score"]:
        return "educational_score"
    if spec["name"] == "fineweb" and row.get("language_score", 0) < 0.9:
        return "language_confidence"
    if spec["name"] == "cosmopedia-stories" and row.get("format") != "story_children":
        return "nonchildren_story_format"
    if spec["name"] == "wikipedia-en":
        text = source_text(row, spec["name"])
        if (
            not isinstance(text, str)
            or len(text.split()) < 100
            or len(re.findall(r"[.!?](?:\s|$)", text)) < 4
        ):
            return "encyclopedia_stub_or_list"
        if re.search(r"may refer to:|^This is a list|^List of", text):
            return "encyclopedia_disambiguation_or_list"
    return None


def canonical(text: str) -> str:
    return " ".join(text.split())


def clean(text: Any, category: str = "general") -> tuple[str, list[str], dict[str, Any]]:
    if not isinstance(text, str):
        return "", ["missing_text"], {}
    if "\x00" in text or "\ufffd" in text or any(0xD800 <= ord(c) <= 0xDFFF for c in text):
        return "", ["corrupt_unicode"], {}
    if any(x in text for x in ("<pad>", "<unk>", "<bos>", "<eos>")):
        return "", ["reserved_token"], {}
    if len(text.encode()) > 1024**2:
        return "", ["oversize"], {}
    text = html.unescape(text).replace("\r\n", "\n").replace("\r", "\n")
    if any(x in text for x in ("<pad>", "<unk>", "<bos>", "<eos>")):
        return "", ["reserved_token"], {}
    if re.search(r"(?i)<(?:script|style|html|body|div|table)\b", text):
        return "", ["html_remnant"], {}
    before = len(text)
    lines = [line.rstrip() for line in text.splitlines()]
    lines = [line for line in lines if not BOILER.search(line.strip()) and not CONTACT.search(line)]
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    words = re.findall(r"[A-Za-z]+(?:['’][A-Za-z]+)?", text)
    if len(text) < 80 or len(words) < (16 if category == "math" else 24):
        return "", ["low_information"], {}
    if category == "narrative" and not re.search(r'[.!?][\'"”’)]*\s*$', text):
        return "", ["incomplete_narrative"], {}
    sentences = [canonical(s).lower() for s in re.split(r"(?<=[.!?])\s+", text) if len(s) >= 30]
    counts = Counter(sentences)
    repeated = sum(len(s) * (n - 1) for s, n in counts.items()) / max(1, len(text))
    if repeated > 0.20 and max(counts.values(), default=0) >= 3:
        return "", ["repeated_sentences"], {"repeated_sentence_fraction": repeated}
    prose = sum(c.isalpha() for c in text) / len(text)
    if prose < (0.20 if category == "math" else 0.45):
        return "", ["nonprose"], {"alphabetic_fraction": prose}
    broken = len(re.findall(r"\b\w+-\s+\w+", text)) / len(words)
    fragments = sum(len(w) == 1 and w.lower() not in ("a", "i") for w in words) / len(words)
    if category != "math" and broken > 0.025 and fragments > 0.025:
        return "", ["ocr_fragmentation"], {"broken_word_fraction": broken}
    code = len(re.findall(r"\\[A-Za-z]+|\b(?:def|class|include|import)\b", text))
    macros = len(
        re.findall(r"\\(?:newcommand|def|DeclareMathOperator|documentclass|usepackage)\b", text)
    )
    if category == "math" and (macros > 8 or code > max(50, len(words) * 0.22)):
        return "", ["math_markup_dominated"], {}
    if category != "math" and code > max(12, len(words) * 0.08):
        return "", ["code_dominated"], {}
    return (
        text,
        [],
        {
            "removed_characters": before - len(text),
            "alphabetic_fraction": prose,
            "repeated_sentence_fraction": repeated,
            "words": len(words),
        },
    )


@lru_cache(maxsize=131072)
def word_hash(word: str) -> int:
    return int.from_bytes(hashlib.blake2b(word.encode(), digest_size=8).digest(), "little")


def rolling(words: list[str], size: int) -> np.ndarray:
    if len(words) < size:
        return np.empty(0, dtype="<u8")
    values = np.array([word_hash(w) for w in words], dtype="<u8")
    result = np.zeros(len(values) - size + 1, dtype="<u8")
    for i in range(size):
        result += values[i : i + len(result)] * np.uint64(pow(1099511628211, i, 2**64))
    return result


def features(text: str) -> tuple[np.ndarray, bytes, list[tuple[bytes, int]]]:
    words = WORD.findall(text.lower())
    shingles = np.unique(rolling(words, 5))
    mins = []
    for i in range(64):
        key = hashlib.blake2b(str(i).encode(), digest_size=16).digest()
        av = np.uint64(int.from_bytes(key[:8], "little") | 1)
        bv = np.uint64(int.from_bytes(key[8:], "little"))
        mins.append(int(np.min(shingles * av + bv)) if len(shingles) else 0)
    signature = np.asarray(mins, dtype="<u8").tobytes()
    hashes = rolling(words, 13)
    if len(hashes) >= 64:
        view = np.lib.stride_tricks.sliding_window_view(hashes, 64)
        positions = np.unique(view.argmin(axis=1) + np.arange(len(view)))
    else:
        positions = np.arange(len(hashes))
    anchors = [(int(hashes[p]).to_bytes(8, "little"), int(p)) for p in positions]
    return shingles, signature, anchors


def bands(signature: bytes) -> list[bytes]:
    return [
        hashlib.blake2b(bytes([i]) + signature[i * 32 : (i + 1) * 32], digest_size=8).digest()
        for i in range(16)
    ]


def shared_passage(a: str, b: str, pa: int, pb: int) -> bool:
    left, right = WORD.findall(a.lower()), WORD.findall(b.lower())
    if left[pa : pa + 13] != right[pb : pb + 13]:
        return False
    lo = 0
    while pa - lo > 0 and pb - lo > 0 and left[pa - lo - 1] == right[pb - lo - 1]:
        lo += 1
    hi = 13
    while pa + hi < len(left) and pb + hi < len(right) and left[pa + hi] == right[pb + hi]:
        hi += 1
    segment = left[pa - lo : pa + hi]
    return len(segment) >= 128 and sum(w.isalpha() and len(w) > 1 for w in segment) >= 40
