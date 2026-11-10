"""Conservative question-paraphrase screening; ordinary factual prose stays eligible."""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

STOP = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "does",
    "do",
    "did",
    "of",
    "in",
    "on",
    "at",
    "to",
    "for",
    "from",
    "by",
    "with",
    "and",
    "or",
    "this",
    "that",
    "which",
    "what",
    "how",
    "many",
    "much",
    "common",
    "typical",
    "national",
    "called",
    "represents",
    "names",
    "one",
}
SYNONYMS = {
    "capital": "capital",
    "capitals": "capital",
    "city": "city",
    "cities": "city",
    "calculate": "calculate",
    "compute": "calculate",
    "evaluate": "calculate",
    "young": "young",
    "baby": "young",
    "offspring": "young",
    "chemical": "chemical",
    "element": "chemical",
    "symbol": "symbol",
    "opposite": "opposite",
    "opposites": "opposite",
}


def keys(text: str) -> set[str]:
    return {SYNONYMS.get(w, w) for w in re.findall(r"[a-z]+|\d+", text.lower()) if w not in STOP}


class QuestionProtection:
    def __init__(self, fixtures: Path):
        rows = [
            json.loads(line)
            for split in ("dev", "test")
            for line in (fixtures / (split + ".jsonl")).read_text().splitlines()
        ]
        self.questions: list[tuple[str, set[str], tuple[str, ...]]] = []
        for row in rows:
            question = row.get("question")
            if question:
                self.questions.append(
                    (row["id"], keys(question), tuple(re.findall(r"\d+", question)))
                )
        frequency = Counter(w for _, words, _ in self.questions for w in words)
        self.lookup: dict[str, list[int]] = defaultdict(list)
        for i, (_, words, _) in enumerate(self.questions):
            for word in sorted(words, key=lambda w: (frequency[w], w))[:2]:
                self.lookup[word].append(i)

    def matches(self, text: str) -> str | None:
        pieces = re.split(r"(?<=[.!?])\s+|\n+", text)
        for i, piece in enumerate(pieces):
            if "?" not in piece and not re.search(
                r"(?i)\b(?:calculate|compute|solve|simplify|evaluate|how many|what is|which)\b",
                piece,
            ):
                continue
            # Include preceding problem context, never a whole long document.
            span = ((pieces[i - 1] + " ") if i else "") + piece
            if len(span) > 1500:
                continue
            words = keys(span)
            numbers = tuple(re.findall(r"\d+", span))
            candidates = {j for w in words for j in self.lookup.get(w, [])}
            for j in candidates:
                ident, expected, expected_numbers = self.questions[j]
                if expected_numbers and expected_numbers != numbers:
                    continue
                overlap = len(words & expected)
                if (
                    len(expected) >= 2
                    and overlap / len(expected) >= 0.85
                    and overlap / max(1, len(words)) >= 0.65
                ):
                    return ident
        return None

    def audit(self, texts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {"document": row["id"], "question": match}
            for row in texts
            if (match := self.matches(row["text"]))
        ]


class FrozenAssessmentProtection:
    """Read frozen private questions; apply the existing exact/template screen unchanged.

    Generator-family checks belong to dedicated mathematics generators, which are not
    admitted by the language source policy. No private text is emitted or tokenized.
    """

    def __init__(self, roots: list[Path]):
        import hashlib

        self.expected: list[tuple[str, str]] = []
        for root in roots:
            manifest = json.loads((root / "freeze-manifest.json").read_text())
            pool = root / "private-assessment.jsonl"
            families = root / "family-exclusions.json"
            if (
                hashlib.sha256(pool.read_bytes()).hexdigest()
                != manifest["private_assessment_sha256"]
                or hashlib.sha256(families.read_bytes()).hexdigest()
                != manifest["family_exclusions_sha256"]
            ):
                raise ValueError("Frozen private assessment changed")
            for line in pool.read_text().splitlines():
                row = json.loads(line)
                if not row.get("question"):
                    raise ValueError("Frozen question missing")
                text = self.normalise(row["question"])
                self.expected.append((text, self.shape(row["question"])))

    @staticmethod
    def normalise(text: str) -> str:
        return " ".join(re.findall(r"[\w]+|[+*/=−-]", text.casefold(), re.UNICODE))

    @classmethod
    def shape(cls, text: str) -> str:
        return re.sub(r"\b\d+(?:\.\d+)?\b", "#", cls.normalise(text))

    def matches(self, text: str) -> bool:
        from difflib import SequenceMatcher

        normalized = self.normalise(text)
        if any(q and q in normalized for q, _ in self.expected):
            return True
        for segment in re.split(r"(?<=[.!?])\s+|\n+", text):
            if not 18 <= len(segment) <= 400:
                continue
            value = self.shape(segment)
            for _, target in self.expected:
                if (
                    len(target) >= 35
                    and 2 * min(len(value), len(target)) / (len(value) + len(target)) >= 0.88
                    and SequenceMatcher(None, value, target, autojunk=False).quick_ratio() >= 0.88
                    and SequenceMatcher(None, value, target, autojunk=False).ratio() >= 0.88
                ):
                    return True
        return False
