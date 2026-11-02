import hashlib
import random
from dataclasses import asdict, dataclass
from typing import Any

import sympy


@dataclass(frozen=True)
class Problem:
    id: str
    grade: int
    topic: str
    difficulty: str
    prompt: str
    operands: tuple[int, int]
    answer: int
    steps: tuple[str, ...]
    split: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def problems(split: str, count: int, seed: int = 42) -> list[Problem]:
    if split not in ("train", "val", "test") or count < 0:
        raise ValueError("Invalid split or count")
    pool = []
    for op in ("+", "-"):
        for a in range(21):
            for b in range(21):
                if (op == "+" and (a > b or a + b > 20)) or (op == "-" and a < b):
                    continue
                key = f"v1:{op}:{a}:{b}"
                identity = hashlib.sha256(key.encode()).hexdigest()
                bucket = int(identity, 16) % 10
                partition = "test" if bucket == 0 else "val" if bucket == 1 else "train"
                if partition != split:
                    continue
                answer = a + b if op == "+" else a - b
                symbolic = sympy.Integer(a) + b if op == "+" else sympy.Integer(a) - b
                assert int(symbolic) == answer
                topic = "addition" if op == "+" else "subtraction"
                steps = (
                    f"Start with {a}.",
                    f"{'Add' if op == '+' else 'Subtract'} {b}.",
                    f"{a} {op} {b} = {answer}.",
                )
                pool.append(
                    Problem(
                        identity,
                        1,
                        topic,
                        "easy" if max(a, b, answer) <= 10 else "medium",
                        f"What is {a} {op} {b}?",
                        (a, b),
                        answer,
                        steps,
                        split,
                    )
                )
    random.Random(seed).shuffle(pool)
    if count > len(pool):
        raise ValueError(f"Requested {count} problems but partition contains only {len(pool)}")
    return pool[:count]


def accuracy(problems_: list[dict[str, Any]], predictions: dict[str, str]) -> dict[str, Any]:
    groups: dict[str, list[bool]] = {}
    for p in problems_:
        try:
            correct = int(predictions.get(p["id"], "")) == p["answer"]
        except ValueError:
            correct = False
        for key in ("overall", f"grade:{p['grade']}", f"topic:{p['topic']}"):
            groups.setdefault(key, []).append(correct)
    return {
        k: {"correct": sum(v), "total": len(v), "accuracy": sum(v) / len(v)}
        for k, v in groups.items()
    }
