#!/usr/bin/env python3
"""Original grade 4–7 instructional candidates, separate from source quotations."""

import argparse
import ast
import hashlib
import json
import operator
from fractions import Fraction
from pathlib import Path

BASE = Path("/mnt/codexa/pytorch/codexa-v3/school-math")
POLICY = BASE / "curriculum-corpus-v2/workers/middle/policy"
URLS = {
    4: "https://www.education.gov.za/LinkClick.aspx?fileticket=HINtK497f94%3D&forcedownload=true&mid=10739&portalid=0&tabid=3207",
    5: "https://www.education.gov.za/LinkClick.aspx?fileticket=Zcwzy598npw%3D&forcedownload=true&mid=10742&portalid=0&tabid=3207",
    6: "https://www.education.gov.za/LinkClick.aspx?fileticket=cT8nlsKsBAs%3D&forcedownload=true&mid=10744&portalid=0&tabid=3207",
    7: "https://www.education.gov.za/LinkClick.aspx?fileticket=ntUvRUteBtQ%3D&forcedownload=true&mid=10747&portalid=0&tabid=3208",
}
# Each arithmetic equality is independently checked as exact rational arithmetic.
# Distinct structures are deliberately hand-authored, not permutations of an exam.
LESSONS = [
    (
        4,
        "place value",
        "A digit has a value determined by its position. Read a number by adding the values of its digits; a zero can hold an empty place.",
        [
            ("Expand 3 745 by place value.", "3000+700+40+5", "3745"),
            ("What is the value of the 8 in 6 082?", "8*10", "80"),
            ("Expand 5 019 by place value.", "5000+10+9", "5019"),
        ],
    ),
    (
        4,
        "addition and subtraction",
        "Align digits of the same place before adding or subtracting. Regroup one unit from a place as ten units in the next place when needed.",
        [
            ("Calculate 3 478 + 2 659.", "3478+2659", "6137"),
            ("Calculate 8 002 − 3 647.", "8002-3647", "4355"),
            ("Check 6 137 − 2 659 by addition.", "3478+2659", "6137"),
        ],
    ),
    (
        4,
        "multiplication",
        "The distributive property lets us split one factor into place values, multiply each part, and add the partial products.",
        [
            ("Calculate 36 × 24.", "36*20+36*4", "864"),
            ("Calculate 12 × 23.", "12*20+12*3", "276"),
            ("Calculate 16 × 25 by doubling and halving.", "8*50", "400"),
        ],
    ),
    (
        5,
        "place value",
        "In a base-ten number each place is ten times the place immediately to its right. A digit contributes its value in that place.",
        [
            ("What is the value of the 8 in 483 216?", "8*10000", "80000"),
            ("Expand 600 405 by place value.", "600000+400+5", "600405"),
            ("Expand 274 018 by place value.", "200000+70000+4000+10+8", "274018"),
        ],
    ),
    (
        5,
        "whole-number operations",
        "Use place-value alignment for addition and subtraction. For multiplication, distribute across the tens and ones before combining partial products.",
        [
            ("Calculate 58 427 + 26 758.", "58427+26758", "85185"),
            ("Calculate 70 002 − 38 974.", "70002-38974", "31028"),
            ("Calculate 326 × 24.", "326*20+326*4", "7824"),
        ],
    ),
    (
        5,
        "common fractions",
        "Fractions with the same denominator count equal-sized parts. Add their numerators and keep the denominator; a fraction of a quantity is found by dividing into equal parts and taking the specified number.",
        [
            ("Calculate three eighths plus two eighths.", "3/8+2/8", "5/8"),
            ("Calculate one and one quarter plus two and one half.", "1+1/4+2+1/2", "15/4"),
            ("Find three fifths of 35.", "3*35/5", "21"),
        ],
    ),
    (
        6,
        "whole-number operations",
        "Regroup using place value when adding or subtracting. Multiplying by a sum is the sum of the separate partial products.",
        [
            ("Calculate 700 002 − 389 745.", "700002-389745", "310257"),
            ("Calculate 530 410 + 278 965.", "530410+278965", "809375"),
            ("Calculate 1 234 × 205.", "1234*200+1234*5", "252970"),
        ],
    ),
    (
        6,
        "ratio and rate",
        "A ratio compares quantities in equal parts. To divide a total in a ratio, add the parts, find one part, then multiply. A rate compares quantities in different units.",
        [
            (
                "Share 25 objects in the ratio 2:3. How many objects are in the first share?",
                "25/(2+3)*2",
                "10",
            ),
            ("A trip covers 180 km in 3 hours. Find its average speed in km/h.", "180/3", "60"),
            (
                "Four identical items cost R28. At the same unit price, what do seven cost in rand?",
                "28/4*7",
                "49",
            ),
        ],
    ),
    (
        6,
        "factors",
        "A factor divides a whole number without remainder. A factorisation expresses a number as a product of factors.",
        [
            ("Check that 7 is a factor of 84 by finding 84 ÷ 7.", "84/7", "12"),
            ("Use a product to check that 2 × 2 × 3 × 7 equals 84.", "2*2*3*7", "84"),
            ("Find 25 × 4 × 36 by grouping 25 and 4 first.", "(25*4)*36", "3600"),
        ],
    ),
    (
        7,
        "fractions",
        "To add or subtract fractions, express them in equal-sized parts using a common denominator. Multiply fractions by multiplying numerators and denominators.",
        [
            ("Calculate two thirds plus three quarters.", "2/3+3/4", "17/12"),
            ("Calculate five sixths minus one quarter.", "5/6-1/4", "7/12"),
            ("Calculate one and one half multiplied by two thirds.", "(1+1/2)*(2/3)", "1"),
        ],
    ),
    (
        7,
        "decimal fractions",
        "A decimal digit gives a multiple of a tenth, hundredth, thousandth or another power of ten. Align decimal places when adding.",
        [
            ("Calculate 3.475 + 1.8.", "3475/1000+18/10", "5275/1000"),
            ("What is the value of the 7 in 4.372?", "7/100", "7/100"),
            ("Calculate 6.2 − 0.35.", "62/10-35/100", "585/100"),
        ],
    ),
    (
        7,
        "percentages",
        "Percent means per hundred. Multiply a quantity by the percentage divided by 100; an increase or decrease changes the original by that amount.",
        [
            ("Find 15% of 240.", "15*240/100", "36"),
            ("Increase 80 by 12.5%.", "80+125*80/1000", "90"),
            ("Decrease 240 by 15%.", "240-15*240/100", "204"),
        ],
    ),
]


def evaluate(text):
    node = ast.parse(text, mode="eval")
    ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
    }

    def visit(n):
        if isinstance(n, ast.Constant) and type(n.value) is int:
            return Fraction(n.value)
        if isinstance(n, ast.BinOp) and type(n.op) in ops:
            return ops[type(n.op)](visit(n.left), visit(n.right))
        raise ValueError(f"Unsupported expression {ast.dump(n)}")

    return visit(node.body)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    out = parser.parse_args().output
    out.mkdir(parents=True, exist_ok=False)
    records = []
    for grade, topic, explanation, examples in LESSONS:
        source = POLICY / f"grade-{grade}-atp.pdf"
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        source_ref = {
            "url": URLS[grade],
            "sha256": source_hash,
            "page": 3 if grade == 5 and topic == "common fractions" else 1,
            "document": source.name,
            "applicability": "2026 document"
            if grade in (4, 5)
            else "2023/24 document; 2026 applicability pending independent reconciliation",
        }
        common = {
            "grade": grade,
            "grade_band": "4-7",
            "topic": topic,
            "origin": "original_authored",
            "source_policy_reference": source_ref,
            "source_fidelity": "not_applicable_original_paraphrase",
            "rights": "original content; CC0 dedication proposed, rights review pending",
            "evaluation_overlap": "not_screened",
            "curriculum_mapping": "provisional policy page reference; independent mapping pending",
            "admission": "blocked: independent content, overlap and admission review pending",
            "review_exposure": "development",
            "visual_dependencies": [],
        }

        def add(r):
            r = {
                "record_id": "middle-original-"
                + hashlib.sha256(
                    json.dumps(r, sort_keys=True, ensure_ascii=False).encode()
                ).hexdigest()[:24],
                **r,
            }
            records.append(r)

        add(
            {
                **common,
                "record_type": "concept_explanation",
                "normalized_text": explanation,
                "mathematical_review": "pending independent review",
            }
        )
        for question, expression, answer in examples:
            if evaluate(expression) != evaluate(answer):
                raise ValueError((question, expression, answer))
            add(
                {
                    **common,
                    "record_type": "worked_example",
                    "question": question,
                    "solution": f"{expression} = {answer}",
                    "answer": answer,
                    "mathematical_answer_verification": "exact-rational-arithmetic-check-passed",
                    "supplied_step_verification": "one exact equality checked; question correspondence independently pending",
                    "verification_method": "Python fractions.Fraction AST arithmetic evaluator",
                    "assumptions": "ordinary base-ten arithmetic; contextual assumptions explicit in question",
                    "units": "km/h"
                    if "km/h" in question
                    else "rand"
                    if "rand" in question
                    else "objects"
                    if "objects" in question
                    else None,
                }
            )
    ids = [r["record_id"] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate record identity")
    output = out / "original-instructional-candidates.jsonl"
    payload = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in records)
    if output.exists() and output.read_text() != payload:
        raise RuntimeError("Output exists with different content; use a new output version")
    output.write_text(payload)
    receipt = {
        "records": len(records),
        "concept_explanations": sum(r["record_type"] == "concept_explanation" for r in records),
        "arithmetic_checked_worked_examples": sum(
            r["record_type"] == "worked_example" for r in records
        ),
        "admitted": 0,
        "output_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "limitations": [
            "Not independently reviewed",
            "Protected-overlap policy not applied",
            "Objective coverage incomplete",
            "No final corpus admission",
        ],
    }
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
