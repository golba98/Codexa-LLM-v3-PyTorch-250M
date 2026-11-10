"""Emit user-provided retrieval seeds; these are not official objectives."""

import argparse
import hashlib
import json
from pathlib import Path

SEEDS = {
    8: [
        (
            "Whole numbers, integers and exponents",
            "Number properties; signed operations; powers; roots; applications.",
        ),
        ("Common and decimal fractions", "Equivalent forms; calculations; rounding; context."),
        ("Numeric and geometric patterns", "Rules; sequences; tables; descriptions."),
        (
            "Functions and relationships",
            "Input-output; formulae; flow diagrams; equivalent representations.",
        ),
        (
            "Algebraic expressions",
            "Notation; substitution; operations; simplification; powers; roots.",
        ),
        (
            "Algebraic equations",
            "Formulation; inverse operations; solutions; tables; applications.",
        ),
        (
            "Constructions and 2D geometry",
            "Accurate figures; properties; triangles; quadrilaterals; classification.",
        ),
        (
            "Straight-line geometry",
            "Angle relationships; intersecting and parallel lines; transversals.",
        ),
        ("Pythagoras", "Right triangles; unknown sides; applications; validity conditions."),
        ("Measurement", "Perimeter; area; surface area; volume; units."),
        (
            "Data handling",
            "Collection; organisation; summaries; graphs; interpretation; reporting.",
        ),
        ("Graphs", "Relationships; representation; interpretation."),
        ("Transformations", "Translation; reflection; rotation; enlargement; reduction."),
        ("3D geometry", "Solids; faces; edges; vertices; models; relationships."),
        ("Probability", "Outcomes; theoretical likelihood; relative frequency."),
    ],
    9: [
        (
            "Number systems and applications",
            "Whole numbers; integers; rational and irrational distinctions; ratios; rates; financial contexts.",
        ),
        ("Common fractions and decimals", "Operations; equivalent forms; estimation; context."),
        ("Exponents", "Laws; scientific notation; prescribed equations and calculations."),
        ("Numeric and geometric patterns", "Rules; general descriptions; sequences; tables."),
        (
            "Functions and relationships",
            "Input-output; formulae; tables; equivalent representations.",
        ),
        (
            "Algebraic expressions",
            "Expansion; factorisation; simplification; algebraic fractions within scope.",
        ),
        ("Algebraic equations", "Linear and other prescribed forms; context; solution checking."),
        ("Graphs", "Plotting; interpreting; relationships; representations."),
        ("Geometric constructions", "Accurate constructions; properties; supporting reasoning."),
        (
            "2D and straight-line geometry",
            "Triangles; quadrilaterals; similarity and congruence where prescribed; angle reasoning.",
        ),
        (
            "Pythagoras and measurement",
            "Side lengths; perimeter; area; surface area; volume; units.",
        ),
        (
            "Transformations and 3D geometry",
            "Movement; scale; solids; properties; representations.",
        ),
        (
            "Data handling",
            "Collection; organisation; graphical displays; statistics; interpretation.",
        ),
        ("Probability", "Outcomes; experimental and theoretical reasoning; context."),
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.output.open("x", encoding="utf-8") as out:
        for grade, topics in SEEDS.items():
            for title, description in topics:
                item = {
                    "id": hashlib.sha256(f"user-seed-v1\0{grade}\0{title}".encode()).hexdigest(),
                    "grade": grade,
                    "grade_band": "8-9",
                    "official_phase": "senior",
                    "topic": title,
                    "seed_subtopics": description,
                    "classification": "seed_unverified",
                    "source": "user supplied grade topic inventory",
                    "official_objective_ids": [],
                    "status": "needs_2026_atp_scope_and_source_reconciliation",
                }
                out.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    print(
        json.dumps(
            {"records": sum(len(v) for v in SEEDS.values()), "status": "unverified retrieval seeds"}
        )
    )


if __name__ == "__main__":
    main()
