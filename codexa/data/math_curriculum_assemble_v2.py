"""Assemble unapproved, evidence-linked curriculum work and content candidates.

Unreviewed policy fragments are explicitly not counted as official objectives.
All outputs are candidate artifacts; this command never admits training data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer

from codexa.data.math_curriculum_senior_v2 import json_bytes, jsonl_bytes, write_idempotent

SENIOR_SUBTOPICS: dict[int, dict[str, str]] = {
    10: {
        "Algebraic expressions": "Real numbers; surds; products; factorisation; algebraic fractions.",
        "Exponents": "Laws; rational exponents; exponential equations.",
        "Number patterns": "Linear sequences; rules; term relationships.",
        "Equations and inequalities": "Linear; quadratic; simultaneous; literal; contextual equations; linear inequalities.",
        "Functions and graphs": "Linear; quadratic; hyperbolic; exponential; trigonometric; interpretation.",
        "Trigonometry": "Ratios; special angles; equations; Cartesian reasoning; 2D applications.",
        "Euclidean geometry": "Triangles; quadrilaterals; midpoint reasoning; conjecture; proof.",
        "Analytical geometry": "Coordinates; distance; gradient; midpoint.",
        "Financial mathematics": "Simple and compound interest; exchange rates; growth contexts.",
        "Statistics": "Data; central tendency; dispersion; grouped data; five-number summaries.",
        "Measurement": "Polygons; solids; surface area; volume; scale effects.",
        "Probability": "Events; Venn diagrams; complements; mutually exclusive events; theoretical and experimental probability.",
    },
    11: {
        "Exponents and surds": "Laws; simplification; rational exponents; equations within scope.",
        "Equations and inequalities": "Quadratic methods; discriminant; nature of roots; inequalities; mixed simultaneous systems.",
        "Number patterns": "Quadratic sequences; differences; general terms.",
        "Functions and graphs": "Quadratic; hyperbolic; exponential; trigonometric; transformations; average gradient.",
        "Analytical geometry": "Line equations; inclination; parallel and perpendicular relationships.",
        "Trigonometry": "Identities; reduction; equations; sine, cosine and area rules; applications.",
        "Euclidean geometry": "Circle theorems; angles; tangents; proof.",
        "Measurement": "Solids; surface area; volume; scaling relationships.",
        "Financial mathematics": "Growth; decay; depreciation; timelines; nominal and effective rates.",
        "Probability": "Dependent and independent events; trees; Venn diagrams; contingency tables.",
        "Statistics": "Histograms; ogives; variance; standard deviation; skewness; outliers.",
        "Linear programming": "Textbook topic; current compulsory scope requires official confirmation.",
    },
    12: {
        "Sequences and series": "Arithmetic; geometric; finite sums; convergent infinite geometric series.",
        "Functions and inverses": "Relations; inverse functions; linear, quadratic and exponential forms; restricted domains; prescribed logarithms.",
        "Financial mathematics": "Annuities; present and future value; loans; investment periods; timelines.",
        "Trigonometry": "Compound and double angles; identities; equations; applications.",
        "Polynomial algebra": "Cubic polynomials; remainder and factor reasoning; prescribed factorisation.",
        "Differential calculus": "First principles; differentiation rules; tangents; stationary points; cubic graphs; optimisation.",
        "Analytical geometry": "Circles; equations; tangents; coordinate reasoning.",
        "Euclidean geometry": "Proportionality; similarity; triangle results; proof; applications.",
        "Statistics": "Regression; scatter plots; correlation; interpretation.",
        "Probability and counting": "Event identities; counting principles; factorial notation; applications.",
    },
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def lines(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def text_and_kind(row: dict[str, Any]) -> tuple[str, str]:
    original = row.get("original_record") or row.get("raw_source_record") or {}
    text = (
        row.get("normalized_text")
        or row.get("original_text")
        or row.get("text")
        or row.get("original_source_text")
        or row.get("question")
        or original.get("original_source_text")
        or original.get("question")
        or ""
    )
    kind = row.get("record_type") or original.get("record_type") or "unknown"
    if row.get("source_solution") is not None:
        solution = row.get("source_solution") or []
        if isinstance(solution, str):
            solution = [solution]
        context = row.get("context") or []
        if isinstance(context, str):
            context = [context]
        text = "\n".join(str(part) for part in [*context, text, *solution] if part)
    return str(text), str(kind)


def canonical_key(text: str, kind: str, identity: str, origin: str = "source_extracted") -> str:
    # Preserve distinct source identities: identical short text can have different context.
    return sha((origin + "\0" + identity + "\0" + kind + "\0" + text).encode("utf-8"))


def seed_id(row: dict[str, Any]) -> str:
    return str(row.get("objective_id") or row.get("id") or sha(json_bytes(row)))


def run(root: Path, recovery: Path, tokenizer_path: Path, output: Path) -> dict[str, Any]:
    workers = root / "workers"
    f = workers / "foundation" / "run-02"
    m = workers / "middle"
    j = workers / "junior"
    s = workers / "senior" / "inventory-01"
    paths = {
        "foundation_seeds": f / "objective_seed_inventory.jsonl",
        "middle_seeds": m / "topic-seeds.jsonl",
        "junior_seeds": j / "topic-seeds.jsonl",
        "senior_seeds": s / "topic-seeds.jsonl",
        "foundation_fragments": workers / "foundation" / "atp-fragments-v1.jsonl",
        "middle_fragments": m / "objective_candidates-v2.jsonl",
        "junior_fragments": j / "inventory-02" / "objective-candidates.jsonl",
        "senior_fragments": s / "objective-candidates.jsonl",
        "reviewed_objectives": m / "staged-v2" / "reviewed-objectives.jsonl",
        "foundation_registry": f / "source_registry.jsonl",
        "middle_registry": m / "staged-v2" / "source_registry.jsonl",
        "junior_registry": j / "inventory-02" / "source-registry.jsonl",
        "senior_registry": s / "source-registry.jsonl",
        "foundation_sections": f / "source_section_manifest.jsonl",
        "middle_sections": m / "staged-v2" / "source_section_manifest.jsonl",
        "junior_sections": j / "inventory-02" / "source-sections.jsonl",
        "senior_sections": s / "source-section-manifest.jsonl",
        "foundation_records": f / "records.jsonl",
        "foundation_concepts": workers / "foundation/guide-concepts-run-01/teacher-concepts.jsonl",
        "middle_recovered": m / "grade7-recovery-v3/records.jsonl",
        "junior_recovered": j / "exercise-entries-02/records.jsonl",
        **{
            f"senior_{grade}_learning": workers
            / "senior"
            / "learning-units-02"
            / f"grade-{grade}-learning-units.jsonl"
            for grade in (10, 11, 12)
        },
        "middle_records": m / "staged-v2" / "source-records.jsonl",
        "junior_records": j / "inventory-02" / "records-source.jsonl",
        **{
            f"senior_{grade}_records": s / f"grade-{grade}-source-records.jsonl"
            for grade in (10, 11, 12)
        },
        "junior_headings": j / "topic-headings.jsonl",
        "junior_unacquired": j / "source-candidates.jsonl",
        "middle_unacquired": m / "unacquired-sources.jsonl",
        "synthetic_records": root.parent / "recovery-synthetic-v1/generated.jsonl",
        "synthetic_checks": root.parent / "recovery-synthetic-v1/checks-v2.jsonl",
        "synthetic_screen": root.parent / "recovery-synthetic-v1/screening-v3.jsonl",
        "synthetic_family": root.parent
        / "protected-math-recovery-v2/recovery-family-map-02/family-flags.jsonl",
        "foundation_policy": workers / "foundation/policy/atp-sources.json",
        "middle_policy": workers / "middle/policy/receipts.json",
        "junior_policy": workers / "junior/policy/acquisition.json",
        "junior_applicability": workers / "junior/policy/applicability.json",
        "senior_policy": workers / "senior/policy/acquisition.json",
        "caps_baselines": root.parent
        / "corpus-v1/coordinator-audit-20261110/caps-reference/manifest.json",
        **{
            f"reviewed_{grade}": recovery / f"grade-{grade:02d}-pending.jsonl"
            for grade in range(1, 13)
        },
    }
    inputs = {str(path): sha(path.read_bytes()) for path in paths.values()}
    tokenizer_hash = sha(tokenizer_path.read_bytes())
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    seeds = [
        row
        for key in ("foundation_seeds", "middle_seeds", "junior_seeds", "senior_seeds")
        for row in lines(paths[key])
    ]
    candidates = [
        row
        for key in (
            "foundation_fragments",
            "middle_fragments",
            "junior_fragments",
            "senior_fragments",
        )
        for row in lines(paths[key])
    ]
    reviewed_objectives = lines(paths["reviewed_objectives"])
    if any(
        row.get("classification") not in ("seed_unverified", "user_supplied_seed_unverified")
        for row in seeds
    ):
        raise ValueError("unexpected seed classification")
    if {int(row["grade"]) for row in seeds} != set(range(1, 13)):
        raise ValueError("seed inventory does not cover all grades")
    reviews: dict[str, dict[str, Any]] = {}
    for grade in range(1, 13):
        for wrapper in lines(paths[f"reviewed_{grade}"]):
            if not wrapper.get("kind", "").startswith(("practice_source_", "explanation_source_")):
                continue
            item = wrapper["record"]
            rid = item.get("record_id") or item.get("id")
            text, _ = text_and_kind(item)
            if not rid or not text:
                raise ValueError("reviewed record missing identity or text")
            reviews[str(rid)] = {
                "kind": wrapper["kind"],
                "text": text,
                "review_evidence": wrapper.get("review_evidence"),
                "remaining_gates": wrapper.get("remaining_gates"),
                "review_snapshot": str(paths[f"reviewed_{grade}"]),
            }
    grouped: dict[str, dict[str, Any]] = {}
    original_ids: dict[str, set[str]] = defaultdict(set)
    text_groups: dict[str, set[str]] = defaultdict(set)
    source_keys = [
        "foundation_records",
        "foundation_concepts",
        "middle_records",
        "middle_recovered",
        "junior_records",
        "junior_recovered",
        *(f"senior_{g}_learning" for g in (10, 11, 12)),
        *(f"senior_{g}_records" for g in (10, 11, 12)),
    ]
    source_rows = 0
    for key in source_keys:
        path = paths[key]
        for line_no, row in enumerate(lines(path), 1):
            text, kind = text_and_kind(row)
            if not text:
                continue
            source_rows += 1
            nested = row.get("original_record") or row.get("raw_source_record") or {}
            evidence = row.get("source_evidence") or []
            first_evidence = evidence[0] if isinstance(evidence, list) and evidence else {}
            rid = str(row.get("record_id") or row.get("id") or nested.get("id") or "")
            row_grade = row.get("grade")
            key_hash = canonical_key(text, kind, rid or f"{path}:{line_no}")
            text_groups[sha((kind + "\0" + text).encode("utf-8"))].add(key_hash)
            if rid:
                original_ids[rid].add(key_hash)
            if key_hash not in grouped:
                grouped[key_hash] = {
                    "record_id": key_hash,
                    "origin": "source_extracted",
                    "grades": [],
                    "grade": None,
                    "grade_band": None,
                    "original_grade": row.get("original_grade")
                    or row.get("source_grade")
                    or nested.get("source_grade")
                    or (str(row_grade) if key.startswith("senior_") else None),
                    "original_curriculum": row.get("original_curriculum")
                    or row.get("source_curriculum")
                    or nested.get("source_curriculum"),
                    "target_curriculum": row.get("target_curriculum")
                    or nested.get("target_curriculum"),
                    "strand": None,
                    "topic_ids": [],
                    "subtopic_ids": [],
                    "prerequisite_ids": [],
                    "scope_status": "unresolved",
                    "difficulty": None,
                    "source_id": None,
                    "edition": None,
                    "page_or_section": None,
                    "source_exercise_id": None,
                    "source_hash": None,
                    "question": row.get("question"),
                    "solution": row.get("source_solution"),
                    "source_context": row.get("context"),
                    "source_visual_assets": row.get("visual_assets") or [],
                    "answer": row.get("answer"),
                    "assumptions": None,
                    "units": None,
                    "rights_evidence_id": None,
                    "verification_methods": [],
                    "verification_evidence_ids": [],
                    "overlap_policy_version": "math-recovery-protected-v1",
                    "original_text": text,
                    "normalized_text": text,
                    "record_type": kind,
                    "objective_ids": [],
                    "source_lineage": [],
                    "review_evidence": [],
                    "acquisition": "available",
                    "extraction": "unreviewed",
                    "source_fidelity": "pending",
                    "answer_verification": "pending",
                    "step_verification": "pending",
                    "rights": "unresolved",
                    "curriculum_mapping": "provisional",
                    "overlap": "not_screened_under_approved_policy",
                    "review_exposure": "development",
                    "admission_status": "blocked",
                    "native_token_count": None,
                }
            rec = grouped[key_hash]
            if type(row_grade) is int and row_grade not in rec["grades"]:
                rec["grades"].append(row_grade)
            for obj in row.get("objective_ids") or []:
                if obj not in rec["objective_ids"]:
                    rec["objective_ids"].append(obj)
            lineage = {
                "worker": key,
                "source_record_id": rid,
                "input_file": str(path),
                "line": line_no,
                "input_sha256": inputs[str(path)],
                "source_id": row.get("source_id")
                or nested.get("source_id")
                or first_evidence.get("source_id"),
                "source_hash": row.get("source_hash")
                or row.get("source_file_sha256")
                or row.get("source_epub_sha256")
                or nested.get("source_file_sha256")
                or first_evidence.get("source_sha256"),
                "source_location": row.get("source_location")
                or nested.get("page_or_section")
                or nested.get("source_location")
                or first_evidence.get("source_page"),
                "source_evidence": evidence,
                "flags": (row.get("flags") or [])
                + (["visual_dependency"] if row.get("visual_dependency") is True else []),
                "worker_extraction_status": row.get("extraction")
                or row.get("extraction_status")
                or row.get("status"),
            }
            rec["source_lineage"].append(lineage)
            if rid in reviews:
                review = reviews[rid]
                if review["text"] != text:
                    rec["source_fidelity"] = "previous_id_text_changed_requires_review"
                    continue
                rec["review_evidence"].append(
                    {"source_record_id": rid, **{k: v for k, v in review.items() if k != "text"}}
                )
                rec["source_fidelity"] = "passed_selected_review_scope"
    collisions = {rid: sorted(keys) for rid, keys in original_ids.items() if len(keys) > 1}
    if collisions:
        # Do not discard either record. No status is upgraded for an identity collision.
        for keys in collisions.values():
            for key_hash in keys:
                if not grouped[key_hash]["review_evidence"]:
                    grouped[key_hash]["source_fidelity"] = "identity_collision_requires_review"
                grouped[key_hash]["identity_conflict"] = (
                    "same historical id has distinct record text; preserve variants"
                )
    potential_duplicates = {key: sorted(ids) for key, ids in text_groups.items() if len(ids) > 1}
    content = []
    visual = []
    unresolved = []
    for rec in grouped.values():
        rec["grades"].sort()
        if len(rec["grades"]) == 1:
            rec["grade"] = rec["grades"][0]
            g = rec["grade"]
            rec["grade_band"] = (
                "1-3" if g <= 3 else "4-7" if g <= 7 else "8-9" if g <= 9 else "10-12"
            )
        if rec["source_lineage"]:
            first = rec["source_lineage"][0]
            rec["source_id"] = first["source_id"]
            rec["source_hash"] = first["source_hash"]
            rec["page_or_section"] = first["source_location"]
        if rec["question"] is None and rec["record_type"] in (
            "practice_only",
            "practice_only_candidate",
            "exercise",
        ):
            rec["question"] = rec["normalized_text"]
        if rec["review_evidence"]:
            rec["verification_methods"] = ["prior independent source review; limited scope"]
            rec["verification_evidence_ids"] = [
                r["review_snapshot"] for r in rec["review_evidence"]
            ]
        rec["native_token_count"] = len(
            tokenizer.encode(rec["normalized_text"], add_special_tokens=False).ids
        )
        flags = {str(flag) for line in rec["source_lineage"] for flag in line.get("flags", [])}
        states = {str(line.get("worker_extraction_status")) for line in rec["source_lineage"]}
        rec["visual_dependencies"] = (
            "possible"
            if "unresolved_image" in flags
            or "visual-dependent" in states
            or any(
                "visual" in f.casefold() or "diagram" in f.casefold() or "mathml" in f.casefold()
                for f in flags
            )
            else "not_established"
        )
        if rec["source_fidelity"] == "passed_selected_review_scope":
            content.append(rec)
        elif rec["visual_dependencies"] == "possible":
            visual.append(rec)
        else:
            unresolved.append(rec)
    canonical = sorted(grouped.values(), key=lambda r: r["record_id"])
    synthetic_rows = lines(paths["synthetic_records"])
    synthetic_checks = {r["id"]: r for r in lines(paths["synthetic_checks"])}
    synthetic_screens = {r["id"]: r for r in lines(paths["synthetic_screen"])}
    synthetic_families = {r["id"]: r for r in lines(paths["synthetic_family"])}
    synthetic_ids = {r["id"] for r in synthetic_rows}
    if (
        len(synthetic_ids) != len(synthetic_rows)
        or synthetic_ids != set(synthetic_checks)
        or synthetic_ids != set(synthetic_screens)
        or synthetic_ids != set(synthetic_families)
    ):
        raise ValueError("synthetic evidence does not cover distinct generated records")
    synthetic = []
    solution_candidates = []
    synthetic_blocked = []
    for row in synthetic_rows:
        check = synthetic_checks[row["id"]]
        screen = synthetic_screens[row["id"]]
        family = synthetic_families[row["id"]]
        text = "\n".join(
            [
                row["question"],
                *row["solution_steps"],
                str(row["answer"]) + (" " + row["answer_unit"] if row.get("answer_unit") else ""),
            ]
        )
        item = {
            "record_id": row["id"],
            "origin": "synthetic",
            "grade": row["grade"],
            "record_type": row["record_type"],
            "objective_ids": [],
            "question": row["question"],
            "solution": row["solution_steps"],
            "answer": row["answer"],
            "units": row.get("answer_unit"),
            "original_record": row,
            "verification_evidence": check,
            "overlap_evidence": {"screen": screen, "family": family},
            "rights": "original fixture; downstream weight rights not determined",
            "curriculum_mapping": "provisional",
            "overlap": "held",
            "admission_status": "blocked",
            "review_exposure": "development",
            "export_text": text,
            "native_token_count": len(tokenizer.encode(text, add_special_tokens=False).ids),
        }
        synthetic.append(item)
        if (
            check.get("answer_and_steps") == "passed"
            and not check.get("errors")
            and not screen.get("flags")
            and family.get("reason") != "excluded_broad_family"
        ):
            solution_candidates.append(item)
        else:
            synthetic_blocked.append(item)
    senior_topics = []
    for row in seeds:
        if row.get("grade") in (10, 11, 12):
            row = {
                **row,
                "id": sha(f"seed:{row['grade']}:{row['topic']}".encode()),
                "seed_subtopics": SENIOR_SUBTOPICS[row["grade"]][row["topic"]],
                "source": "user-supplied starting topic and subtopic seeds; official classification unreviewed",
                "classification": "seed_unverified",
            }
        senior_topics.append(row)
    seeds = senior_topics
    by_grade: dict[str, Any] = {}
    for grade in range(1, 13):
        by_grade[str(grade)] = {
            "grade": grade,
            "official_phase": "foundation"
            if grade <= 3
            else "intermediate"
            if grade <= 6
            else "senior"
            if grade <= 9
            else "FET",
            "grade_band": "1-3"
            if grade <= 3
            else "4-7"
            if grade <= 7
            else "8-9"
            if grade <= 9
            else "10-12",
            "seed_items": [row for row in seeds if row.get("grade") == grade],
            "official_objective_candidates": [
                row for row in candidates if row.get("grade") == grade
            ],
            "reviewed_objectives": [
                row for row in reviewed_objectives if row.get("grade") == grade
            ],
            "official_topic_headings_reviewed_at_heading_level": [
                row for row in lines(paths["junior_headings"]) if row.get("grade") == grade
            ],
        }
    curriculum = {
        "version": "v2-candidate-1",
        "grades": by_grade,
        "official_denominator": "not_reconciled",
        "notes": [
            "Seeds and machine-extracted policy fragments are work items, not confirmed official objectives.",
            "Circular S19 applicability and ATP page anomalies require reconciliation.",
        ],
    }
    registries = [
        {
            "worker": key,
            "declared_source": row,
            "rights_gate": "unresolved_pending_exact_edition_and_use",
        }
        for key in (
            "foundation_registry",
            "middle_registry",
            "junior_registry",
            "senior_registry",
            "middle_unacquired",
            "junior_unacquired",
        )
        for row in lines(paths[key])
    ]
    sections = [
        {"worker": key, "entry": row, "objective_mapping": "unreviewed_unless_explicitly_linked"}
        for key in ("foundation_sections", "middle_sections", "junior_sections", "senior_sections")
        for row in lines(paths[key])
    ]
    headings = lines(paths["junior_headings"])
    pack = []
    for row in seeds:
        grade = row["grade"]
        topic = row["topic"]
        parent_sid = seed_id(row)
        raw_subtopics = (
            row.get("subtopic") or row.get("seed_subtopics") or row.get("subtopic_seeds")
        )
        if isinstance(raw_subtopics, str):
            subtopics = [
                piece.strip().rstrip(".") for piece in raw_subtopics.split(";") if piece.strip()
            ]
        elif isinstance(raw_subtopics, list):
            subtopics = [str(piece).strip() for piece in raw_subtopics if str(piece).strip()]
        else:
            subtopics = [""]
        for subtopic in subtopics:
            sid = sha(f"{parent_sid}\0{subtopic}".encode())
            candidate_section_ids: list[str] = []
            for heading in headings:
                heading_name = str(heading.get("official_topic_heading", "")).casefold()
                if (
                    heading.get("grade") == grade
                    and subtopic
                    and subtopic.casefold() == heading_name
                ):
                    candidate_section_ids.extend(heading.get("candidate_source_section_ids") or [])
            linked = []
            for rec in content:
                if (
                    grade == 7
                    and "whole numbers" in topic.casefold()
                    and subtopic.casefold() == "calculations"
                    and rec["objective_ids"]
                ):
                    linked.append(rec["record_id"])
            pack.append(
                {
                    "seed_id": sid,
                    "parent_seed_id": parent_sid,
                    "grade": grade,
                    "topic": topic,
                    "subtopic": subtopic,
                    "classification": "seed_unverified",
                    "official_objective_ids": [],
                    "candidate_source_section_ids": list(dict.fromkeys(candidate_section_ids)),
                    "content_checked_record_ids": linked,
                    "source_links_status": "provisional_not_coverage",
                    "instructional_state": "instructionally_partial"
                    if linked
                    else "missing_or_unmapped",
                    "missing_material": "objective mapping, explanations, examples, practice, solutions and visuals not fully reconciled",
                }
            )
    coverage_rows = []
    for item in pack:
        coverage_rows.append(
            {
                "grade": item["grade"],
                "topic": item["topic"],
                "subtopic_seed": item["subtopic"] or "",
                "objective_id": "unresolved",
                "classification": "seed_unverified",
                "official_scope": "unreconciled",
                "policy_reference": "pending exact objective mapping",
                "candidate_sections": len(item["candidate_source_section_ids"]),
                "content_checked_records": len(item["content_checked_record_ids"]),
                "rights_cleared": 0,
                "finally_admitted": 0,
                "instructional_state": item["instructional_state"],
                "missing_material": item["missing_material"],
            }
        )
    # The separately reviewed Grade 7 objective is a revision objective, not a complete official denominator.
    for obj in reviewed_objectives:
        refs = [r["record_id"] for r in content if obj["id"] in r["objective_ids"]]
        coverage_rows.append(
            {
                "grade": obj["grade"],
                "topic": obj.get("topic") or "",
                "subtopic_seed": obj.get("subtopic") or "",
                "objective_id": obj["id"],
                "classification": obj.get("classification"),
                "official_scope": "baseline_confirmed_current_applicability_pending",
                "policy_reference": str(obj.get("policy_source_url") or "")
                + "#page="
                + str(obj.get("pdf_page") or ""),
                "candidate_sections": 0,
                "content_checked_records": len(refs),
                "rights_cleared": 0,
                "finally_admitted": 0,
                "instructional_state": "instructionally_partial" if refs else "missing_or_unmapped",
                "missing_material": "full explanatory, worked-example and remaining-skill coverage; current policy reconciliation",
            }
        )
    for fragment in candidates:
        policy = fragment.get("policy") or {}
        candidate_id = (
            fragment.get("id") or fragment.get("candidate_id") or sha(json_bytes(fragment))
        )
        reference_url = (
            policy.get("url")
            or fragment.get("policy_url")
            or fragment.get("policy_source_url")
            or ""
        )
        page = policy.get("pdf_page") or policy.get("page") or fragment.get("pdf_page") or ""
        coverage_rows.append(
            {
                "grade": fragment.get("grade"),
                "topic": fragment.get("topic") or "unmapped",
                "subtopic_seed": fragment.get("objective_text")
                or fragment.get("source_text_candidate")
                or "",
                "objective_id": candidate_id,
                "classification": "unresolved_policy_fragment_not_confirmed_objective",
                "official_scope": "unreviewed",
                "policy_reference": str(reference_url) + "#page=" + str(page),
                "candidate_sections": 0,
                "content_checked_records": 0,
                "rights_cleared": 0,
                "finally_admitted": 0,
                "instructional_state": "unreviewed_policy_fragment",
                "missing_material": "verify exact objective and classify core/revision/enrichment before coverage assessment",
            }
        )
    csv_stream = io.StringIO(newline="")
    writer = csv.DictWriter(csv_stream, fieldnames=list(coverage_rows[0]))
    writer.writeheader()
    writer.writerows(coverage_rows)
    output.mkdir(parents=True, exist_ok=True)
    policy_ledger = {
        "foundation": json.loads(paths["foundation_policy"].read_text()),
        "middle": json.loads(paths["middle_policy"].read_text()),
        "junior": json.loads(paths["junior_policy"].read_text()),
        "junior_applicability": json.loads(paths["junior_applicability"].read_text()),
        "senior": json.loads(paths["senior_policy"].read_text()),
        "caps_baselines": json.loads(paths["caps_baselines"].read_text()),
        "status": "applicability and amendments unresolved where not explicitly supported by official evidence",
        "policy_documents_are_training_content": False,
    }
    payloads = {
        "policy/applicability-ledger.json": json_bytes(policy_ledger),
        "curriculum_map.json": json_bytes(curriculum),
        "source_registry.jsonl": jsonl_bytes(registries),
        "source_section_manifest.jsonl": jsonl_bytes(sections),
        "grade_topic_coverage.csv": csv_stream.getvalue().encode(),
        "prerequisite_progression.json": json_bytes(
            {
                "relationships": [],
                "unresolved": [
                    {
                        "grade": 7,
                        "objective_id": o["id"],
                        "relation": "revision; prerequisite link requires exact earlier-grade objective",
                    }
                    for o in reviewed_objectives
                ],
                "status": "incomplete_no_inferred_links",
            }
        ),
        "topics/topic-pack-index.jsonl": jsonl_bytes(pack),
        "records/source-canonical.jsonl": jsonl_bytes(canonical),
        "records/synthetic-candidates.jsonl": jsonl_bytes(synthetic),
        "exports/unresolved-synthetic.jsonl": jsonl_bytes(synthetic_blocked),
        "exports/content-checked-candidates.jsonl": jsonl_bytes(content),
        "exports/practice-only-candidates.jsonl": jsonl_bytes(
            [
                r
                for r in canonical
                if r["record_type"] in ("practice_only", "practice_only_candidate")
                and r["answer"] is None
                and not r["solution"]
            ]
        ),
        "exports/visual-dependent.jsonl": jsonl_bytes(visual),
        "exports/unresolved.jsonl": jsonl_bytes(unresolved),
        "exports/rights-cleared-candidates.jsonl": b"",
        "exports/supervised-solution-candidates.jsonl": jsonl_bytes(solution_candidates),
        "exports/practice-only-cleared.jsonl": b"",
        "exports/finally-admitted.jsonl": b"",
        "manifests/identity-collisions.json": json_bytes(collisions),
        "manifests/possible-cross-identity-duplicates.json": json_bytes(potential_duplicates),
    }
    for name, data in payloads.items():
        write_idempotent(output / name, data)
    grade_summary = {}
    for grade in range(1, 13):
        subset = [r for r in canonical if grade in r["grades"]]
        checked = [r for r in content if grade in r["grades"]]
        grade_summary[str(grade)] = {
            "seed_items": sum(x["grade"] == grade for x in seeds),
            "unreviewed_policy_fragments": sum(x.get("grade") == grade for x in candidates),
            "reviewed_baseline_objectives": sum(
                x.get("grade") == grade for x in reviewed_objectives
            ),
            "source_identity_candidates": len(subset),
            "content_checked": len(checked),
            "content_checked_text_tokens": sum(r["native_token_count"] for r in checked),
            "rights_cleared": 0,
            "synthetic_instances_held": sum(x["grade"] == grade for x in synthetic),
            "solution_checked_candidates_held": sum(
                x["grade"] == grade for x in solution_candidates
            ),
            "finally_admitted": 0,
            "verified_training_tokens": 0,
        }
    summary = {
        "source_rows": source_rows,
        "distinct_source_identity_candidates": len(canonical),
        "same_identity_text_duplicates": source_rows - len(canonical),
        "same_text_distinct_identity_groups_pending_review": len(potential_duplicates),
        "identity_collisions": len(collisions),
        "source_registry_entries": len(registries),
        "source_section_entries": len(sections),
        "seed_items_unverified": len(seeds),
        "unreviewed_policy_fragments": len(candidates),
        "reviewed_baseline_objectives": len(reviewed_objectives),
        "content_checked_records": len(content),
        "synthetic_instances_held": len(synthetic),
        "solution_checked_candidates_held": len(solution_candidates),
        "synthetic_excluded_or_flagged": len(synthetic_blocked),
        "rights_cleared": 0,
        "finally_admitted": 0,
        "by_grade": grade_summary,
        "tokenizer_sha256": tokenizer_hash,
        "token_count_definition": "native tokenizer on normalized_text field only; no BOS/EOS; checked candidates are not approved training tokens",
        "limitations": [
            "Official objective denominator is not reconciled; policy fragments and seeds are not confirmed objectives.",
            "Only identical source identity, type and text are merged; same-text distinct identities remain pending duplicate review and are not verified unique content.",
            "Topic/source links are preliminary navigation signals, not evidence of instructional completeness.",
            "Content checks retain their original limited review scope; rights, current policy, overlap and admission remain open.",
            "Missing official requirements and gaps have not been fully enumerated.",
        ],
    }
    overview = io.StringIO(newline="")
    overview_rows = []
    for grade in range(1, 13):
        info = grade_summary[str(grade)]
        topic_rows = [p for p in pack if p["grade"] == grade]
        overview_rows.append(
            {
                "grade": grade,
                "official_objective_denominator": "not_reconciled",
                "baseline_revision_objectives_identified": info["reviewed_baseline_objectives"],
                "unreviewed_policy_fragments": info["unreviewed_policy_fragments"],
                "seed_items_unverified": info["seed_items"],
                "topic_packs_confirmed_complete": 0,
                "seed_packs_partial": sum(
                    p["instructional_state"] == "instructionally_partial" for p in topic_rows
                ),
                "missing_topics": "not enumerated",
                "source_identity_candidates_not_deduplicated": info["source_identity_candidates"],
                "synthetic_instances_held": info["synthetic_instances_held"],
                "content_checked": info["content_checked"],
                "content_checked_tokens": info["content_checked_text_tokens"],
                "rights_cleared": 0,
                "finally_admitted": 0,
                "verified_training_tokens": 0,
            }
        )
    overview_writer = csv.DictWriter(overview, fieldnames=list(overview_rows[0]))
    overview_writer.writeheader()
    overview_writer.writerows(overview_rows)
    overview_data = overview.getvalue().encode()
    write_idempotent(output / "reports/grade-overview.csv", overview_data)
    payloads["reports/grade-overview.csv"] = overview_data
    write_idempotent(output / "reports/summary.json", json_bytes(summary))
    manifest = {
        "inputs_sha256": inputs,
        "tokenizer_sha256": tokenizer_hash,
        "outputs_sha256": {name: sha(data) for name, data in payloads.items()},
        "summary_sha256": sha(json_bytes(summary)),
        "status": "partial_candidate_not_training_approved",
    }
    write_idempotent(output / "manifests/manifest.json", json_bytes(manifest))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reviewed", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.root, args.reviewed, args.tokenizer, args.output)))


if __name__ == "__main__":
    main()
