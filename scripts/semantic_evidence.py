"""Resolve a complete admission census's factual uncertainty without resampling."""

import argparse
import json
from pathlib import Path
from typing import Any

from codexa.data.production import atomic, digest
from codexa.data.semantic import (
    assessed,
    cleared,
    load_admission,
    primary_evidence_ids,
    require_primary_evidence,
)
from codexa.data.v2_storage import Storage

FILES = (
    "protocol.json",
    "packet-integrity.json",
    "blind.jsonl",
    "mapping.jsonl",
    "sampling.json",
    "token-verification.json",
    "curator-a.jsonl",
    "curator-b.jsonl",
    "admission.jsonl",
    "decision.json",
)


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def preservation_files(directory: Path) -> set[str]:
    """Bind raw shards, amendments and coverage without changing the blind packet."""
    names = set(FILES) | set(json.loads((directory / "packet-integrity.json").read_text()))
    supporting = directory / "supporting-evidence.json"
    if supporting.exists():
        names.add(supporting.name)
        for name, sha in json.loads(supporting.read_text())["files"].items():
            if Path(name).name != name or digest(directory / name) != sha:
                raise ValueError("Supporting review evidence changed or has an unsafe name")
            names.add(name)
    if any(Path(name).name != name or (directory / name).is_symlink() for name in names):
        raise ValueError("Review evidence must be local regular files")
    return names


def inputs(directory: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[Any]]:
    protocol = json.loads((directory / "protocol.json").read_text())
    load_admission(directory, protocol["parent_manifest_sha256"])
    mapping = read_rows(directory / "mapping.jsonl")
    expected = {row["id"] for row in mapping}
    judges = [assessed(read_rows(directory / f"curator-{j}.jsonl"), expected) for j in "ab"]
    return mapping, read_rows(directory / "blind.jsonl"), judges


def eligible(judgments: list[dict[str, Any]]) -> bool:
    """A false usable flag due to factual uncertainty is not a separate defect."""
    return any(r["checks"]["correctness"] == "uncertain" for r in judgments) and all(
        r["coverage_complete"]
        and not r["severe"]
        and not r["critical"]
        and r["checks"]["correctness"] in {"pass", "uncertain", "not_applicable"}
        and all(
            value not in {"defect", "uncertain"}
            for check, value in r["checks"].items()
            if check != "correctness"
        )
        and all(
            r["checks"][check] == "pass"
            for check in ("completeness", "education", "coherence", "contamination")
        )
        for r in judgments
    )


def destination(root: Path, output: Path, allocation: int = 64 * 1024**2) -> None:
    Storage.load(root).check(allocation)
    if not output.resolve().is_relative_to((root / "semantic-admission").resolve()):
        raise ValueError("Evidence artifacts must stay on the preparation volume")
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)


def write_packet(output: Path, rows: dict[str, list[dict[str, Any]]], protocol: dict) -> None:
    atomic(output / "protocol.json", protocol)
    for name, values in rows.items():
        with (output / name).open("x") as stream:
            stream.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in values)
    atomic(
        output / "sampling.json",
        {"candidates": len(rows["mapping.jsonl"]), "resampling": False},
    )
    names = {"protocol.json", "sampling.json", *rows}
    names.update(p.name for p in output.iterdir() if p.is_file())
    atomic(output / "packet-integrity.json", {name: digest(output / name) for name in names})


def prepare(root: Path, base: Path, output: Path, policy: Path) -> None:
    mapping, blind, judges = inputs(base)
    if json.loads(policy.read_text())["original_packet_sha256"] != digest(
        base / "packet-integrity.json"
    ):
        raise ValueError("Resolution policy belongs to another census")
    selected = {row["id"] for row in mapping if eligible([judge[row["id"]] for judge in judges])}
    if not selected:
        raise ValueError("No correctness-only uncertainty to resolve")
    destination(root, output)
    (output / "resolution-policy.json").write_bytes(policy.read_bytes())
    (output / "evidence-preparation-source.py").write_bytes(Path(__file__).read_bytes())
    protocol = json.loads((base / "protocol.json").read_text()) | {
        "version": "semantic-evidence-resolution-1",
        "base_decision_sha256": digest(base / "decision.json"),
        "base_packet_integrity_sha256": digest(base / "packet-integrity.json"),
        "resolution_policy_sha256": digest(policy),
        "scope": "Every correctness-only uncertainty in the complete original census; no resampling or approval scores. Both judges reread each full document. Clearing requires primary evidence for every identified uncertainty, with claim, source_url, finding and limitation recorded in each evidence entry. Preserve remaining uncertainty or any newly discovered defect.",
    }
    write_packet(
        output,
        {
            "mapping.jsonl": [row for row in mapping if row["id"] in selected],
            "blind.jsonl": [row for row in blind if row["id"] in selected],
            "questions.jsonl": [
                {"id": ident, "questions": [judge[ident]["reason"] for judge in judges]}
                for ident in sorted(selected)
            ],
        },
        protocol,
    )


def compose(root: Path, base: Path, resolution: Path, output: Path) -> None:
    mapping, blind, old_judges = inputs(base)
    new_mapping, new_blind, new_judges = inputs(resolution)
    protocol = json.loads((resolution / "protocol.json").read_text())
    base_protocol = json.loads((base / "protocol.json").read_text())
    if (
        protocol["base_decision_sha256"] != digest(base / "decision.json")
        or protocol["base_packet_integrity_sha256"] != digest(base / "packet-integrity.json")
        or protocol["parent_manifest_sha256"] != base_protocol["parent_manifest_sha256"]
    ):
        raise ValueError("Resolution belongs to another original census")
    selected = {
        row["id"] for row in mapping if eligible([judge[row["id"]] for judge in old_judges])
    }
    if new_mapping != [row for row in mapping if row["id"] in selected] or new_blind != [
        row for row in blind if row["id"] in selected
    ]:
        raise ValueError("Resolve the entire unchanged eligible packet, without selection")
    for row in new_mapping:
        for judge in new_judges:
            rating = judge[row["id"]]
            if cleared(rating, row["source"]):
                require_primary_evidence(rating)
    preserved = [
        (prefix, directory, preservation_files(directory))
        for prefix, directory in (("original-", base), ("resolution-", resolution))
    ]
    destination(root, output)
    (output / "evidence-composition-source.py").write_bytes(Path(__file__).read_bytes())
    for prefix, directory, names in preserved:
        for name in names:
            (output / (prefix + name)).write_bytes((directory / name).read_bytes())
    write_packet(
        output,
        {"mapping.jsonl": mapping, "blind.jsonl": blind},
        base_protocol
        | {
            "version": "semantic-admission-evidence-2",
            "primary_evidence_required_ids": sorted(selected),
            "original_decision_sha256": digest(base / "decision.json"),
            "resolution_decision_sha256": digest(resolution / "decision.json"),
            "scope": "Original full census, with every correctness-only uncertainty independently rereviewed against primary evidence. Existing labels preserved in original evidence; composite rows are verbatim actual assessments. This is not approval review.",
        },
    )
    for slot in (0, 1):
        with (output / f"curator-{'ab'[slot]}.jsonl").open("x") as stream:
            for row in mapping:
                ident = row["id"]
                judgment = (new_judges if ident in selected else old_judges)[slot][ident]
                stream.write(json.dumps(judgment, ensure_ascii=False) + "\n")
    # Fresh text-to-token verification and ordinary census finalization are mandatory.


def union(root: Path, components: list[Path], output: Path) -> None:
    """Join disjoint completed censuses, preserving their actual independent judgments."""
    if len(components) < 2:
        raise ValueError("At least two complete admission components required")
    packets = [inputs(component) for component in components]
    protocols = [json.loads((p / "protocol.json").read_text()) for p in components]
    if len({p["parent_manifest_sha256"] for p in protocols}) != 1:
        raise ValueError("Admission components belong to different parents")
    combined: dict[str, list[dict[str, Any]]] = {
        name: [] for name in ("mapping.jsonl", "blind.jsonl", "curator-a.jsonl", "curator-b.jsonl")
    }
    identities: set[str] = set()
    documents: set[str] = set()
    required_evidence = set()
    for (mapping, blind, judges), protocol in zip(packets, protocols):
        ids = {r["id"] for r in mapping}
        docs = {r["document_id"] for r in mapping}
        if identities & ids or documents & docs:
            raise ValueError("Admission components overlap; do not review or count replacements")
        identities.update(ids)
        documents.update(docs)
        required_evidence.update(primary_evidence_ids(protocol, ids))
        combined["mapping.jsonl"].extend(mapping)
        combined["blind.jsonl"].extend(blind)
        for slot in (0, 1):
            combined[f"curator-{'ab'[slot]}.jsonl"].extend(judges[slot][r["id"]] for r in mapping)
    preserved = [preservation_files(p) for p in components]
    destination(root, output, 256 * 1024**2)
    (output / "admission-union-source.py").write_bytes(Path(__file__).read_bytes())
    for index, (component, names) in enumerate(zip(components, preserved)):
        for name in names:
            (output / f"component-{index}-{name}").write_bytes((component / name).read_bytes())
    protocol = protocols[0] | {
        "version": "semantic-admission-union-1",
        "require_primary_evidence": False,
        "primary_evidence_required_ids": sorted(required_evidence),
        "component_decision_sha256": [digest(p / "decision.json") for p in components],
        "scope": "Disjoint complete censuses of the same immutable parent; actual judgments concatenated without substitutions or changed labels. This is not approval review. Fresh full token verification and finalization required.",
    }
    write_packet(output, combined, protocol)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "compose", "union"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--resolution", type=Path)
    parser.add_argument("--component", type=Path, action="append", default=[])
    args = parser.parse_args()
    if args.action == "union":
        if len(args.component) < 2:
            parser.error("At least two --component paths required")
        union(args.root, args.component, args.output)
    elif args.action == "prepare":
        if args.policy is None or args.base is None:
            parser.error("--base and --policy required")
        prepare(args.root, args.base, args.output, args.policy)
    else:
        if args.resolution is None or args.base is None:
            parser.error("--base and --resolution required")
        compose(args.root, args.base, args.resolution, args.output)


if __name__ == "__main__":
    main()
