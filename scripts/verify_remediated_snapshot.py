"""Independent subsequence proof for rejection-only remediation, plus full ID audit."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from codexa.data.production import atomic, digest, verify
from codexa.data.v2_finalize import token_documents
from codexa.data.v2_storage import Storage
from scripts.audit_v2_snapshot import audit


def check(root: Path, parent: Path, snapshot: Path) -> dict:
    Storage.load(root).check()
    old, new = verify(parent), verify(snapshot)
    if new["identity"]["remediation_parent_manifest"] != digest(parent / "manifest.json"):
        raise ValueError("Parent manifest identity mismatch")
    if new["tokenizer_hash"] != old["tokenizer_hash"]:
        raise ValueError("Frozen tokenizer changed")
    if new["identity"]["quarantine_sha256"] != digest(snapshot / "quarantine.jsonl"):
        raise ValueError("Quarantine ledger changed")
    if new["identity"]["remediation_policy"] != digest(snapshot / "policy.json"):
        raise ValueError("Remediation policy differs from manifest")
    policy = json.loads((snapshot / "policy.json").read_text())
    source = snapshot / "policy-source.py"
    if not source.exists():
        source = Path("codexa/data/remediate.py")
    if policy["code_sha256"] != digest(source):
        raise ValueError("Remediation source differs from executed policy")
    if policy.get("quality_code_sha256") and policy["quality_code_sha256"] != digest(
        snapshot / "quality-source.py"
    ):
        raise ValueError("Quality screening source differs from executed policy")
    if policy.get("math_code_sha256") and policy["math_code_sha256"] != digest(
        snapshot / "math-source.py"
    ):
        raise ValueError("Symbolic screening source differs from executed policy")
    if policy.get("representation_code_sha256") and policy["representation_code_sha256"] != digest(
        snapshot / "representation-source.py"
    ):
        raise ValueError("Representation screening source differs from executed policy")
    if policy.get("semantic_code_sha256") and (
        policy["semantic_code_sha256"] != digest(snapshot / "semantic-source.py")
        or policy["semantic_admission_decision_sha256"]
        != digest(snapshot / "semantic-decision.json")
    ):
        raise ValueError("Semantic policy or source changed")
    if policy.get("semantic_code_sha256"):
        semantic_decision = json.loads((snapshot / "semantic-decision.json").read_text())
        if semantic_decision["packet_integrity_sha256"] != digest(
            snapshot / "semantic-packet-integrity.json"
        ) or semantic_decision["admission_sha256"] != digest(snapshot / "semantic-admission.jsonl"):
            raise ValueError("Semantic ledger or packet changed")
        if semantic_decision["token_verification_sha256"] != digest(
            snapshot / "semantic-token-verification.json"
        ):
            raise ValueError("Semantic input proof changed")
        for name, sha in json.loads(
            (snapshot / "semantic-packet-integrity.json").read_text()
        ).items():
            if Path(name).name != name or digest(snapshot / ("semantic-" + name)) != sha:
                raise ValueError("Semantic packet component changed")
        for name, sha in semantic_decision["reviewer_hashes"].items():
            if Path(name).name != name or digest(snapshot / ("semantic-" + name)) != sha:
                raise ValueError("Semantic reviewer evidence changed")
    quarantined = {}
    for row in map(json.loads, (snapshot / "quarantine.jsonl").read_text().splitlines()):
        if row["id"] in quarantined or not row.get("reason") or not row.get("source_record"):
            raise ValueError("Invalid quarantine provenance")
        quarantined[row["id"]] = row
    original = {s: token_documents(parent, entries) for s, entries in old["splits"].items()}
    retained = {s: token_documents(snapshot, entries) for s, entries in new["splits"].items()}
    source_tokens: Counter[str] = Counter()
    checked = removed = 0
    with (parent / "documents.jsonl").open() as f, (snapshot / "documents.jsonl").open() as g:
        for row in map(json.loads, f):
            before = next(original[row["split"]])
            if len(before) != row["tokens"]:
                raise ValueError("Original mapping/offset mismatch")
            if row["id"] in quarantined:
                if any(
                    quarantined[row["id"]].get(k) != row[k] for k in ("source", "split", "tokens")
                ):
                    raise ValueError("Quarantine identity differs from parent")
                removed += 1
                continue
            line = next(g, None)
            if line is None or json.loads(line) != row:
                raise ValueError("Retained identity, source, split or order changed")
            after = next(retained[row["split"]])
            if not np.array_equal(before, after):
                raise ValueError("Retained token bytes changed")
            if row["split"] == "train":
                source_tokens[row["source"]] += row["tokens"]
            checked += 1
            if checked % 250000 == 0:
                print(json.dumps({"verified_documents": checked, "removed": removed}), flush=True)
        if next(g, None) is not None:
            raise ValueError("Unexpected added document")
    if removed != len(quarantined):
        raise ValueError("Quarantined document not found in original snapshot")
    for streams in (original, retained):
        if any(next(stream, None) is not None for stream in streams.values()):
            raise ValueError("Unexpected added token document")
    if dict(source_tokens) != new["source_tokens"]:
        raise ValueError("Source token accounting changed")
    ids = audit(snapshot, root)
    return {
        "status": "passed",
        "parent_manifest_sha256": digest(parent / "manifest.json"),
        "manifest_sha256": digest(snapshot / "manifest.json"),
        "retained_documents_bitwise_compared": checked,
        "quarantined_documents": removed,
        "no_added_or_rewritten_text": True,
        "source_split_order_preserved": True,
        "historical_evaluation_exclusion_inherited_by_exact_subsequence": True,
        "token_audit": ids,
        "training_started": False,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "parent", "snapshot", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    result = check(a.root, a.parent, a.snapshot)
    atomic(a.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
