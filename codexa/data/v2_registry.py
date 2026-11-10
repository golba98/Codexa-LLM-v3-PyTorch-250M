"""Explicit append-only source-registry transactions; unrelated edits are rejected."""

import copy
import json
import os
from typing import Any

from codexa.data.production import atomic, digest, sync_directory
from codexa.data.v2_acquire import validate_spec
from codexa.data.v2_index import Index


def validate_extension(old: dict[str, Any], new: dict[str, Any]) -> None:
    count = len(old["sources"])
    if new["sources"][:count] != old["sources"] or len(new["sources"]) <= count:
        raise ValueError("An extension must preserve every prior pinned source entry")
    before = copy.deepcopy(old["config"])
    after = copy.deepcopy(new["config"])
    prior = before.pop("sources")
    extended = after.pop("sources")
    if before != after or extended[: len(prior)] != prior:
        raise ValueError("Existing source configuration changed")
    old_names = {s["name"] for s in prior}
    new_names = [s["name"] for s in extended[len(prior) :]]
    if old_names.intersection(new_names) or len(set(new_names)) != len(new_names):
        raise ValueError("Extension aliases must be new and distinct")
    hashes = {s["sha256"] for s in old["sources"]}
    for source in new["sources"][count:]:
        validate_spec(source)
        if source["name"] not in new_names or source["sha256"] in hashes:
            raise ValueError("Unregistered alias or duplicate source payload")
        hashes.add(source["sha256"])


def recover(index: Index) -> None:
    pending = index.value("registry_pending")
    if not pending:
        return
    root = index.storage.root
    directory = root / "reports/registries"
    old_path = directory / (pending["old"] + ".json")
    new_path = directory / (pending["new"] + ".json")
    if digest(old_path) != pending["old"] or digest(new_path) != pending["new"]:
        raise ValueError("Registry transaction snapshot integrity failed")
    old = json.loads(old_path.read_text())
    new = json.loads(new_path.read_text())
    validate_extension(old, new)
    actual = digest(root / "source-registry.json")
    if actual not in (pending["old"], pending["new"]):
        raise ValueError("Unrelated registry mutation during extension")
    identity = index.value("preparation_identity")
    if identity["source_registry"] not in (pending["old"], pending["new"]):
        raise ValueError("Registry extension parent fingerprint mismatch")
    atomic(root / "source-registry.json", new)
    if digest(root / "source-registry.json") != pending["new"]:
        raise ValueError("Registry transaction serialization mismatch")
    identity["source_registry"] = pending["new"]
    chain = index.value("registry_revisions", [])
    if pending not in chain:
        chain.append(pending)
    index.set("registry_revisions", chain)
    index.set("preparation_identity", identity)
    index.set("registry_pending", None)
    index.db.commit()


def extend(index: Index, addition: dict[str, Any]) -> dict[str, Any]:
    recover(index)
    root = index.storage.root
    old_path = root / "source-registry.json"
    old_hash = digest(old_path)
    identity = index.value("preparation_identity")
    if not identity or identity["source_registry"] != old_hash:
        raise ValueError("Extension requires the exact current preparation identity")
    old = json.loads(old_path.read_text())
    new = copy.deepcopy(old)
    new["sources"].extend(addition["sources"])
    new["config"]["sources"].extend(addition["config"]["sources"])
    validate_extension(old, new)
    directory = root / "reports/registries"
    directory.mkdir(exist_ok=True)
    original = directory / (old_hash + ".json")
    if original.exists() and digest(original) != old_hash:
        raise ValueError("Historical registry changed")
    if not original.exists():
        with original.open("xb") as output:
            output.write(old_path.read_bytes())
            output.flush()
            os.fsync(output.fileno())
    candidate = root / "tmp/registry-candidate.json"
    atomic(candidate, new)
    new_hash = digest(candidate)
    target = directory / (new_hash + ".json")
    if target.exists() and digest(target) != new_hash:
        raise ValueError("Registry extension snapshot changed")
    if not target.exists():
        with target.open("xb") as output:
            output.write(candidate.read_bytes())
            output.flush()
            os.fsync(output.fileno())
    sync_directory(directory)
    pending = {
        "old": old_hash,
        "new": new_hash,
        "policy": "explicit append-only source addition; native training fingerprints unchanged",
    }
    index.set("registry_pending", pending)
    index.db.commit()
    recover(index)
    return pending
