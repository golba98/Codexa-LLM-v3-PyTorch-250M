"""Gate, launch and evaluate the explicitly authorized language continuation."""

import argparse
import fcntl
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from codexa.data.language_protocol import SUPPORTED_VERSIONS, V4_VERSION
from codexa.data.production import atomic, digest, verify
from codexa.data.quality_protocol import VERSION as QUALITY_VERSION
from codexa.data.quality_protocol import validate_specification
from codexa.data.v2_storage import GIB, VOLUME_UUID, Storage
from scripts.review_continuation_corpus import adjudicate

PARENT = Path("/mnt/codexa/pytorch/codexa-v3/pretraining/20261103-500m-233201/final/checkpoint.pt")
PARENT_SHA = "54fca007f272fb3784c759d18db44912091227aedc233b62c9731514b1835c73"
TOKENIZER = PARENT.parent.parent / "tokenizer/tokenizer.json"
BUDGET = 50000000
GPU_LEASE_ROOT = PARENT.parents[3] / "dataset-v2/20261104-133814"


def continuation_budget(protocol: dict[str, Any]) -> int:
    """Read the exact budget from the same authorization that binds the quality panel."""
    from codexa.data.language_protocol import validate_authorization

    if protocol.get("version") not in SUPPORTED_VERSIONS:
        return BUDGET
    authorization = validate_authorization(Path(protocol["authorization_path"]), protocol)
    return int(authorization["additional_target_tokens"])


def run_budget(record: dict[str, Any]) -> int:
    """Keep historical controls compatible without accepting arbitrary new run budgets."""
    checks = record["checks"]
    budget = checks.get("additional_targets", BUDGET)
    if type(budget) is not int or (
        budget != BUDGET
        and not (budget == 100_000_000 and checks.get("quality_protocol") == V4_VERSION)
    ):
        raise ValueError("Run budget is outside the authorized 50M/100M continuation scope")
    return int(budget)


def expected_accounting(budget: int) -> tuple[int, int, int]:
    """Native exact-budget continuation: updates, cumulative targets, microsteps."""
    return (budget + 32767) // 32768, 500006912 + budget, (budget + 2047) // 2048


def quality_gate(data: Path, review: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Dispatch only to explicitly authorized protocol versions."""
    protocol = json.loads((review / "protocol.json").read_text())
    if protocol.get("version") in SUPPORTED_VERSIONS:
        language_quality_gate(data, review, manifest, protocol)
        return protocol
    legacy_quality_gate(data, review, manifest)
    return protocol


def language_quality_gate(
    data: Path, review: Path, manifest: dict[str, Any], protocol: dict[str, Any]
) -> None:
    from codexa.data.language_protocol import adjudicate as language_adjudicate
    from codexa.data.language_protocol import validate_authorization
    from codexa.data.language_protocol import validate_specification as validate_language
    from codexa.repository import source_identity
    from scripts.combine_quality_reviews import verify_reviewers
    from scripts.verify_review_packet import review_directories

    validate_language(protocol)
    authorization = Path(protocol["authorization_path"])
    if digest(authorization) != protocol["authorization_sha256"]:
        raise ValueError("Language protocol authorization changed")
    validate_authorization(authorization, protocol)
    manifest_sha = digest(data / "manifest.json")
    if protocol["manifest_sha256"] != manifest_sha:
        raise ValueError("Quality review belongs to another dataset")
    if set(protocol["sources"]) != set(manifest["source_tokens"]):
        raise ValueError("Language source panels differ from the actual corpus")
    independent = json.loads((data / "independent-verification.json").read_text())
    verifier_path = Path(__file__).resolve().parents[1] / "codexa/data/language.py"
    if (
        independent.get("status") != "passed"
        or independent.get("manifest_sha256") != manifest_sha
        or independent.get("tokenizer_sha256") != manifest["tokenizer_hash"]
        or not verifier_path.is_file()
        or independent.get("verifier_sha256") != digest(verifier_path)
        or independent.get("source", {}).get("digest") != source_identity()["digest"]
        or not all(
            independent.get(field) is True
            for field in (
                "all_tokens_reencoded",
                "protection_verified",
                "split_dedup_verified",
                "prior_exposure_verified",
                "rights_verified",
            )
        )
    ):
        raise ValueError("Independent language corpus verification is missing or incompatible")
    integrity = json.loads((review / "packet-integrity.json").read_text())
    if not {"protocol.json", "mapping.jsonl", "blind.jsonl", "sampling-log.json"} <= set(integrity):
        raise ValueError("Language review integrity lacks required inputs")
    for name, sha in integrity.items():
        if Path(name).name != name or digest(review / name) != sha:
            raise ValueError("Review packet integrity changed")
    decision = json.loads((review / "decision.json").read_text())
    rendering = json.loads((review / "rendering.json").read_text())
    if (
        decision.get("ratings_sha256") != digest(review / "ratings.jsonl")
        or decision.get("protocol_sha256") != digest(review / "protocol.json")
        or decision.get("rendering_sha256") != digest(review / "rendering.json")
        or rendering.get("rendered_blind_sha256") != digest(review / "rendered-blind.jsonl")
        or rendering.get("original_blind_sha256") != digest(review / "blind.jsonl")
    ):
        raise ValueError("Frozen language ratings or rendering changed")
    mapping = list(map(json.loads, (review / "mapping.jsonl").read_text().splitlines()))
    ratings = list(map(json.loads, (review / "ratings.jsonl").read_text().splitlines()))
    result = language_adjudicate(mapping, ratings, protocol)
    if not result["training_allowed"] or not decision.get("training_allowed"):
        raise ValueError("Mandatory complete language quality review has not passed")
    verify_reviewers(review)
    language_freshness_gate(data, review, manifest, protocol)
    text_proof = json.loads((review / "sample-token-verification.json").read_text())
    if (
        text_proof.get("status") != "passed"
        or text_proof.get("manifest_sha256") != manifest_sha
        or text_proof.get("tokenizer_sha256") != manifest["tokenizer_hash"]
        or text_proof.get("documents") != 100 * len(protocol["sources"])
        or text_proof.get("windows") != 300
        or text_proof.get("packets")
        != [
            {
                "packet_integrity_sha256": digest(p / "packet-integrity.json"),
                "rendered_blind_sha256": digest(p / "rendered-blind.jsonl"),
            }
            for p in review_directories(review)
        ]
        or text_proof.get("verifier_sha256")
        != digest(Path(__file__).with_name("verify_review_packet.py"))
    ):
        raise ValueError("Blinded language document/token proof is missing or stale")


def language_freshness_gate(
    data: Path, review: Path, manifest: dict[str, Any], protocol: dict[str, Any]
) -> None:
    """Recheck the frozen eligibility evidence and every registered inspection packet."""
    from codexa.data.language_protocol import validate_authorization
    from scripts.review_continuation_corpus import require_known_census_exclusions

    record = validate_authorization(Path(protocol["authorization_path"]), protocol)
    supplied: dict[str, list[Path]] = {}
    for key in ("excluded_reviews", "excluded_inspection_censuses"):
        supplied[key] = []
        for packet in protocol.get(key, []):
            path = Path(packet["path"])
            if digest(path / "packet-integrity.json") != packet["integrity_sha256"]:
                raise ValueError("Historical inspected packet changed after language sampling")
            integrity = json.loads((path / "packet-integrity.json").read_text())
            for name, sha in integrity.items():
                if Path(name).name != name or digest(path / name) != sha:
                    raise ValueError("Historical inspected packet contents changed")
            supplied[key].append(path)
    roots = [*(Path(p) for p in record["historical_roots"]), data.parent.parent]
    recorded_development = protocol.get("excluded_development_files", [])
    recorded_paths: set[Path] = set()
    for item in recorded_development:
        path = Path(item["path"])
        if digest(path) != item["sha256"]:
            raise ValueError("Inspected development evidence changed after approval sampling")
        recorded_paths.add(path.resolve())
    known_development: set[Path] = set()
    for root in roots:
        paths = set((root / "reports/profiles").glob("*.jsonl"))
        for development in (root / "reports").glob("development*"):
            if development.is_dir():
                paths.update(development.rglob("*.jsonl"))
        examples = root / "reports/cleaning-examples.jsonl"
        if examples.exists():
            paths.add(examples)
        for path in paths:
            known_development.add(path.resolve())
            if path.with_suffix(".json").exists():
                known_development.add(path.with_suffix(".json").resolve())
    if recorded_paths != known_development:
        raise ValueError("Language approval omitted newly inspected development evidence")
    for root in roots:
        require_known_census_exclusions(root, supplied["excluded_inspection_censuses"])
        known = {p.parent.resolve() for p in (root / "reviews").rglob("packet-integrity.json")}
        known.discard(review.resolve())
        if not known <= {p.resolve() for p in supplied["excluded_reviews"]}:
            raise ValueError("Language approval omitted a known inspected review")
    sampling = json.loads((review / "sampling-log.json").read_text())
    if sampling.get("accepted_windows") != 300:
        raise ValueError("Language sampling window evidence is incomplete")
    if len(sampling.get("window_overlap_rejections", [])) > sampling.get("window_draws", 0) * 0.05:
        raise ValueError("Inspected-passage exclusions distort the training-window sample")
    for source in protocol["sources"]:
        total = sampling["source_documents"][source]
        tokens = sampling["source_tokens"][source]
        excluded = sampling["excluded_source_documents"].get(source, 0)
        excluded_tokens = sampling["excluded_source_tokens"].get(source, 0)
        if (
            type(total) is not int
            or type(tokens) is not int
            or total < 100
            or tokens <= 0
            or tokens != manifest["source_tokens"][source]
            or type(excluded) is not int
            or not 0 <= excluded <= total * 0.05
            or type(excluded_tokens) is not int
            or not 0 <= excluded_tokens <= tokens * 0.05
            or sampling["eligible_source_documents"][source] != total - excluded
        ):
            raise ValueError("Language approval eligibility distorts the corpus distribution")
        if (
            sampling.get("document_passage_exclusions", {}).get(source, 0)
            > sampling.get("document_passage_candidates", {}).get(source, 0) * 0.05
        ):
            raise ValueError("Inspected-passage exclusions distort the source document sample")


def legacy_quality_gate(data: Path, review: Path, manifest: dict[str, Any]) -> None:
    independent = json.loads((data / "independent-verification.json").read_text())
    if (
        independent.get("status") != "passed"
        or independent["manifest_sha256"] != digest(data / "manifest.json")
        or not independent.get("no_added_or_rewritten_text")
    ):
        raise ValueError("Independent remediation verification is missing or incompatible")
    decision = json.loads((review / "decision.json").read_text())
    protocol = json.loads((review / "protocol.json").read_text())
    validate_specification(protocol)
    if protocol["version"] != QUALITY_VERSION or protocol["kind"] != "primary":
        raise ValueError("Current strict primary protocol is required")
    integrity = json.loads((review / "packet-integrity.json").read_text())
    for name, sha in integrity.items():
        if digest(review / name) != sha:
            raise ValueError("Review packet integrity changed")
    if decision["ratings_sha256"] != digest(review / "ratings.jsonl") or decision[
        "protocol_sha256"
    ] != digest(review / "protocol.json"):
        raise ValueError("Frozen ratings or protocol changed")
    rendering = json.loads((review / "rendering.json").read_text())
    if decision.get("rendering_sha256") != digest(review / "rendering.json") or rendering[
        "rendered_blind_sha256"
    ] != digest(review / "rendered-blind.jsonl"):
        raise ValueError("Frozen review rendering changed")
    mapping = [json.loads(x) for x in (review / "mapping.jsonl").read_text().splitlines()]
    ratings = [json.loads(x) for x in (review / "ratings.jsonl").read_text().splitlines()]
    scored = adjudicate(mapping, ratings, protocol)
    if {r["source"] for r in mapping if r["kind"] == "document"} != set(manifest["source_tokens"]):
        raise ValueError("Source review panels differ from the actual corpus")
    if (
        len(mapping) != 500
        or not scored["training_allowed"]
        or not decision["training_allowed"]
        or not protocol.get("complete_required")
    ):
        raise ValueError("Mandatory complete quality review has not passed")
    if protocol["manifest_sha256"] != digest(data / "manifest.json"):
        raise ValueError("Quality review belongs to another dataset")
    from scripts.combine_quality_reviews import verify_reviewers

    verify_reviewers(review)
    extended = review / "extended"
    extended_protocol = json.loads((extended / "protocol.json").read_text())
    validate_specification(extended_protocol)
    if extended_protocol["kind"] != "extended" or extended_protocol["manifest_sha256"] != digest(
        data / "manifest.json"
    ):
        raise ValueError("Larger independent review is missing or stale")
    extended_integrity = json.loads((extended / "packet-integrity.json").read_text())
    for name, sha in extended_integrity.items():
        if digest(extended / name) != sha:
            raise ValueError("Larger review packet changed")
    extended_ratings = [
        json.loads(x) for x in (extended / "ratings.jsonl").read_text().splitlines()
    ]
    extended_mapping = [
        json.loads(x) for x in (extended / "mapping.jsonl").read_text().splitlines()
    ]
    extended_decision = json.loads((extended / "decision.json").read_text())
    extended_rendering = json.loads((extended / "rendering.json").read_text())
    if (
        extended_decision.get("rendering_sha256") != digest(extended / "rendering.json")
        or extended_rendering["rendered_blind_sha256"] != digest(extended / "rendered-blind.jsonl")
        or extended_rendering["original_blind_sha256"] != digest(extended / "blind.jsonl")
    ):
        raise ValueError("Larger review rendering changed")
    if extended_decision.get("ratings_sha256") != digest(
        extended / "ratings.jsonl"
    ) or extended_decision.get("protocol_sha256") != digest(extended / "protocol.json"):
        raise ValueError("Larger review ratings or protocol changed")
    if not adjudicate(extended_mapping, extended_ratings, extended_protocol)[
        "training_allowed"
    ] or not extended_decision.get("training_allowed"):
        raise ValueError("Larger independent review failed")
    verify_reviewers(extended)
    text_proof = json.loads((review / "sample-token-verification.json").read_text())
    if (
        text_proof.get("status") != "passed"
        or text_proof.get("manifest_sha256") != digest(data / "manifest.json")
        or text_proof.get("documents") != 600
        or text_proof.get("windows") != 300
        or text_proof.get("packets")
        != [
            {
                "packet_integrity_sha256": digest(p / "packet-integrity.json"),
                "rendered_blind_sha256": digest(p / "rendered-blind.jsonl"),
            }
            for p in (review, extended)
        ]
        or text_proof.get("verifier_sha256")
        != digest(Path(__file__).with_name("verify_review_packet.py"))
    ):
        raise ValueError("Blinded document/token identity proof is missing or stale")


def language_baseline_review_gate(root: Path) -> None:
    """Poor baseline scores are allowed; missing independent assessment is not."""
    from codexa.evaluation.language import load_fixtures, score_reviews

    packet = root / "benchmarks/language-baseline-review"
    plan = json.loads((root / "benchmarks/baseline-review-plan.json").read_text())
    manifest = json.loads((packet / "manifest.json").read_text())
    scored = json.loads((packet / "score.json").read_text())
    fixtures = root / "benchmarks/language-v1"
    output = root / "benchmarks" / ("language-baseline-" + PARENT_SHA[:16])
    expected = {r["id"] + ":sample" for r in load_fixtures(fixtures) if r["split"] == "test"}
    if (
        plan.get("selection")
        != {"split": "test", "setting": "sample", "seed": 42, "max_tokens": 256}
        or plan.get("reviewers_required") != 2
        or plan.get("fixture_manifest_sha256") != digest(fixtures / "manifest.json")
        or manifest.get("records") != 100
        or manifest.get("fixture_manifest_sha256") != digest(fixtures / "manifest.json")
        or manifest.get("inputs")
        != [{"path": str(output.resolve()), "outputs_sha256": digest(output / "outputs.jsonl")}]
        or manifest.get("selection", {}).get("split") != "test"
        or manifest.get("selection", {}).get("setting") != "sample"
        or {r["key"] for r in scored["records"]} != expected
        or any(r["checkpoint_sha256"] != PARENT_SHA for r in scored["records"])
    ):
        raise ValueError("Independent baseline review differs from preregistered coverage")
    reviewers = {name: Path(row["path"]) for name, row in scored["reviewers"].items()}
    if score_reviews(packet, reviewers) != scored:
        raise ValueError("Independent baseline scores or reviewer evidence changed")


def language_baseline_gate(root: Path) -> dict[str, Any]:
    """Require all frozen language responses before updating the parent model."""
    from codexa.evaluation.language import SETTINGS, load_fixtures

    fixtures = root / "benchmarks/language-v1"
    items = load_fixtures(fixtures)
    output = root / "benchmarks" / ("language-baseline-" + PARENT_SHA[:16])
    verification = json.loads((output / "verification.json").read_text())
    identity = json.loads((output / "identity.json").read_text())
    expected = {r["id"] + ":" + setting for r in items for setting in ("greedy", "sample")}
    records = list(map(json.loads, (output / "outputs.jsonl").read_text().splitlines()))
    if (
        verification.get("complete") is not True
        or verification.get("records") != len(expected)
        or verification.get("expected_records") != len(expected)
        or verification.get("checkpoint_sha256") != PARENT_SHA
        or verification.get("fixture_manifest_sha256") != digest(fixtures / "manifest.json")
        or verification.get("tokenizer_sha256") != digest(TOKENIZER)
        or verification.get("outputs_sha256") != digest(output / "outputs.jsonl")
        or verification.get("weights_unchanged") is not True
        or verification.get("backward_calls") != 0
        or verification.get("optimizer_updates") != 0
        or verification.get("forbidden_attempts") != []
        or verification.get("settings") != SETTINGS
        or any(
            identity.get(k) != verification.get(k)
            for k in ("checkpoint_sha256", "fixture_manifest_sha256", "settings", "device")
        )
        or {r["key"] for r in records} != expected
        or len(records) != len(expected)
        or any(r.get("settings") != SETTINGS for r in records)
    ):
        raise ValueError("Complete preserved language baseline is missing or stale")
    budget = json.loads((root / "benchmarks/language-pair-budget.json").read_text())
    if (
        budget.get("limit_seconds") != 14400
        or not 0 <= budget.get("used_seconds", -1) < 14400
        or budget.get("active_since")
    ):
        raise ValueError("Combined language evaluation budget is unavailable")
    language_baseline_review_gate(root)
    return {
        "comparison_keys": sorted(expected),
        "outputs_sha256": verification["outputs_sha256"],
        "fixture_manifest_sha256": verification["fixture_manifest_sha256"],
    }


def preflight(
    root: Path, data: Path, bundle: Path, review: Path, require_smoke: bool = True
) -> dict[str, Any]:
    from codexa.continuation import validate
    from codexa.repository import source_identity

    Storage.load(root).check(64 * 1024**2)
    verification_path = data / "source-verification.json"
    code_verification = json.loads(verification_path.read_text())
    if (
        code_verification.get("status") != "passed"
        or code_verification["source"]["digest"] != source_identity()["digest"]
    ):
        raise ValueError("Final repository verification is missing or stale")
    if shutil.disk_usage(root).free < (40 + 18) * GIB:
        raise OSError("Insufficient free space for reserve plus bounded training allocation")
    manifest = verify(data)
    protocol = quality_gate(data, review, manifest)
    budget = continuation_budget(protocol)
    report = validate(bundle / "initial.pt", PARENT, data, TOKENIZER, PARENT_SHA)
    if (
        report["phase"].get("additional_target_tokens") != budget
        or report["phase"]["optimizer_policy"] != "preserve"
        or report["phase"]["parent_tokens"] != 500006912
        or report["phase"]["parent_updates"] != 15259
        or report.get("optimizer_parameter_states") != 170
    ):
        raise ValueError(
            "Bundle must preserve the 500M optimizer and match the exact authorized target budget"
        )
    verification = json.loads((bundle / "verification.json").read_text())
    if verification["checkpoint_sha256"] != report["checkpoint_sha256"]:
        raise ValueError("Bundle integrity record changed")
    if require_smoke:
        smoke_report = json.loads((bundle / "cuda-smoke.json").read_text())
        if (
            smoke_report.get("status") != "passed"
            or smoke_report.get("dataset_sha256") != digest(data / "manifest.json")
            or smoke_report.get("bundle_sha256") != report["checkpoint_sha256"]
            or smoke_report.get("source_digest") != source_identity()["digest"]
        ):
            raise ValueError("Full-shape CUDA smoke evidence is missing or stale")
    if protocol["version"] in SUPPORTED_VERSIONS:
        baseline = language_baseline_gate(root)
        comparison_keys = baseline["comparison_keys"]
    else:
        old = root / "benchmarks" / ("baseline-" + PARENT_SHA[:16])
        baseline_verification = json.loads((old / "verification.json").read_text())
        proxy = json.loads((old / "answer-proxies.json").read_text())
        if baseline_verification["fixture_manifest_sha256"] != digest(
            root / "benchmarks/v3/manifest.json"
        ) or proxy["outputs_sha256"] != digest(old / "outputs.jsonl"):
            raise ValueError("Frozen comparison baseline identity changed")
        comparison_keys = []
        old_rows = [json.loads(x) for x in (old / "outputs.jsonl").read_text().splitlines()]
        for kind in ("sentence", "fact", "context", "paragraph", "math"):
            items = sorted({r["item_id"] for r in old_rows if r["kind"] == kind})[:4]
            comparison_keys.extend(
                r["key"]
                for r in old_rows
                if r["item_id"] in items
                and r["setting"] in ("greedy", "sample")
                and r["seed"] == 42
            )
    return {
        "status": "passed",
        "parent_sha256": PARENT_SHA,
        "bundle_sha256": report["checkpoint_sha256"],
        "dataset_sha256": digest(data / "manifest.json"),
        "tokenizer_sha256": manifest["tokenizer_hash"],
        "decision_sha256": digest(review / "decision.json"),
        "source": source_identity(),
        "free_bytes": shutil.disk_usage(root).free,
        "additional_targets": budget,
        "cumulative_target": 500006912 + budget,
        "comparison_keys": comparison_keys,
        "quality_protocol": protocol["version"],
    }


def launch(
    root: Path, data: Path, bundle: Path, review: Path, *, defer_postrun: bool = False
) -> None:
    checks = preflight(root, data, bundle, review)
    budget = run_budget({"checks": checks})
    label = f"exact{budget // 1_000_000}m"
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run = PARENT.parents[2] / (stamp + "-continuation-" + label)
    run.mkdir(exist_ok=False)
    session = "codexa-v3-" + label + "-" + stamp
    tmux = ["tmux", "-S", str(run / "tmux.sock")]
    env = {
        k: v
        for k, v in os.environ.items()
        if k.endswith(("_CACHE", "_CACHE_DIR"))
        or k
        in (
            "TMPDIR",
            "TMP",
            "TEMP",
            "SQLITE_TMPDIR",
            "HF_HOME",
            "TORCH_HOME",
            "XDG_CACHE_HOME",
            "MPLCONFIGDIR",
            "PYTHONPYCACHEPREFIX",
            "PYTHONDONTWRITEBYTECODE",
            "CUDA_CACHE_PATH",
            "TOKENIZERS_PARALLELISM",
            "CUBLAS_WORKSPACE_CONFIG",
        )
    }
    record = {
        "root": str(root),
        "data": str(data),
        "bundle": str(bundle),
        "review": str(review),
        "run": str(run),
        "session": session,
        "tmux_socket": str(run / "tmux.sock"),
        "repo": str(Path.cwd()),
        "python": sys.executable,
        "checks": checks,
        "defer_postrun": defer_postrun,
        "environment": env,
        "created_epoch": time.time(),
    }
    atomic(run / "control.json", record)
    command = shlex.join(
        [sys.executable, "-B", str(Path(__file__).resolve()), "worker", "--run", str(run)]
    )
    shell = (
        command
        + '; printf "\\nSession finished. Reports remain in this run directory.\\n"; exec /bin/bash -i'
    )
    subprocess.run(
        [
            *tmux,
            "new-session",
            "-d",
            "-s",
            session,
            "-x",
            "140",
            "-y",
            "42",
            "-c",
            str(Path.cwd()),
            "/bin/bash",
            "-c",
            shell,
        ],
        check=True,
    )
    subprocess.run(
        [
            *tmux,
            "pipe-pane",
            "-t",
            session,
            "-o",
            "cat >> " + shlex.quote(str(run / "terminal-raw.log")),
        ],
        check=True,
    )
    terminal_status: dict[str, Any] = {
        "session": session,
        "window_verified": False,
        "attach_command": shlex.join([*tmux, "attach-session", "-t", session]),
    }
    try:
        terminal = subprocess.Popen(
            [
                "gnome-terminal",
                "--window",
                "--geometry=140x42",
                f"--title=Codexa v3 — exact {budget // 1_000_000}M continuation",
                "--",
                *tmux,
                "attach-session",
                "-t",
                session,
            ],
            start_new_session=True,
        )
        terminal_status["launcher_pid"] = terminal.pid
    except OSError as exc:
        terminal_status["error"] = str(exc)
    atomic(run / "terminal-request.json", terminal_status)
    print(json.dumps(record, indent=2))


def isolated_preflight(root: Path, data: Path, bundle: Path, review: Path) -> dict[str, Any]:
    # Native verification intentionally installs irreversible backward guards.
    # Keep those guards in a CPU child; the fresh CUDA smoke process must retain
    # normal autograd and optimizer entry points.
    code = (
        "import json,sys; from pathlib import Path; "
        "from scripts.continuation_healthcheck import preflight; "
        "print(json.dumps(preflight(*(Path(p) for p in sys.argv[1:]),require_smoke=False)))"
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", code, *(str(p) for p in (root, data, bundle, review))],
        capture_output=True,
        text=True,
        check=True,
        env=os.environ | {"CODEXA_NO_TRAINING": "1", "CUDA_VISIBLE_DEVICES": ""},
    )
    return json.loads(result.stdout)


def smoke(root: Path, data: Path, bundle: Path, review: Path) -> None:
    checks = isolated_preflight(root, data, bundle, review)
    import torch

    from codexa.config import Config
    from codexa.training.trainer import Trainer

    if not torch.cuda.is_available() or "4080" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Requested RTX 4080 is unavailable")
    output = bundle / "cuda-smoke"
    if output.exists() or (bundle / "cuda-smoke.json").exists():
        raise FileExistsError("Use a fresh bundle for a new smoke attempt")
    Storage.load(root).check(4 * GIB)
    os.environ.pop("CODEXA_NO_TRAINING", None)
    with (GPU_LEASE_ROOT / "tmp/gpu-evaluation.lock").open("a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        trainer = Trainer(
            Config.load(str(bundle / "config.yaml")),
            str(data),
            str(TOKENIZER),
            str(output),
            "cuda",
            checkpoint_volume_uuid=VOLUME_UUID,
        )
        trainer.resume(str(bundle / "initial.pt"))
        torch.cuda.reset_peak_memory_stats()
        trainer.train(until=2, ui="none")
        free, total = torch.cuda.mem_get_info()
        state = torch.load(output / "latest.pt", map_location="cpu", weights_only=True, mmap=True)
        if (
            trainer.step != 2
            or trainer.tokens != 500072448
            or len(state["optimizer"]["state"]) != 170
            or any(float(s["step"]) != 15261 for s in state["optimizer"]["state"].values())
        ):
            raise ValueError("CUDA smoke accounting failed")
        if free < GIB:
            raise RuntimeError("CUDA smoke left less than 1 GiB VRAM headroom")
        atomic(
            bundle / "cuda-smoke.json",
            {
                "status": "passed",
                "dataset_sha256": checks["dataset_sha256"],
                "bundle_sha256": checks["bundle_sha256"],
                "source_digest": checks["source"]["digest"],
                "updates": 2,
                "discarded_smoke_tokens": 65536,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "free_cuda_bytes": free,
                "total_cuda_bytes": total,
                "checkpoint_sha256": digest(output / "latest.pt"),
                "production_starts_from_untouched_initial_bundle": True,
            },
        )


def worker(run: Path) -> None:
    record = json.loads((run / "control.json").read_text())
    _root, data, bundle = (Path(record[k]) for k in ("root", "data", "bundle"))
    budget = run_budget(record)
    protocol = json.loads((Path(record["review"]) / "protocol.json").read_text())
    if continuation_budget(protocol) != budget:
        raise ValueError("Run budget differs from the bound language authorization")
    os.environ.update(record["environment"])
    os.environ.pop("CODEXA_NO_TRAINING", None)
    os.environ.pop("NO_COLOR", None)
    os.environ["TERM"] = "xterm-256color"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    import torch

    from codexa.config import Config
    from codexa.repository import source_identity
    from codexa.training.trainer import Trainer

    torch.set_num_threads(4)
    if source_identity()["digest"] != record["checks"]["source"]["digest"]:
        raise ValueError("Repository changed between preflight and worker launch")
    if not torch.cuda.is_available() or "4080" not in torch.cuda.get_device_name(0):
        raise RuntimeError("The requested RTX 4080 CUDA device is unavailable")
    for path, sha in (
        (bundle / "initial.pt", record["checks"]["bundle_sha256"]),
        (data / "manifest.json", record["checks"]["dataset_sha256"]),
    ):
        if digest(path) != sha:
            raise ValueError("Launch inputs changed after preflight")
    storage = Storage(run, VOLUME_UUID, 40 * GIB, 18 * GIB)
    storage.check(3 * GIB)

    class Monitor:
        trainer: Any = None
        failure: Exception | None = None

        def on_event(self, event: Any) -> None:
            try:
                self.check_event(event)
            except Exception as exc:
                # EventBus intentionally isolates ordinary UI observers. This
                # observer enforces safety: request a stop even if it is disabled.
                self.failure = exc
                self.trainer.stop = True
                raise

        def check_event(self, event: Any) -> None:
            if event.event == "step_completed":
                if event.step % 25 == 0:
                    storage.check(3 * GIB)
                if event.step == 2:
                    self.trainer.checkpoint()
            if event.event != "checkpoint_saved":
                return
            path = run / event.data["identifier"]
            state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            expected = 500006912 + min(event.step * 32768, budget)
            if state["tokens"] != expected or state["step"] != event.step:
                raise ValueError("Checkpoint counters invalid")
            if not all(bool(torch.isfinite(t).all()) for t in state["model"].values()):
                raise ValueError("Checkpoint model tensors are nonfinite")
            if any(
                float(s["step"]) != 15259 + event.step for s in state["optimizer"]["state"].values()
            ):
                raise ValueError("Real optimizer update counters invalid")
            for s in state["optimizer"]["state"].values():
                if (
                    not torch.isfinite(s["exp_avg"]).all()
                    or not torch.isfinite(s["exp_avg_sq"]).all()
                    or bool((s["exp_avg_sq"] < 0).any())
                ):
                    raise ValueError("Checkpoint optimizer tensors invalid")
            sha = digest(path)
            atomic(
                run / "verified-latest.json",
                {
                    "file": path.name,
                    "sha256": sha,
                    "step": event.step,
                    "tokens": state["tokens"],
                    "fingerprints": state["fingerprints"],
                },
            )
            if event.step in (250, 750, 1250):
                preserved = run / f"step-{event.step:04d}.pt"
                if not preserved.exists():
                    os.link(path, preserved)
            if event.step == 2:
                parent = torch.load(PARENT, map_location="cpu", weights_only=True, mmap=True)
                if torch.equal(
                    parent["model"]["embedding.weight"], state["model"]["embedding.weight"]
                ):
                    raise ValueError("No measured parameter change after optimizer updates")
                atomic(
                    run / "running-verification.json",
                    {
                        "status": "passed",
                        "pid": os.getpid(),
                        "device": torch.cuda.get_device_name(0),
                        "phase_steps": 2,
                        "cumulative_tokens": state["tokens"],
                        "optimizer_steps": 15261,
                        "weights_changed": True,
                        "allocated_cuda_bytes": torch.cuda.memory_allocated(),
                        "reserved_cuda_bytes": torch.cuda.memory_reserved(),
                        "checkpoint_sha256": sha,
                    },
                )

    with (GPU_LEASE_ROOT / "tmp/gpu-evaluation.lock").open("a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        monitor = Monitor()
        trainer = Trainer(
            Config.load(str(bundle / "config.yaml")),
            str(data),
            str(TOKENIZER),
            str(run),
            "cuda",
            callbacks=(monitor,),
            checkpoint_volume_uuid=VOLUME_UUID,
        )
        import weakref

        monitor.trainer = weakref.proxy(trainer)
        verified = run / "verified-latest.json"
        resume = bundle / "initial.pt"
        if verified.exists():
            prior = json.loads(verified.read_text())
            resume = run / prior["file"]
            if digest(resume) != prior["sha256"]:
                raise ValueError("Recovery checkpoint failed checksum verification")
        trainer.resume(str(resume))
        started = time.monotonic()
        trainer.train(ui="terminal")
        if monitor.failure is not None:
            raise RuntimeError(
                "Continuation safety monitor failed; training stopped"
            ) from monitor.failure
        if (trainer.step, trainer.tokens, trainer.microsteps) != expected_accounting(budget):
            raise ValueError("Final exact-budget accounting did not pass")
        final = run / "final.pt"
        if not final.exists():
            os.link(run / "latest.pt", final)
        atomic(
            run / "completion.json",
            {
                "step": trainer.step,
                "tokens": trainer.tokens,
                "microsteps": trainer.microsteps,
                "additional_tokens": budget,
                "wall_seconds_this_session": time.monotonic() - started,
                "wall_seconds_from_launch": time.time() - record["created_epoch"],
                "final_sha256": digest(final),
                "parent_sha256": digest(PARENT),
            },
        )
    del trainer
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    queued_postrun(run, record)


def queued_postrun(run: Path, record: dict[str, Any]) -> None:
    """Honor explicit deferral while preserving historical automatic comparisons."""
    if record.get("defer_postrun", False):
        atomic(
            run / "postrun-deferred.json", {"status": "awaiting_user", "training_complete": True}
        )
        print("Training complete; evaluation deferred pending the user notification.", flush=True)
        return
    print("Training complete; starting the queued read-only baseline comparison.", flush=True)
    subprocess.run(
        [sys.executable, "-B", str(Path(__file__).resolve()), "postrun", "--run", str(run)],
        check=True,
        env=os.environ | {"CODEXA_NO_TRAINING": "1"},
    )


def language_loss(
    root: Path, checkpoint: Path, datasets: list[Path], output: Path, maximum_seconds: int = 900
) -> dict[str, Any]:
    """Charge held-out loss to the same persistent language evaluation allowance."""
    from codexa.evaluation.loss import evaluate

    budget_path = root / "benchmarks/language-pair-budget.json"
    with (GPU_LEASE_ROOT / "tmp/gpu-evaluation.lock").open("a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        budget = json.loads(budget_path.read_text())
        if budget.get("limit_seconds") != 14400:
            raise ValueError("Combined language evaluation budget changed")
        if budget.get("active_since"):
            budget["used_seconds"] += max(0, time.time() - budget.pop("active_since"))
        allowance = min(maximum_seconds, 14400 - budget["used_seconds"])
        if allowance <= 0:
            atomic(budget_path, budget)
            return {"complete": False, "reason": "combined evaluation time envelope exhausted"}
        report_path = output / "heldout.json"
        if report_path.exists():
            prior = json.loads(report_path.read_text())
            if (
                prior.get("complete") is True
                and prior.get("checkpoint_sha256") == digest(checkpoint)
                and prior.get("weights_unchanged") is True
                and [r.get("manifest_sha256") for r in prior.get("datasets", [])]
                == [digest(p / "manifest.json") for p in datasets]
            ):
                return prior
        if output.exists():
            base = output
            attempt = 1
            while output.exists():
                output = base.with_name(base.name + "-attempt-" + str(attempt))
                attempt += 1
        output.mkdir(parents=True, exist_ok=False)
        atomic(output / "budget.json", {"limit_seconds": allowance, "used_seconds": 0.0})
        started = time.monotonic()
        atomic(budget_path, budget | {"active_since": time.time()})
        try:
            return evaluate(root, checkpoint, datasets, "heldout", int(allowance), output)
        finally:
            budget["used_seconds"] += time.monotonic() - started
            atomic(budget_path, budget)


def language_paragraphs(root: Path, checkpoint: Path, output: Path) -> dict[str, Any]:
    """Replay the historical 256-token paragraph panel within the shared pair budget."""
    from codexa.evaluation.benchmark import _evaluate

    historical = GPU_LEASE_ROOT / "benchmarks/v3"
    fixtures = root / "benchmarks/v3"
    if not fixtures.exists():
        shutil.copytree(historical, fixtures)
    if digest(fixtures / "manifest.json") != digest(historical / "manifest.json"):
        raise ValueError("Historical paragraph fixture identity changed")
    with (GPU_LEASE_ROOT / "tmp/gpu-evaluation.lock").open("a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        budget_path = root / "benchmarks/language-pair-budget.json"
        budget = json.loads(budget_path.read_text())
        if budget.get("active_since"):
            budget["used_seconds"] += max(0, time.time() - budget.pop("active_since"))
        allowance = min(5400, int(14400 - budget["used_seconds"] - 1800))
        if allowance <= 0:
            atomic(budget_path, budget)
            return {"complete": False, "reason": "combined evaluation budget exhausted"}
        started = time.monotonic()
        atomic(budget_path, budget | {"active_since": time.time()})
        try:
            return _evaluate(root, checkpoint, allowance, output, kinds={"paragraph"})
        finally:
            budget["used_seconds"] += time.monotonic() - started
            atomic(budget_path, budget)


def language_postrun(run: Path, record: dict[str, Any]) -> None:
    from codexa.evaluation.language import compare, evaluate, review_packet

    root, data = Path(record["root"]), Path(record["data"])
    fixtures = root / "benchmarks/language-v1"
    before = root / "benchmarks" / ("language-baseline-" + PARENT_SHA[:16])
    final = run / "final.pt"
    output = root / "benchmarks" / ("language-successor-" + digest(final)[:16])
    budget = json.loads((root / "benchmarks/language-pair-budget.json").read_text())
    allowance = min(5400, int(14400 - budget["used_seconds"] - 1800))
    generation = (
        evaluate(root, fixtures, final, output, seconds=allowance, lease_root=GPU_LEASE_ROOT)
        if allowance > 0
        else {"complete": False, "reason": "combined budget has insufficient generation allowance"}
    )
    paragraph_output = root / "benchmarks" / ("language-paragraphs-" + digest(final)[:16])
    paragraphs = language_paragraphs(root, final, paragraph_output)
    old_loss = language_loss(
        root,
        PARENT,
        [PARENT.parent.parent / "tokens", data],
        root / "benchmarks" / (run.name + "-parent-loss"),
    )
    new_loss = language_loss(
        root,
        final,
        [PARENT.parent.parent / "tokens", data],
        root / "benchmarks" / (run.name + "-successor-loss"),
    )
    comparison = {
        "parent": str(PARENT),
        "successor": str(final),
        "generation": generation,
        "historical_paragraph_generation": paragraphs,
        "historical_paragraph_parent_summary": {
            key: value
            for key, value in json.loads(
                (
                    GPU_LEASE_ROOT / "benchmarks" / ("baseline-" + PARENT_SHA[:16]) / "summary.json"
                ).read_text()
            ).items()
            if ":paragraph:" in key
        },
        "historical_paragraph_successor_summary": json.loads(
            (paragraph_output / "summary.json").read_text()
        )
        if (paragraph_output / "summary.json").exists()
        else None,
        "parent_loss": old_loss,
        "successor_loss": new_loss,
        "training": json.loads((run / "completion.json").read_text()),
        "fresh_evaluation_budget_seconds": 14400,
        "historical_evaluation_budget_unchanged": True,
        "rubric_status": "pending two independent blinded reviews; no competence claim",
    }
    if (output / "outputs.jsonl").exists():
        comparison["paired"] = compare(before, output)
        packet = root / "benchmarks" / (run.name + "-language-review")
        if not packet.exists():
            review_packet(fixtures, [before, output], packet, split="test", setting="sample")
        comparison["independent_review_packet"] = str(packet)
    metrics = {
        r["step"]: r for r in map(json.loads, (run / "metrics.jsonl").read_text().splitlines())
    }
    comparison["measured_step_throughput"] = run_budget(record) / sum(
        x["seconds"] for x in metrics.values()
    )
    comparison["measured_end_to_end_throughput"] = (
        run_budget(record) / comparison["training"]["wall_seconds_from_launch"]
    )
    atomic(run / "comparison.json", comparison)
    print(json.dumps(comparison, indent=2), flush=True)


def postrun(run: Path) -> None:
    record = json.loads((run / "control.json").read_text())
    if record["checks"].get("quality_protocol") in SUPPORTED_VERSIONS:
        language_postrun(run, record)
        return
    from codexa.evaluation.benchmark import evaluate, summarize
    from codexa.evaluation.loss import evaluate as loss
    from codexa.evaluation.scoring import score

    record = json.loads((run / "control.json").read_text())
    root, data = Path(record["root"]), Path(record["data"])
    final = run / "final.pt"
    sha = digest(final)
    old = root / "benchmarks" / ("baseline-" + PARENT_SHA[:16])
    comparison_root = root / "benchmarks" / ("comparison-" + run.name)
    new = comparison_root / ("baseline-" + sha[:16])
    parent_loss_root = comparison_root / ("baseline-" + PARENT_SHA[:16])
    comparison_root.mkdir(exist_ok=True)
    parent_loss_root.mkdir(exist_ok=True)
    new.mkdir(exist_ok=True)
    if not (new / "budget.json").exists():
        atomic(new / "budget.json", {"limit_seconds": 7200, "used_seconds": 0.0})
    if not (parent_loss_root / "budget.json").exists():
        atomic(parent_loss_root / "budget.json", {"limit_seconds": 7200, "used_seconds": 0.0})
    # Reserve at least 30 minutes of the successor ledger for identical-stream
    # loss evaluation, even when generation consumes its full 90-minute slice.
    generation = evaluate(root, final, seconds=5400, output_path=new)
    scored = score(root, new)
    tag = "comparison-" + run.name
    newloss = loss(
        root,
        final,
        [PARENT.parent.parent / "tokens", data],
        tag,
        maximum_seconds=1800,
        output_path=new,
    )
    oldloss = loss(
        root,
        PARENT,
        [PARENT.parent.parent / "tokens", data],
        tag,
        maximum_seconds=7200,
        output_path=parent_loss_root,
    )
    comparison = {
        "parent": str(PARENT),
        "successor": str(final),
        "generation_complete": generation.get("complete"),
        "new_summary": summarize(new),
        "parent_summary": json.loads((old / "summary.json").read_text()),
        "new_answer_proxies": scored,
        "parent_answer_proxies": json.loads((old / "answer-proxies.json").read_text()),
        "new_loss": newloss,
        "parent_loss": oldloss,
        "training": json.loads((run / "completion.json").read_text()),
        "raw_successor_outputs": str(new / "outputs.jsonl"),
        "raw_parent_outputs": str(old / "outputs.jsonl"),
        "fresh_evaluation_budget_seconds": 14400,
        "historical_evaluation_budget_unchanged": True,
    }
    comparison["validation_regressions"] = [
        {
            "dataset": a["dataset"],
            "delta_nats": a["loss"] - b["loss"],
            "failed_tolerance": a["loss"] - b["loss"] > 0.05,
        }
        for a, b in zip(newloss["datasets"], oldloss["datasets"])
        if a["complete"] and b["complete"]
    ]
    metrics = {
        r["step"]: r
        for r in (json.loads(x) for x in (run / "metrics.jsonl").read_text().splitlines())
    }
    comparison["measured_step_throughput"] = run_budget(record) / sum(
        x["seconds"] for x in metrics.values()
    )
    comparison["measured_end_to_end_throughput"] = (
        run_budget(record) / comparison["training"]["wall_seconds_from_launch"]
    )
    old_outputs = {
        r["key"]: r
        for r in (json.loads(x) for x in (old / "outputs.jsonl").read_text().splitlines())
    }
    new_outputs = {
        r["key"]: r
        for r in (json.loads(x) for x in (new / "outputs.jsonl").read_text().splitlines())
    }
    atomic(
        run / "representative-comparisons.json",
        [
            {"key": key, "parent": old_outputs[key], "successor": new_outputs.get(key)}
            for key in record["checks"]["comparison_keys"]
        ],
    )
    comparison["perplexity_note"] = (
        "Computed on identical full-window held-out streams; generated-answer proxies do not establish verified reasoning."
    )
    atomic(run / "comparison.json", comparison)
    print(json.dumps(comparison, indent=2), flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["preflight", "smoke", "launch", "worker", "postrun"])
    p.add_argument("--root", type=Path)
    p.add_argument("--data", type=Path)
    p.add_argument("--bundle", type=Path)
    p.add_argument("--review", type=Path)
    p.add_argument("--run", type=Path)
    p.add_argument(
        "--defer-postrun",
        action="store_true",
        help="Finish training and await user notification before read-only evaluation",
    )
    a = p.parse_args()
    if a.action in ("worker", "postrun"):
        if a.run is None:
            p.error("--run required")
        (worker if a.action == "worker" else postrun)(a.run)
    else:
        if any(v is None for v in (a.root, a.data, a.bundle, a.review)):
            p.error("--root, --data, --bundle and --review required")
        if a.action == "launch":
            launch(a.root, a.data, a.bundle, a.review, defer_postrun=a.defer_postrun)
        elif a.action == "smoke":
            smoke(a.root, a.data, a.bundle, a.review)
        else:
            print(json.dumps(preflight(a.root, a.data, a.bundle, a.review), indent=2))


if __name__ == "__main__":
    main()
