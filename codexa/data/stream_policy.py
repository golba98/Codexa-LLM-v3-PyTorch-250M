"""Configured Wikimedia admission, native exclusions and separately checked launch approval."""

import hashlib
import json
from pathlib import Path
from typing import Any

from codexa.data.language import Reference
from codexa.data.language_protocol import (
    GENERAL_RUBRIC_SHA256,
    V5_VERSION,
    V6_VERSION,
    V7_VERSION,
    adjudicate,
    combine_reviews,
    validate_authorization,
    validate_specification,
)
from codexa.data.production import digest
from codexa.data.stream_admission import (
    OnlineDuplicates,
    stackexchange_discussion,
    wikimedia_article,
)
from codexa.data.stream_probe import validate_config
from codexa.data.streaming import Opener, StreamingData, open_provider
from codexa.data.v2_index import Protection
from codexa.data.v2_quality import canonical, features
from codexa.evaluation.protection import FrozenAssessmentProtection, QuestionProtection
from codexa.tokenizer.bpe import CodexaTokenizer


def read_bound(entry: dict[str, str]) -> Path:
    original = Path(entry["path"])
    path = original.resolve()
    if original.is_symlink() or digest(path) != entry["sha256"]:
        raise ValueError("Streaming evidence fingerprint mismatch")
    return path


def launch_approval(config: dict[str, Any]) -> None:
    if {Path(p).name for p in config.get("assessment_roots", [])} != {
        "protected-math-recovery-v1",
        "protected-math-recovery-v2",
    }:
        raise ValueError("Frozen private assessments must be bound before streaming launch")

    evidence = config["approval"]
    protocol = json.loads(read_bound(evidence["protocol"]).read_text())
    validate_specification(protocol)
    if protocol["version"] not in {V5_VERSION, V6_VERSION, V7_VERSION}:
        raise ValueError("Streaming requires a pinned source protocol")
    if protocol["version"] == V7_VERSION:
        rubric = read_bound(evidence["rubric"])
        authorization = json.loads(read_bound(evidence["quality_authorization"]).read_text())
        if (
            digest(rubric) != GENERAL_RUBRIC_SHA256
            or digest(read_bound(evidence["original_panel"])) != protocol["original_panel_sha256"]
            or authorization.get("status") != "authorized-for-reevaluation-only"
            or authorization.get("source") != config["dataset"]
            or authorization.get("revision") != config["revision"]
            or authorization.get("approved_application_sha256") != GENERAL_RUBRIC_SHA256
            or not authorization.get("user_decision")
        ):
            raise ValueError("Explicit approved-rubric reevaluation authorization required")
    sources = list(protocol["source_bindings"].values())
    if len(sources) != 1 or sources[0] != {
        "repo": config["dataset"],
        "revision": config["revision"],
    }:
        raise ValueError("Protocol and streaming source differ")
    binding = {k: v for k, v in config.items() if k != "approval"}
    if (
        protocol.get("stream_config_sha256")
        != hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
    ):
        raise ValueError("Source review is not bound to this streaming configuration")
    if protocol.get("training_config_identity") != config.get("training_config_identity"):
        raise ValueError("Source authorization and compute configuration differ")
    mapping = [json.loads(x) for x in read_bound(evidence["mapping"]).read_text().splitlines()]
    reviewers = evidence["reviewers"]
    if len(reviewers) != 2 or len({r["identity"] for r in reviewers}) != 2:
        raise ValueError("Two independent reviewers required")
    inputs = []
    for reviewer in reviewers:
        ratings = [json.loads(x) for x in read_bound(reviewer).read_text().splitlines()]
        if adjudicate(mapping, ratings, protocol)["status"] != "passed":
            raise ValueError("Source quality review did not pass")
        inputs.append(ratings)
    if (
        adjudicate(mapping, combine_reviews(inputs, version=protocol["version"]), protocol)[
            "status"
        ]
        != "passed"
    ):
        raise ValueError("Conservative source quality review did not pass")
    capacity = json.loads(read_bound(evidence["capacity"]).read_text())
    budget = 500_000_000 if protocol["version"] == V7_VERSION else 1_000_000_000
    if (
        capacity.get("source_bindings") != protocol["source_bindings"]
        or capacity.get("lower_95_eligible_train_tokens", 0) < budget
        or capacity.get("stream_config_sha256") != protocol["stream_config_sha256"]
    ):
        raise ValueError("Conservative eligible-token availability is not established")
    if protocol["version"] == V7_VERSION:
        losses = capacity.get("inspection_representativeness", {})
        if (
            losses.get("status") != "passed-on-bounded-evidence"
            or losses.get("maximum_permitted_fraction") != 0.05
            or any(
                type(losses.get(key)) not in (int, float) or not 0 <= losses[key] <= 0.05
                for key in (
                    "estimated_inspection_token_loss_fraction",
                    "estimated_inspection_document_loss_fraction",
                )
            )
        ):
            raise ValueError("Inspection exclusions exceed or lack the preserved 5-percent gate")
    compute = json.loads(read_bound(evidence["compute"]).read_text())
    if (
        compute.get("status") != "passed"
        or not compute.get("optimizer_schema_verified")
        or not compute.get("checkpoint_resume_verified")
        or not compute.get("cuda_memory_verified")
        or compute.get("training_config_identity") != config.get("training_config_identity")
        or not config.get("training_config_identity")
        or compute.get("parent_sha256") != config.get("parent_checkpoint_sha256")
        or not config.get("parent_checkpoint_sha256")
    ):
        raise ValueError("Compute and checkpoint readiness is incomplete")
    rights = json.loads(read_bound(evidence["rights"]).read_text())
    if (
        rights.get("status") != "approved"
        or rights.get("source_bindings") != protocol["source_bindings"]
        or rights.get("scope") != "internal-pretraining"
        or not rights.get("reviewer")
        or not rights.get("attribution_policy")
        or not rights.get("original_rights_basis")
    ):
        raise ValueError("Reviewed source rights required")
    auth = read_bound(evidence["authorization"])
    validate_authorization(auth, protocol)
    record = json.loads(auth.read_text())
    if protocol["version"] == V7_VERSION and (
        config.get("additional_target_tokens") != budget
        or config.get("cumulative_target_tokens") != record["cumulative_target_tokens"]
        or compute.get("optimizer_updates", 0) < 2
        or compute.get("exact_gpu_resume_verified") is not True
    ):
        raise ValueError(
            "500M launch requires bound exact budget and real optimizer/resume evidence"
        )


class WikimediaAdmission:
    def __init__(
        self,
        config: dict[str, Any],
        shards: list[dict[str, str]],
        opener: Opener = open_provider,
        *,
        validated_files: bool = False,
    ):
        self.ledger = OnlineDuplicates(shards, opener=opener)
        if not validated_files:
            for entry in config["exclusion_files"]:
                read_bound(entry)
        bound = {str(Path(e["path"]).resolve()) for e in config["exclusion_files"]}
        self.references = []
        review_ids = Path(config["review_ids"])
        if str(review_ids.resolve()) not in bound:
            raise ValueError("Unbound prior review identities")
        self.review_ids = set(json.loads(review_ids.read_text()))
        if {r["role"] for r in config["references"]} != {"historical", "prior50m", "prior100m"}:
            raise ValueError("Historical and both prior candidate heldouts must be bound")
        for ref in config["references"]:
            root, old = Path(ref["root"]), Path(ref["historical"])
            for path in (root / "indexes/documents.sqlite", old):
                if str(path.resolve()) not in bound:
                    raise ValueError("Unbound reference index")
            self.references.append(Reference(root, old, include_candidate_holdouts=True))
        self.protections = []
        for path in config["protected_indexes"]:
            if str(Path(path).resolve()) not in bound:
                raise ValueError("Unbound protected index")
            self.protections.append(Protection.load(Path(path)))
        fixtures = Path(config["fixtures"])
        for split in ("dev", "test"):
            if str((fixtures / f"{split}.jsonl").resolve()) not in bound:
                raise ValueError("Unbound question fixtures")
        if not self.protections:
            raise ValueError("Protected evaluation index required")
        self.questions = QuestionProtection(fixtures)
        assessment_roots = [Path(p) for p in config.get("assessment_roots", [])]
        for root in assessment_roots:
            for name in (
                "private-assessment.jsonl",
                "family-exclusions.json",
                "freeze-manifest.json",
            ):
                if str((root / name).resolve()) not in bound:
                    raise ValueError("Unbound frozen assessment evidence")
        self.assessments = FrozenAssessmentProtection(assessment_roots)
        self.cursor = (0, 0)

    def position(self, shard: int, record: int) -> None:
        self.cursor = (shard, record)

    def group(self, row: dict[str, Any]) -> str:
        return "enwiki:" + row["id"]

    def metadata_eligible(self, row: dict[str, Any]) -> bool:
        return wikimedia_article(row)

    def protected(self, text: str) -> bool:
        return (
            any(p.matches(text) for p in self.protections)
            or self.questions.matches(text) is not None
            or self.assessments.matches(text)
        )

    def __call__(self, row: dict[str, Any]) -> bool:
        if not self.metadata_eligible(row):
            return False
        text = row["text"]
        ident = hashlib.sha256(canonical(text).encode()).digest()
        if ident.hex() in self.review_ids or self.review_identity(row) in self.review_ids:
            return False
        shingles, signature, anchors = features(text)
        if any(r.duplicate(ident, text, shingles, signature, anchors) for r in self.references):
            return False
        return self.ledger.add(text, *self.cursor, attribution=self.attribution(row)) is None

    def review_identity(self, row: dict[str, Any]) -> str:
        return row["id"]

    def attribution(self, row: dict[str, Any]) -> dict[str, Any]:
        attribution = dict(row["metadata"])
        page_id = row["id"].split("-", 1)[1]
        attribution["attribution_url"] = f"https://en.wikipedia.org/?curid={page_id}"
        attribution["history_url"] = (
            f"https://en.wikipedia.org/w/index.php?curid={page_id}&action=history"
        )
        attribution["provider_record_id"] = row["id"]
        return attribution

    def begin(self) -> None:
        self.ledger.begin()

    def commit(self) -> None:
        self.ledger.commit()

    def rollback(self) -> None:
        self.ledger.rollback()

    def state_dict(self) -> dict[str, Any]:
        return self.ledger.state_dict()

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.ledger.load_state_dict(state)


class StackExchangeAdmission(WikimediaAdmission):
    """Same native exclusions and duplicate controls; source-specific metadata only."""

    def metadata_eligible(self, row: dict[str, Any]) -> bool:
        return stackexchange_discussion(row)

    def group(self, row: dict[str, Any]) -> str:
        return "stackexchange:" + row["metadata"]["site"] + ":" + row["id"]

    def review_identity(self, row: dict[str, Any]) -> str:
        return self.group(row)

    def attribution(self, row: dict[str, Any]) -> dict[str, Any]:
        return {**row["metadata"], "provider_record_id": row["id"]}


def configured_streams(
    config_path: str,
    tokenizer: str,
    sequence: int,
    seed: int,
    *,
    require_launch: bool = True,
    opener: Opener = open_provider,
    evaluation_split: str = "val",
) -> tuple[StreamingData, StreamingData]:
    if evaluation_split not in {"val", "test"}:
        raise ValueError("Choose an independent validation or test stream")
    config = json.loads(Path(config_path).read_text())
    for entry in config["policy_sources"]:
        read_bound(entry)
    expected = {
        str(Path(__file__).resolve()),
        str(Path(__file__).with_name("stream_admission.py").resolve()),
        str(Path(__file__).with_name("streaming.py").resolve()),
        str(Path(__file__).with_name("v2_index.py").resolve()),
        str(Path(__file__).with_name("v2_quality.py").resolve()),
        str(Path(__file__).with_name("language_protocol.py").resolve()),
        str((Path(__file__).parents[1] / "evaluation/protection.py").resolve()),
    }
    if {str(Path(e["path"]).resolve()) for e in config["policy_sources"]} != expected:
        raise ValueError("All streaming policy source identities must be pinned")
    policies = {
        (
            "common-pile/wikimedia_filtered",
            "0641bb84bd9b7162bcddf8be7836822161a9a342",
        ): WikimediaAdmission,
        (
            "common-pile/stackexchange_filtered",
            "c0ac7373830c688a43fc12d1988c4b19ccd884ab",
        ): StackExchangeAdmission,
    }
    policy_class = policies.get((config["dataset"], config["revision"]))
    if policy_class is None:
        raise ValueError("Unsupported source binding")
    if require_launch:
        launch_approval(config)
    for entry in config["exclusion_files"]:
        read_bound(entry)
    shards = validate_config({k: config[k] for k in ("dataset", "revision", "shards")})
    # The policy stores locators in the exact same shuffled shard order as the reader.
    import random

    ordered = [dict(s) for s in shards]
    random.Random(seed).shuffle(ordered)
    tok = CodexaTokenizer.load(tokenizer)
    if digest(tokenizer) != config["tokenizer_sha256"]:
        raise ValueError("Tokenizer changed")
    identity = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    streams = []
    for split in ("train", evaluation_split):
        policy = policy_class(config, ordered, opener=opener, validated_files=True)
        stream = StreamingData(
            shards,
            tok,
            sequence,
            seed,
            split,
            policy,
            policy.protected,
            policy_identity=identity,
            opener=opener,
        )
        stream.launch_check = lambda: launch_approval(config)
        stream.required_config_identity = config.get("training_config_identity")
        streams.append(stream)
    return streams[0], streams[1]
