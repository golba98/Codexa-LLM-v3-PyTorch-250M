import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from codexa.release.artifacts import load_export


def preflight(artifact: str, verification_report: str) -> dict[str, Any]:
    from codexa.repository import source_identity

    model, tok, metadata = load_export(artifact)
    report = json.loads(Path(verification_report).read_text())
    if report.get("status") != "passed" or report.get("source") != source_identity():
        raise ValueError("Missing, failed or stale verification report")
    required = [
        "stage",
        "version",
        "training_origin",
        "training_tokens",
        "datasets",
        "compute",
        "source_commit",
        "evaluation",
        "uses",
        "weight_license",
    ]
    if any(k not in metadata for k in required):
        raise ValueError("Incomplete metadata")
    card = (Path(artifact) / "README.md").read_text()
    if "{{" in card or "UNKNOWN" in card:
        raise ValueError("Incomplete model card")
    x = torch.tensor([[2, *tok.encode("Test")[:8]]])
    with torch.no_grad():
        expected = model(x).tolist()
    code = "import json,sys,torch; from codexa.release.artifacts import load_export; m,_,_=load_export(sys.argv[1]); x=torch.tensor(json.loads(sys.argv[2])); print(json.dumps(m(x).detach().tolist()))"
    actual = json.loads(
        subprocess.check_output(
            [sys.executable, "-c", code, artifact, json.dumps(x.tolist())], text=True
        )
    )
    torch.testing.assert_close(torch.tensor(actual), torch.tensor(expected), rtol=1e-6, atol=1e-7)
    if not torch.equal(torch.tensor(actual).argmax(-1), torch.tensor(expected).argmax(-1)):
        raise ValueError("Greedy output mismatch")
    blockers = []
    if metadata["stage"] == "fixture":
        if metadata["training_tokens"] != 0:
            raise ValueError("Fixture must claim zero training tokens")
        blockers.append("Random fixture is not a public trained-model release")
    else:
        if (
            not metadata["training_tokens"]
            or not metadata.get("evaluation")
            or not metadata.get("datasets")
        ):
            blockers.append("Missing training/evaluation/provenance evidence")
        if not metadata.get("weight_license_text"):
            blockers.append("Weight license text missing")
        if not metadata.get("weight_license_reviewed") or not metadata.get("weight_license"):
            blockers.append("Weight licensing review incomplete")
        if not metadata.get("security_reporting_reviewed"):
            blockers.append("Private security reporting not reviewed")
        if metadata.get("source_commit") != report["source"]["commit"]:
            blockers.append("Source evidence mismatch")
        if report["source"]["dirty"]:
            blockers.append("Source tree is dirty")
    return {
        "validation": "passed",
        "publish_ready": not blockers,
        "blockers": blockers,
        "parameters": model.parameter_count(),
        "version": metadata["version"],
        "publication_performed": False,
    }
