"""Document-level eligibility and reviewed rights coverage for Common Corpus.

Eligibility is a development filter, never a source-rights approval. Every production
document additionally needs coverage by a separately reviewed, hash-bound rights file.
"""

import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from codexa.data.production import digest

REPO = "PleIAs/common_corpus"
REVISION = "307910e4c5d040d6f318e6edf2a2b97849155771"
POLICY = "codexa-common-modern-english-1"
COLUMNS = (
    "text",
    "identifier",
    "collection",
    "open_type",
    "curator",
    "license",
    "date",
    "title",
    "creator",
    "language",
    "language_type",
    "word_count",
    "token_count",
)
LICENSES = {
    "CC0",
    "CC0-1.0",
    "CC-BY-4.0",
    "CC-BY-3.0",
    "CC-BY-2.5",
    "CC-BY-2.0",
    "Public Domain",
    "US federal public domain",
}
EXCLUDED_COLLECTION = re.compile(
    r"(?i)patent|uspto|\bsec\b|finance|caselaw|case law|court|legal|wikidata|github|stack.exchange|math"
)
MATH_TOPIC = re.compile(
    r"(?i)\b(?:mathematics|mathematical|algebra|geometry|calculus|number theory|theorem|trigonometry)\b"
)
MATH_INSTRUCTION = re.compile(
    r"(?i)(?:solve|calculate|evaluate|simplify|find the value|prove that)\b[^.!?\n]{0,120}(?:equation|\\frac|\\sum|\\sqrt|\bpolynomial\b|\bintegral\b)|\\\[|\\begin\{(?:equation|align)"
)


def eligible(row: dict[str, Any], spec: dict[str, Any]) -> str | None:
    """Reject unresolved rights labels; accepted labels still require original evidence."""
    if spec.get("repo") != REPO or spec.get("revision") != REVISION:
        return "common_source_identity"
    if row.get("language") != "English" or row.get("language_type") != "Written":
        return "common_not_written_english"
    year = row.get("date")
    if type(year) is not int or not 2000 <= year <= 2026:
        return "common_not_modern"
    collection = row.get("collection")
    if not isinstance(collection, str) or not collection or EXCLUDED_COLLECTION.search(collection):
        return "common_excluded_collection"
    if row.get("open_type") not in {"Open Science", "Open Web", "Open Government", "Open Culture"}:
        return "common_nonprose_type"
    if spec.get("collections") and collection not in spec["collections"]:
        return "common_collection_not_selected"
    if row.get("license") not in LICENSES:
        return "common_unresolved_license"
    if not isinstance(row.get("identifier"), str) or not row["identifier"].strip():
        return "common_missing_origin"
    text = row.get("text")
    if (
        not isinstance(text, str)
        or MATH_INSTRUCTION.search(text)
        or MATH_TOPIC.search(str(row.get("title", "")))
    ):
        return "common_math_or_missing_text"
    return None


def grouping(row: dict[str, Any]) -> str:
    """Use stable origin identity; all extracted parts of that origin share a split."""
    ident = str(row.get("identifier", "")).strip()
    # Common science extraction can suffix chunks with _1, _2, etc.
    return re.sub(r"(?:\.txt)?_\d+$", "", ident).split("#", 1)[0]


def metadata(row: dict[str, Any]) -> dict[str, Any]:
    return {key: row.get(key) for key in COLUMNS if key != "text"}


def reviewed_grants(spec: dict[str, Any], evidence: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate the rights file and original-licence proof contract, without admitting rows."""
    if (
        evidence.get("status") != "approved"
        or evidence.get("repo") != REPO
        or evidence.get("revision") != REVISION
    ):
        return []
    if evidence.get("policy") != POLICY or not evidence.get("reviewer"):
        return []
    grants = []
    for grant in evidence.get("coverage", []):
        collection = grant.get("collection")
        licenses, prefixes = grant.get("licenses"), grant.get("origin_prefixes")
        if (
            not isinstance(collection, str)
            or not collection
            or EXCLUDED_COLLECTION.search(collection)
            or (spec.get("collections") and collection not in spec["collections"])
            or not isinstance(licenses, list)
            or not licenses
            or any(license_name not in LICENSES for license_name in licenses)
            or not isinstance(prefixes, list)
            or not prefixes
            or any(not isinstance(prefix, str) or not prefix for prefix in prefixes)
            or grant.get("scope") != "all_admitted_documents"
            or grant.get("verification") != "original_license_verified"
            or not grant.get("attribution")
            or urlparse(str(grant.get("basis_url", ""))).scheme != "https"
        ):
            continue
        proof = grant.get("proof", {})
        path = Path(proof.get("path", ""))
        if path.is_file() and digest(path) == proof.get("sha256"):
            grants.append(grant)
    return grants


def covered(row: dict[str, Any], spec: dict[str, Any], evidence: dict[str, Any]) -> bool:
    """Require exact declared licence plus reviewed original-document scope evidence."""
    if eligible(row, spec):
        return False
    for grant in reviewed_grants(spec, evidence):
        if (
            grant["collection"] != row.get("collection")
            or row.get("license") not in grant["licenses"]
        ):
            continue
        origin = str(row.get("identifier", ""))
        if any(origin.startswith(prefix) for prefix in grant["origin_prefixes"]):
            return True
    return False
