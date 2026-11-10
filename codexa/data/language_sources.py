"""Small explicit adapters for the authorized version-3 English sources.

Metadata classification is not rights approval. Production rows additionally need a
hash-bound original-document or original-book grant; dataset licences are insufficient.
"""

import html
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from codexa.data.production import digest
from codexa.data.v2_policy import admission_error
from codexa.data.v2_quality import CONTACT, clean

DCLM_ADAPTER = "codexa-dclm-english-1"
PRESSBOOKS_ADAPTER = "codexa-pressbooks-english-1"
ADAPTERS = {
    DCLM_ADAPTER: ("HuggingFaceTB/dclm-edu", "dbad8ad71224482740cd9c9d353591adbf62fe04"),
    PRESSBOOKS_ADAPTER: (
        "common-pile/pressbooks_filtered",
        "1a1d3b50d77f834370f8eb4c0d174668dd1676bb",
    ),
}
POLICY = "codexa-language-sources-3"
INTERNAL_POLICY = "codexa-publisher-internal-research-1"
INTERNAL_POLICY_V2 = "codexa-publisher-internal-research-2"
POLICY_V2_PROPOSAL_SHA256 = "30ab68c6a2f259893638dd0debaf557ab9425e5d20b117917e61d4e3ed870c82"
ALLOWED_LICENSES = {"CC0-1.0", "CC-BY-2.0", "CC-BY-2.5", "CC-BY-3.0", "CC-BY-4.0", "public-domain"}
MATH_TOPIC = re.compile(
    r"(?i)\b(?:mathematics|mathematical|algebra|calculus|geometry|statistics|trigonometry|number theory|group theory)\b"
)
MATH_TASK = re.compile(
    r"(?i)\b(?:solve|calculate|simplify|prove that)\b[^.!?\n]{0,100}\b(?:equation|polynomial|integral|theorem|surface area|volume of (?:a |the )?cube)\b|\\\[|\\begin\{(?:align|equation)"
)
FOOTER = re.compile(
    r"(?i)^(?:copyright\s*(?:©|\(c\))?\s*(?:\d{4}|all rights reserved)\b.*|©\s*\d{4}.*|add flashcard\s+cite\s+random|jump to:\s*navigation,?\s*search|from wikipedia, the free encyclopedia|want to create or adapt books like this\?.*|share this(?: article)?(?: on .*)?|previous(?: chapter| page)?|next(?: chapter| page)?)$"
)
CONTRARY_NOTICE = re.compile(
    r"(?i)creativecommons\.org/licenses/(?:by-)?(?:nc|nd|sa)|\bcc[-\s]+by[-\s]+(?:nc|nd|sa)\b|Creative Commons\s+(?:Attribution[-\s]+)?(?:Non[-\s]?Commercial|No[-\s]?Derivatives|Share[-\s]?Alike)\b|\ball rights reserved\b|\b(?:reprinted|reproduced|used) with permission\b|\brepublished from\b|\breprinted from\b|\boriginally published (?:in|by|at)\b"
)
SA_NOTICE = re.compile(
    r"(?i)https?://(?:www\.)?creativecommons\.org/licenses/by-sa/[1-4]\.[0-9]/?|\bcc[-\s]+by[-\s]+sa(?:[-\s]+[1-4]\.[0-9])?\b|Creative Commons\s+(?:Attribution[-\s]+)?Share[-\s]?Alike\b"
)
SA_SPECIAL_CONDITION = re.compile(
    r"(?i)creativecommons\.org/licenses/by-sa/[1-4]\.[0-9]/(?!legalcode|deed(?:\.|/|$))[a-z]{2}(?:/|[\s<]|$)|(?:CC[-\s]+BY[-\s]+SA|Attribution[-\s]+Share[-\s]?Alike)[^\n]{0,100}\b(?:ported|additional restrictions|non[- ]standard conditions)\b"
)

SA_VERSIONS = {"1.0", "2.0", "2.5", "3.0", "4.0"}
RESTRICTED_USE = re.compile(
    r"(?i)\b(?:CC[-\s]+BY[-\s]+SA|Attribution[-\s]+Share[-\s]?Alike)[^\n]{0,100}\b(?:NC|ND|non[-\s]?commercial|no[-\s]?derivatives?)\b|\bnon[-\s]?commercial(?:\s+use)?\s+only\b|\b(?:not|no)\s+(?:for\s+)?commercial\s+use\b|\bno[-\s]+derivative(?:s|\s+works)?(?:\s+(?:are\s+)?(?:permitted|allowed))?\b|\b(?:no|without)\s+(?:modifications|adaptations)\b|\b(?:may|must)\s+not\s+(?:modify|adapt|create\s+derivative)\b"
)


def noncanonical_sa_notice(notice: str) -> bool:
    """Unknown versions and any noncanonical jurisdiction/condition URL quarantine."""
    for match in re.finditer(
        r"(?i)(?:https?://)?(?:www\.)?creativecommons\.org/licenses/by-sa/[^\s<>\"')]+", notice
    ):
        url = match.group().rstrip(".,;:!?")
        if not re.match(r"(?i)https?://", url):
            url = "https://" + url
        parts = urlsplit(url).path.strip("/").split("/")
        if len(parts) < 3 or parts[2] not in SA_VERSIONS:
            return True
        suffix = parts[3:]
        if suffix:
            if len(suffix) != 1:
                return True
            allowed = suffix[0] == "legalcode" or bool(
                re.fullmatch(r"deed(?:\.[a-z]{2,3}(?:[_-][A-Za-z]{2})?)?", suffix[0])
            )
            if suffix[0] == "legalcode.en" and parts[2] == "4.0":
                allowed = True
            if not allowed:
                return True
    for match in re.finditer(
        r"(?i)(?:\bCC[-\s]+BY[-\s]+SA|(?:Creative Commons\s+)?Attribution[-\s]+Share[-\s]?Alike)\b(?:[-\s]+(\d+(?:\.\d+)*))?",
        notice,
    ):
        if match.group(1) is not None and match.group(1) not in SA_VERSIONS:
            return True
        # A jurisdiction/name is part of this licence label, not an unported grant.
        # URLs were validated above; do not parse their domain punctuation as prose.
        tail = re.sub(
            r"(?i)(?:https?://)?(?:www\.)?creativecommons\.org/licenses/by-sa/[^\s<>\"')]+",
            "",
            notice[match.end() :],
        )
        qualifiers = re.split(r"[.!?\n\r]", tail, maxsplit=1)[0]
        words = re.findall(r"[A-Za-z]+|\d+", qualifiers)
        if any(
            word.lower() not in {"unported", "international", "public", "license", "licence"}
            for word in words
        ):
            return True
    return False


APPENDIX = re.compile(
    r"(?im)^\s*(?:references|bibliography|external links|further reading|see also)(?:\[edit\])?\s*:?\s*$"
)
IMAGE_CREDIT = re.compile(
    r"(?i)^\s*(?:[-*•]\s*)?(?:(?:image|photo(?:graph)?|illustration|screenshot)\s*(?::|[\"“‘']|by\b|of\b)|(?:figure|fig\.)\s+\d+(?:\.\d+)*(?:[.:]|\s))"
)
IMAGE_ATTRIBUTION = re.compile(
    r"(?i)^\s*(?:[-*•]\s*)?(?:(?:image|photo(?:graph)?|illustration|screenshot)\s*:?\s*|(?:figure|fig\.)\s+\d+(?:\.\d+)*(?:[.:]|\s)\s*)"
    r"[^\n]{1,300}?\s+by\s+[^\n]{1,200}?\s+(?:is\s+)?(?:licensed|used|released)\s+under\s+(?:a\s+)?"
    r"(?:CC[\s-]+BY(?:[\s-]+(?:NC(?:[\s-]+(?:SA|ND))?|SA|ND))?(?:[\s-]+\d\.\d)?(?:\s+International)?(?:\s+licen[cs]e)?|(?:the\s+)?public domain)\.?\s*$"
)


def scoped_lines(raw: str, *, allow_sharealike: bool = False) -> tuple[list[str], int, str | None]:
    """Remove explicit asset credits; ambiguous or text restrictions quarantine."""
    kept = []
    removed = 0
    for line in raw.splitlines():
        notice = html.unescape(html.unescape(line))
        credit = (
            len(notice) <= 1500
            and "\n" not in notice
            and "\r" not in notice
            and IMAGE_CREDIT.search(notice)
            and (
                IMAGE_ATTRIBUTION.fullmatch(notice)
                or re.fullmatch(
                    r"(?i)\s*Image\s+by\s+[^.\n]{1,200}\s+is public domain\.?\s*", notice
                )
            )
            and not re.search(
                r"(?i)\b(?:chapter|article|text|adaptation|republish\w*|reprint\w*|book|excerpt|section|paragraph|course|work|material|content|lesson|document|website|curriculum|module|entire|everything|whole)\b",
                notice,
            )
        )
        if credit:
            removed += 1
        elif (
            RESTRICTED_USE.search(notice)
            or (
                allow_sharealike
                and (SA_SPECIAL_CONDITION.search(notice) or noncanonical_sa_notice(notice))
            )
            or CONTRARY_NOTICE.search(SA_NOTICE.sub("", notice) if allow_sharealike else notice)
        ):
            return [], removed, "language_contrary_rights_notice"
        else:
            kept.append(line)
    return kept, removed, None


def canonical_license(raw: Any) -> dict[str, Any]:
    """Resolve explicit versions and known aliases, never infer an absent BY version."""
    result: dict[str, Any] = {"raw": raw, "family": None, "spdx": None, "status": "unresolved"}
    if not isinstance(raw, str):
        return result
    value = raw.strip()
    urls = re.findall(r"https?://[^\s<>\"')]+", value)
    for url in urls:
        parsed = urlsplit(url.rstrip(".,;"))
        if parsed.hostname not in {"creativecommons.org", "www.creativecommons.org"}:
            continue
        parts = parsed.path.strip("/").split("/")
        if (
            len(parts) >= 3
            and parts[0] == "licenses"
            and parts[1] in {"by", "by-sa", "by-nc", "by-nc-sa", "by-nd", "by-nc-nd"}
        ) and parts[2] in {"1.0", "2.0", "2.5", "3.0", "4.0"}:
            result.update(
                family="CC-" + parts[1].upper(),
                spdx="CC-" + parts[1].upper() + "-" + parts[2],
                status="explicit",
            )
            return result
        if len(parts) >= 3 and parts[:2] == ["publicdomain", "zero"] and parts[2] == "1.0":
            result.update(family="CC0", spdx="CC0-1.0", status="explicit")
            return result
        if len(parts) >= 3 and parts[:2] == ["publicdomain", "mark"] and parts[2] == "1.0":
            result.update(family="public-domain", status="basis_required")
            return result
    value = re.sub(r"[ _]+", "-", value).upper()
    if value in {"CC0", "CC0-1.0"}:
        result.update(family="CC0", spdx="CC0-1.0", status="alias")
    elif value in {"PUBLIC-DOMAIN", "US-FEDERAL-PUBLIC-DOMAIN"}:
        result.update(family="public-domain", status="basis_required")
    elif re.fullmatch(r"CC-BY(?:-SA|-NC(?:-SA|-ND)?|-ND)?(?:-\d\.\d)?", value):
        version = re.search(r"-([1-4]\.\d)$", value)
        family = value[: version.start()] if version else value
        result.update(
            family=family,
            spdx=value if version else None,
            status="explicit" if version else "missing_version",
        )
    return result


def valid_source(spec: dict[str, Any]) -> bool:
    return ADAPTERS.get(str(spec.get("adapter", ""))) == (spec.get("repo"), spec.get("revision"))


def source_metadata(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "text"}


def grouping(row: dict[str, Any], spec: dict[str, Any]) -> str:
    if spec.get("adapter") == PRESSBOOKS_ADAPTER:
        return str(row.get("metadata", {}).get("book_url", "")).rstrip("/")
    return str(row.get("url") or row.get("id") or "").split("#", 1)[0]


def _sa_book(row: dict[str, Any], spec: dict[str, Any]) -> bool:
    """Mechanical source selection only; compiled original grants still gate training."""
    return (
        spec.get("adapter") == PRESSBOOKS_ADAPTER
        and spec.get("rights_policy") == INTERNAL_POLICY_V2
        and canonical_license(row.get("metadata", {}).get("license"))["family"] == "CC-BY-SA"
        and not SA_SPECIAL_CONDITION.search(str(row.get("metadata", {}).get("license", "")))
        and not noncanonical_sa_notice(str(row.get("metadata", {}).get("license", "")))
        and not RESTRICTED_USE.search(str(row.get("metadata", {}).get("license", "")))
    )


def clean_source(row: dict[str, Any], spec: dict[str, Any]) -> tuple[str, list[str], str | None]:
    """Remove identified page furniture, not arbitrary incomplete sentence tails."""
    raw = row.get("text")
    if not isinstance(raw, str):
        return "", [], "missing_text"
    if "\x00" in raw or "\ufffd" in raw or any(0xD800 <= ord(c) <= 0xDFFF for c in raw):
        return "", [], "corrupt_unicode"
    removed = 0
    if spec.get("adapter") == PRESSBOOKS_ADAPTER:
        kept, removed, error = scoped_lines(raw, allow_sharealike=_sa_book(row, spec))
        if error:
            return "", [], error
        raw = "\n".join(kept)
    text = html.unescape(raw).replace("\r\n", "\n").replace("\r", "\n")
    without_returns = re.sub(r"(?m)(?:^|(?<=\s))[↵↑]+(?=\s*(?:\n|$))", "", text)
    decisions = ["footnote_navigation_glyph"] if without_returns != text else []
    if removed:
        decisions.append("standalone_image_credits_excluded")
    text = without_returns
    lines = []
    for line in text.splitlines():
        if FOOTER.fullmatch(line.strip()):
            decisions.append("recognized_page_furniture")
            continue
        if CONTACT.search(html.unescape(line)):
            # Legacy cleaning drops complete lines. Never silently lose adjacent prose.
            if re.fullmatch(
                r"(?i)\s*(?:e-?mail|phone|tel|fax|mobile)?\s*:?\s*(?:[\w.+-]+@[\w.-]+\.[a-z]{2,}|[+\d ()-]+)\s*",
                html.unescape(line),
            ):
                decisions.append("standalone_contact")
                continue
            return "", decisions, "language_mixed_contact_line"
        lines.append(line.rstrip())
    text = "\n".join(lines)
    appendix = APPENDIX.search(text)
    if appendix:
        text = text[: appendix.start()]
        decisions.append("trailing_reference_appendix")
    if spec.get("adapter") == PRESSBOOKS_ADAPTER:
        # A media heading never licenses all later text or proves the chapter ended.
        text = re.sub(r"(?im)^\s*(?:(?:Media|Image|Photo)\s+)?Attributions\s*$", "", text)
    return text.strip(), decisions, None


def prose_admission_error(text: str, row: dict[str, Any], options: dict[str, Any]) -> str | None:
    """Optional measurable prose proxies and observed scaffolding quarantine.

    These checks do not certify grammar or factual correctness. The independent
    full-document quality protocol remains mandatory and unchanged.
    """
    minimum = options.get("minimum_prose_sentences", 0)
    fraction = options.get("minimum_prose_fraction", 0)
    scaffolding = options.get("reject_instructional_scaffolding", False)
    if (
        type(minimum) is not int
        or minimum < 0
        or type(fraction) not in {int, float}
        or not 0 <= fraction <= 1
        or type(scaffolding) is not bool
    ):
        return "language_prose_filter_policy"
    if not minimum and not fraction and not scaffolding:
        return None
    words = re.findall(r"[A-Za-z]+(?:['’][A-Za-z]+)?", text)
    sentence_words = 0
    sentences = 0
    imperative = re.compile(
        r"(?i)^(?:read|watch|answer|write|list|describe|identify|discuss|explain|calculate|solve|complete|review|fill|match|select|choose|click|define|distinguish|compare|summarize|demonstrate|evaluate|analyze|refer to)\b"
    )
    for unit in re.split(r"(?<=[.!?])\s+|\n+", text):
        unit = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s+", "", unit).strip()
        if "?" in unit or not re.search(r"[.!][\'\"”’\])]*\s*$", unit) or imperative.search(unit):
            continue
        count = len(re.findall(r"[A-Za-z]+(?:['’][A-Za-z]+)?", unit))
        if count >= 6:
            sentences += 1
            sentence_words += count
    if scaffolding:
        raw = str(row.get("text", text))
        decoded = html.unescape(html.unescape(raw))
        if re.search(r"(?i)<EMAIL_ADDRESS>|\[REDACTED\]", decoded):
            return "language_redacted_scaffolding"
        if re.search(
            r"(?im)^\s*(?:[-*•]|\d+[.)])?\s*(?:read|watch|view)\s+(?:the\s+(?:(?:following|provided|assigned|linked)\s+)?|this\s+|(?:provided|assigned|following)\s+)[^\n.!?]{0,160}\b(?:paper|video|article|lecture)\b[^\n.!?]{0,240}\b(?:then|and)\s+(?:answer|respond|complete|discuss)\b",
            decoded,
        ):
            return "language_required_external_resource"
        diagnostic = re.search(
            r"(?i)\bDSM(?:[-\s]+[IVX0-9]+)?\b|\bICD(?:[-\s]+\d+)?\b|\bdiagnostic criteria\b",
            decoded,
        )
        criterion_lines = re.findall(r"(?m)^\s*[A-F][.)]\s+", decoded)
        criterion_heading = re.search(
            r"(?im)^\s*(?:\d+[.)]?\s+)?diagnostic criteria(?:\s+(?:for|of)\b[^\n]*)?\s*:?\s*$",
            decoded,
        )
        if diagnostic and (len(criterion_lines) >= 2 or criterion_heading):
            return "language_diagnostic_template"
        learning = re.search(
            r"(?i)\blearning (?:outcomes|objectives)\b|\byou will be able to\b", text
        )
        bullets = re.findall(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+([^\n]+)", text)
        bullet_words = sum(
            len(re.findall(r"[A-Za-z]+(?:['’][A-Za-z]+)?", line))
            for line in bullets
            if imperative.search(line.strip())
        )
        if (
            len(words) <= 250
            and learning
            and len(bullets) >= 2
            and bullet_words >= max(1, len(words)) * 0.5
        ):
            return "language_learning_outcome_scaffolding"
        paragraphs = [part.strip() for part in re.split(r"\n+", decoded) if part.strip()]
        for number, paragraph in enumerate(paragraphs):
            if not re.search(
                r"(?i)\b(?:plaque|inscription)\b[^\n.!?]{0,120}\b(?:inscribed|reads|reading|says)\s*[:,]\s*$",
                paragraph,
            ):
                continue
            following = paragraphs[number + 1] if number + 1 < len(paragraphs) else ""
            if not following or re.match(
                r"(?i)^(?:the\s+(?:urn|plaque|inscription|monument)|this\s+(?:work|plaque|inscription|project))\b",
                following,
            ):
                return "language_missing_announced_quotation"
    if sentences < minimum:
        return "language_insufficient_prose_sentences"
    if sentence_words / max(1, len(words)) < fraction:
        return "language_insufficient_prose_fraction"
    return None


def cleaned_v3(row: dict[str, Any], spec: dict[str, Any]) -> tuple[str, str | None]:
    if not valid_source(spec):
        return "", "language_source_identity"
    adapter = spec["adapter"]
    if row.get("error"):
        return "", str(row["error"])
    if MATH_TASK.search(str(row.get("text", ""))) or MATH_TOPIC.search(str(row.get("title", ""))):
        return "", "language_math_specialization"
    if adapter == DCLM_ADAPTER:
        score = row.get("edu_int_score")
        threshold = spec.get("min_edu_score", 3)
        if type(threshold) is not int or threshold not in {3, 4}:
            return "", "educational_score_policy"
        if type(score) is not int or score < threshold:
            return "", "educational_score"
        heading = str(row.get("text", "")).split("\n", 1)[0]
        if MATH_TOPIC.search(heading) or re.search(
            r"(?i)(?:group[-_/ ]theory|surface[-_/ ]area|volume[-_/ ]of[-_/ ](?:a[-_/ ])?cube)",
            str(row.get("url", "")),
        ):
            return "", "language_math_specialization"
        if row.get("language") != "en":
            return "", "language_not_english"
    else:
        metadata = row.get("metadata")
        if not isinstance(metadata, dict) or not grouping(row, spec):
            return "", "pressbooks_missing_origin"
        topic = str(metadata.get("title", "")) + " " + str(metadata.get("subject", ""))
        if re.search(r"(?i)\b(?:directory|medical terminology)\b", str(metadata.get("title", ""))):
            return "", "language_reference_or_directory"
        if re.search(
            r"(?i)\banswer key\b|\bresponses to (?:transitions|exercises)\b",
            str(row.get("text", ""))[:512],
        ):
            return "", "language_answer_key"
        if MATH_TOPIC.search(topic) or MATH_TASK.search(str(row.get("text", ""))):
            return "", "language_math_specialization"
        license_info = canonical_license(metadata.get("license"))
        if license_info["family"] not in {"CC-BY", "CC0", "public-domain"} and not _sa_book(
            row, spec
        ):
            return "", "language_license_not_selected"
    text, _decisions, error = clean_source(row, spec)
    if error:
        return "", error
    text, errors, _stats = clean(text, spec.get("category", "general"))
    if errors:
        return text, errors[0]
    options = spec.get("filter", {})
    error = prose_admission_error(text, row, options)
    return text, error or admission_error(text, row, options)


def _proof_data(proof: Any, cache: dict[tuple[str, str], bytes]) -> bytes | None:
    if not isinstance(proof, dict):
        return None
    path = Path(proof.get("path", ""))
    sha = proof.get("sha256", "")
    key = (str(path), sha)
    if key in cache:
        return cache[key]
    if not path.is_file() or digest(path) != sha:
        return None
    data = path.read_bytes()
    cache[key] = data
    try:
        parsed = json.loads(data)
    except (ValueError, UnicodeError):
        parsed = None
    if isinstance(parsed, dict):
        links = [
            parsed.get(name)
            for name in (
                "policy_proof",
                "collector_proof",
                "additional_policy_proof",
                "rights_review",
                "authorization_proof",
                "proposal_proof",
            )
            if parsed.get(name)
        ]
        containers = [parsed]
        if isinstance(parsed.get("reviewed_grant"), dict):
            containers.append(parsed["reviewed_grant"])
        for container in containers:
            for name in ("proofs", "primary_policy_proofs"):
                nested = container.get(name, [])
                links.extend(nested.values() if isinstance(nested, dict) else nested)
        for link in links:
            if (
                isinstance(link, dict)
                and "path" in link
                and "sha256" in link
                and _proof_data(link, cache) is None
            ):
                cache.pop(key, None)
                return None
    return data


def _reviewed_public_domain(
    spec: dict[str, Any],
    grant: dict[str, Any],
    raw: bytes,
    cache: dict[tuple[str, str], bytes],
) -> bool:
    try:
        wrapper = json.loads(raw)
        review_raw = _proof_data(wrapper.get("rights_review"), cache)
        review = json.loads(review_raw) if review_raw is not None else {}
        record = wrapper["reviewed_grant"]
        binding = record["source_binding"]
        return (
            review.get("version") == "codexa-pressbooks-additional-rights-review-1"
            and record in review.get("public_domain_grants", [])
            and record["book_url"].rstrip("/") == grant["book_url"].rstrip("/")
            and all(
                binding.get(key) == spec.get(field)
                for key, field in (
                    ("source_repository", "repo"),
                    ("source_revision", "revision"),
                    ("source_file_sha256", "sha256"),
                )
            )
            and grant.get("chapter_url_prefixes") == [record["canonical_book_root"] + "chapter/"]
            and grant.get("original_work_only") is True
        )
    except (KeyError, TypeError, ValueError, UnicodeError):
        return False


def _internal_policy(
    spec: dict[str, Any], evidence: dict[str, Any], cache: dict[tuple[str, str], bytes]
) -> bool:
    selected = spec.get("rights_policy")
    if (
        spec.get("adapter") != PRESSBOOKS_ADAPTER
        or selected not in {INTERNAL_POLICY, INTERNAL_POLICY_V2}
        or evidence.get("rights_policy") != selected
        or evidence.get("redistribution_allowed") is not False
    ):
        return False
    raw = _proof_data(evidence.get("policy_proof"), cache)
    collector = _proof_data(evidence.get("collector_proof"), cache)
    if raw is None or collector is None:
        return False
    try:
        policy = json.loads(raw)
    except (ValueError, UnicodeError):
        return False
    expected_version = (
        "codexa-pressbooks-original-rights-policy-1"
        if selected == INTERNAL_POLICY
        else "codexa-pressbooks-original-rights-policy-2"
    )
    if not (
        policy.get("version") == expected_version
        and policy.get("status") == "conditionally_approved_local_research_policy"
        and policy.get("repo") == spec["repo"]
        and policy.get("revision") == spec["revision"]
        and bool(policy.get("reviewer"))
    ):
        return False
    if selected == INTERNAL_POLICY:
        return True
    binding = {key: spec.get(key) for key in ("repo", "revision", "sha256")}
    if (
        policy.get("rights_policy") != INTERNAL_POLICY_V2
        or policy.get("source_binding") != binding
        or policy.get("approved") is not True
        or policy.get("redistribution_allowed") is not False
    ):
        return False
    authorization_proof = policy.get("authorization_proof")
    proposal_proof = policy.get("proposal_proof")
    if (
        not isinstance(proposal_proof, dict)
        or proposal_proof.get("sha256") != POLICY_V2_PROPOSAL_SHA256
    ):
        return False
    authorization_raw = _proof_data(authorization_proof, cache)
    proposal_raw = _proof_data(proposal_proof, cache)
    if authorization_raw is None or proposal_raw is None:
        return False
    try:
        authorization = json.loads(authorization_raw)
        proposal = json.loads(proposal_raw)
    except (ValueError, UnicodeError):
        return False
    proposal_binding = proposal.get("source_binding", {})
    return (
        authorization.get("version") == "codexa-pressbooks-rights-policy-2-authorization"
        and authorization.get("status") == "authorized"
        and authorization.get("approved") is True
        and authorization.get("rights_policy") == INTERNAL_POLICY_V2
        and bool(authorization.get("user_decision"))
        and authorization.get("source_binding") == binding
        and authorization.get("proposal_proof") == proposal_proof
        and authorization.get("proposed_authorization_text")
        == proposal.get("proposed_authorization_text")
        and proposal.get("version") == "codexa-pressbooks-rights-policy-2-proposal"
        and proposal_binding.get("repository") == spec["repo"]
        and proposal_binding.get("revision") == spec["revision"]
        and proposal_binding.get("source_file_sha256") == spec.get("sha256")
    )


def _archived_declaration(
    spec: dict[str, Any], grant: dict[str, Any], raw: bytes, family: str = "CC-BY"
) -> bool:
    try:
        proof = json.loads(raw)
    except (ValueError, UnicodeError):
        return False
    metadata = proof.get("publisher_directory_metadata", {})
    return (
        proof.get("version") == "pressbooks-archived-publisher-declaration-1"
        and proof.get("source_repository") == spec["repo"]
        and proof.get("source_revision") == spec["revision"]
        and proof.get("source_file_sha256") == spec.get("sha256")
        and type(proof.get("source_ordinal")) is int
        and proof["source_ordinal"] > 0
        and str(proof.get("book_url", "")).rstrip("/") == str(grant.get("book_url", "")).rstrip("/")
        and str(metadata.get("book_url", "")).rstrip("/")
        == str(grant.get("book_url", "")).rstrip("/")
        and bool(metadata.get("author"))
        and bool(metadata.get("title"))
        and proof.get("reconstructed_declared_family") == family
        and canonical_license(metadata.get("license"))["family"] == family
        and proof.get("effective_version")
        == (None if family in {"CC-BY", "CC-BY-SA"} else "CC0-1.0")
    )


def _api_version_proof(grant: dict[str, Any], raw: bytes) -> bool:
    try:
        proof = json.loads(raw)
    except (ValueError, UnicodeError):
        return False
    licence = proof.get("license", {})
    if not isinstance(licence, dict) or canonical_license(licence.get("url"))["spdx"] != grant.get(
        "license"
    ):
        return False
    if not proof.get("name") or not proof.get("author"):
        return False
    book = str(grant.get("book_url", ""))
    book_root = re.split(r"/(?:front-matter|chapter|back-matter|part)/", book, maxsplit=1)[
        0
    ].rstrip("/")
    return any(
        str(item.get("href", "")).startswith(book_root + "/wp-json/")
        for item in proof.get("_links", {}).get("self", [])
        if isinstance(item, dict)
    )


def reviewed_grants(
    spec: dict[str, Any],
    evidence: dict[str, Any],
    *,
    _cache: dict[tuple[str, str], bytes] | None = None,
) -> list[dict[str, Any]]:
    if not valid_source(spec) or any(
        evidence.get(k) != spec.get(k) for k in ("repo", "revision", "adapter")
    ):
        return []
    if (
        evidence.get("status") != "approved"
        or not evidence.get("reviewer")
        or evidence.get("policy") != POLICY
    ):
        return []
    cache = {} if _cache is None else _cache
    if (
        evidence.get("additional_policy_proof")
        and _proof_data(evidence["additional_policy_proof"], cache) is None
    ):
        return []
    grants = []
    for grant in evidence.get("coverage", []):
        if not isinstance(grant, dict) or not grant.get("attribution"):
            continue
        family_grant = grant.get("license") in {"CC-BY", "CC-BY-SA"}
        if family_grant:
            if (
                grant.get("license") == "CC-BY-SA"
                and spec.get("rights_policy") != INTERNAL_POLICY_V2
            ):
                continue
            if (
                grant.get("version_status") != "unresolved"
                or grant.get("scope") != "internal_research_training"
                or not _internal_policy(spec, evidence, cache)
            ):
                continue
            if grant.get("verification") not in {
                "original_license_verified",
                "archived_publisher_declaration_verified",
            }:
                continue
        elif (
            grant.get("license") not in ALLOWED_LICENSES
            or grant.get("verification") != "original_license_verified"
            or grant.get("scope") not in {"original_book", "original_documents"}
        ):
            continue
        if urlsplit(str(grant.get("basis_url", ""))).scheme != "https":
            continue
        if spec["adapter"] == PRESSBOOKS_ADAPTER:
            if not isinstance(grant.get("book_url"), str) or not grant["book_url"]:
                continue
        elif (
            not isinstance(grant.get("origin_prefixes"), list)
            or not grant["origin_prefixes"]
            or any(
                not isinstance(prefix, str)
                or urlsplit(prefix).scheme not in {"http", "https"}
                or not urlsplit(prefix).hostname
                for prefix in grant["origin_prefixes"]
            )
        ):
            continue
        raw = _proof_data(grant.get("proof"), cache)
        if raw is None:
            continue
        if (
            grant.get("license") == "public-domain"
            and spec["adapter"] == PRESSBOOKS_ADAPTER
            and not _reviewed_public_domain(spec, grant, raw, cache)
        ):
            continue
        if family_grant and not _archived_declaration(spec, grant, raw, grant["license"]):
            continue
        if grant.get("license") == "CC0-1.0":
            try:
                archived = json.loads(raw)
            except (ValueError, UnicodeError):
                archived = {}
            if (
                isinstance(archived, dict)
                and archived.get("version") == "pressbooks-archived-publisher-declaration-1"
                and not _archived_declaration(spec, grant, raw, "CC0")
            ):
                continue
        if grant.get("collector_normalized_version") and (
            _proof_data(grant.get("collector_proof", evidence.get("collector_proof")), cache)
            is None
            or not _api_version_proof(grant, raw)
        ):
            continue
        grants.append(grant)
    return grants


def _matching(
    row: dict[str, Any], spec: dict[str, Any], grants: list[dict[str, Any]]
) -> dict[str, Any] | None:
    allow_sa = _sa_book(row, spec) and any(grant.get("license") == "CC-BY-SA" for grant in grants)
    if (
        spec["adapter"] == PRESSBOOKS_ADAPTER
        and scoped_lines(str(row.get("text", "")), allow_sharealike=allow_sa)[2]
    ):
        return None
    for grant in grants:
        chapter_prefixes = grant.get("chapter_url_prefixes")
        if chapter_prefixes is not None:
            origin = str(row.get("id") or row.get("metadata", {}).get("url") or row.get("url", ""))
            if (
                not isinstance(chapter_prefixes, list)
                or not chapter_prefixes
                or not any(
                    isinstance(prefix, str)
                    and origin.startswith(prefix)
                    and urlsplit(origin).netloc == urlsplit(prefix).netloc
                    for prefix in chapter_prefixes
                )
            ):
                continue
        if grant.get("original_work_only"):
            origin = str(row.get("id") or row.get("metadata", {}).get("url") or row.get("url", ""))
            if re.search(
                r"(?i)editor|series[-_]introduction|discussion[-_]questions|new[-_]commentary",
                origin,
            ):
                continue
        if spec["adapter"] == PRESSBOOKS_ADAPTER:
            if grant.get("license") == "CC-BY-SA" and not _sa_book(row, spec):
                continue
            if grouping(row, spec) != grant["book_url"].rstrip("/"):
                continue
            label = canonical_license(row.get("metadata", {}).get("license"))
            expected = (
                grant["license"]
                if grant["license"] in {"CC-BY", "CC-BY-SA"}
                else "public-domain"
                if grant["license"] == "public-domain"
                else "CC0"
                if grant["license"] == "CC0-1.0"
                else grant["license"].rsplit("-", 1)[0]
            )
            if label["family"] != expected:
                continue
            if (
                label["spdx"]
                and label["spdx"] != grant["license"]
                and grant["license"] not in {"CC-BY", "CC-BY-SA"}
                and not grant.get("collector_normalized_version")
            ):
                continue
        else:
            origin = str(row.get("url", ""))
            if not any(
                origin.startswith(prefix) and urlsplit(origin).netloc == urlsplit(prefix).netloc
                for prefix in grant["origin_prefixes"]
            ):
                continue
        return grant
    return None


def compile_checker(
    spec: dict[str, Any], evidence: dict[str, Any]
) -> Callable[[dict[str, Any]], dict[str, Any] | None]:
    """Fresh proof validation and book index for one prepare/verification invocation."""
    cache: dict[tuple[str, str], bytes] = {}
    grants = reviewed_grants(spec, evidence, _cache=cache)
    declarations = {}
    for grant in grants:
        if grant.get("license") in {"CC-BY", "CC-BY-SA", "CC0-1.0"}:
            raw = _proof_data(grant["proof"], cache)
            assert raw is not None
            try:
                proof = json.loads(raw)
            except (ValueError, UnicodeError):
                proof = {}
            if proof.get("version") == "pressbooks-archived-publisher-declaration-1":
                declarations[id(grant)] = proof["publisher_directory_metadata"]
    by_book: dict[str, list[dict[str, Any]]] = {}
    for grant in grants:
        if spec.get("adapter") == PRESSBOOKS_ADAPTER:
            by_book.setdefault(grant["book_url"].rstrip("/"), []).append(grant)

    excluded = {str(book).rstrip("/") for book in evidence.get("excluded_book_urls", [])}

    def check(row: dict[str, Any]) -> dict[str, Any] | None:
        if grouping(row, spec) in excluded:
            return None
        selected = (
            by_book.get(grouping(row, spec), [])
            if spec.get("adapter") == PRESSBOOKS_ADAPTER
            else grants
        )
        grant = _matching(row, spec, selected)
        if grant is not None and id(grant) in declarations:
            expected = declarations[id(grant)]
            actual = row.get("metadata", {})
            if any(actual.get(key) != expected.get(key) for key in ("author", "title")):
                return None
        return grant

    return check


def matching_grant(
    row: dict[str, Any], spec: dict[str, Any], evidence: dict[str, Any]
) -> dict[str, Any] | None:
    return compile_checker(spec, evidence)(row)


def covered(row: dict[str, Any], spec: dict[str, Any], evidence: dict[str, Any]) -> bool:
    return matching_grant(row, spec, evidence) is not None


# Explicit creator-licensed narrative source for V4 callers only. Deliberately not
# registered in ADAPTERS: the established narrative reader/cleaner remains separate.
TINYSTORIES_REPO = "roneneldan/TinyStories"
TINYSTORIES_REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
TINYSTORIES_PATH = "TinyStoriesV2-GPT4-train.txt"
TINYSTORIES_SHA256 = "6418d412de72888f52b5142c761ac21a582f7d1166f0bfbdb5f03ccfdec90443"
TINYSTORIES_BYTES = 2227753162
TINYSTORIES_LICENSE = "CDLA-Sharing-1.0"
TINYSTORIES_RIGHTS_POLICY = "codexa-tinystories-creator-cdla-1"


def validate_narrative_rights(spec: dict[str, Any], evidence: dict[str, Any]) -> bool:
    """Validate the reviewed creator grant, never infer it from source authorization.

    V4 dispatch is the caller's responsibility. This does not register another V3
    adapter, change narrative normalization, authorize training or approve a release.
    """
    binding = {
        "repo": TINYSTORIES_REPO,
        "revision": TINYSTORIES_REVISION,
        "path": TINYSTORIES_PATH,
        "sha256": TINYSTORIES_SHA256,
        "bytes": TINYSTORIES_BYTES,
    }
    if any(spec.get(key) != value or evidence.get(key) != value for key, value in binding.items()):
        return False
    if (
        spec.get("rights_policy") != TINYSTORIES_RIGHTS_POLICY
        or spec.get("license") != TINYSTORIES_LICENSE
        or evidence.get("version") != "codexa-tinystories-creator-rights-evidence-1"
        or evidence.get("status") != "approved"
        or not evidence.get("reviewer")
        or evidence.get("rights_policy") != TINYSTORIES_RIGHTS_POLICY
        or evidence.get("license") != TINYSTORIES_LICENSE
        or evidence.get("scope") != "internal_research_training"
        or evidence.get("redistribution_allowed") is not False
    ):
        return False
    attribution = evidence.get("attribution")
    if not isinstance(attribution, str) or any(
        name not in attribution.casefold()
        for name in ("ronen eldan", "yuanzhi li", "roneneldan/tinystories")
    ):
        return False
    proofs = evidence.get("proofs")
    required = {
        "dataset_card",
        "generation_paper",
        "legalcode",
        "creator_declaration",
        "independent_review",
    }
    if not isinstance(proofs, dict) or not required <= set(proofs):
        return False
    cache: dict[tuple[str, str], bytes] = {}
    for role in required:
        if _proof_data(proofs[role], cache) is None:
            return False
    raw = _proof_data(proofs["independent_review"], cache)
    assert raw is not None
    try:
        decision = json.loads(raw)
    except (ValueError, UnicodeError):
        return False
    if (
        decision.get("version") != "codexa-tinystories-creator-rights-decision-1"
        or decision.get("status") != "approved_for_internal_research"
        or not decision.get("reviewer")
        or decision.get("source_binding") != binding
        or decision.get("license") != TINYSTORIES_LICENSE
        or decision.get("creator_text_grant_established") is not True
        or decision.get("internal_research_training_permitted") is not True
        or decision.get("redistribution_allowed") is not False
        or decision.get("universal_noninfringement_certified") is not False
    ):
        return False
    return _proof_data(decision.get("rights_review_proof"), cache) is not None
