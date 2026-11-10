"""Conservative representation and substance checks, without document-ID exceptions."""

import re
from fractions import Fraction
from typing import TypedDict

from codexa.data.math_integrity import affine_findings
from codexa.data.representation import rejection as representation_rejection

VERSION = "educational-integrity-3"
AFFECTED = {"finemath-4plus", "fineweb-edu", "openwebmath"}


class Finding(TypedDict):
    reason: str
    evidence: str


def findings(text: str, source: str | None) -> list[Finding]:
    if source not in AFFECTED:
        return []
    result: list[Finding] = []
    representation = representation_rejection(text)
    if representation is not None:
        result.append({"reason": representation, "evidence": text[:240]})

    def flag(reason: str, pattern: str, minimum: int = 1) -> None:
        matches = list(re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE))
        if len(matches) >= minimum:
            result.append({"reason": reason, "evidence": matches[0][0][:240]})

    flag(
        "interface_timestamp_contamination",
        r"^\s*[•*-]?\s*(?:\d+|an?|one)\s+(?:seconds?|minutes?|hours?)\s+ago\s*$",
        3,
    )
    flag(
        "unfilled_answer_editor",
        r"(?:explain in detail\s*\.{3}|then teach the underlying concepts|I want someone to double check my answer)",
    )
    flag(
        "search_catalogue",
        r"^\s*(?:Questions tagged\s*\[|Showing\s+\d+\s*[-–]\s*\d+\s+of\s+\d+\s+(?:resources|results)|Search results\s*(?:for|:))",
    )
    flag("catalogue_teasers", r"^\s*\d[\d,]*\s+(?:views?|votes?)\s*$", 4)
    terminal_sequence = re.search(r"(?:\d+\s*[,;+]\s*){2,}(?:\.{3}|…)\s*\Z", text)
    if terminal_sequence is None:
        flag("truncated_document", r"(?:\.{3,}|…)\s*\Z")
    last_line = text.strip().rsplit("\n", 1)[-1]
    if len(last_line.split()) >= 2 and not re.search(r"[=$\\]", last_line):
        flag(
            "unfinished_terminal_clause",
            r"\b(?:Since for|because the|due to|such that|and the|of the|in the|with the)\s*\Z",
        )
    listed_questions = re.findall(r"(?im)^\s*[-•]?\s*\d+[.)]?\s+([^\n?]{8,}\?)\s*$", text)
    duplicated_questions = [q for q in listed_questions if text.count(q) >= 2]
    if len(set(duplicated_questions)) >= 8:
        result.append(
            {
                "reason": "duplicated_question_catalogue",
                "evidence": "Repeated numbered question index; unresolved answer coherence requires semantic review",
            }
        )
    flag(
        "advertised_absent_resources",
        r"(?:everything is ready to download|(?:buy|purchase)\s+(?:this|the|our)\s+(?:worksheet|printable|resource|lesson pack)|(?:included are|includes?)\s+the\s+printables)",
    )
    if source == "fineweb-edu":
        # Mark repeated orphaned closing tags, but preserve lessons explaining
        # markup or containing intentional code examples.
        if not re.search(r"(?i)\b(?:html|markup|code|tags?)\b|```", text):
            closing = re.findall(r"</(?:article|section|div|table|span)>\s*", text, re.IGNORECASE)
            if len(closing) >= 2:
                result.append({"reason": "orphaned_html_closing_tags", "evidence": closing[0]})
        if len(text.split()) < 450:
            flag(
                "paywalled_content_missing",
                r"\b(?:subscribe|sign in|log in)\s+(?:now\s+)?to\s+(?:read|access|unlock)\s+(?:the\s+)?(?:full|complete|rest of (?:the|this))\s+(?:article|story|report)",
            )
    if len(text.split()) < 450:
        flag(
            "promoted_lesson_not_supplied",
            r"(?:to get|download)[^.\n]{0,200}(?:lesson|worksheet|curriculum)[^.\n]{0,100}click here",
        )
        invitations = re.findall(
            r"(?i)\b(?:try out|have a look|here is information|series of articles|you may be (?:especially )?interested)\b",
            text,
        )
        if len(invitations) >= 3:
            result.append(
                {
                    "reason": "educational_resource_directory",
                    "evidence": "Multiple resource invitations without the referenced lessons",
                }
            )
    # A list of advertised teaching materials is not a supplied lesson. Complete
    # self-contained activity instructions are not rejected merely for listing tasks.
    if (
        re.search(r"(?i)\b(?:resources|printables|worksheets|lesson plans)\b", text[:250])
        and len(text.split()) < 180
        and re.search(r"(?i)\b(?:our favorite|top \d+|ready to download|great supplement)\b", text)
    ):
        result.append({"reason": "resource_promotion_without_lesson", "evidence": text[:240]})
    if re.search(r"(?i)\b(?:words in bold|terms in bold|vocabulary definitions)\b", text[:250]):
        lines = [s.strip() for s in text.splitlines()[1:] if s.strip()]
        unlabeled = [
            s
            for s in lines
            if re.match(r"(?:A |An |The |Spores? |A process |A chemical )", s)
            and not re.search(r"[:=—]|\*\*", s)
        ]
        if len(unlabeled) >= 4:
            result.append(
                {"reason": "missing_definition_terms", "evidence": "\n".join(unlabeled[:2])[:240]}
            )
    # Repeated anonymous entries following an explicit list promise indicate
    # extracted labels have disappeared, independently of the topic involved.
    if re.search(r"(?i)(?:following is the list|following are the|list of)[^\n]{0,80}:?", text):
        anonymous = re.findall(
            r"(?im)^\s*(?:It is (?:also )?a[n]? |The next (?:family member|item)|This is a[n]? )",
            text,
        )
        if len(anonymous) >= 3:
            result.append(
                {
                    "reason": "missing_list_entry_labels",
                    "evidence": "Repeated anonymous entries after a list promise",
                }
            )
    flag("unresolved_notation_discussion", r"your formatting is unclear|or something else\?", 2)
    if source in AFFECTED:
        for reason, evidence in affine_findings(text):
            result.append({"reason": reason, "evidence": evidence[:240]})
        flag(
            "missing_numbered_visual_inputs",
            r"(?:equations? given|as (?:shown|in)|shown (?:in|by))[^.\n]{0,70}\b(?:Fig\.?|Figure|Diagram)\s*\d+",
            2,
        )
        repeated_renderings = []
        for expression in re.findall(r"\$([^$\n]{2,250})\$", text):
            compact = re.sub(r"\s+", "", expression).strip(".,")
            if (
                len(compact) >= 6
                and len(compact) % 2 == 0
                and compact[: len(compact) // 2] == compact[len(compact) // 2 :]
            ):
                repeated_renderings.append(expression)
        if len(repeated_renderings) >= 3:
            result.append(
                {"reason": "duplicated_math_rendering", "evidence": repeated_renderings[0][:240]}
            )
        flag(
            "absent_proof_visual",
            r"(?:the|this)\s+(?:above|below)\s+(?:figure|diagram)\s+(?:allows us to|shows that|gives us)",
        )
        environments = re.findall(r"\\(begin|end)\{([A-Za-z*]+)\}", text)
        stack = []
        for action, name in environments:
            if action == "begin":
                stack.append(name)
            elif not stack or stack.pop() != name:
                result.append({"reason": "unbalanced_math_environment", "evidence": name})
                break
        else:
            if stack:
                result.append({"reason": "unbalanced_math_environment", "evidence": stack[-1]})
        for expression in re.findall(r"(?<!\\)\$([^$]+)(?<!\\)\$", text):
            if not re.search(r"\\[A-Za-z]+", expression):
                continue
            braces = re.sub(r"\\[{}]", "", expression)
            depth = 0
            for character in braces:
                depth += (character == "{") - (character == "}")
                if depth < 0:
                    break
            if depth != 0:
                result.append({"reason": "unbalanced_math_braces", "evidence": expression[:240]})
                break
        # x2 may be a legitimate identifier. It becomes an extraction risk when
        # repeatedly used as a power in an explicit polynomial/power lesson.
        if re.search(r"(?i)\b(?:polynomials?|exponents?|powers? of|degree of)\b", text):
            flag("flattened_power_notation", r"(?<![A-Za-z_])(?:\d+)?[xyzab]\d+(?![A-Za-z_])", 5)
        numbered = re.findall(r"(?:\\angle|mangle|\bangle)\s*[1-9]\b", text)
        if len(numbered) >= 3 and not re.search(
            r"(?i)(?:denote|label(?:led)?|defined?|let)\b[^.\n]{0,100}(?:angle|\\angle)", text
        ):
            result.append({"reason": "undefined_numbered_angles", "evidence": numbered[0]})
        flag(
            "missing_worked_expression",
            r"(?:long division|we (?:get|obtain)|the result is)\s*:?\s*\n\s*\n\s*(?:Therefore|Hence|Thus|Solution|\d+\.)",
            2,
        )
        flag(
            "missing_required_visual",
            r"(?:find|calculate|determine|measure|what is|which)[^.!?\n]{0,130}(?:in|from|on)\s+(?:the\s+)?(?:diagram|figure|picture|graph)\s+(?:below|above|shown)",
        )
        # Check only fully explicit rational arithmetic, never eval arbitrary
        # expressions or reinterpret an unsupported mathematical notation.
        arithmetic = r"(?<![\w./])(-?\d+(?:/\d+)?)\s+([+−-])\s+(-?\d+(?:/\d+)?)\s*=\s*(-?\d+(?:/\d+)?)(?![\w/]|\.\d)"
        for match in re.finditer(arithmetic, text):
            try:
                a, b, c = (Fraction(match[i]) for i in (1, 3, 4))
            except ZeroDivisionError:
                continue
            actual = a + b if match[2] == "+" else a - b
            paragraph = text[
                max(0, text.rfind("\n\n", 0, match.start())) : text.find("\n\n", match.end())
                if "\n\n" in text[match.end() :]
                else len(text)
            ]
            corrected = re.search(
                r"(?i)\b(?:incorrect|mistake|wrong|false|not true|correction)\b", paragraph
            )
            if (
                actual != c
                and not corrected
                and not re.search(
                    r"(?i)\b(?:modulo|modular|modulus|incorrect|wrong|correction)\b|\\pmod", text
                )
            ):
                result.append({"reason": "explicit_arithmetic_contradiction", "evidence": match[0]})
    return result
