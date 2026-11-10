"""Small cross-source checks for contradictory identity fields and erased math slots."""

import re
import unicodedata


def same_subject(a: str, b: str) -> bool:
    def first_name(value: str) -> str:
        parts = value.casefold().split()
        if not parts:
            return ""
        return "".join(c for c in unicodedata.normalize("NFKD", parts[0]) if c.isalpha())

    if first_name(a) != first_name(b):
        return False
    a = "".join(c for c in a.casefold() if c.isalpha())
    b = "".join(c for c in b.casefold() if c.isalpha())
    if len(a) < 6 or len(b) < 6:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) <= 1
    if abs(len(a) - len(b)) == 1:
        longer, shorter = (a, b) if len(a) > len(b) else (b, a)
        return any(longer[:i] + longer[i + 1 :] == shorter for i in range(len(longer)))
    return False


def explicit_birth_year(details: str) -> str | None:
    # A parent's dates or later career events do not describe this birth.
    birth_clause = re.split(
        r"(?i)\b(?:to|son|daughter|father|mother|parents|where|who|and|raised)\b", details
    )[0]
    month = r"(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?"
    year = re.search(
        rf"(?i)(?:\A\s*|\bin\s+(?:the year\s+)?|\bon\s+(?:{month}\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+|\d{{1,2}}(?:st|nd|rd|th)?\s+{month}\s+))((?:1\d{{3}}|20\d{{2}}))\b",
        birth_clause,
    )
    return year[1] if year else None


def rejection(text: str) -> str | None:
    lead = re.match(r"\A([^()\n]{3,100})\(([^)\n]{5,100})\)", text)
    if lead is not None:
        years = re.findall(r"\b(?:1\d{3}|20\d{2})\b", lead[2])
        if years and (len(years) >= 2 or re.search(r"(?i)\bborn\b", lead[2])):
            for birth in re.finditer(r"(?m)^([^\n.]{3,100}?)\s+was born\s+([^\n.]{0,100})", text):
                body_year = explicit_birth_year(birth[2])
                qualified = re.search(
                    r"(?i)some sources|disput|uncertain|alternativ|according to", birth[0]
                )
                if (
                    body_year is not None
                    and not qualified
                    and same_subject(lead[1], birth[1])
                    and body_year != years[0]
                ):
                    return "contradictory_birth_fields"
    if re.search(r"(?i)fill in (?:the )?blanks|complete (?:the )?blanks", text):
        return None
    gaps = re.findall(
        r"(?i)\b(?:where|if|let|since|while|when)[ \t]{2,}(?:is|be|denote[sd]?|represents?|has)\b|"
        r"\b(?:in|of|from|by|between|as|on|at|and|or)[ \t]{2,}(?:[.,;:]|(?:and|or|to|on|is|was)\b)|"
        r"\b(?:of|with|over|around|and|to be)[ \t]{2,}of\b|"
        r"\brising to[ \t]{2,}(?:sustained|per|[.,])|"
        r"\b(?:reserves?|width|height|length|depth|volume|mass|quantity|rate|output|capacity) of[ \t]+[.,;]|"
        r"\b(?:transcribed|transliterated|pronounced|written|represented|measured) as[ \t]*(?:[,;]|\))",
        text,
    )
    if len(gaps) >= 3:
        return "repeated_erased_content_slots"
    if re.search(
        r"(?is)(?:incorrect|wording|text):\s*\n\s*\n"
        r"instead of (?:the )?original (?:text|wording):\s*\n\s*\n\[\[Image:",
        text,
    ):
        return "erased_required_content_block"
    # Empty introduced blocks are only rejected with corroborating missing
    # content; ordinary prose after a colon and optional illustrations survive.
    if len(gaps) >= 1 and re.search(
        r"(?is)(?:incorrect|wording|text|equation|formula|criterion)[^\n]{0,80}:\s*\n\s*\n"
        r"(?:instead of (?:the )?original (?:text|wording):|\[\[Image:|where[ \t]{2,})",
        text,
    ):
        return "erased_required_content_block"
    for match in re.finditer(
        r"(?i)\bcatalyzes the (?:following )?chemical reaction\s*\n\s*\n([^\n]+)", text
    ):
        reaction = match[1]
        if not re.search(
            r"[=⇌⇄⇆→←↔⟶⟵⟷⟹⟺⇒⇔]|->|<=>|"
            r"\\(?:rightleftharpoons|leftrightharpoons|(?:long|x)?(?:leftrightarrow|rightarrow|leftarrow))|"
            r"(?i:\b(?:to|produces?|yields?|forms?|converted)\b)",
            reaction,
        ):
            return "erased_reaction_relation"
    for claim in re.finditer(
        r"(?i)\bis a sovereign (?:nation|country|state)[^.\n]{0,150}"
        r"(?:township|county)[^\n]{0,100}in (?:the )?(?:U\.S\.|US|United States) state[^\n]*",
        text,
    ):
        context = text[
            max(text.rfind("\n", 0, claim.start()) + 1, claim.start() - 180) : claim.end()
        ]
        if not re.search(r"(?i)self.proclaimed|unrecogni[sz]ed|micronation|claims?", context):
            return "contradictory_sovereign_locality_claim"
    slots = re.findall(
        r"(?i)\bfrom[ \t]{2,}to\b|\b(?:ring|field|point|number),?\s*(?:for example|say),\s*,\s*then\b|"
        r"(?:can be described as (?:the|a) (?:functor|map)|consider (?:the|a) group scheme)\s*\n\s*\n(?=(?:can be|If |The |For |In ))",
        text,
    )
    if len(slots) >= 2:
        return "erased_mathematical_slots"
    return None
