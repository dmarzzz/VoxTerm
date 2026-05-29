"""Built-in profiles for transcript redaction (local-LLM PII detection).

The LLM's job here is deliberately narrow: it does NOT rewrite the
transcript, it only *names the sensitive spans* as a JSON list. The engine
then does the actual masking by exact string replacement (see
``redaction/engine.py``). This split keeps the transcript verbatim — a small
local model can't paraphrase, drop, or hallucinate content into the output,
because it never produces the output.

Prompts are tuned for SMALL local models (0.6B–7B via MLX/Ollama): strict
JSON-only output, verbatim spans, and an explicit empty-result form, all of
which these models otherwise fumble.
"""

from __future__ import annotations

from dataclasses import dataclass


# The fixed category vocabulary. The engine coerces any out-of-set label the
# model emits down to OTHER, so adding/removing here only affects the prompt.
# Two families: IDENTIFIERS (who/where) and SENSITIVITY content-classes
# (what-about-them). The redaction *tiers* (redaction/tiers.py) decide which of
# these get masked for a given audience — detection finds them all.
CATEGORIES: tuple[str, ...] = (
    # --- identifiers ---
    "NAME",          # person names
    "EMAIL",         # email addresses
    "PHONE",         # phone / fax numbers
    "ADDRESS",       # street / mailing addresses
    "LOCATION",      # specific places that identify a person
    "DATE",          # specific dates, especially dates of birth
    "ID",            # SSN, account / card / licence / passport numbers
    "CREDENTIAL",    # passwords, API keys, tokens, secrets
    "URL",           # web links
    "HANDLE",        # social handles / usernames (@someone)
    # --- proper nouns (kept until the most-public tier) ---
    "ORG",           # organizations, companies, employers
    "PROJECT",       # project / product / software / protocol names
    # --- sensitivity content-classes (LLM-only; regex can't see these) ---
    "SUBSTANCE",     # drugs, alcohol, substance use
    "HEALTH",        # medical / mental-health details
    "SEXUAL",        # sexual orientation / activity
    "LEGAL",         # criminal or legal exposure
    "FINANCIAL",     # personal financial specifics (salary, debt, balances)
    "AFFILIATION",   # political / religious affiliation
    "RELATIONSHIP",  # family / personal relationships
    "OTHER",         # anything else identifying or sensitive
    "CUSTOM",        # user word-list match (always censored when redacting)
)


@dataclass(frozen=True)
class RedactionProfile:
    id: str
    label: str
    description: str
    system: str
    user: str  # uses {transcript} (and, for custom, {custom}) placeholders


# System prompt: identity + the hard "identify, don't rewrite" contract +
# strict JSON-only output rules. Kept terse — long system prompts dilute
# compliance on sub-3B models.
_SYSTEM = (
    "You are a strict PII detection engine. Your input is a transcript "
    "produced by speech recognition; it may contain word errors, "
    "timestamps, and speaker labels like [Alice] or 'Speaker 1'. You read "
    "it and identify spans of sensitive or personally-identifying "
    "information.\n\n"
    "You DO NOT summarize, translate, rewrite, correct, or modify the "
    "transcript in any way. Your ONLY output is a list of the sensitive "
    "spans.\n\n"
    "Output rules — follow EXACTLY:\n"
    "- Output ONLY a JSON array. No preamble, no commentary, no markdown "
    "code fences. Never write things like \"Here are the spans\".\n"
    "- Each element is an object: "
    '{"text": "<exact verbatim span>", "type": "<CATEGORY>"}.\n'
    "- Copy each span VERBATIM — character for character as it appears in "
    "the transcript — so it can be found by exact string match. Do not "
    "normalize, reformat, re-case, or paraphrase it.\n"
    "- CATEGORY is one of: NAME, EMAIL, PHONE, ADDRESS, LOCATION, DATE, ID, "
    "CREDENTIAL, URL, HANDLE, ORG (organizations/companies), PROJECT "
    "(products/software/protocols), SUBSTANCE (drugs/alcohol), HEALTH, "
    "SEXUAL, LEGAL, FINANCIAL, AFFILIATION (political/religious), "
    "RELATIONSHIP, OTHER.\n"
    "- Prefer the smallest span that captures the sensitive value — the "
    "name itself, not the whole sentence.\n"
    "- If the transcript contains no sensitive information, output exactly: []"
)


# Transcript is fenced and flagged as data, not instructions — reduces
# prompt-injection-by-transcript on small models.
_TRANSCRIPT = (
    "\n\nThe text between triple quotes is the transcript. Identify its "
    "sensitive spans; do not repeat or rewrite it.\n"
    '"""\n{transcript}\n"""'
)


_STANDARD_FOCUS = (
    "Find every span of these kinds: people's names; email addresses; "
    "phone numbers; street or mailing addresses; account, SSN, card, "
    "licence, or passport numbers; passwords, API keys, or other secrets; "
    "and URLs. Also flag organizations and specific locations when they "
    "would identify a particular person."
)


PROFILES: tuple[RedactionProfile, ...] = (
    RedactionProfile(
        id="standard",
        label="Standard",
        description="Names, contacts, IDs, secrets, identifying orgs/places",
        system=_SYSTEM,
        user=_STANDARD_FOCUS + _TRANSCRIPT,
    ),
    RedactionProfile(
        id="contact_only",
        label="Contacts only",
        description="Just names, emails, phones, and addresses",
        system=_SYSTEM,
        user=(
            "Find ONLY direct contact identifiers: people's names, email "
            "addresses, phone numbers, and street or mailing addresses. "
            "Ignore everything else." + _TRANSCRIPT
        ),
    ),
    RedactionProfile(
        id="aggressive",
        label="Aggressive",
        description="Standard + dates, ages, relationships, re-identifiers",
        system=_SYSTEM,
        user=(
            _STANDARD_FOCUS
            + " Additionally flag dates (especially birthdays and ages), "
            "job titles tied to a named person, family relationships, and "
            "any other detail that could help re-identify a specific "
            "individual. When in doubt, include it." + _TRANSCRIPT
        ),
    ),
    RedactionProfile(
        id="custom",
        label="Custom instruction…",
        description="Describe what to redact in your own words",
        system=_SYSTEM,
        user=(
            "Find every span matching this instruction: {custom}\n"
            "Use the closest matching CATEGORY, or OTHER." + _TRANSCRIPT
        ),
    ),
)


_BY_ID = {p.id: p for p in PROFILES}


def resolve_profile(profile_id: str) -> RedactionProfile:
    """Look up a profile by id. Falls back to ``standard`` if unknown."""
    return _BY_ID.get(profile_id, _BY_ID["standard"])


# Comprehensive detection used by the tier dial: find EVERYTHING across both
# families in a single pass, so cycling tiers only re-filters which categories
# get masked (no re-inference). The tier policy — not the prompt — decides
# what's actually redacted.
DETECTION_PROFILE = RedactionProfile(
    id="_detect_all",
    label="All",
    description="detect every identifier and sensitive content-class",
    system=_SYSTEM,
    user=(
        "Find every span of every kind: people's names, emails, phones, "
        "addresses, identifying locations, dates (esp. births), account/SSN/"
        "card/licence numbers, passwords/keys/secrets, URLs, social handles, "
        "organizations, project/product/software/protocol names, and any "
        "mention of substances/drugs, health, sexual matters, legal/criminal "
        "exposure, personal finances, political/religious affiliation, or "
        "family/personal relationships. Tag each with its closest CATEGORY. "
        "Be thorough — it is better to over-detect here; a later step decides "
        "what actually gets masked." + _TRANSCRIPT
    ),
)
