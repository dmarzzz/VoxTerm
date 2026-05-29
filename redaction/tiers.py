"""Disclosure tiers — redaction as a policy keyed to *audience*, not amount.

The dial has concentric rings: from RAW (just me) outward to WORLD (the open
internet). Each tier is a nested mask-policy over the category vocabulary in
``prompts.py``. Detection finds every category in one pass; the tier decides
which of those found spans actually get masked — so cycling the dial only
re-filters (no re-inference).

The sets are strictly nested:  RAW ⊂ INNER ⊂ ROOM ⊂ WORLD.
"""

from __future__ import annotations

from dataclasses import dataclass


# Category groupings the tiers compose from.
_SECRETS = {"CREDENTIAL", "ID"}  # hazardous regardless of who's trusted
_PERSONAL = {
    # personal identifiers
    "NAME", "EMAIL", "PHONE", "ADDRESS", "LOCATION", "DATE", "URL", "HANDLE",
    # personal-sensitive content-classes
    "SUBSTANCE", "HEALTH", "SEXUAL", "LEGAL", "FINANCIAL", "AFFILIATION",
    "RELATIONSHIP", "OTHER",
}
_PROPER = {"ORG", "PROJECT"}  # the "work" — kept until the most-public tier


@dataclass(frozen=True)
class Tier:
    id: str
    label: str
    color: str          # hex for the dial meter / badge
    rank: int           # 0=RAW … 3=WORLD; fills rank+1 of 4 meter segments
    audience: str       # who's on the other side
    description: str    # what it does, one line
    masks: frozenset[str]  # categories masked at this tier


TIERS: tuple[Tier, ...] = (
    Tier(
        id="raw",
        label="RAW",
        color="#6a6760",
        rank=0,
        audience="just me / archive",
        description="no redaction (explicit)",
        masks=frozenset(),
    ),
    Tier(
        id="inner",
        label="INNER",
        color="#74b6a6",
        rank=1,
        audience="a trusted recipient",
        description="strip only hard secrets (keys, IDs)",
        masks=frozenset(_SECRETS),
    ),
    Tier(
        id="room",
        label="ROOM",
        color="#e0a44a",
        rank=2,
        audience="a meetup / the cohort",
        description="strip the people, keep the work",
        masks=frozenset(_SECRETS | _PERSONAL),
    ),
    Tier(
        id="world",
        label="WORLD",
        color="#d8806e",
        rank=3,
        audience="the public internet",
        description="strip every identifier and proper noun",
        masks=frozenset(_SECRETS | _PERSONAL | _PROPER),
    ),
)

_BY_ID = {t.id: t for t in TIERS}
TIER_COUNT = len(TIERS)


def resolve_tier(tier_id: str) -> Tier:
    """Look up a tier by id. Falls back to ``room`` if unknown."""
    return _BY_ID.get(tier_id, _BY_ID["room"])


def tier_masks(tier: Tier, category: str) -> bool:
    """Whether ``category`` is masked at ``tier``."""
    return category.upper() in tier.masks


def next_tier(tier: Tier) -> Tier:
    """The next tier when cycling the dial (wraps WORLD → RAW)."""
    return TIERS[(tier.rank + 1) % TIER_COUNT]


def filter_spans(
    tier: Tier, spans: list[tuple[str, str]]
) -> list[tuple[str, str]]:
    """Keep only the spans whose category this tier masks."""
    return [(t, ty) for (t, ty) in spans if tier_masks(tier, ty)]
