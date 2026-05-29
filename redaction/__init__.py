"""Local LLM redaction for VoxTerm transcripts."""

from .engine import (
    Finding,
    Redactor,
    RedactionError,
    RedactionResult,
    apply_redactions,
    apply_word_lists,
    custom_censor_spans,
    drop_allowed,
    get_redactor,
    overwrite_and_delete,
)
from .prompts import (
    CATEGORIES,
    DETECTION_PROFILE,
    PROFILES,
    RedactionProfile,
    resolve_profile,
)
from .tiers import (
    TIERS,
    Tier,
    filter_spans,
    next_tier,
    resolve_tier,
    tier_masks,
)

__all__ = [
    "CATEGORIES",
    "DETECTION_PROFILE",
    "Finding",
    "PROFILES",
    "Redactor",
    "RedactionError",
    "RedactionProfile",
    "RedactionResult",
    "TIERS",
    "Tier",
    "apply_redactions",
    "apply_word_lists",
    "custom_censor_spans",
    "drop_allowed",
    "filter_spans",
    "get_redactor",
    "next_tier",
    "overwrite_and_delete",
    "resolve_profile",
    "resolve_tier",
    "tier_masks",
]
