"""EMA item codebook for LENS.

The LENS study uses a 14-item ecological-momentary-assessment (EMA) battery
derived from PHQ-9 (Q1-Q9, Q11 sub-items, Q12-Q13 anxiety items) plus a
negative-event yes/no item (Q14).  Every Likert item is recorded on a 0-100
slider and discretised into four equal quartiles, mapped to frequency adverbs
("not at all" / "sometimes" / "often" / "constantly").  Q3 (sleep last night)
uses an intensity scale instead ("minimal" / "moderate" / "significant" /
"severe").

These constants are lifted verbatim from the rule-based template generator and
the row builder so downstream code does not have to re-derive them at runtime.
See the paper, Appendix B / Table 2.
"""
from __future__ import annotations

import math
from typing import Dict, Optional


# --------------------------------------------------------------------------- #
# Item inventory
# --------------------------------------------------------------------------- #
# The 14 EMA items, in the canonical column order.  Q11 has three sub-items
# Q11(0)/Q11(1)/Q11(2); Q14 is a yes/no item handled separately from the
# Likert items.
EMA_LIKERT_ITEMS = [
    "Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8", "Q9", "Q10",
    "Q11(0)", "Q11(1)", "Q11(2)", "Q12", "Q13",
]
EMA_YESNO_ITEMS = ["Q14"]
EMA_ITEMS = EMA_LIKERT_ITEMS + EMA_YESNO_ITEMS

# Column names as they appear in feature_rows.csv (prefixed with ``ema_``).
EMA_COLUMNS = [f"ema_{q}" for q in EMA_ITEMS]

# PHQ-9 core items used for the summary severity score.
PHQ9_ITEMS = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8", "Q9"]

# Q11 sub-item meanings (positively framed comprehension checks).
Q11_SUBITEMS = {
    "Q11(0)": "interest or pleasure in doing things",
    "Q11(1)": "energy",
    "Q11(2)": "concentration",
}


# --------------------------------------------------------------------------- #
# 0-100 -> quartile -> label discretisation
# --------------------------------------------------------------------------- #
# Quartile bins on the 0-100 slider scale (inclusive bounds).
QUARTILE_BINS = [(0, 25), (26, 50), (51, 75), (76, 100)]

# Frequency-adverb labels for the four quartiles.
FREQUENCY_MAP: Dict[tuple, str] = {
    (0, 25): "not at all",
    (26, 50): "sometimes",
    (51, 75): "often",
    (76, 100): "constantly",
}

# Sleep-intensity labels (used only by Q3, which refers to "last night").
SLEEP_INTENSITY_MAP: Dict[tuple, str] = {
    (0, 25): "minimal",
    (26, 50): "moderate",
    (51, 75): "significant",
    (76, 100): "severe",
}

# 0-100 quartile -> 0-3 PHQ-9 item score.
PHQ9_QUARTILE_TO_SCORE: Dict[tuple, int] = {
    (0, 25): 0,    # not at all
    (26, 50): 1,   # several days
    (51, 75): 2,   # more than half the days
    (76, 100): 3,  # nearly every day
}

# PHQ-9 total -> severity interpretation (standard ranges).
PHQ9_SEVERITY: Dict[tuple, str] = {
    (0, 0): "No depression",
    (1, 4): "Minimal depression",
    (5, 9): "Mild depression",
    (10, 14): "Moderate depression",
    (15, 19): "Moderately severe depression",
    (20, 27): "Severe depression",
}


def _isna(score) -> bool:
    if score is None:
        return True
    try:
        return math.isnan(float(score))
    except (TypeError, ValueError):
        return True


def _lookup(score, mapping: Dict[tuple, str], default: str) -> str:
    if _isna(score):
        return default
    try:
        s = float(score)
    except (TypeError, ValueError):
        return default
    if s < 0:
        return default
    for (lo, hi), label in mapping.items():
        if lo <= s <= hi:
            return label
    return default


def frequency_adverb(score) -> str:
    """Map a 0-100 slider score to a frequency adverb (default 'not at all')."""
    return _lookup(score, FREQUENCY_MAP, "not at all")


def sleep_intensity(score) -> str:
    """Map a 0-100 sleep-disturbance score to an intensity word (default 'no')."""
    return _lookup(score, SLEEP_INTENSITY_MAP, "no")


def phq9_item_score(score) -> int:
    """Map a single 0-100 PHQ-9 item to its 0-3 score (NaN/missing -> 0)."""
    if _isna(score):
        return 0
    try:
        s = float(score)
    except (TypeError, ValueError):
        return 0
    for (lo, hi), v in PHQ9_QUARTILE_TO_SCORE.items():
        if lo <= s <= hi:
            return v
    # Anything above 100 falls in the top bin; below 0 in the bottom.
    return 3 if s > 100 else 0


def phq9_total(scores: Dict[str, float]) -> int:
    """Sum the 0-3 PHQ-9 item scores.

    ``scores`` may be keyed either by bare item name (``"Q1"``) or by the
    ``ema_``-prefixed column name (``"ema_Q1"``).
    """
    total = 0
    for q in PHQ9_ITEMS:
        if q in scores:
            total += phq9_item_score(scores[q])
        elif f"ema_{q}" in scores:
            total += phq9_item_score(scores[f"ema_{q}"])
    return total


def phq9_interpretation(total_score: int) -> str:
    """Return the standard PHQ-9 severity label for a 0-27 total score."""
    for (lo, hi), label in PHQ9_SEVERITY.items():
        if lo <= total_score <= hi:
            return label
    return "Score out of range"


# --------------------------------------------------------------------------- #
# Rule-based item / summary narrative templates
# --------------------------------------------------------------------------- #
# Sentence templates per item.  ``{frequency}`` is filled from FREQUENCY_MAP,
# ``{intensity}`` from SLEEP_INTENSITY_MAP (Q3 only); Q14 is fixed text.
QUESTION_TEMPLATES: Dict[str, str] = {
    "Q1": "The user {frequency} had little interest or pleasure in doing things",
    "Q2": "The user {frequency} felt down, depressed, or hopeless",
    "Q3": "The user had {intensity} trouble with sleep last night",
    "Q4": "The user {frequency} felt tired or had little energy",
    "Q5": "The user {frequency} had poor appetite or was overeating",
    "Q6": "The user {frequency} felt bad about themselves",
    "Q7": "The user {frequency} had trouble concentrating",
    "Q8": "The user {frequency} moved or spoke slowly, or was fidgeting more",
    "Q9": "The user {frequency} had thoughts of hurting themselves or that "
          "they would be better off dead",
    "Q10": "The user {frequency} had headaches, abdominal discomfort, or body aches",
    "Q11(0)": "The user {frequency} had interest or pleasure in doing things",
    "Q11(1)": "The user {frequency} had a lot of energy",
    "Q11(2)": "The user {frequency} was able to concentrate well",
    "Q12": "The user {frequency} felt nervous, anxious, or on edge",
    "Q13": "The user {frequency} was not able to stop or control worrying",
    "Q14": "The user experienced a negative event",
}

# Order used when concatenating item sentences into a summary narrative.
NARRATIVE_ITEM_ORDER = [
    "Q1", "Q2", "Q4", "Q5", "Q6", "Q7", "Q8", "Q9", "Q10",
    "Q3", "Q11(0)", "Q11(1)", "Q11(2)", "Q12", "Q13",
]


def _norm_key(key: str) -> str:
    return key[4:] if key.startswith("ema_") else key


def item_sentence(item: str, score) -> str:
    """Render one item's rule-based sentence (empty string if not applicable)."""
    q = _norm_key(item)
    if q == "Q14":
        return QUESTION_TEMPLATES["Q14"] if str(score).strip().lower() == "yes" else ""
    template = QUESTION_TEMPLATES.get(q, "")
    if not template:
        return ""
    if _isna(score):
        return ""
    if q == "Q3":
        return template.format(intensity=sleep_intensity(score))
    return template.format(frequency=frequency_adverb(score))


def question_response(item: str, score) -> Optional[str]:
    """Per-item answer for QA-style outputs (None if the item is missing)."""
    q = _norm_key(item)
    if q == "Q14":
        if _isna(score):
            return None
        return (QUESTION_TEMPLATES["Q14"] if str(score).strip().lower() == "yes"
                else "The user did not experience a negative event")
    if _isna(score):
        return None
    return item_sentence(item, score) or None


def summary_narrative(scores: Dict[str, float]) -> str:
    """Build the rule-based summary narrative from a dict of item scores.

    Keys may be bare item names or ``ema_``-prefixed.  Mirrors the behaviour of
    the study's ``EMATemplateGenerator.generate_narrative``.
    """
    normalised = {_norm_key(k): v for k, v in scores.items()}
    parts = []
    for q in NARRATIVE_ITEM_ORDER:
        if q not in normalised or _isna(normalised[q]):
            continue
        s = item_sentence(q, normalised[q])
        if s:
            parts.append(s)
    if "Q14" in normalised and str(normalised["Q14"]).strip().lower() == "yes":
        parts.append(QUESTION_TEMPLATES["Q14"])
    total = phq9_total(normalised)
    parts.append(f"Overall, the user presented with {phq9_interpretation(total).lower()}")
    return ". ".join(parts) + "."
