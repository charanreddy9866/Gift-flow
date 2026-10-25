"""Fund and campaign coding.

Known keywords are handled by plain rules (predictable, free, easy to audit).
The AI only suggests codes when no rule applies, and its suggestion must come
from the allowed list or it is flagged.
"""
from __future__ import annotations

RULES: list[tuple[tuple[str, ...], str, str]] = [
    # keywords in the memo/description  -> fund, campaign
    (("thanksgiving", "turkey"), "MEALS", "THANKSGIVING-2026"),
    (("building", "capital", "renovation"), "BUILDING", "CAPITAL-2026"),
    (("christmas", "holiday", "holidays"), "GENERAL", "HOLIDAY-2026"),
    (("meal", "meals", "food", "kitchen"), "MEALS", "ANNUAL-2026"),
    (("shelter", "beds", "coats", "blankets"), "SHELTER", "ANNUAL-2026"),
]


def rule_coding(*texts: str | None) -> tuple[str, str] | None:
    blob = " ".join(t.lower() for t in texts if t)
    for keywords, fund, campaign in RULES:
        if any(k in blob for k in keywords):
            return fund, campaign
    return None
