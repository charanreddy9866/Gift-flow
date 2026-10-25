"""Data shapes shared across the pipeline."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

DocType = Literal["check", "gik"]


class Extraction(BaseModel):
    """What the AI must return for every document. Unknown fields stay null."""

    doc_type: DocType
    donor_name: Optional[str] = None
    address: Optional[str] = None
    amount: Optional[float] = None          # numeric box on a check
    amount_words: Optional[str] = None      # written-out amount, verbatim
    date: Optional[str] = None              # YYYY-MM-DD
    check_number: Optional[str] = None
    memo: Optional[str] = None
    item_description: Optional[str] = None  # gift-in-kind only
    quantity: Optional[int] = None          # gift-in-kind only
    stated_value: Optional[float] = None    # gift-in-kind, value stated by donor
    suggested_donor_id: Optional[int] = None
    suggested_fund: Optional[str] = None
    suggested_campaign: Optional[str] = None
    ai_confidence: float = Field(0.0, ge=0.0, le=1.0)
    reasoning: str = ""


# JSON Schema used for OpenAI "structured output" (strict mode requires every
# property to be listed as required; nullable fields use ["type", "null"]).
EXTRACTION_JSON_SCHEMA = {
    "name": "gift_extraction",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "doc_type": {"type": "string", "enum": ["check", "gik"]},
            "donor_name": {"type": ["string", "null"]},
            "address": {"type": ["string", "null"]},
            "amount": {"type": ["number", "null"]},
            "amount_words": {"type": ["string", "null"]},
            "date": {"type": ["string", "null"], "description": "YYYY-MM-DD"},
            "check_number": {"type": ["string", "null"]},
            "memo": {"type": ["string", "null"]},
            "item_description": {"type": ["string", "null"]},
            "quantity": {"type": ["integer", "null"]},
            "stated_value": {"type": ["number", "null"]},
            "suggested_donor_id": {"type": ["integer", "null"]},
            "suggested_fund": {"type": ["string", "null"]},
            "suggested_campaign": {"type": ["string", "null"]},
            "ai_confidence": {"type": "number"},
            "reasoning": {"type": "string"},
        },
        "required": [
            "doc_type", "donor_name", "address", "amount", "amount_words", "date",
            "check_number", "memo", "item_description", "quantity", "stated_value",
            "suggested_donor_id", "suggested_fund", "suggested_campaign",
            "ai_confidence", "reasoning",
        ],
    },
}


class Flag(BaseModel):
    code: str
    severity: Literal["block", "warn"]   # "block" must be fixed before approval
    message: str


class DonorCandidate(BaseModel):
    donor_id: int
    name: str
    address: Optional[str] = None
    score: float  # 0-100
