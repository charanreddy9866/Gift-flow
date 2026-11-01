"""Human review: approve (after re-checking the reviewer's version), or reject.
Every correction is recorded, which is how accuracy gets measured."""
from __future__ import annotations

import json
import sqlite3
from collections import Counter

from .config import Settings
from .db import audit, loads, now
from .matching import search_donors
from .models import Flag
from .validate import has_blockers, ocr_dollar_amounts, validate

TRACKED_FIELDS = ["donor_name", "amount", "date", "check_number", "memo", "item_description",
                  "quantity", "stated_value", "suggested_donor_id", "suggested_fund", "suggested_campaign"]


class ApprovalBlocked(Exception):
    def __init__(self, flags: list[Flag]):
        self.flags = flags
        super().__init__("; ".join(f.message for f in flags if f.severity == "block"))


def approve(conn: sqlite3.Connection, doc_id: int, final: dict, reviewer: str, settings: Settings,
            donor_confirmed: bool = False, amount_confirmed: bool = False,
            coding_confirmed: bool = False) -> list[Flag]:
    """Re-validate the reviewer's version. Blocks approval if any blocking issue remains.

    `donor_confirmed`: the reviewer ticked "I've verified this donor", which lets an AI
    donor pick that its own search never returned through (and is audited).
    `amount_confirmed`: the reviewer ticked "I've verified this amount", which lets an AI
    amount that doesn't appear in the OCR text through (and is audited).
    `coding_confirmed`: the reviewer ticked "I've checked the fund and campaign", which is
    required whenever an AI-suggested fund or campaign is kept (and is audited)."""
    row = conn.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
    if row is None:
        raise ValueError(f"document {doc_id} not found")
    if row["status"] == "posted":
        raise ValueError("Already posted; it can't be changed here.")
    proposal = loads(row["proposal"], {})
    merged = {**proposal, **final}
    candidates = search_donors(conn, merged.get("donor_name"), merged.get("address"))
    flags = validate(conn, merged, candidates=candidates, match_threshold=settings.match_threshold,
                     ai_conf_threshold=settings.ai_conf_threshold, document_id=doc_id,
                     ai_donor_id=loads(row["extraction"], {}).get("suggested_donor_id"),
                     tool_calls=loads(row["tool_calls"], []), donor_confirmed=donor_confirmed,
                     ocr_text=row["ocr_text"], ai_amount=loads(row["extraction"], {}).get("amount"),
                     amount_confirmed=amount_confirmed,
                     ai_coding=(proposal.get("suggested_fund"), proposal.get("suggested_campaign"))
                     if proposal.get("coding_source") == "ai" else None,
                     coding_confirmed=coding_confirmed)
    if has_blockers(flags):
        raise ApprovalBlocked(flags)
    if coding_confirmed and any(f.code == "coding_ai_chosen" for f in flags):
        audit(conn, reviewer, "coding_confirmed", "document", doc_id,
              {"fund": merged.get("suggested_fund"), "campaign": merged.get("suggested_campaign"),
               "ai_confidence": proposal.get("ai_confidence")})
    if donor_confirmed and any(f.code == "donor_not_searched" for f in flags):
        audit(conn, reviewer, "donor_confirmed_unsearched", "document", doc_id,
              {"donor_id": merged.get("suggested_donor_id")})
    if amount_confirmed and any(f.code == "amount_not_in_ocr" for f in flags):
        audit(conn, reviewer, "amount_confirmed_not_in_ocr", "document", doc_id,
              {"amount": merged.get("amount")})
    if amount_confirmed and any(f.code == "amount_multiple_in_ocr" for f in flags):
        audit(conn, reviewer, "amount_confirmed_multiple_in_ocr", "document", doc_id,
              {"amount": merged.get("amount"),
               "ocr_amounts": sorted({round(v, 2) for v in ocr_dollar_amounts(row["ocr_text"])})})
    corrections = [f for f in TRACKED_FIELDS if proposal.get(f) != merged.get(f)]
    conn.execute(
        "UPDATE documents SET status='approved', final=?, corrections=?, flags=?, reviewer=?, reviewed_at=? WHERE id=?",
        (json.dumps(merged), json.dumps(corrections), json.dumps([f.model_dump() for f in flags]),
         reviewer, now(), doc_id),
    )
    audit(conn, reviewer, "document_approved", "document", doc_id, {"corrections": corrections})
    conn.commit()
    return flags


def create_donor(conn: sqlite3.Connection, name: str, address: str | None, reviewer: str) -> int:
    """New donors are only ever created by a person, never by the AI."""
    if not name or not name.strip():
        raise ValueError("Donor name is required")
    cur = conn.execute("INSERT INTO donors(name, address) VALUES (?, ?)", (name.strip(), address))
    audit(conn, reviewer, "donor_created", "donor", cur.lastrowid, {"name": name, "address": address})
    conn.commit()
    return cur.lastrowid


def create_donor_for_document(conn: sqlite3.Connection, doc_id: int, name: str, address: str | None,
                              reviewer: str, settings: Settings) -> int:
    """A person creates the donor from the review screen. The new donor becomes the
    document's suggested donor, and its flags are re-checked so the screen is current."""
    donor_id = create_donor(conn, name, address, reviewer)
    row = conn.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
    if row is None or row["status"] in ("approved", "posted"):
        return donor_id
    proposal = {**loads(row["proposal"], {}), "suggested_donor_id": donor_id}
    candidates = search_donors(conn, proposal.get("donor_name"), proposal.get("address"))
    flags = validate(conn, proposal, candidates=candidates, ocr_confidence=row["ocr_confidence"],
                     low_conf_words=loads(row["low_conf_words"], []),
                     match_threshold=settings.match_threshold, ai_conf_threshold=settings.ai_conf_threshold,
                     ocr_conf_threshold=settings.ocr_conf_threshold, document_id=doc_id,
                     ai_donor_id=loads(row["extraction"], {}).get("suggested_donor_id"),
                     tool_calls=loads(row["tool_calls"], []),
                     ocr_text=row["ocr_text"], ai_amount=loads(row["extraction"], {}).get("amount"))
    conn.execute("UPDATE documents SET proposal=?, candidates=?, flags=? WHERE id=?",
                 (json.dumps(proposal), json.dumps([c.model_dump() for c in candidates]),
                  json.dumps([f.model_dump() for f in flags]), doc_id))
    conn.commit()
    return donor_id


def reject(conn: sqlite3.Connection, doc_id: int, reviewer: str, reason: str) -> None:
    conn.execute("UPDATE documents SET status='rejected', reviewer=?, reviewed_at=? WHERE id=? AND status!='posted'",
                 (reviewer, now(), doc_id))
    audit(conn, reviewer, "document_rejected", "document", doc_id, {"reason": reason})
    conn.commit()


def accuracy_metrics(conn: sqlite3.Connection) -> dict:
    """Share of AI proposals accepted with no changes, and which fields get corrected most."""
    rows = conn.execute("SELECT corrections FROM documents WHERE status IN ('approved','posted')").fetchall()
    reviewed = len(rows)
    corrected = Counter()
    untouched = 0
    for r in rows:
        fields = loads(r["corrections"], [])
        if not fields:
            untouched += 1
        corrected.update(fields)
    return {
        "reviewed": reviewed,
        "accepted_unchanged": untouched,
        "acceptance_rate": round(untouched / reviewed, 3) if reviewed else None,
        "corrections_by_field": dict(corrected.most_common()),
    }
