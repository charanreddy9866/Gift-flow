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
from .validate import has_blockers, validate

TRACKED_FIELDS = ["donor_name", "amount", "date", "check_number", "memo", "item_description",
                  "quantity", "stated_value", "suggested_donor_id", "suggested_fund", "suggested_campaign"]


class ApprovalBlocked(Exception):
    def __init__(self, flags: list[Flag]):
        self.flags = flags
        super().__init__("; ".join(f.message for f in flags if f.severity == "block"))


def approve(conn: sqlite3.Connection, doc_id: int, final: dict, reviewer: str, settings: Settings) -> list[Flag]:
    """Re-validate the reviewer's version. Blocks approval if any blocking issue remains."""
    row = conn.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
    if row is None:
        raise ValueError(f"document {doc_id} not found")
    if row["status"] == "posted":
        raise ValueError("Already posted; it can't be changed here.")
    proposal = loads(row["proposal"], {})
    merged = {**proposal, **final}
    candidates = search_donors(conn, merged.get("donor_name"), merged.get("address"))
    flags = validate(conn, merged, candidates=candidates, match_threshold=settings.match_threshold,
                     ai_conf_threshold=settings.ai_conf_threshold, document_id=doc_id)
    if has_blockers(flags):
        raise ApprovalBlocked(flags)
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
