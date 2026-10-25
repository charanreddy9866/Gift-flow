"""Posting approved gifts to the CRM, plus batch reconciliation.

Posting is idempotent: each document can create at most one gift, and the same
check (donor + check number + amount) can never be posted twice. Running
"post" again after a failure only posts what's missing.

The CSV export follows the shape of a typical CRM gift-batch import (constituent,
date, amount, gift type, payment type, check number, fund, campaign), so a
reviewer can import it into a gift batch and commit it: a second approval point.
"""
from __future__ import annotations

import csv
import io
import sqlite3

from .db import audit, loads, now


def post_batch(conn: sqlite3.Connection, batch_id: int, actor: str) -> dict:
    results = {"posted": [], "skipped_duplicate": [], "errors": []}
    docs = conn.execute(
        "SELECT id, filename, final FROM documents WHERE batch_id=? AND status='approved' ORDER BY id",
        (batch_id,),
    ).fetchall()
    for d in docs:
        g = loads(d["final"], {})
        try:
            with conn:  # one transaction per gift
                conn.execute(
                    """INSERT INTO gifts(document_id, donor_id, gift_type, amount, gift_date, check_number,
                                         fund, campaign, description, posted_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    (d["id"], g["suggested_donor_id"], g["doc_type"],
                     g.get("amount") if g["doc_type"] == "check" else g.get("stated_value"),
                     g.get("date"), g.get("check_number"), g["suggested_fund"], g["suggested_campaign"],
                     g.get("item_description") or g.get("memo"), now()),
                )
                conn.execute("UPDATE documents SET status='posted' WHERE id=?", (d["id"],))
                audit(conn, actor, "gift_posted", "document", d["id"], {"donor_id": g["suggested_donor_id"]})
            results["posted"].append(d["filename"])
        except sqlite3.IntegrityError as exc:
            results["skipped_duplicate"].append(d["filename"])
            audit(conn, actor, "post_skipped_duplicate", "document", d["id"], {"error": str(exc)})
            conn.commit()
        except Exception as exc:  # noqa: BLE001 - keep posting the rest, report this one
            results["errors"].append({"file": d["filename"], "error": str(exc)})
            audit(conn, actor, "post_failed", "document", d["id"], {"error": str(exc)})
            conn.commit()
    return results


def reconcile(conn: sqlite3.Connection, batch_id: int) -> dict:
    """Checks posted (and approved) must add up to the bank deposit slip."""
    batch = conn.execute("SELECT * FROM batches WHERE id=?", (batch_id,)).fetchone()
    posted = conn.execute(
        """SELECT COALESCE(SUM(g.amount), 0) FROM gifts g JOIN documents d ON d.id = g.document_id
           WHERE d.batch_id=? AND g.gift_type='check'""", (batch_id,)).fetchone()[0]
    pending = 0.0
    for r in conn.execute("SELECT final FROM documents WHERE batch_id=? AND status='approved'", (batch_id,)):
        f = loads(r["final"], {})
        if f.get("doc_type") == "check":
            pending += f.get("amount") or 0
    diff = round(batch["deposit_total"] - posted - pending, 2)
    return {
        "deposit_total": round(batch["deposit_total"], 2),
        "posted_checks": round(posted, 2),
        "approved_not_posted": round(pending, 2),
        "difference": diff,
        "status": "balanced" if abs(diff) < 0.005 else ("short" if diff > 0 else "over"),
    }


def export_gift_csv(conn: sqlite3.Connection, batch_id: int) -> str:
    rows = conn.execute(
        """SELECT g.*, dn.name AS donor_name FROM gifts g
           JOIN documents d ON d.id = g.document_id JOIN donors dn ON dn.id = g.donor_id
           WHERE d.batch_id=? ORDER BY g.id""", (batch_id,)).fetchall()
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["ConstituentID", "ConstituentName", "GiftDate", "GiftAmount", "GiftType", "PaymentType",
                "CheckNumber", "FundID", "CampaignID", "Reference"])
    for r in rows:
        w.writerow([r["donor_id"], r["donor_name"], r["gift_date"], f"{r['amount'] or 0:.2f}",
                    "Gift-in-Kind" if r["gift_type"] == "gik" else "Cash",
                    "" if r["gift_type"] == "gik" else "Personal Check",
                    r["check_number"] or "", r["fund"], r["campaign"], r["description"] or ""])
    return out.getvalue()
