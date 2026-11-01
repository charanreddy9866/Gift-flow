"""Validation: code checks the AI's work before any person sees it.

"block" flags must be fixed before a gift can be approved.
"warn" flags are shown first in the review queue so a person looks closely.
"""
from __future__ import annotations

import re
import sqlite3
from datetime import date, timedelta

from .amounts import parse_money, words_to_amount
from .db import allowed_codes, loads
from .models import DonorCandidate, Flag

STALE_CHECK_DAYS = 180
MAX_REASONABLE_AMOUNT = 250_000


def _block(code: str, msg: str) -> Flag:
    return Flag(code=code, severity="block", message=msg)


def _warn(code: str, msg: str) -> Flag:
    return Flag(code=code, severity="warn", message=msg)


def find_duplicates(conn: sqlite3.Connection, p: dict, document_id: int | None) -> list[Flag]:
    """Same donor + check number + amount = the same check scanned twice."""
    if p.get("doc_type") != "check" or not p.get("check_number") or p.get("amount") is None:
        return []
    flags: list[Flag] = []
    if p.get("suggested_donor_id"):
        posted = conn.execute(
            "SELECT id FROM gifts WHERE gift_type='check' AND donor_id=? AND check_number=? AND amount=?",
            (p["suggested_donor_id"], p["check_number"], p["amount"]),
        ).fetchone()
        if posted:
            flags.append(_block("duplicate_posted",
                                f"Check #{p['check_number']} for ${p['amount']:,.2f} was already posted (gift {posted['id']})."))
    # Compare with earlier scans, and with anything already approved or posted.
    # (The first copy of a check stays approvable; the second copy is flagged.)
    if document_id is None:
        rows = conn.execute(
            "SELECT id, filename, proposal, final FROM documents WHERE status != 'rejected'").fetchall()
    else:
        rows = conn.execute(
            "SELECT id, filename, proposal, final FROM documents WHERE status != 'rejected' AND id != ? "
            "AND (id < ? OR status IN ('approved', 'posted'))",
            (document_id, document_id),
        ).fetchall()
    for r in rows:
        other = loads(r["final"]) or loads(r["proposal"]) or {}
        same_donor = (other.get("suggested_donor_id") == p.get("suggested_donor_id")) or (
            (other.get("donor_name") or "").lower() == (p.get("donor_name") or "").lower())
        if (other.get("check_number") == p.get("check_number")
                and other.get("amount") == p.get("amount") and same_donor):
            flags.append(_block("duplicate_scan",
                                f"Looks like the same check as {r['filename']} (same donor, check number and amount)."))
            break
    return flags


_OCR_DOLLARS = re.compile(r"\$\s*(\d[\d,]*(?:\.\d{2})?)")


def ocr_dollar_amounts(text: str | None) -> list[float]:
    """Every '$ 1,250.00'-style amount printed in the OCR text."""
    return [v for v in (parse_money(m) for m in _OCR_DOLLARS.findall(text or "")) if v is not None]


def searched_donor_ids(tool_calls: list[dict] | None) -> set[int]:
    """Donor IDs that the AI's own search_donors calls actually returned."""
    ids: set[int] = set()
    for call in tool_calls or []:
        result = call.get("result")
        if call.get("tool") == "search_donors" and isinstance(result, list):
            ids.update(c["donor_id"] for c in result if isinstance(c, dict) and "donor_id" in c)
    return ids


def validate(conn: sqlite3.Connection, p: dict, *, candidates: list[DonorCandidate] | None = None,
             ocr_confidence: float | None = None, low_conf_words: list[str] | None = None,
             match_threshold: float = 85, ai_conf_threshold: float = 0.8,
             ocr_conf_threshold: float = 70, document_id: int | None = None,
             ai_donor_id: int | None = None, tool_calls: list[dict] | None = None,
             donor_confirmed: bool = False, ocr_text: str | None = None,
             ai_amount: float | None = None, amount_confirmed: bool = False,
             today: date | None = None) -> list[Flag]:
    """`ai_donor_id` and `tool_calls` are the AI's raw donor choice and its function
    calls. When given, an AI-chosen donor must be one its own search returned, unless
    a reviewer has confirmed that donor (`donor_confirmed`).

    `ocr_text` and `ai_amount`: when given, an AI-proposed check amount must appear as a
    dollar amount in the OCR text. An amount the reviewer typed in is not checked, and a
    reviewer who has verified the AI's amount against the image can confirm it
    (`amount_confirmed`), e.g. when OCR misread the digits."""
    flags: list[Flag] = []
    today = today or date.today()
    funds, campaigns = allowed_codes(conn)
    is_check = p.get("doc_type") == "check"

    # --- required fields
    required = ["donor_name", "amount", "date", "check_number"] if is_check else ["donor_name", "item_description", "date"]
    missing = [f for f in required if p.get(f) in (None, "")]
    if missing:
        flags.append(_block("missing_fields", "Missing: " + ", ".join(missing) + ". Read it from the image."))

    # --- amount checks
    amount = p.get("amount")
    if is_check and amount is not None:
        if amount <= 0 or amount > MAX_REASONABLE_AMOUNT:
            flags.append(_block("amount_invalid", f"Amount ${amount:,.2f} is outside the expected range."))
        words_value = words_to_amount(p.get("amount_words"))
        if p.get("amount_words") and words_value is None:
            flags.append(_warn("amount_words_unreadable", "Couldn't read the written-out amount to cross-check."))
        elif words_value is not None and abs(words_value - amount) > 0.009:
            flags.append(_block("amount_mismatch",
                                f"Amount box says ${amount:,.2f} but the written amount says ${words_value:,.2f}. Check the image."))
        # More than one amount on a check (e.g. a fake one in the memo) means the OCR
        # cross-check below can be satisfied by the wrong figure. Make a person look.
        distinct = sorted({round(v, 2) for v in ocr_dollar_amounts(ocr_text)})
        if len(distinct) > 1:
            flags.append(_warn("amount_multiple_in_ocr",
                               "The scanned text shows more than one dollar amount: "
                               + ", ".join(f"${v:,.2f}" for v in distinct)
                               + ". Check the image for the real check amount."))
        # The model supplies both the amount box and the written amount, so they can agree
        # and still be invented (e.g. by a prompt injection). Check against what OCR read.
        if ocr_text and ai_amount is not None and abs(amount - ai_amount) <= 0.009:
            printed = ocr_dollar_amounts(ocr_text)
            if not printed:
                flags.append(_warn("amount_not_in_ocr_unverified",
                                   "Couldn't find a dollar amount in the OCR text to cross-check. Check the image."))
            elif not any(abs(v - amount) <= 0.009 for v in printed):
                shown = ", ".join(f"${v:,.2f}" for v in printed[:3])
                if amount_confirmed:
                    flags.append(_warn("amount_not_in_ocr",
                                       f"The AI says ${amount:,.2f} but the scanned text shows {shown}; "
                                       "a reviewer confirmed the amount from the image."))
                else:
                    flags.append(_block("amount_not_in_ocr",
                                        f"The AI says ${amount:,.2f}, but the scanned text shows {shown}. "
                                        "Check the image: enter the amount yourself, or confirm you've verified it."))

    # --- OCR confidence (digits matter most: they are amounts and dates)
    low = low_conf_words or []
    risky = [w for w in low if any(ch.isdigit() for ch in w)]
    if risky:
        flags.append(_warn("ocr_low_conf_digits", "Low-confidence numbers from OCR: " + ", ".join(risky[:6])))
    elif ocr_confidence is not None and ocr_confidence < ocr_conf_threshold:
        flags.append(_warn("ocr_low_conf", f"Overall OCR confidence is low ({ocr_confidence:.0f}/100)."))

    # --- date checks
    if p.get("date"):
        try:
            d = date.fromisoformat(p["date"])
            if d > today:
                flags.append(_block("date_future", f"Date {d} is in the future."))
            elif is_check and today - d > timedelta(days=STALE_CHECK_DAYS):
                flags.append(_warn("stale_check", f"Check is dated {d}, more than {STALE_CHECK_DAYS} days ago."))
        except ValueError:
            flags.append(_block("date_invalid", f"Date '{p['date']}' isn't a valid date."))

    # --- donor
    if not p.get("suggested_donor_id"):
        if candidates:
            flags.append(_block("donor_required",
                                f"Pick the donor. Closest match: {candidates[0].name} ({candidates[0].score:.0f}/100)."))
        else:
            flags.append(_block("donor_new", "No matching donor found. Confirm and create the donor record first."))
    elif candidates:
        chosen = next((c for c in candidates if c.donor_id == p["suggested_donor_id"]), None)
        if chosen is None or chosen.score < match_threshold:
            flags.append(_warn("donor_uncertain", "Donor match is below the confidence threshold. Confirm it."))
        elif len(candidates) > 1 and candidates[1].score >= match_threshold - 5:
            flags.append(_warn("donor_ambiguous",
                               f"Two similar donors: {candidates[0].name} and {candidates[1].name}."))
    # The AI may only pick a donor its own search returned. A different ID (e.g. from a
    # prompt injection) is blocked; a donor the reviewer picks themselves is not checked.
    # A reviewer who has verified the AI's pick can confirm it, which turns this into a warning.
    chosen_id = p.get("suggested_donor_id")
    # Whoever chose it, the donor must exist. Confirmation can't override this.
    if chosen_id and conn.execute("SELECT 1 FROM donors WHERE id=?", (chosen_id,)).fetchone() is None:
        flags.append(_block("donor_not_found",
                            f"Donor #{chosen_id} doesn't exist in the donor database. Pick an existing donor."))
    if (tool_calls is not None and ai_donor_id is not None and chosen_id == ai_donor_id
            and chosen_id not in searched_donor_ids(tool_calls)):
        if donor_confirmed:
            flags.append(_warn("donor_not_searched",
                               f"The AI chose donor #{chosen_id} without its search returning it; "
                               "a reviewer confirmed this donor."))
        else:
            flags.append(_block("donor_not_searched",
                                f"The AI chose donor #{chosen_id}, but its donor search never returned that donor. "
                                "Pick the donor yourself, or confirm you've verified this one."))

    # --- coding
    fund, campaign = p.get("suggested_fund"), p.get("suggested_campaign")
    if not fund or not campaign:
        flags.append(_block("coding_required", "Fund and campaign are required."))
    else:
        if fund not in funds:
            flags.append(_block("fund_invalid", f"Fund '{fund}' isn't in the fund list."))
        if campaign not in campaigns:
            flags.append(_block("campaign_invalid", f"Campaign '{campaign}' isn't in the campaign list."))
    if p.get("coding_source") == "ai" and (p.get("ai_confidence") or 0) < ai_conf_threshold:
        flags.append(_warn("coding_low_conf", "The AI wasn't sure about the fund/campaign. Check the memo."))

    # --- gift-in-kind
    if not is_check:
        if p.get("stated_value") in (None, 0):
            flags.append(_warn("gik_no_value", "No value stated by the donor. Staff record a value; the AI never assigns one."))
        if not p.get("quantity"):
            flags.append(_warn("gik_no_quantity", "Quantity is missing."))

    # --- duplicates
    flags.extend(find_duplicates(conn, p, document_id))
    return flags


def has_blockers(flags: list[Flag]) -> bool:
    return any(f.severity == "block" for f in flags)
