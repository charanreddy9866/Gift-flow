"""The intake pipeline: scan -> OCR -> redact -> AI extract (+ donor lookup)
-> rule-based coding -> validation -> review queue.

Nothing here posts a gift. Posting only happens after a person approves.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .coding import rule_coding
from .config import Settings
from .db import allowed_codes, audit, now
from .llm import Extractor
from .matching import search_donors
from .ocr import run_ocr
from .validate import validate


def create_batch(conn: sqlite3.Connection, name: str, deposit_total: float, actor: str = "system") -> int:
    cur = conn.execute("INSERT INTO batches(name, deposit_total, created_at) VALUES (?,?,?)",
                       (name, deposit_total, now()))
    audit(conn, actor, "batch_created", "batch", cur.lastrowid, {"name": name, "deposit_total": deposit_total})
    conn.commit()
    return cur.lastrowid


def build_proposal(extraction: dict) -> dict:
    """What the reviewer sees: the AI's fields, with rule-based coding when a rule applies."""
    p = dict(extraction)
    coded = rule_coding(p.get("memo"), p.get("item_description"))
    if coded:
        p["suggested_fund"], p["suggested_campaign"] = coded
        p["coding_source"] = "rule"
    else:
        p["coding_source"] = "ai"
    return p


def process_document(conn: sqlite3.Connection, batch_id: int, image_path: str | Path,
                     extractor: Extractor, settings: Settings) -> int:
    path = Path(image_path)
    try:
        ocr = run_ocr(path, settings.ocr_conf_threshold)
        ocr_error = None
    except Exception as exc:  # e.g. Tesseract not installed, unreadable file
        from .ocr import OcrResult
        ocr, ocr_error = OcrResult(text="", mean_confidence=0.0, engine="failed"), str(exc)

    def tool_executor(name: str, args: dict):
        if name == "search_donors":
            return [c.model_dump() for c in search_donors(conn, args.get("name"), args.get("address"))]
        return {"error": f"unknown tool {name}"}

    funds, campaigns = allowed_codes(conn)
    try:
        extraction, tool_calls = extractor.extract(ocr.text, funds, campaigns, tool_executor)
        extraction_dict = extraction.model_dump()
        error = None
    except Exception as exc:  # the document still goes to review, with the error shown
        extraction_dict, tool_calls, error = {"doc_type": "check"}, [], str(exc)

    proposal = build_proposal(extraction_dict)
    candidates = search_donors(conn, proposal.get("donor_name"), proposal.get("address"))
    flags = validate(conn, proposal, candidates=candidates, ocr_confidence=ocr.mean_confidence,
                     low_conf_words=ocr.low_conf_words, match_threshold=settings.match_threshold,
                     ai_conf_threshold=settings.ai_conf_threshold,
                     ocr_conf_threshold=settings.ocr_conf_threshold,
                     ai_donor_id=extraction_dict.get("suggested_donor_id"), tool_calls=tool_calls)
    from .models import Flag
    if error:
        flags.insert(0, Flag(code="extraction_failed", severity="block",
                             message=f"AI extraction failed: {error}. Enter the fields by hand."))
    if ocr_error:
        flags.insert(0, Flag(code="ocr_failed", severity="block",
                             message=f"OCR couldn't read this image ({ocr_error[:120]}). Rescan or enter by hand."))

    cur = conn.execute(
        """INSERT INTO documents(batch_id, filename, image_path, doc_type, ocr_text, ocr_confidence, low_conf_words,
                                 extraction, proposal, candidates, flags, tool_calls)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
           ON CONFLICT(batch_id, filename) DO NOTHING""",
        (batch_id, path.name, str(path.resolve()), proposal.get("doc_type"), ocr.text, ocr.mean_confidence,
         json.dumps(ocr.low_conf_words), json.dumps(extraction_dict), json.dumps(proposal),
         json.dumps([c.model_dump() for c in candidates]),
         json.dumps([f.model_dump() for f in flags]), json.dumps(tool_calls, default=str)),
    )
    doc_id = cur.lastrowid
    audit(conn, f"ai:{extractor.name}", "document_processed", "document", doc_id,
          {"file": path.name, "ocr_engine": ocr.engine, "ocr_confidence": ocr.mean_confidence,
           "flags": [f.code for f in flags]})
    conn.commit()
    return doc_id


def process_batch(conn: sqlite3.Connection, batch_id: int, paths: list[str | Path],
                  extractor: Extractor, settings: Settings) -> list[int]:
    return [process_document(conn, batch_id, p, extractor, settings) for p in sorted(map(str, paths))]
