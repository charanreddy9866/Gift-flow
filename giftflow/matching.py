"""Donor matching: find the existing donor record a gift belongs to.

Scores combine the name (with nickname handling, so "Bob" finds "Robert") and,
when available, the address. The pipeline never creates donors automatically:
anything below the match threshold goes to a person.
"""
from __future__ import annotations

import re
import sqlite3

from rapidfuzz import fuzz

from .models import DonorCandidate

NICKNAMES = {
    "bob": "robert", "rob": "robert", "bobby": "robert",
    "bill": "william", "will": "william", "liz": "elizabeth", "beth": "elizabeth",
    "jim": "james", "jimmy": "james", "mike": "michael", "tom": "thomas",
    "tony": "anthony", "kate": "katherine", "sue": "susan", "dave": "david",
    "dan": "daniel", "chris": "christopher", "jen": "jennifer", "joe": "joseph",
}
MIN_CANDIDATE_SCORE = 55  # below this it's treated as a possible new donor
_NOISE = {"mr", "mrs", "ms", "dr", "and", "the", "&"}


def normalize_name(name: str) -> str:
    words = re.findall(r"[a-z]+", name.lower())
    words = [NICKNAMES.get(w, w) for w in words if w not in _NOISE]
    return " ".join(words)


def normalize_address(addr: str) -> str:
    a = addr.lower()
    for long, short in [("avenue", "ave"), ("street", "st"), ("drive", "dr"),
                        ("road", "rd"), ("boulevard", "blvd"), ("lane", "ln")]:
        a = re.sub(rf"\b{long}\b", short, a)
    return re.sub(r"[^a-z0-9 ]", " ", a)


def score(name: str, address: str | None, donor_name: str, donor_address: str | None) -> float:
    name_score = fuzz.token_sort_ratio(normalize_name(name), normalize_name(donor_name))
    if address and donor_address:
        addr_score = fuzz.token_set_ratio(normalize_address(address), normalize_address(donor_address))
        return round(0.7 * name_score + 0.3 * addr_score, 1)
    return round(name_score * 0.9, 1)  # name only: cap the score so it needs a look


def search_donors(conn: sqlite3.Connection, name: str | None, address: str | None = None,
                  limit: int = 3) -> list[DonorCandidate]:
    if not name:
        return []
    rows = conn.execute("SELECT id, name, address FROM donors").fetchall()
    scored = [
        DonorCandidate(donor_id=r["id"], name=r["name"], address=r["address"],
                       score=score(name, address, r["name"], r["address"]))
        for r in rows
    ]
    scored.sort(key=lambda c: c.score, reverse=True)
    return [c for c in scored[:limit] if c.score >= MIN_CANDIDATE_SCORE]
