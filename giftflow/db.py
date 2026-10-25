"""SQLite storage: a small stand-in CRM (donors, funds, campaigns, posted gifts)
plus the workflow tables (batches, documents) and an audit log.

In production the donor list and posted gifts live in the CRM (e.g. Raiser's
Edge NXT). Here they live in SQLite so the whole flow runs on a laptop.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS donors (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    address TEXT,
    email TEXT
);
CREATE TABLE IF NOT EXISTS funds (code TEXT PRIMARY KEY, name TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS campaigns (code TEXT PRIMARY KEY, name TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS batches (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    deposit_total REAL NOT NULL,      -- from the bank deposit slip
    created_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open'
);

CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY,
    batch_id INTEGER NOT NULL REFERENCES batches(id),
    filename TEXT NOT NULL,
    image_path TEXT,
    doc_type TEXT,
    ocr_text TEXT,                    -- already redacted: no bank numbers stored
    ocr_confidence REAL,
    low_conf_words TEXT,              -- JSON list
    extraction TEXT,                  -- JSON: exactly what the AI returned
    proposal TEXT,                    -- JSON: AI output + rule-based coding, shown to the reviewer
    candidates TEXT,                  -- JSON: donor match candidates
    flags TEXT,                       -- JSON: validation flags
    tool_calls TEXT,                  -- JSON: function calls the AI made
    final TEXT,                       -- JSON: what the reviewer approved
    corrections TEXT,                 -- JSON: fields the reviewer changed
    status TEXT NOT NULL DEFAULT 'needs_review', -- needs_review|approved|rejected|posted
    reviewer TEXT,
    reviewed_at TEXT,
    UNIQUE(batch_id, filename)
);

CREATE TABLE IF NOT EXISTS gifts (
    id INTEGER PRIMARY KEY,
    document_id INTEGER NOT NULL UNIQUE REFERENCES documents(id), -- idempotent posting
    donor_id INTEGER NOT NULL REFERENCES donors(id),
    gift_type TEXT NOT NULL,          -- check | gik
    amount REAL,
    gift_date TEXT,
    check_number TEXT,
    fund TEXT,
    campaign TEXT,
    description TEXT,
    posted_at TEXT NOT NULL
);
-- The same check can never be posted twice.
CREATE UNIQUE INDEX IF NOT EXISTS ux_gift_check
    ON gifts(donor_id, check_number, amount) WHERE gift_type = 'check';

CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY,
    ts TEXT NOT NULL,
    actor TEXT NOT NULL,
    action TEXT NOT NULL,
    entity TEXT NOT NULL,
    entity_id INTEGER,
    details TEXT
);
"""

SEED_DONORS = [
    ("Robert Smith", "412 Palm Ave, Miami, FL 33130"),
    ("Robert Martinez", "88 Coral Way, Miami, FL 33145"),
    ("Maria Lopez", "1500 Bay Rd, Miami Beach, FL 33139"),
    ("Johnson Family Foundation", "200 Biscayne Blvd, Miami, FL 33131"),
    ("Angela Davis-Reed", "17 Sunset Dr, Coral Gables, FL 33143"),
    ("Grace Community Church", "950 NW 7th St, Miami, FL 33136"),
    ("Linda Park", "3301 Ocean Dr, Fort Lauderdale, FL 33308"),
    ("Thomas Nguyen", "61 Pine St, Hollywood, FL 33020"),
    ("Susan Whitfield", "725 Lake Ave, Tamarac, FL 33321"),
    ("Robert Smithson", "9 Harbor Ln, Key Biscayne, FL 33149"),
]
SEED_FUNDS = [
    ("GENERAL", "General Operating"),
    ("MEALS", "Meals Program"),
    ("SHELTER", "Shelter Services"),
    ("BUILDING", "Building Fund"),
]
SEED_CAMPAIGNS = [
    ("ANNUAL-2026", "Annual Appeal 2026"),
    ("THANKSGIVING-2026", "Thanksgiving 2026"),
    ("HOLIDAY-2026", "Holiday Giving 2026"),
    ("CAPITAL-2026", "Capital Campaign 2026"),
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def init_db(conn: sqlite3.Connection, seed: bool = True) -> None:
    conn.executescript(SCHEMA)
    if seed and conn.execute("SELECT COUNT(*) FROM donors").fetchone()[0] == 0:
        conn.executemany("INSERT INTO donors(name, address) VALUES (?, ?)", SEED_DONORS)
        conn.executemany("INSERT INTO funds VALUES (?, ?)", SEED_FUNDS)
        conn.executemany("INSERT INTO campaigns VALUES (?, ?)", SEED_CAMPAIGNS)
    conn.commit()


def audit(conn: sqlite3.Connection, actor: str, action: str, entity: str,
          entity_id: int | None, details: Any = None) -> None:
    conn.execute(
        "INSERT INTO audit_log(ts, actor, action, entity, entity_id, details) VALUES (?,?,?,?,?,?)",
        (now(), actor, action, entity, entity_id,
         json.dumps(details, default=str) if details is not None else None),
    )


def allowed_codes(conn: sqlite3.Connection) -> tuple[list[str], list[str]]:
    funds = [r[0] for r in conn.execute("SELECT code FROM funds ORDER BY code")]
    campaigns = [r[0] for r in conn.execute("SELECT code FROM campaigns ORDER BY code")]
    return funds, campaigns


def loads(value: str | None, default: Any = None) -> Any:
    return json.loads(value) if value else default
