from datetime import date, timedelta

import pytest

from giftflow.amounts import parse_money, words_to_amount
from giftflow.coding import rule_coding
from giftflow.matching import normalize_name, search_donors
from giftflow.ocr import redact_bank_numbers
from giftflow.validate import STALE_CHECK_DAYS, has_blockers, validate


@pytest.mark.parametrize("text,expected", [
    ("Two hundred fifty and 00/100", 250.00),
    ("One thousand and 00/100 dollars", 1000.00),
    ("Seventy-five and 50/100", 75.50),
    ("Five thousand two hundred and 00/100", 5200.00),
    ("Fifty and 00/100", 50.00),
    ("Banana and 00/100", None),       # unexpected word: don't guess
    ("", None),
])
def test_words_to_amount(text, expected):
    assert words_to_amount(text) == expected


def test_parse_money():
    assert parse_money("$ 1,250.00") == 1250.00
    assert parse_money("none") is None


def test_redaction_removes_bank_numbers_but_keeps_amounts_and_dates():
    out = redact_bank_numbers("Routing 063100277 Acct 004512378901 Date 10/20/2026 $ 250.00 No. 1043")
    assert "063100277" not in out and "004512378901" not in out
    assert "10/20/2026" in out and "250.00" in out and "1043" in out


def test_nickname_and_address_matching(conn):
    assert normalize_name("Mr. Bob Martinez") == "robert martinez"
    top = search_donors(conn, "Bob Martinez", "88 Coral Way, Miami, FL 33145")[0]
    assert top.name == "Robert Martinez" and top.score >= 95


def test_address_breaks_a_tie_between_similar_names(conn):
    cands = search_donors(conn, "R. Smith", "9 Harbor Ln, Key Biscayne, FL 33149")
    assert cands[0].name == "Robert Smithson"   # the address points to Smithson, not Smith


def test_unknown_donor_returns_no_candidates(conn):
    assert search_donors(conn, "Kevin O'Brien", "45 Elm St, Plantation, FL 33317") == []


def test_rule_coding():
    assert rule_coding("Thanksgiving meals") == ("MEALS", "THANKSGIVING-2026")
    assert rule_coding(None, "Winter coats") == ("SHELTER", "ANNUAL-2026")
    assert rule_coding("") is None


def _check(**over):
    p = {"doc_type": "check", "donor_name": "Robert Smith", "amount": 250.0,
         "amount_words": "Two hundred fifty and 00/100", "date": "2026-10-20", "check_number": "1043",
         "suggested_donor_id": 1, "suggested_fund": "MEALS", "suggested_campaign": "THANKSGIVING-2026",
         "coding_source": "rule", "ai_confidence": 0.95}
    p.update(over)
    return p


def codes(flags):
    return {f.code for f in flags}


def test_clean_check_has_no_flags(conn):
    cands = search_donors(conn, "Robert Smith", "412 Palm Ave, Miami, FL 33130")
    assert validate(conn, _check(), candidates=cands, today=date(2026, 10, 25)) == []


def test_amount_mismatch_blocks(conn):
    f = validate(conn, _check(amount=175.0), today=date(2026, 10, 25))
    assert "amount_mismatch" in codes(f)


def test_future_date_blocks_and_old_check_blocks(conn):
    assert "date_future" in codes(validate(conn, _check(date="2027-01-01"), today=date(2026, 10, 25)))
    f = validate(conn, _check(date="2026-01-02"), today=date(2026, 10, 25))
    assert any(x.code == "stale_check" and x.severity == "block" for x in f)


TODAY = date(2026, 10, 25)


def _days_ago(n):
    return (TODAY - timedelta(days=n)).isoformat()


def test_check_exactly_180_days_old_passes(conn):
    cands = search_donors(conn, "Robert Smith", "412 Palm Ave, Miami, FL 33130")
    assert validate(conn, _check(date=_days_ago(STALE_CHECK_DAYS)), candidates=cands, today=TODAY) == []


def test_check_181_days_old_is_blocked(conn):
    f = validate(conn, _check(date=_days_ago(STALE_CHECK_DAYS + 1)), today=TODAY)
    stale = [x for x in f if x.code == "stale_check"]
    assert len(stale) == 1 and stale[0].severity == "block"
    assert has_blockers(f)


def test_gik_181_days_old_is_not_blocked(conn):
    p = {"doc_type": "gik", "donor_name": "Grace Community Church", "item_description": "Winter coats",
         "quantity": 40, "stated_value": 800.0, "date": _days_ago(STALE_CHECK_DAYS + 1),
         "suggested_donor_id": 6, "suggested_fund": "SHELTER", "suggested_campaign": "ANNUAL-2026",
         "coding_source": "rule"}
    f = validate(conn, p, today=TODAY)
    assert "stale_check" not in codes(f)
    assert not has_blockers(f)


def test_invalid_fund_blocks(conn):
    assert "fund_invalid" in codes(validate(conn, _check(suggested_fund="MADE-UP"), today=date(2026, 10, 25)))


def test_missing_donor_blocks(conn):
    assert "donor_new" in codes(validate(conn, _check(suggested_donor_id=None), today=date(2026, 10, 25)))


def test_gik_without_value_warns_but_does_not_block(conn):
    p = {"doc_type": "gik", "donor_name": "Grace Community Church", "item_description": "Winter coats",
         "quantity": 40, "date": "2026-10-23", "suggested_donor_id": 6, "suggested_fund": "SHELTER",
         "suggested_campaign": "ANNUAL-2026", "coding_source": "rule"}
    f = validate(conn, p, today=date(2026, 10, 25))
    assert "gik_no_value" in codes(f)
    assert all(x.severity == "warn" for x in f)


def test_low_confidence_digits_warn(conn):
    f = validate(conn, _check(), low_conf_words=["Memo:", "25O.00"], today=date(2026, 10, 25))
    assert "ocr_low_conf_digits" in codes(f)
