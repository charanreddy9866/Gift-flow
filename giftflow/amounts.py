"""Turn the written-out amount on a check into a number, so it can be compared
with the numeric amount box. A mismatch usually means an OCR misread."""
from __future__ import annotations

import re

_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}
_SCALES = {"hundred": 100, "thousand": 1_000, "million": 1_000_000}


def words_to_amount(text: str | None) -> float | None:
    """'Two hundred fifty and 00/100 dollars' -> 250.0. Returns None if unreadable."""
    if not text:
        return None
    s = text.lower().replace("-", " ").replace(",", " ")
    cents = 0.0
    m = re.search(r"(\d{1,2})\s*/\s*100", s)
    if m:
        cents = int(m.group(1)) / 100
        s = s[: m.start()]
    words = [w for w in re.findall(r"[a-z]+", s) if w not in {"and", "dollars", "dollar", "only", "no"}]
    if not words:
        return None
    total, current, seen_number = 0, 0, False
    for w in words:
        if w in _UNITS:
            current += _UNITS[w]
            seen_number = True
        elif w in _TENS:
            current += _TENS[w]
            seen_number = True
        elif w == "hundred":
            current = (current or 1) * 100
            seen_number = True
        elif w in _SCALES:
            total += (current or 1) * _SCALES[w]
            current = 0
            seen_number = True
        else:
            return None  # an unexpected word: don't guess
    if not seen_number:
        return None
    return round(total + current + cents, 2)


def parse_money(text: str | None) -> float | None:
    """'$1,250.00' -> 1250.0"""
    if text is None:
        return None
    m = re.search(r"(\d[\d,]*\.?\d{0,2})", str(text))
    if not m:
        return None
    try:
        return round(float(m.group(1).replace(",", "")), 2)
    except ValueError:
        return None
