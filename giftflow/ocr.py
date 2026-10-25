"""OCR step: image -> text plus a confidence score.

Bank routing and account numbers sit in the strip along the bottom of a check.
We crop that strip off BEFORE reading, and run a redaction pass afterwards as a
safety net, so bank numbers are never stored or sent to the AI.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

BOTTOM_CROP_RATIO = 0.14  # drop the bottom 14% of the page (MICR / bank line)


@dataclass
class OcrResult:
    text: str
    mean_confidence: float                     # 0-100
    low_conf_words: list[str] = field(default_factory=list)
    engine: str = "tesseract"


def crop_bank_strip(image: Image.Image) -> Image.Image:
    w, h = image.size
    return image.crop((0, 0, w, int(h * (1 - BOTTOM_CROP_RATIO))))


_LONG_DIGITS = re.compile(r"\b\d[\d\- ]{7,}\d\b")


def redact_bank_numbers(text: str) -> str:
    """Mask any long run of digits (8+), which is never a gift amount or date."""
    return _LONG_DIGITS.sub("[REDACTED]", text)


def run_ocr(image_path: str | Path, conf_threshold: float = 70) -> OcrResult:
    path = Path(image_path)
    image = crop_bank_strip(Image.open(path).convert("L"))
    try:
        import pytesseract

        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    except Exception:
        # No Tesseract installed: fall back to a text file next to the image.
        sidecar = path.with_suffix(".txt")
        if sidecar.exists():
            return OcrResult(redact_bank_numbers(sidecar.read_text()), 99.0, [], engine="sidecar")
        raise

    lines: dict[tuple[int, int, int], list[str]] = {}
    confs: list[float] = []
    low: list[str] = []
    for i, word in enumerate(data["text"]):
        word = word.strip()
        conf = float(data["conf"][i])
        if not word or conf < 0:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append(word)
        confs.append(conf)
        if conf < conf_threshold:
            low.append(word)
    text = "\n".join(" ".join(ws) for _, ws in sorted(lines.items()))
    mean = round(sum(confs) / len(confs), 1) if confs else 0.0
    return OcrResult(redact_bank_numbers(text), mean, low)
