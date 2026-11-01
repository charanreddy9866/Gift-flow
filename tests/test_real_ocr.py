"""Real Tesseract on the generated samples: OCR quality, bank-strip cropping and the
upload safety net, with no fakes. Skipped when Tesseract isn't installed.

These check what OCR reads, not how the mock extractor parses it: Tesseract may put a
label and its value on separate lines, so fields are looked for anywhere in the text."""
import io

import pytest
from PIL import Image

from giftflow.ocr import BankNumbersVisible, _LONG_DIGITS, run_ocr, save_masked_upload
from scripts.generate_samples import CHECKS, GIK_FORMS


def _tesseract_installed() -> bool:
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except Exception:  # noqa: BLE001 - not installed, not on PATH, or broken
        return False


pytestmark = pytest.mark.skipif(not _tesseract_installed(), reason="Tesseract isn't installed")

ROUTING, ACCOUNT = "063100277", "004512378901"


def _missing(text: str, expected) -> list[str]:
    return [x for x in expected if x and x.lower() not in text.lower()]


@pytest.mark.parametrize("check", CHECKS, ids=[c[0] for c in CHECKS])
def test_real_ocr_reads_check_fields(samples, check):
    file, donor, _address, number, date, amount, _words, memo = check
    text = run_ocr(samples / f"{file}.png").text
    assert not _missing(text, [donor, number, date, f"$ {amount}", memo]), text


@pytest.mark.parametrize("form", GIK_FORMS, ids=[g[0] for g in GIK_FORMS])
def test_real_ocr_reads_gik_fields(samples, form):
    file, donor, _address, date, item, qty, value = form
    text = run_ocr(samples / f"{file}.png").text
    assert not _missing(text, [donor, date, item, qty, f"$ {value}" if value else None]), text


@pytest.mark.parametrize("check", CHECKS, ids=[c[0] for c in CHECKS])
def test_real_ocr_never_returns_bank_numbers(samples, check):
    result = run_ocr(samples / f"{check[0]}.png")
    for text in [result.text, *result.low_conf_words]:
        assert ROUTING not in text and ACCOUNT not in text
        assert not _LONG_DIGITS.search(text), text


@pytest.mark.parametrize("check", CHECKS, ids=[c[0] for c in CHECKS])
def test_real_safety_net_saves_masked_checks(samples, tmp_path, check):
    """Each sample's bank line is readable unmasked, so the safety net has something to
    catch; once masked it's gone and the scan is saved."""
    import pytesseract
    path = samples / f"{check[0]}.png"
    assert _LONG_DIGITS.search(pytesseract.image_to_string(Image.open(path)))
    saved = save_masked_upload(path.read_bytes(), tmp_path / "check.png")
    assert saved.exists()


def test_real_safety_net_refuses_a_bank_line_above_the_mask(samples, tmp_path):
    """A scan whose bank line sits higher than the masked strip (e.g. a wide bottom margin)
    is refused, and nothing is written."""
    image = Image.open(samples / "01_check_robert_smith.png")
    w, h = image.size
    moved = image.copy()
    moved.paste(image.crop((0, int(h * 0.86), w, h)), (0, 380))   # bank line into the middle
    buf = io.BytesIO()
    moved.save(buf, format="PNG")
    dest = tmp_path / "uploads" / "check.png"
    with pytest.raises(BankNumbersVisible):
        save_masked_upload(buf.getvalue(), dest)
    assert not dest.exists() and not dest.parent.exists()
