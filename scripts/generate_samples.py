"""Create a synthetic batch of donation checks and gift-in-kind forms.

All names, addresses, banks and account numbers are fictional. The batch has
deliberate problems so the demo shows every safety check:

  05 - amount box and written amount disagree      -> amount_mismatch (block)
  06 - the same check scanned twice                -> duplicate_scan (block)
  07 - a donor who isn't in the CRM yet            -> donor_new (block)
  08 - "R. Smith": two similar donors, address decides -> donor_required (block)
  02 - "Bob" on the check, "Robert" in the CRM     -> nickname match
  09 - in-kind gift with no stated value           -> gik_no_value (warn)

Also writes a .txt "sidecar" with the expected OCR text next to each image, so
the demo still runs on machines without Tesseract installed.

Usage:  python scripts/generate_samples.py [output_dir]
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PAYEE = "Bayside Hope Center"
DEPOSIT_TOTAL = 6625.00  # what the bank deposit slip says (legal amounts, duplicate deposited once)

FONT_CANDIDATES = {
    "print": ["DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/Library/Fonts/Arial.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf",
              "C:/Windows/Fonts/arial.ttf"],
    "bold": ["DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
             "/System/Library/Fonts/Supplemental/Arial Bold.ttf", "C:/Windows/Fonts/arialbd.ttf"],
    "hand": ["/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf",
             "/System/Library/Fonts/Supplemental/Times New Roman Italic.ttf",
             "C:/Windows/Fonts/timesi.ttf", "DejaVuSans.ttf"],
    "mono": ["DejaVuSansMono.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
             "/System/Library/Fonts/Menlo.ttc", "C:/Windows/Fonts/consola.ttf"],
}


def font(kind: str, size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES[kind]:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


CHECKS = [
    # file, donor, address, number, date, box amount, written amount, memo
    ("01_check_robert_smith", "ROBERT SMITH", "412 Palm Ave, Miami, FL 33130", "1043", "10/20/2026",
     "250.00", "Two hundred fifty and 00/100", "Thanksgiving meals"),
    ("02_check_bob_martinez", "BOB MARTINEZ", "88 Coral Way, Miami, FL 33145", "2210", "10/21/2026",
     "100.00", "One hundred and 00/100", ""),
    ("03_check_maria_lopez", "MARIA LOPEZ", "1500 Bay Rd, Miami Beach, FL 33139", "518", "10/21/2026",
     "1,000.00", "One thousand and 00/100", "for the holidays"),
    ("04_check_johnson_foundation", "JOHNSON FAMILY FOUNDATION", "200 Biscayne Blvd, Miami, FL 33131", "7781",
     "10/22/2026", "5,000.00", "Five thousand and 00/100", "Building fund"),
    ("05_check_angela_davis_reed", "ANGELA DAVIS-REED", "17 Sunset Dr, Coral Gables, FL 33143", "334",
     "10/22/2026", "175.00", "One hundred fifty and 00/100", "Shelter beds"),
    ("06_check_robert_smith_rescan", "ROBERT SMITH", "412 Palm Ave, Miami, FL 33130", "1043", "10/20/2026",
     "250.00", "Two hundred fifty and 00/100", "Thanksgiving meals"),
    ("07_check_kevin_obrien", "KEVIN O'BRIEN", "45 Elm St, Plantation, FL 33317", "1501", "10/23/2026",
     "50.00", "Fifty and 00/100", "General donation"),
    ("08_check_r_smith", "R. SMITH", "9 Harbor Ln, Key Biscayne, FL 33149", "88", "10/23/2026",
     "75.00", "Seventy-five and 00/100", "Meals program"),
]

GIK_FORMS = [
    # file, donor, address, date, item, quantity, stated value ("" = not stated)
    ("09_gik_grace_church", "Grace Community Church", "950 NW 7th St, Miami, FL 33136", "10/23/2026",
     "Winter coats, adult sizes", "40", ""),
    ("10_gik_linda_park", "Linda Park", "3301 Ocean Dr, Fort Lauderdale, FL 33308", "10/22/2026",
     "Canned food, assorted", "25", "300.00"),
]


def draw_check(out: Path, donor, address, number, date, amount, words, memo) -> str:
    w, h = 1400, 620
    img = Image.new("RGB", (w, h), (250, 251, 246))
    d = ImageDraw.Draw(img)
    d.rectangle([8, 8, w - 9, h - 9], outline=(120, 140, 160), width=3)
    d.text((40, 30), donor, font=font("bold", 30), fill="black")
    d.text((40, 70), address, font=font("print", 24), fill="black")
    d.text((1150, 30), f"No. {number}", font=font("bold", 30), fill="black")
    d.text((900, 140), "Date:", font=font("print", 26), fill="black")
    d.text((990, 132), date, font=font("hand", 36), fill=(20, 30, 110))
    d.line([(985, 175), (1330, 175)], fill="black", width=2)
    d.text((40, 220), f"PAY TO THE ORDER OF   {PAYEE}", font=font("print", 26), fill="black")
    d.line([(345, 255), (1080, 255)], fill="black", width=2)
    d.rectangle([1110, 205, 1350, 260], outline="black", width=2)
    d.text((1122, 213), f"$ {amount}", font=font("hand", 36), fill=(20, 30, 110))
    d.text((40, 300), f"{words} ----------", font=font("hand", 36), fill=(20, 30, 110))
    d.text((1180, 305), "DOLLARS", font=font("print", 26), fill="black")
    d.line([(40, 345), (1170, 345)], fill="black", width=2)
    d.text((40, 395), "FIRST COASTAL BANK  •  Miami, FL", font=font("print", 22), fill=(90, 90, 90))
    d.text((40, 470), "MEMO:", font=font("print", 26), fill="black")
    d.text((140, 462), memo, font=font("hand", 34), fill=(20, 30, 110))
    d.line([(135, 505), (640, 505)], fill="black", width=2)
    d.line([(820, 505), (1330, 505)], fill="black", width=2)
    d.text((1020, 405), donor.title().split()[0], font=font("hand", 40), fill=(20, 30, 110))
    # Bank line (routing / account / check no.) - in the bottom strip that gets cropped off.
    d.text((260, 560), f"|: 063100277 |:  004512378901 ||'  {number}", font=font("mono", 30), fill="black")
    img.save(out.with_suffix(".png"))
    sidecar = (f"{donor} No. {number}\n{address}\nDate: {date}\n"
               f"PAY TO THE ORDER OF {PAYEE} $ {amount}\n{words} ---------- DOLLARS\n"
               f"FIRST COASTAL BANK Miami, FL\nMEMO: {memo}\n")
    out.with_suffix(".txt").write_text(sidecar)
    return sidecar


def draw_gik(out: Path, donor, address, date, item, qty, value) -> str:
    w, h = 1400, 900
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([8, 8, w - 9, h - 9], outline=(120, 140, 160), width=3)
    d.text((40, 30), "IN-KIND DONATION FORM", font=font("bold", 36), fill="black")
    d.text((40, 85), PAYEE, font=font("print", 26), fill=(70, 70, 70))
    rows = [("Donor Name:", donor), ("Address:", address), ("Date Received:", date),
            ("Item Description:", item), ("Quantity:", qty),
            ("Estimated Value (provided by donor): $", value)]
    y = 170
    for label, val in rows:
        d.text((40, y), label, font=font("print", 26), fill="black")
        x = 40 + int(d.textlength(label, font=font("print", 26))) + 16
        d.text((x, y - 6), val, font=font("hand", 34), fill=(20, 30, 110))
        d.line([(x, y + 34), (1340, y + 34)], fill=(160, 160, 160), width=1)
        y += 85
    d.text((40, 700), "Received by: ________________", font=font("print", 24), fill="black")
    img.save(out.with_suffix(".png"))
    sidecar = "IN-KIND DONATION FORM\n" + PAYEE + "\n" + "\n".join(f"{a} {b}" for a, b in rows) + "\n"
    out.with_suffix(".txt").write_text(sidecar)
    return sidecar


def main(out_dir: str = "samples") -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for row in CHECKS:
        draw_check(out / row[0], *row[1:])
    for row in GIK_FORMS:
        draw_gik(out / row[0], *row[1:])
    (out / "DEPOSIT_SLIP.txt").write_text(f"Deposit total: {DEPOSIT_TOTAL:.2f}\n")
    print(f"Wrote {len(CHECKS) + len(GIK_FORMS)} documents to {out}/ (deposit total ${DEPOSIT_TOTAL:,.2f})")
    return out


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "samples")
