# GiftFlow: AI-assisted intake for donation checks and gift-in-kind forms

**[▶ Live demo](https://giftflow-demo.streamlit.app/)** · runs in offline demo mode on synthetic data. Click **Run intake**, then open **Review**. (If it's been idle, it takes about 30 seconds to wake up.)

GiftFlow reads scanned donation checks and in-kind donation forms, extracts the data with OCR and an LLM, suggests the donor match and fund/campaign coding, **checks the AI's work in code**, and puts every gift in front of a **person for approval** before anything is posted. Posted gifts reconcile against the bank deposit slip, and every step is written to an audit log.

> All names, addresses, banks and account numbers in this repo are **synthetic**. The sample batch includes deliberate problems so the demo shows every safety check.

![Review queue](docs/02_review.png)

## Why it exists

Manually reading each check, matching the donor, choosing the fund and campaign, and typing it into the CRM takes hours a week, and every keystroke is a chance for an error. GiftFlow lets the AI do the reading and suggesting, and lets people make the decisions.

## How it works

```mermaid
flowchart LR
    A[1 Scan batch] --> B[2 OCR<br/>bank strip cropped<br/>numbers redacted]
    B --> C[3 AI extracts fields<br/>structured output]
    C <-->|function call| D[(CRM<br/>search_donors)]
    C --> E[4 Rules code<br/>fund + campaign]
    E --> F[5 Code validates<br/>amounts · dupes · dates · codes]
    F --> G{6 Person<br/>approves?}
    G -->|approve| H[7 Post gift<br/>idempotent]
    G -->|reject| X[Rescan / not a gift]
    H --> I[8 Reconcile with<br/>deposit slip]
    F -.-> L[(Audit log)]
    G -.-> L
    H -.-> L
```

| Part | Job | File |
| --- | --- | --- |
| OCR | Image → text with word-level confidence. Crops the bank strip (MICR line) **before** reading, then redacts any 8+ digit number as a safety net | `giftflow/ocr.py` |
| AI extraction | LLM returns a **strict JSON schema** (structured output). It can **call `search_donors`** (function calling) to look up the CRM; our code runs the lookup. Retries 429/5xx with exponential backoff. Document text is fenced as data, not instructions, and any `<<<`/`>>>` inside it is neutralised so it can't close the fence | `giftflow/llm.py` |
| Offline mode | Deterministic mock extractor with the same interface and the same function call, so the app runs with no API key or internet | `giftflow/llm.py` |
| Donor matching | Fuzzy name + address scoring with nickname handling (Bob → Robert). Never creates donors automatically | `giftflow/matching.py` |
| Coding | Keyword rules first (predictable, free, auditable). AI suggestion only when no rule applies, and only from the allowed list | `giftflow/coding.py` |
| Validation | Required fields, amount box vs written amount, OCR low-confidence digits, future/stale dates, donor certainty, valid codes, GIK value, duplicates | `giftflow/validate.py` |
| Review | Re-validates the reviewer's version; blocking issues prevent approval; records every correction | `giftflow/review.py` |
| Posting | One transaction per gift; unique constraints make re-running safe and stop the same check posting twice | `giftflow/posting.py` |
| Reconciliation | Posted + approved check totals must equal the deposit slip | `giftflow/posting.py` |
| App | Streamlit: intake, review queue, post & reconcile, accuracy, audit log | `app.py` |

## Run it

```bash
# 1. Install Tesseract OCR (optional: without it, the demo uses the .txt OCR files next to each sample)
brew install tesseract          # macOS
# sudo apt install tesseract-ocr   (Ubuntu)

# 2. Install Python packages
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Run the app (offline mock AI by default)
streamlit run app.py
```

Click **Run intake** with "Use the sample batch" on, then open **Review**.

**Use the real OpenAI model:** copy `.env.example` to `.env`, set `LLM_PROVIDER=openai` and `OPENAI_API_KEY=...`. The key is read from the environment and never stored in code.

Other commands:

```bash
python scripts/generate_samples.py samples   # regenerate the synthetic batch
python scripts/run_batch.py samples 6625     # process a folder from the terminal
pytest -q                                    # 31 tests, no API key needed
```

## The 5-minute demo

1. **Intake:** run the sample batch (10 documents, deposit slip $6,625.00).
2. **Review:** flagged items are first.
   - `05` Angela Davis-Reed: amount box says **$175.00**, written amount says **$150.00** → blocked. The written amount is the legal amount, so correct it to 150 and approve.
   - `06` Robert Smith: the same check scanned twice → blocked as a duplicate → reject it.
   - `07` Kevin O'Brien: no matching donor → a person creates the donor (it is selected automatically), then approves.
   - `08` "R. Smith": two similar donors; the **address** points to Robert Smithson → pick him and approve.
   - `02` "Bob Martinez" matched to **Robert** Martinez through nickname handling.
   - `09` In-kind coats with no stated value → warning; the AI never assigns a value.
3. **Post & reconcile:** post 9 gifts → posted checks = **$6,625.00** = deposit slip → **balanced**. Download the gift import CSV.
4. **Accuracy:** the share of AI proposals accepted unchanged, and which fields reviewers corrected most.
5. **Audit log:** every AI step, approval, donor creation and posting, with who and when.

![Reconciliation](docs/03_reconcile.png)

## Common errors and how GiftFlow handles them

| Error | Caught by | Resolution |
| --- | --- | --- |
| Handwritten amount misread | Low-confidence digits from OCR; amount box ≠ written amount; batch total ≠ deposit | Flag for review; written amount governs; rescan if unreadable |
| Wrong donor | Match score below threshold; two similar donors | Name + address scoring; person picks; never auto-create |
| Same check twice | Same donor + check number + amount in an earlier scan or already posted | Blocked at review; unique index in the database as the last line of defence |
| AI invents or mangles a field | Strict schema + code validation + allowed code lists | Prompt says "null if unsure"; invalid values block approval |
| API rate limit / outage | 429 / 5xx / timeout | Exponential backoff with jitter; if it still fails, the document goes to review with the fields blank |
| Posting fails halfway | Per-gift transactions | Re-run posting: only missing gifts post (idempotent) |
| Gift-in-kind with no value | Empty value field | Warning; staff record the value |
| Bank numbers exposed | Cropped before OCR, redacted after (OCR text and low-confidence word list) | Never stored and never sent to the AI; `tests/test_safety.py` checks every model request, DB row and audit entry |
| Prompt injection in a document | Document text is fenced; fence markers inside it are neutralised | Strict schema, allowed code lists, code validation and human approval limit what a bad extraction can do |

## What the AI sees

Only bank numbers are redacted. The model receives the donor name, address, amount, date, memo and check number from the document, and the `search_donors` tool returns up to 3 CRM candidates (name, address, score) to it. Use a provider and account whose data-handling terms fit donor data before switching from mock mode.

## Design decisions

- **OCR + LLM instead of one step:** OCR gives a confidence score per word and raw text you can inspect, so it's clear whether OCR misread or the AI misunderstood. Text is also cheaper to send than images.
- **Rules before AI for coding:** known keywords are free, instant and predictable. The AI handles the long tail.
- **Human approval for everything:** it's money and donor records. The path to automation is to measure acceptance rates per category, then auto-approve only what's proven.
- **Function calling for the donor lookup:** the model asks; our code runs the query and decides what to return. The model never touches the database directly.
- **Idempotent posting + unique constraints:** a retry or a double click can't double-post a gift.

## The same pattern for invoices and orders

| GiftFlow | Wholesale / ecommerce business |
| --- | --- |
| Donation check, GIK form | Vendor invoice, purchase order, packing slip |
| Donor match | Vendor or B2B customer match |
| Fund / campaign | GL account / order tag / routing |
| Batch total = deposit slip | Invoice = PO = receiving; payments = orders |
| Duplicate check | Duplicate invoice, duplicate webhook event |
| Person approves before posting | Human approval before anything touches money or customers |
| CRM gift import | Accounting draft bill, ecommerce order update |

## Next steps

- Post directly to the CRM API (OAuth, retries, error report) instead of the CSV import.
- Auto-approve proven high-confidence cases once acceptance rates support it.
- Per-field accuracy tracking against a labelled test set on every prompt change.
- Image-only extraction with a vision model, A/B tested against OCR + LLM on the same samples.

## Tech

Python · Streamlit · SQLite · Tesseract (pytesseract) · OpenAI API (structured output + function calling) · RapidFuzz · Pydantic · Pillow · pytest
