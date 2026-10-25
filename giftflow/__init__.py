"""GiftFlow: AI-assisted intake for donation checks and gift-in-kind forms.

scan -> OCR -> redact bank numbers -> AI extraction (structured output + function
calling) -> rule-based coding -> validation -> human review -> idempotent posting
-> reconciliation against the bank deposit, with an audit log throughout.
"""
__version__ = "1.0.0"
