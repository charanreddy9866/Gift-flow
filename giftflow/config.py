"""Runtime settings, read from environment variables (or a .env file)."""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    db_path: str
    llm_provider: str          # "mock" (offline) or "openai"
    openai_model: str
    ocr_conf_threshold: float  # 0-100, per word
    match_threshold: float     # 0-100, donor match score
    ai_conf_threshold: float   # 0-1, model self-reported confidence


def load_settings() -> Settings:
    """Read settings at call time so tests and the UI can change them."""
    return Settings(
        db_path=os.getenv("GIFTFLOW_DB", "giftflow.db"),
        llm_provider=os.getenv("LLM_PROVIDER", "mock").lower(),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        ocr_conf_threshold=float(os.getenv("OCR_CONF_THRESHOLD", "70")),
        match_threshold=float(os.getenv("MATCH_THRESHOLD", "85")),
        ai_conf_threshold=float(os.getenv("AI_CONF_THRESHOLD", "0.8")),
    )
