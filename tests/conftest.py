import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from giftflow.config import Settings  # noqa: E402
from giftflow.db import connect, init_db  # noqa: E402


@pytest.fixture
def conn(tmp_path):
    c = connect(str(tmp_path / "test.db"))
    init_db(c)
    yield c
    c.close()


@pytest.fixture
def settings(tmp_path):
    return Settings(db_path=str(tmp_path / "test.db"), llm_provider="mock", openai_model="gpt-4o-mini",
                    ocr_conf_threshold=70, match_threshold=85, ai_conf_threshold=0.8)


@pytest.fixture(scope="session")
def samples(tmp_path_factory):
    from scripts.generate_samples import main
    return main(str(tmp_path_factory.mktemp("samples")))
