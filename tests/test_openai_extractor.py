"""Tests the real OpenAI code path with a fake client: the function-calling
loop, structured output parsing, and retry with backoff. No API key needed."""
import json
from types import SimpleNamespace as NS

import pytest

from giftflow.llm import OpenAIExtractor, with_retries

FINAL = {
    "doc_type": "check", "donor_name": "Robert Smith", "address": "412 Palm Ave, Miami, FL 33130",
    "amount": 250.0, "amount_words": "Two hundred fifty and 00/100", "date": "2026-10-20",
    "check_number": "1043", "memo": "Thanksgiving meals", "item_description": None, "quantity": None,
    "stated_value": None, "suggested_donor_id": 1, "suggested_fund": "MEALS",
    "suggested_campaign": "THANKSGIVING-2026", "ai_confidence": 0.96,
    "reasoning": "Exact name and address match; memo mentions Thanksgiving.",
}


def _resp(message):
    return NS(choices=[NS(message=message)], usage=None)


class FakeCompletions:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class RateLimited(Exception):
    status_code = 429


def make_client(script):
    comp = FakeCompletions(script)
    return NS(chat=NS(completions=comp)), comp


def test_tool_call_then_structured_answer():
    tool_msg = NS(content=None, refusal=None, tool_calls=[
        NS(id="call_1", function=NS(name="search_donors",
                                     arguments=json.dumps({"name": "Robert Smith", "address": "412 Palm Ave"})))])
    final_msg = NS(content=json.dumps(FINAL), refusal=None, tool_calls=None)
    client, comp = make_client([_resp(tool_msg), _resp(final_msg)])

    executed = []

    def executor(name, args):
        executed.append((name, args))
        return [{"donor_id": 1, "name": "Robert Smith", "score": 100}]

    ext, log = OpenAIExtractor(client=client).extract("OCR TEXT", ["MEALS"], ["THANKSGIVING-2026"], executor)

    assert executed == [("search_donors", {"name": "Robert Smith", "address": "412 Palm Ave"})]
    assert ext.suggested_donor_id == 1 and ext.amount == 250.0
    assert log[0]["tool"] == "search_donors"
    # second request carries the tool result back to the model
    assert comp.calls[1]["messages"][-1]["role"] == "tool"
    # structured output is requested on every call
    assert comp.calls[0]["response_format"]["type"] == "json_schema"
    # document text is fenced as data
    assert "<<<" in comp.calls[0]["messages"][1]["content"]


def test_retries_rate_limit_then_succeeds():
    final_msg = NS(content=json.dumps(FINAL), refusal=None, tool_calls=None)
    client, comp = make_client([RateLimited("429"), RateLimited("429"), _resp(final_msg)])
    sleeps = []
    import giftflow.llm as llm
    orig = llm.time.sleep
    llm.time.sleep = sleeps.append
    try:
        ext, _ = OpenAIExtractor(client=client).extract("x", [], [], lambda *a: [])
    finally:
        llm.time.sleep = orig
    assert ext.check_number == "1043"
    assert len(sleeps) == 2 and sleeps[1] > sleeps[0]   # backoff grows


def test_non_retryable_error_is_raised_immediately():
    calls = []

    def boom():
        calls.append(1)
        raise ValueError("bad request")

    with pytest.raises(ValueError):
        with_retries(boom, sleep=lambda s: None)
    assert len(calls) == 1
