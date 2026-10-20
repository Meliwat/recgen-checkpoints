import json

import numpy as np

from recgen import llm_reader
from recgen.llm_reader import cache_path, read, unit_prompt


def test_unit_prompt_is_deterministic_and_shows_every_example_once():
    corpus = np.arange(40 * 64).reshape(40, 64) % 50000
    unit = {"id": "70m:real::3", "keys": np.array([[w, 40] for w in range(5)])}
    a, b = unit_prompt(unit, corpus), unit_prompt(unit, corpus)
    snippet_lines = [line for line in a.splitlines() if line[:2] in {f"{i}." for i in range(1, 6)}]
    assert a == b and "5 text snippets" in a and len(snippet_lines) == 5 and all("[[" in line for line in snippet_lines)


def test_read_caches_answers_and_retries_unparseable_replies(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_reader, "BACKOFF_SECONDS", (0, 0, 0))
    replies = iter(["no json", '{"label": "C", "patterns": ["x", "y"]}'])
    calls = []

    def runner(text):
        calls.append(text)
        return {"reply": next(replies), "models": ["m"], "usage": {}, "cost_usd": 0}

    first = read("70m:pooled:50/50:1-2", "prompt", tmp_path, runner)
    assert first["label"] == "C" and len(calls) == 2
    assert read("70m:pooled:50/50:1-2", "prompt", tmp_path, runner)["label"] == "C" and len(calls) == 2  # cached
    assert json.loads(cache_path(tmp_path, "70m:pooled:50/50:1-2").read_text())["label"] == "C"


def test_read_records_a_missing_label_after_every_attempt_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_reader, "BACKOFF_SECONDS", (0, 0, 0))

    def runner(text):
        raise RuntimeError("rate limited")

    record = read("70m:real::3", "prompt", tmp_path, runner)
    assert record["label"] is None and record["error"] == "rate limited"


def test_a_changed_prompt_is_read_again(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_reader, "BACKOFF_SECONDS", (0, 0, 0))
    labels = iter(["A", "B"])

    def runner(text):
        return {"reply": json.dumps({"label": next(labels)}), "models": [], "usage": {}, "cost_usd": 0}

    assert read("u", "first prompt", tmp_path, runner)["label"] == "A"
    assert read("u", "second prompt", tmp_path, runner)["label"] == "B"
