import hashlib

import pytest

from recgen import llm_reader, reader_v24
from recgen.rubric_v24 import ARMS, GROUPS, RETEST_PER_GROUP


@pytest.fixture
def fake_prompts(monkeypatch):
    monkeypatch.setattr(llm_reader, "unit_prompt", lambda unit, corpus, make_prompt: make_prompt([unit["id"]]))


def units():
    return [{"id": f"70m:{g}::{i}", "group": g} for g in GROUPS for i in range(20)]


def test_main_jobs_interleave_every_series_then_read_the_retests(fake_prompts):
    jobs = reader_v24.main_jobs(units(), corpus=None)
    first, again = jobs[:60 * len(ARMS)], jobs[60 * len(ARMS):]
    assert not any(j["retest"] for j in first) and all(j["retest"] for j in again)
    assert len(again) == 3 * RETEST_PER_GROUP * len(ARMS)
    assert sorted((j["id"], j["arm"]) for j in first) == sorted((u["id"], a) for u in units() for a in ARMS)
    assert len({j["arm"] for j in first[:12]}) >= 4  # the series are interleaved, not read one after another
    assert all(("ONE MAIN JOB" in j["text"]) == j["arm"].endswith("E3") for j in jobs)
    assert all(j["directory"].parts[-2:] == (j["arm"], "llm_retest" if j["retest"] else "llm") for j in jobs)


def test_check_random_prompts_needs_v20s_hash(fake_prompts):
    unit = {"id": "70m:random::0", "group": "random"}
    text = llm_reader.unit_prompt(unit, None, reader_v24.prompt_for("v7"))
    v20 = {unit["id"]: {"prompt_sha256": hashlib.sha256(text.encode()).hexdigest()}}
    assert reader_v24.check_random_prompts([unit], None, v20) == 1
    with pytest.raises(RuntimeError):
        reader_v24.check_random_prompts([unit], None, {unit["id"]: {"prompt_sha256": "0" * 64}})


def test_spent_sums_the_spend_log(tmp_path, monkeypatch):
    monkeypatch.setattr(reader_v24, "SPEND_LOG", tmp_path / "spend.jsonl")
    assert reader_v24.spent() == 0.0
    (tmp_path / "spend.jsonl").write_text('{"cost_usd": 0.5}\n{"cost_usd": null}\n{"cost_usd": 0.25}\n')
    assert reader_v24.spent() == 0.75
