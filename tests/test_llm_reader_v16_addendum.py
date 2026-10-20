import json

from recgen import llm_reader, llm_reader_v16_addendum
from recgen.llm_reader_v16 import SpendGuard, read_all
from recgen.llm_reader_v16_addendum import collect, jobs_for
from recgen.rubric_v16_addendum import ARMS


def fake_units(n: int) -> list[dict]:
    return [{"id": f"70m:real::{i}", "kind": "real", "mix": ""} for i in range(n)]


def use_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_reader_v16_addendum, "CONCURRENT_DIR", tmp_path)
    monkeypatch.setattr(llm_reader_v16_addendum, "SPEND_LOG", tmp_path / "spend.jsonl")
    monkeypatch.setattr(llm_reader, "unit_prompt", lambda unit, corpus, make_prompt: make_prompt([unit["id"]]))


def test_jobs_read_every_unit_in_every_arm_with_its_rubric(tmp_path, monkeypatch):
    use_tmp(tmp_path, monkeypatch)
    jobs = jobs_for(fake_units(30), corpus=None)
    assert len(jobs) == 120 and {(j["arm"], j["id"]) for j in jobs} == {(a, f"70m:real::{i}") for a in ARMS for i in range(30)}
    for j in jobs:
        assert j["directory"] == tmp_path / j["arm"] / "llm"
        assert ("NOT YET UNDERSTOOD" in j["text"]) == j["arm"].endswith("E3")
        assert ("C) UNRELATED JOBS" in j["text"]) == j["arm"].endswith("v7")
    assert [j["arm"] for j in jobs[:4]] != list(ARMS)  # shuffled, not arm by arm


def test_read_all_parses_each_arm_with_its_rubric_and_collect_keeps_thinking(tmp_path, monkeypatch):
    use_tmp(tmp_path, monkeypatch)
    monkeypatch.setattr(llm_reader, "BACKOFF_SECONDS", (0, 0, 0))
    jobs = [j for j in jobs_for(fake_units(1), corpus=None) if j["arm"] in ("haiku-v7", "haiku-E3")]
    for j in jobs:
        j["directory"].mkdir(parents=True)

    def runner(model, text):
        label = "C" if "NOT YET UNDERSTOOD" in text else "B"
        return {"reply": json.dumps({"label": label, "patterns": ["x"]}), "models": ["claude-haiku"],
                "usage": {"output_tokens": 9, "output_tokens_details": {"thinking_tokens": 7}}, "cost_usd": 0.01}

    guard = SpendGuard(cap=1.0, spent=0.0, log_path=tmp_path / "spend.jsonl")
    records = read_all(jobs, guard, runner, workers=1, arms=ARMS)
    assert {j["arm"]: r["label"] for j, r in zip(jobs, records)} == {"haiku-v7": "B", "haiku-E3": "C"}
    out = collect(jobs)
    assert out["arms"]["haiku-E3"]["70m:real::0"]["thinking_tokens"] == 7 and out["arms"]["sonnet-v7"] == {}
    assert abs(out["spend_usd"] - 0.02) < 1e-9
