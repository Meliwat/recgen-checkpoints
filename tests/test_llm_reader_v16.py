import json

from recgen import llm_reader, llm_reader_v16
from recgen.llm_reader_v16 import SpendGuard, child_env, cli_for, final_record, read_all
from recgen.rubric_v16 import RESERVE_USD


def test_cli_differs_from_v7_only_in_the_model():
    cli = cli_for("sonnet")
    assert cli[cli.index("--model") + 1] == "sonnet"
    assert [x for x in cli if x != "sonnet"] == [x for x in llm_reader.CLI if x != llm_reader.LLM_MODEL]


def test_child_env_drops_effort_settings_and_turns_caching_off(monkeypatch):
    monkeypatch.setenv("CLAUDE_EFFORT", "xhigh")
    monkeypatch.setenv("CLAUDE_CODE_EFFORT_LEVEL", "high")
    env = child_env()
    assert "CLAUDE_EFFORT" not in env and "CLAUDE_CODE_EFFORT_LEVEL" not in env
    assert env["DISABLE_PROMPT_CACHING"] == "1" and env.get("PATH")


def test_spend_guard_reserves_for_readings_in_flight(tmp_path):
    guard = SpendGuard(cap=1.0, spent=1.0 - 2.5 * RESERVE_USD, log_path=tmp_path / "spend.jsonl")
    assert guard.try_start() and guard.try_start() and not guard.try_start()
    guard.finish()
    guard.add(0.01, {"id": "u"})
    assert abs(guard.spent - (1.0 - 2.5 * RESERVE_USD + 0.01)) < 1e-12
    assert json.loads((tmp_path / "spend.jsonl").read_text())["cost_usd"] == 0.01


def jobs_in(tmp_path, n: int) -> list[dict]:
    return [{"arm": "haiku-E4", "id": f"70m:real::{i}", "group": "real", "retest": False, "directory": tmp_path,
             "text": f"prompt {i}"} for i in range(n)]


def test_read_all_parses_with_the_arm_rubric_counts_every_call_and_caches(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_reader, "BACKOFF_SECONDS", (0, 0, 0))
    replies = iter(["no json", '{"label": "D", "patterns": ["x", "y"]}'] + ['{"label": "A", "patterns": ["x"]}'] * 2)

    def runner(model, text):
        assert model == "haiku"
        return {"reply": next(replies), "models": ["claude-haiku"], "usage": {}, "cost_usd": 0.01}

    guard = SpendGuard(cap=10.0, spent=0.0)
    jobs = jobs_in(tmp_path, 3)
    records = read_all(jobs, guard, runner, workers=1)
    assert [r["label"] for r in records] == ["D", "A", "A"]
    assert abs(guard.spent - 0.04) < 1e-12  # the unparseable first reply is counted too
    assert all(final_record(j) is not None for j in jobs)
    assert read_all(jobs, guard, runner, workers=1)[0]["label"] == "D"  # cached, no new call


def test_read_all_stops_when_the_guard_trips(tmp_path):
    def runner(model, text):
        return {"reply": '{"label": "B", "patterns": []}', "models": [], "usage": {}, "cost_usd": 0.8 * RESERVE_USD}

    guard = SpendGuard(cap=4 * RESERVE_USD, spent=0.0)
    records = read_all(jobs_in(tmp_path, 5), guard, runner, workers=1)
    assert [r is None for r in records] == [False, False, False, False, True]  # 3.2 + 1 reserve > 4
    assert guard.spent <= guard.cap


def test_cli_errors_are_not_final_and_stop_the_run(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_reader, "BACKOFF_SECONDS", (0, 0, 0))
    monkeypatch.setattr(llm_reader_v16, "MAX_FAILED", 1)

    def runner(model, text):
        raise RuntimeError("rate limited")

    jobs = jobs_in(tmp_path, 3)
    records = read_all(jobs, SpendGuard(cap=10.0, spent=0.0), runner, workers=1)
    assert records[0]["label"] is None and records[1:] == [None, None]
    assert final_record(jobs[0]) is None  # read again on the next run
