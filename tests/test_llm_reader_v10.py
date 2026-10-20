import json

from recgen import llm_reader, llm_reader_v10
from recgen.llm_reader_v10 import MAX_FAILED, UNPARSEABLE, child_env, cli_for, collect, final_record, read_all


def job(tmp_path, unit_id="70m:real::3", text="prompt", retest=False):
    return {"id": unit_id, "group": "real", "text": text, "directory": tmp_path, "retest": retest}


def no_backoff(monkeypatch):
    monkeypatch.setattr(llm_reader, "BACKOFF_SECONDS", (0, 0, 0))


def test_cli_differs_from_v7_only_in_the_model():
    for model in ("sonnet", "opus"):
        cli = cli_for(model)
        diff = [(a, b) for a, b in zip(llm_reader.CLI, cli) if a != b]
        assert len(cli) == len(llm_reader.CLI) and diff == [("haiku", model)]


def test_child_env_drops_only_the_effort_override(monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_EFFORT_LEVEL", "max")
    monkeypatch.setenv("SOMETHING_ELSE", "kept")
    env = child_env()
    assert "CLAUDE_CODE_EFFORT_LEVEL" not in env and env["SOMETHING_ELSE"] == "kept"


def test_a_cli_error_is_not_final_but_an_unparseable_reply_is(tmp_path, monkeypatch):
    no_backoff(monkeypatch)

    def failing(text):
        raise RuntimeError("rate limited")

    a = job(tmp_path, "a")
    llm_reader.read(a["id"], a["text"], tmp_path, failing)
    assert final_record(a) is None
    b = job(tmp_path, "b")
    llm_reader.read(b["id"], b["text"], tmp_path, lambda text: {"reply": "no json", "models": [], "usage": {},
                                                                  "cost_usd": 0})
    assert final_record(b)["error"] == UNPARSEABLE
    assert final_record(job(tmp_path, "b", text="a changed prompt")) is None


def test_read_all_rereads_a_failed_reading_and_keeps_final_ones(tmp_path, monkeypatch):
    no_backoff(monkeypatch)
    failed, done = job(tmp_path, "failed"), job(tmp_path, "done")

    def failing(text):
        raise RuntimeError("timed out")

    llm_reader.read(failed["id"], failed["text"], tmp_path, failing)
    llm_reader.read(done["id"], done["text"], tmp_path, lambda text: {"reply": '{"label": "A"}', "models": ["m"],
                                                                        "usage": {}, "cost_usd": 0})
    calls = []

    def runner(text):
        calls.append(text)
        return {"reply": '{"label": "C"}', "models": ["m"], "usage": {}, "cost_usd": 0}

    records = read_all([failed, done], runner, workers=1)
    assert [r["label"] for r in records] == ["C", "A"] and len(calls) == 1


def test_read_all_stops_starting_readings_after_repeated_cli_errors(tmp_path, monkeypatch):
    no_backoff(monkeypatch)

    def failing(text):
        raise RuntimeError("usage limit reached")

    jobs = [job(tmp_path, f"u{i}") for i in range(MAX_FAILED + 5)]
    records = read_all(jobs, failing, workers=1)
    assert sum(r is not None for r in records) == MAX_FAILED and records[-1] is None


def test_collect_splits_first_readings_from_retests_and_counts_tokens(tmp_path, monkeypatch):
    no_backoff(monkeypatch)
    usage = {"input_tokens": 2, "cache_creation_input_tokens": 100, "output_tokens": 30,
             "output_tokens_details": {"thinking_tokens": 20}}
    first, again = job(tmp_path / "first", "u"), job(tmp_path / "again", "u", retest=True)
    for j in (first, again):
        j["directory"].mkdir()
        llm_reader.read(j["id"], j["text"], j["directory"], lambda text: {
            "reply": json.dumps({"label": "B"}), "models": ["claude-x"], "usage": usage, "cost_usd": 0.01})
    answers = collect("sonnet", [first, again, job(tmp_path / "first", "never read")])
    assert list(answers["first"]) == ["u"] and list(answers["retest"]) == ["u"]
    a = answers["first"]["u"]
    assert (a["label"], a["input_tokens"], a["output_tokens"], a["thinking_tokens"]) == ("B", 102, 30, 20)


def test_answers_go_to_a_file_per_reader():
    assert llm_reader_v10.answers_path("opus").name == "v10_opus_answers.json"
