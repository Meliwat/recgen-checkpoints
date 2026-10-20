import json

import pytest

from recgen import llm_reader, reader_v21, rubric_v7, rubric_v16
from recgen.llm_reader_v16 import SpendGuard, read_all
from recgen.rubric_v16 import E3_DECISION, V7_DECISION
from recgen.rubric_v21 import ARMS_A, RERUNS, agent_prompt
from tests.test_probe_v8 import fake_view


def test_prompt_for_gives_v7s_and_v16s_exact_prompts():
    snippets = ["a [[b]] c", "d [[e]] f"]
    assert reader_v21.prompt_for("v7") is rubric_v7.prompt
    assert reader_v21.prompt_for("E3")(snippets) == rubric_v16.prompt("E3", snippets)


def test_a_unit_view_with_the_agent_prompt_changes_only_the_decision_paragraph():
    default, essay = fake_view().prompt(), fake_view(make_prompt=agent_prompt).prompt()
    assert V7_DECISION in default and essay == default.replace(V7_DECISION, E3_DECISION)


def test_read_all_reads_each_arm_with_its_model_and_rubric(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_reader, "BACKOFF_SECONDS", (0, 0, 0))
    seen = []

    def runner(model, text):
        seen.append(model)
        return {"reply": '{"label": "C", "patterns": ["x"]}', "models": [model], "usage": {}, "cost_usd": 0.01}

    jobs = [{"arm": arm, "id": f"70m:real::{i}", "group": "real", "retest": False, "directory": tmp_path / "pilot" / arm,
             "text": f"prompt {i}"} for i, arm in enumerate(ARMS_A)]
    for j in jobs:
        j["directory"].mkdir(parents=True)
    guard = SpendGuard(cap=1.0, spent=0.0, log_path=tmp_path / "spend.jsonl")
    records = read_all(jobs, guard, runner, workers=1, arms=ARMS_A, pilot_dir=tmp_path / "pilot")
    assert [r["label"] for r in records] == ["C", "C"] and seen == ["claude-opus-5-5"] * 2
    assert all(json.loads(line)["kind"] == "pilot" for line in (tmp_path / "spend.jsonl").read_text().splitlines())


def test_spent_counts_arm_a_calls_and_every_arm_b_reading(tmp_path, monkeypatch):
    monkeypatch.setattr(reader_v21, "SPEND_LOG", tmp_path / "spend.jsonl")
    monkeypatch.setattr(reader_v21, "B_DIR", tmp_path / "b")
    assert reader_v21.spent() == 0.0
    (tmp_path / "spend.jsonl").write_text('{"cost_usd": 0.5}\n{"cost_usd": null}\n')
    for folder, cost in (("agent", 1.0), ("requeued/agent", 0.25)):
        (tmp_path / "b" / folder).mkdir(parents=True)
        (tmp_path / "b" / folder / "u.json").write_text(json.dumps({"cost_usd": cost}))
    assert reader_v21.spent() == 1.75


def test_requeue_moves_infrastructure_failures_at_most_reruns_times(tmp_path):
    cache, requeued = tmp_path / "agent", tmp_path / "requeued"
    cache.mkdir()
    failed = {"label": None, "attempts": [{"error": "no result from the CLI"}]}
    agent_failure = {"label": None, "attempts": [{"error": "no valid label", "reply": "hmm"}]}
    (cache / "keep.json").write_text(json.dumps(agent_failure))
    for n in range(RERUNS + 1):
        (cache / "u.json").write_text(json.dumps(failed))
        moved = reader_v21.requeue(cache, requeued)
        assert moved == (["u.json"] if n < RERUNS else [])
    assert (cache / "keep.json").exists() and len(list((requeued / "agent").glob("u.*.json"))) == RERUNS


def test_read_size_b_waits_out_memory_errors_and_deletes_the_weights(tmp_path, monkeypatch):
    calls = []

    def read_model(*args):
        calls.append(args[-1])
        (tmp_path / "pythia-70m").mkdir(exist_ok=True)
        if len(calls) == 1:
            raise RuntimeError("MPS backend out of memory")
        return ["record"]

    monkeypatch.setattr(reader_v21, "read_model", read_model)
    monkeypatch.setattr(reader_v21, "WEIGHTS_DIR", tmp_path)
    monkeypatch.setattr(reader_v21, "LOAD_WAIT_SECONDS", 0)
    assert reader_v21.read_size_b("70m", [], [], {}, None, None) == ["record"]
    assert calls == [agent_prompt, agent_prompt] and not (tmp_path / "pythia-70m").exists()
    monkeypatch.setattr(reader_v21, "read_model", lambda *args: (_ for _ in ()).throw(RuntimeError("other")))
    with pytest.raises(RuntimeError, match="other"):
        reader_v21.read_size_b("70m", [], [], {}, None, None)


def test_collect_files_arm_a_answers_by_series_and_reading(tmp_path, monkeypatch):
    monkeypatch.setattr(reader_v21, "SPEND_LOG", tmp_path / "spend.jsonl")
    monkeypatch.setattr(reader_v21, "B_DIR", tmp_path / "b")
    jobs = []
    for arm, retest in (("opus-E3", False), ("opus-v7", True)):
        directory = tmp_path / arm
        directory.mkdir()
        job = {"arm": arm, "id": "70m:real::1", "retest": retest, "directory": directory, "text": "p"}
        llm_reader.read(job["id"], "p", directory, lambda text: {"reply": '{"label": "A"}', "models": ["m"],
                                                                  "usage": {}, "cost_usd": 0.02})
        jobs.append(job)
    out = reader_v21.collect(jobs, [], {"first": {}}, {})
    assert out["arm_a"]["opus-E3"]["first"]["70m:real::1"]["label"] == "A"
    assert out["arm_a"]["opus-v7"]["retest"]["70m:real::1"]["cost_usd"] == 0.02
    assert out["arm_a"]["opus-v7"]["first"] == {} and out["spend_usd"] == 0.0
