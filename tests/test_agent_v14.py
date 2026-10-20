import json

from recgen.agent_v14 import requeue, spent
from recgen.rubric_v14 import RERUNS


def write(folder, name, label, *errors, cost=0.1):
    folder.mkdir(parents=True, exist_ok=True)
    record = {"label": label, "cost_usd": cost, "attempts": [{"error": e, "reply": ""} for e in errors]}
    (folder / f"{name}.json").write_text(json.dumps(record))


def test_requeue_moves_only_infrastructure_failures(tmp_path):
    cache, requeued = tmp_path / "agent_opus", tmp_path / "requeued"
    write(cache, "labeled", "C", None)
    write(cache, "agent_failed", None, "no valid label", "error_max_turns", "no valid label")
    write(cache, "cli_failed", None, "error_during_execution", "error_during_execution", "error_during_execution")
    assert requeue(cache, requeued) == ["cli_failed.json"]
    assert sorted(p.name for p in cache.glob("*.json")) == ["agent_failed.json", "labeled.json"]
    assert (requeued / "agent_opus" / "cli_failed.1.json").exists()


def test_requeue_stops_after_the_allowed_reruns(tmp_path):
    cache, requeued = tmp_path / "agent_opus", tmp_path / "requeued"
    for n in range(RERUNS):
        write(cache, "unit", None, "error_during_execution")
        assert requeue(cache, requeued) == ["unit.json"]
    write(cache, "unit", None, "error_during_execution")
    assert requeue(cache, requeued) == [] and (cache / "unit.json").exists()
    assert len(list((requeued / "agent_opus").glob("unit.*.json"))) == RERUNS


def test_spent_counts_every_reading_including_requeued_ones(tmp_path):
    write(tmp_path / "pilot", "a", "A", None, cost=0.25)
    write(tmp_path / "agent_opus", "b", "C", None, cost=0.5)
    write(tmp_path / "requeued" / "agent_opus", "c.1", None, "error_during_execution", cost=0.125)
    assert spent(tmp_path) == 0.875 and spent(tmp_path / "missing") == 0.0
