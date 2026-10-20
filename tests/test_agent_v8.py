import json

from recgen.agent_v8 import cache_path, read
from recgen.probe_v8 import Backend
from tests.test_probe_v8 import fake_view


def test_read_retries_failed_attempts_and_caches_a_valid_label(tmp_path):
    view = fake_view()
    backend = Backend({view.unit["id"]: view}, reader=None)
    tools = ["mcp__probe__examples", "mcp__probe__activation"]
    replies = iter([{"reply": "", "error": "error_max_turns", "tools_offered": tools},
                    {"reply": '{"label": "A"}', "error": None, "tools_offered": []},
                    {"reply": "no json", "error": None, "tools_offered": tools},
                    {"reply": '{"label": "C", "patterns": ["a", "b"], "confidence": "medium"}', "error": None,
                     "tools_offered": tools}])

    def runner(text, url, handle):
        backend.call(handle, "examples", {"start": 51, "count": 3})
        return {**next(replies), "cost_usd": 0.01, "usage": {"input_tokens": 5, "cache_read_input_tokens": 7},
                "models": ["m"], "messages": [], "num_turns": 2}

    (tmp_path / "failed").mkdir()
    failed = read(view, tmp_path / "failed", backend, "url", runner)
    assert failed["label"] is None and failed["error"] == "no valid label"
    assert [a["error"] for a in failed["attempts"]] == ["error_max_turns", "probe tools not offered", "no valid label"]
    record = read(view, tmp_path, backend, "url", runner)
    assert record["label"] == "C" and record["confidence"] == "medium" and record["error"] is None
    assert len(record["attempts"]) == 1 and abs(record["cost_usd"] - 0.01) < 1e-9 and record["input_tokens"] == 12
    assert len(record["tool_calls"]) == 1 and record["tool_calls"][0]["answer"].startswith("rank 51 (")
    assert json.loads(cache_path(tmp_path, view.unit["id"]).read_text())["label"] == "C"
    assert read(view, tmp_path, None, None, runner=None)["label"] == "C"  # cached: no backend, no runner


def test_read_gives_up_after_the_attempts(tmp_path):
    view = fake_view()
    backend = Backend({view.unit["id"]: view}, reader=None)

    def runner(text, url, handle):
        raise RuntimeError("CLI failed")

    record = read(view, tmp_path, backend, "url", runner)
    assert record["label"] is None and record["error"] == "CLI failed" and len(record["attempts"]) == 3
