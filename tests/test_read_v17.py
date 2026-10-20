from recgen import llm_reader
from recgen.read_v17 import SpendGuard


def test_spend_guard_stops_calling_once_the_limit_is_reached(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_reader, "BACKOFF_SECONDS", (0, 0, 0))
    calls = []

    def runner(text):
        calls.append(text)
        return {"reply": '{"label": "A", "patterns": ["x"]}', "models": ["m"], "usage": {}, "cost_usd": 0.6}

    guard = SpendGuard(1.0, runner)
    first, second = guard.read("70m:band::1", "a", tmp_path), guard.read("70m:band::2", "b", tmp_path)
    assert first["label"] == second["label"] == "A" and len(calls) == 2
    assert guard.read("70m:band::3", "c", tmp_path) is None and len(calls) == 2
    assert SpendGuard(1.0, runner).read("70m:band::1", "a", tmp_path)["label"] == "A" and len(calls) == 2  # cached
