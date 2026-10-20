import json

import numpy as np

from recgen import simulate_v15
from recgen.agent_v8 import SpendGuard, cache_path
from recgen.probe_v8 import UnitView
from recgen.rubric_v8 import N_WINDOWS, SCALE
from recgen.rubric_v15 import N_HELD_OUT, parse_judgment, parse_predictions
from recgen.simulate_v15 import SPEND_CAP_REACHED, held_out, read, shown_lines

T = 64


def fake_view() -> tuple[UnitView, np.ndarray]:
    """A two-neuron pool over a full list of 4,096 windows, five of them shown."""
    rng = np.random.default_rng(1)
    window_max = rng.gamma(2.0, 1.0, (N_WINDOWS, 3)).astype(np.float16)
    window_pos = rng.integers(32, T, (N_WINDOWS, 3)).astype(np.int16)
    z = np.zeros((5, 2), np.float16)
    z[:, 0] = np.arange(1, 6)
    v6 = {"neurons": np.array([0, 1, 2]), "z": z, "gain": np.ones(2, np.float32),
          "weights": np.array([[1, 0], [2, 0], [0, 1]], np.float32), "constant": np.zeros(3, np.float32)}
    unit = {"id": "70m:pooled:50/50:0-2", "neurons": (0, 2), "rows": np.arange(5),
            "keys": np.array([[w, 40 + i] for i, w in enumerate((3, 7, 11, 13, 17))])}
    corpus = np.arange(N_WINDOWS * T).reshape(N_WINDOWS, T) % 50000
    return UnitView(unit, v6, window_max, window_pos, corpus), window_max


def test_held_out_examples_are_unshown_windows_with_their_drivers():
    view, window_max = fake_view()
    examples = held_out(view, window_max)
    assert len(examples) == N_HELD_OUT and not {e["window"] for e in examples} & {3, 7, 11, 13, 17}
    for e in examples:
        assert np.isclose(e["value"], max(e["neuron_values"])) and e["driver"] == int(np.argmax(e["neuron_values"]))
        assert (e["window"], e["position"]) == view.ranked[e["rank"] - 1][0] and "[[" in e["snippet"]
    assert {e["driver"] for e in examples} == {0, 1} and max(e["value"] for e in examples) <= SCALE


def test_shown_lines_are_the_snippets_of_v8s_agent_prompt():
    view, _ = fake_view()
    lines = shown_lines(view)
    assert len(lines) == 5 and all(line in view.prompt().split("\n") for line in lines)


def test_read_retries_invalid_replies_then_caches_the_parsed_one(tmp_path, monkeypatch):
    monkeypatch.setattr(simulate_v15, "BACKOFF_SECONDS", (0, 0))
    replies = iter([{"reply": "", "error": "overloaded"}, {"reply": "[1, 2]", "error": None},
                    {"reply": "[1, 2, 3]", "error": None}])

    def runner(text, model):
        return {**next(replies), "models": [model], "usage": {"input_tokens": 4, "cache_read_input_tokens": 6},
                "cost_usd": 0.01}

    record = read("70m:real::1", "prompt", tmp_path, "m", lambda r: parse_predictions(r, n=3), runner)
    assert record["parsed"] == [1.0, 2.0, 3.0] and record["error"] is None and record["input_tokens"] == 10
    assert [a["error"] for a in record["attempts"]] == ["overloaded", "invalid reply", None]
    assert np.isclose(record["cost_usd"], 0.03) and json.loads(cache_path(tmp_path, "70m:real::1").read_text())["parsed"]
    assert read("70m:real::1", "prompt", tmp_path, "m", parse_predictions, runner=None) == record  # cached
    # A false judgment is a valid answer, not a failure.
    judged = read("70m:real::2", "p", tmp_path, "m", parse_judgment,
                  lambda t, m: {"reply": '{"unrelated_patterns": false}', "error": None, "cost_usd": 0.0})
    assert judged["parsed"] is False and len(judged["attempts"]) == 1


def test_read_gives_up_after_the_attempts_and_the_guard_stops_new_readings(tmp_path, monkeypatch):
    monkeypatch.setattr(simulate_v15, "BACKOFF_SECONDS", (0, 0))
    failed = read("u", "p", tmp_path, "m", parse_predictions, lambda t, m: {"reply": "no", "error": None, "cost_usd": 0.5})
    assert failed["parsed"] is None and failed["error"] == "invalid reply" and len(failed["attempts"]) == 3
    stopped = read("v", "p", tmp_path, "m", parse_predictions, runner=None, guard=SpendGuard(cap=1.0, spent=1.0))
    assert stopped["error"] == SPEND_CAP_REACHED and not cache_path(tmp_path, "v").exists()
