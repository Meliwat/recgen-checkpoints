import json

import numpy as np

from recgen.agent_v18 import AuditView, failed_outside_the_model, requeue, spent
from recgen.probe_v8 import Backend, UnitView
from recgen.rubric_v8 import N_WINDOWS
from recgen.rubric_v15 import N_HELD_OUT, held_out_ranks
from recgen.rubric_v18 import MAX_TOOL_CALLS, N_LISTED, RERUNS, TOOLS

T = 64


def view_data():
    """A two-neuron pool over a full list of 4,096 windows, five of them shown (as in test_simulate_v15)."""
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
    return unit, v6, window_max, window_pos, corpus


def test_audit_view_hides_v15s_held_out_windows_and_keeps_the_rest_in_order():
    data = view_data()
    full, ours = UnitView(*data), AuditView(*data)
    held = {rank for rank, _ in held_out_ranks(data[0]["id"])}
    assert len(held) == N_HELD_OUT and ours.held_out == sorted(held)
    assert len(ours.ranked) == N_LISTED and ours.ranked == [e for r, e in enumerate(full.ranked, 1) if r not in held]
    assert ours.ranked[:5] == full.ranked[:5]
    assert not {key for key, _ in ours.ranked} & {full.ranked[r - 1][0] for r in held}
    assert ours.examples(N_LISTED, 1).startswith(f"rank {N_LISTED} (") and "ranks go from 1 to" in ours.examples(N_LISTED + 1, 1)


def test_audit_view_prompt_has_v8s_snippet_lines_and_the_protocol():
    data = view_data()
    v8_lines = [line for line in UnitView(*data).prompt().split("\n") if line[:3] in {f"{i}. " for i in range(1, 6)}]
    ours = AuditView(*data).prompt()
    assert len(v8_lines) == 5 and ours.endswith("Snippets:\n" + "\n".join(v8_lines)) and "ranks 51 to 500" in ours


def test_backend_takes_v18s_budget_and_tools():
    view = AuditView(*view_data())
    backend = Backend({view.unit["id"]: view}, reader=None, max_calls=MAX_TOOL_CALLS, tools=TOOLS)
    backend.open("h", view.unit["id"])
    answers = [backend.call("h", "examples", {"start": 51, "count": 1}) for _ in range(MAX_TOOL_CALLS + 1)]
    assert f"({MAX_TOOL_CALLS - 1} tool calls left)" in answers[0] and answers[-1].startswith("No tool calls left")
    assert backend.tools is TOOLS and len(backend.close("h")) == MAX_TOOL_CALLS + 1


def test_requeue_moves_infrastructure_failures_at_most_reruns_times(tmp_path):
    cache, requeued = tmp_path / "agent_opus", tmp_path / "requeued"
    cache.mkdir()
    cli_error = {"label": None, "cost_usd": 0.5, "attempts": [{"error": "Command timed out", "reply": ""}]}
    agent_error = {"label": None, "cost_usd": 0.25, "attempts": [{"error": "no valid label", "reply": "hmm"}]}
    simulation_error = {"parsed": None, "cost_usd": 0.0, "attempts": [{"error": "OSError: x", "reply": ""}]}
    (cache / "a.json").write_text(json.dumps(cli_error))
    (cache / "b.json").write_text(json.dumps(agent_error))
    (cache / "c.json").write_text(json.dumps(simulation_error))
    assert requeue(cache, requeued) == ["a.json", "c.json"] and sorted(p.name for p in cache.iterdir()) == ["b.json"]
    for _ in range(RERUNS):
        (cache / "a.json").write_text(json.dumps(cli_error))
        requeue(cache, requeued)
    assert (cache / "a.json").exists() and len(list((requeued / "agent_opus").glob("a.*.json"))) == RERUNS
    assert abs(spent(tmp_path) - (0.5 * (RERUNS + 1) + 0.25)) < 1e-9


def test_simulations_that_did_not_parse_are_the_models_failure():
    assert not failed_outside_the_model({"parsed": None, "attempts": [{"error": "invalid reply"}] * 3})
    assert not failed_outside_the_model({"parsed": [1.0], "attempts": [{"error": "OSError"}, {"error": None}]})
