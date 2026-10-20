import numpy as np
import torch

from recgen import mcp_v8
from recgen.probe_v8 import Backend, UnitView, v7_order
from recgen.rubric_v8 import EXAMPLES_PER_CALL, MAX_TOOL_CALLS, NO_CALLS_LEFT, SCALE

W, T = 60, 64


def fake_view(neurons=(0, 1)) -> UnitView:
    """Two sampled neurons whose pre-activations at stored residual row r are r + 1 and 2 * (r + 1)."""
    rng = np.random.default_rng(0)
    window_max = rng.gamma(2.0, 1.0, (W, 3)).astype(np.float16)
    window_pos = rng.integers(32, T, (W, 3)).astype(np.int16)
    z = np.zeros((5, 2), np.float16)
    z[:, 0] = np.arange(1, 6)
    v6 = {"neurons": np.array([0, 1, 2]), "z": z, "gain": np.ones(2, np.float32),
          "weights": np.array([[1, 0], [2, 0], [0, 1]], np.float32), "constant": np.zeros(3, np.float32)}
    unit = {"id": "70m:pooled:50/50:0-1", "neurons": neurons, "rows": np.arange(5),
            "keys": np.array([[w, 40] for w in (3, 7, 11, 13, 17)])}
    corpus = np.arange(W * T).reshape(W, T) % 50000
    return UnitView(unit, v6, window_max, window_pos, corpus)


def test_v7_order_is_a_seeded_permutation():
    order = v7_order("70m:real::3", 50)
    assert sorted(order) == list(range(50)) and np.array_equal(order, v7_order("70m:real::3", 50))


def test_unit_view_ranks_the_shown_examples_first_then_every_other_window():
    view = fake_view()
    first_values = [v for _, v in view.first]
    assert [k for k, _ in view.ranked[:5]] == [k for k, _ in sorted(view.first, key=lambda kv: -kv[1])]
    rest = view.ranked[5:]
    assert len(view.ranked) == W and {k[0] for k, _ in rest}.isdisjoint({3, 7, 11, 13, 17})
    assert all(a >= b for (_, a), (_, b) in zip(rest, rest[1:])) and np.isclose(max(v for _, v in rest), SCALE)
    # The pool's value at a stored row is the larger of its neurons' normalized activations.
    gelu = lambda x: torch.nn.functional.gelu(torch.tensor(x)).numpy()  # noqa: E731
    expected = SCALE * np.maximum(gelu(np.arange(1.0, 6)) / view.norms[0], gelu(2 * np.arange(1.0, 6)) / view.norms[1]) / view.top
    assert np.allclose(first_values, expected, atol=1e-3)


def test_examples_caps_the_count_and_reports_ranks_out_of_range():
    view = fake_view()
    assert len(view.examples(1, 100).split("\n")) == EXAMPLES_PER_CALL
    assert view.examples(59, 5).split("\n")[0].startswith("rank 59 (")
    assert "ranks go from 1 to 60" in view.examples(500, 3)


def test_prompt_shows_every_example_with_its_value():
    lines = [line for line in fake_view().prompt().split("\n") if line[:3] in {f"{i}. " for i in range(1, 6)}]
    assert len(lines) == 5 and all(line.split(" ", 2)[1].startswith("(") and "[[" in line for line in lines)


def test_backend_enforces_the_tool_budget_and_logs_every_call():
    view = fake_view()
    backend = Backend({view.unit["id"]: view}, reader=None)
    backend.open("h", view.unit["id"])
    answers = [backend.call("h", "examples", {"start": 1, "count": 2}) for _ in range(MAX_TOOL_CALLS + 1)]
    assert "(7 tool calls left)" in answers[0] and NO_CALLS_LEFT in answers[MAX_TOOL_CALLS - 1]
    assert answers[-1] == NO_CALLS_LEFT
    backend.open("g", view.unit["id"])
    assert backend.call("g", "examples", {"start": "x"}).startswith("Invalid arguments")
    assert backend.call("g", "steer", {}).startswith("Unknown tool")
    calls = backend.close("h")
    assert len(calls) == MAX_TOOL_CALLS + 1 and calls[0]["answer"] == answers[0] and calls[0]["tool"] == "examples"


def test_mcp_server_echoes_the_protocol_and_forwards_calls(monkeypatch):
    sent = []
    monkeypatch.setattr(mcp_v8, "backend", lambda url, payload=None: sent.append(payload) or (
        {"tools": ["t"]} if payload is None else {"text": "answer"}))
    init = mcp_v8.respond({"id": 1, "method": "initialize", "params": {"protocolVersion": "2025-11-25"}}, "u", "h")
    assert init["result"]["protocolVersion"] == "2025-11-25"
    assert mcp_v8.respond({"method": "notifications/initialized"}, "u", "h") is None
    assert mcp_v8.respond({"id": 2, "method": "tools/list"}, "u", "h")["result"] == {"tools": ["t"]}
    call = mcp_v8.respond({"id": 3, "method": "tools/call", "params": {"name": "examples", "arguments": {"start": 1}}},
                          "u", "h")
    assert call["result"]["content"][0]["text"] == "answer"
    assert sent[-1] == {"handle": "h", "tool": "examples", "arguments": {"start": 1}}
    assert mcp_v8.respond({"id": 4, "method": "server/discover"}, "u", "h")["error"]["code"] == -32601
