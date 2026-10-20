import copy

import numpy as np

from recgen.analyze_v18 import analyze, held_out_leak, tool_use
from recgen.rubric_v8 import MODELS, NO_CALLS_LEFT
from recgen.rubric_v15 import N_HELD_OUT, STRATA
from recgen.rubric_v18 import MUCH_BETTER, NO_DIFFERENCE

GROUPS = {"real": "real", "pooled": "pooled 50/50", "related": "related"}


def record(parsed, cost=0.01):
    return {"parsed": parsed, "error": None, "attempts": [{"error": None, "cost_usd": cost, "reply": ""}], "cost_usd": cost,
            "input_tokens": 10, "output_tokens": 5, "models": ["m"], "prompt_sha256": "x", "reply": ""}


def fake_data(v18_labels, v14_labels, v18_score_noise=0.0, v15_score_noise=1.0, n=8, missing=()):
    """n units of each kind per size. Each held-out value is the larger of two neuron values; the
    simulations add Gaussian noise of the given size to the truth."""
    rng = np.random.default_rng(0)
    strata = [s for s in STRATA for _ in range(N_HELD_OUT // len(STRATA))]
    answers, v15, v14 = {"units": {}}, {"units": {}}, {"first": {}, "retest": {}}
    for model in MODELS:
        for kind, group in GROUPS.items():
            for i in range(n):
                unit_id = f"{model}:{kind}:{i}"
                neurons = rng.uniform(0, 10, (N_HELD_OUT, 2))
                if kind == "real":
                    neurons[:, 1] = 0
                values = neurons.max(axis=1)
                held_out = [{"rank": 51 + 3 * j, "stratum": s, "value": float(v), "neuron_values": [float(x) for x in nv],
                             "driver": int(np.argmax(nv)), "snippet": f"{unit_id} snippet {j}"}
                            for j, (s, v, nv) in enumerate(zip(strata, values, neurons))]
                noisy = lambda size: [float(np.clip(v + rng.normal(0, size), 0, 10)) for v in values]  # noqa: E731
                v15["units"][unit_id] = {"held_out": held_out, "first": {
                    "explanation": {"parsed": "an explanation"}, "simulations": {"sonnet": record(noisy(v15_score_noise))},
                    "judge": record(False)}}
                answered = unit_id not in missing
                agent = {"label": v18_labels[kind] if answered else None, "description": "fires on things" if answered else None,
                         "confidence": "high", "error": None, "attempts": [{"error": None, "cost_usd": 0.2}], "cost_usd": 0.2,
                         "input_tokens": 100, "output_tokens": 50, "models": ["claude-opus-5-5"], "messages": [], "reply": "",
                         "tool_calls": [{"tool": "examples", "arguments": {"start": 51, "count": 20}, "answer": "rank 51 (5.0): x"},
                                        {"tool": "activation", "arguments": {"texts": ["a", "b", "c"]}, "answer": "Text 1"}]}
                entry = {"id": unit_id, "model": model, "kind": kind, "group": group, "retested": i == 0,
                         "held_out_ranks": [h["rank"] for h in held_out],
                         "first": {"agent": agent, "simulation": record(noisy(v18_score_noise)) if answered else None,
                                   "judge": record(kind != "real") if answered else None}}
                v14["first"][unit_id] = {"label": v14_labels[kind], "confidence": "high", "prompt_sha256": "p", "cost_usd": 0.1,
                                         "output_tokens": 9, "calls": 5, "test texts": 10, "deepest rank": 300,
                                         "read ranks 51-500": True}
                if i == 0:
                    entry["retest"] = {"agent": copy.deepcopy(agent), "simulation": record(noisy(v18_score_noise))}
                    v14["retest"][unit_id] = {"label": v14_labels[kind]}
                answers["units"][unit_id] = entry
    return answers, v15, v14


V8_AUC = {"sonnet agent": {"mean_over_models": 0.64, "interval": [0.58, 0.69]}}


def test_a_reader_that_labels_every_pool_c_beats_one_that_cant_tell():
    answers, v15, v14 = fake_data({"real": "A", "pooled": "C", "related": "B"}, {"real": "B", "pooled": "B", "related": "B"})
    s = analyze(answers, v15, v14, V8_AUC)["summary"]
    assert s["primary"]["v18 label auc"][0] == 1.0 and s["primary"]["v14 label auc"][0] == 0.5
    assert s["primary"]["v18 minus v14"] == 0.5 and s["verdict"] == MUCH_BETTER
    assert s["detection"]["flag"]["v18"]["mean_over_models"] == 1.0 and s["detection"]["flag"]["v14"]["mean_over_models"] == 0.0
    assert s["one job"]["real"]["v18"]["share"] == 1.0 and s["one job"]["pooled 50/50"]["v18"]["share"] == 0.0
    assert s["retest"]["v18"]["same label"] == 1.0 and s["retest"]["units"] == 3 * len(GROUPS)


def test_identical_readers_show_no_difference():
    labels = {"real": "A", "pooled": "C", "related": "B"}
    s = analyze(*fake_data(labels, labels), V8_AUC)["summary"]
    assert s["primary"]["v18 minus v14"] == 0 and s["verdict"] == NO_DIFFERENCE


def test_a_description_that_predicts_the_truth_scores_above_a_noisy_one():
    labels = {"real": "A", "pooled": "C", "related": "B"}
    s = analyze(*fake_data(labels, labels, v18_score_noise=0.0, v15_score_noise=3.0), V8_AUC)["summary"]
    paired = s["simulation score"]["paired score difference, v18 minus v15"]
    assert paired["real"]["v18 mean"] > 0.99 and paired["real"]["interval"][0] > 0 and paired["pooled 50/50"]["interval"][0] > 0
    assert s["simulation score"]["v18"]["pooled 50/50 minus real"] > -0.01  # perfect predictions don't separate pools
    tracked = s["exploratory"]["tracked neurons"]["v18"]["pooled 50/50"]
    assert tracked["tracked neuron r"] > tracked["other neuron r"] > 0  # the pool's value follows both neurons


def test_units_without_an_answer_are_left_out_and_counted():
    labels = {"real": "A", "pooled": "C", "related": "B"}
    missing = ("70m:real:1", "410m:pooled:2")
    s = analyze(*fake_data(labels, labels, missing=missing), V8_AUC)["summary"]
    assert s["checks"]["units without a v18 answer"] == 2 and s["checks"]["units without a v18 score"] == 2
    assert s["primary"]["units"] == 2 * 8 * len(MODELS) - 2
    assert s["tool use and cost"]["readings"]["simulator, first"]["readings"] == len(GROUPS) * 8 * len(MODELS) - 2


def test_tool_use_maps_the_agents_ranks_past_held_out_windows():
    calls = [{"tool": "examples", "arguments": {"start": 51, "count": 3}, "answer": "rank 51"},
             {"tool": "activation", "arguments": {"texts": ["a", "b"]}, "answer": "Text 1"},
             {"tool": "examples", "arguments": {"start": 4000, "count": 20}, "answer": NO_CALLS_LEFT}]
    use = tool_use(calls, held_out=[52, 60])
    assert use["deepest rank"] == 54 and use["ranks read past 50"] == 3 and use["test texts"] == 2 and use["calls"] == 3
    assert use["read ranks 51-500"] and tool_use(calls)["deepest rank"] == 53


def test_held_out_leak_counts_held_out_snippets_in_examples_answers():
    labels = {"real": "A", "pooled": "C", "related": "B"}
    answers, v15, _ = fake_data(labels, labels, n=1)
    unit_id = "70m:real:0"
    answers["units"][unit_id]["first"]["agent"]["tool_calls"][0]["answer"] = f"rank 51 (5.0): {unit_id} snippet 3\nrank 52 (4.0): x"
    leak = held_out_leak(answers, v15)
    assert leak["held-out snippets shown to the agent"] == 1
    assert leak["matches"][0]["held-out rank"] == 51 + 3 * 3 and leak["matches"][0]["shown at full-list rank"] == 52


def test_post_hoc_transitions_and_the_primary_without_mixed_model_readings():
    answers, v15, v14 = fake_data({"real": "A", "pooled": "C", "related": "B"}, {"real": "B", "pooled": "C", "related": "B"})
    answers["units"]["70m:pooled:0"]["first"]["agent"]["models"] = ["claude-opus-5-5", "other"]
    s = analyze(answers, v15, v14, V8_AUC)["summary"]["post hoc"]
    assert s["label transitions, v14 -> v18"]["real"] == {"B -> A": 8 * len(MODELS)}
    assert s["readings whose usage names a second model"] == ["70m:pooled:0"] and s["primary without them"]["units"] == 2 * 8 * 3 - 1
    assert s["catch-all words in one-job descriptions"]["real"] == {"one-job answers": 8 * len(MODELS), "share with a catch-all word": 0.0}


def test_after_review_pairs_label_shares_and_reads_protocol_traces():
    answers, v15, v14 = fake_data({"real": "A", "pooled": "C", "related": "B"}, {"real": "B", "pooled": "C", "related": "B"})
    first = answers["units"]["70m:real:0"]["first"]["agent"]
    first["messages"] = [{"text": "My candidates: X, Y"}, {"tool": "activation", "input": {}}]
    first["tool_calls"] = first["tool_calls"][::-1]
    s = analyze(answers, v15, v14, V8_AUC)["summary"]["post hoc, after an independent review"]
    assert s["label shares, v18 minus v14 (paired)"]["real, A"]["difference"] == 1.0
    assert s["label shares, v18 minus v14 (paired)"]["pooled 50/50, C"] == {"difference": 0.0, "interval": [0.0, 0.0], "units": 24}
    traces = s["protocol traces"]["first"]
    assert traces["first tool call is activation"] == 1 and traces["visible text before the first tool call mentions candidates"] == 1
    assert traces["deepest rank asked for in the agent's list at most 500"] == traces["readings"] == 72
