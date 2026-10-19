import numpy as np

from recgen.analyze_v4 import evaluate_pair, exploratory_stats, neuron_results
from recgen.rubric import PACKING_LIKE, RECURSIVE_GENERALIZATION
from recgen.rubric_v4 import STEPS

T = len(STEPS)
FORMS = 5  # facet 1 forms over checkpoints 4 to 6; facet 0 is formed from the start


def units(start, k):
    return np.arange(start, start + k)


def synthetic_steps(shared_while_forming: bool) -> list[dict]:
    """Neuron 1 has facets 0 and 1; neuron 2's two facets are the null. Facet 1 shares half of
    facet 0's drivers around its formation (if asked), then nothing by the end."""
    steps = []
    for t in range(T):
        gains = {(1, 0): 1.0, (1, 1): float(np.clip((t - 3) / 3, 0, 1)), (2, 0): 1.0, (2, 1): 1.0}
        share = shared_while_forming and 4 <= t <= 6
        summary = {"gains": gains, "mlp": {}, "heads": {}}
        for kind, k in (("mlp", 100), ("heads", 5)):
            own0 = units(0, k)
            own1 = np.concatenate([units(0, k // 2), units(10_000, k - k // 2)]) if share else units(10_000, k)
            summary[kind][1, (1, 0)], summary[kind][1, (1, 1)] = own0, own1
            summary[kind][1, (2, 0)], summary[kind][1, (2, 1)] = units(20_000, k), units(30_000, k)
        steps.append(summary)
    return steps


V1_ROW = {"hand_picked": True, "label": PACKING_LIKE, "emergence_steps": [STEPS[1], STEPS[6]]}
V2_ROW = {"v2_label": PACKING_LIKE, "adjacent_pairs": {"0-1": {"v2": False}}, "driver_overlaps": {"0-1": 0.0}}
KEYS = [(1, 0), (1, 1), (2, 0), (2, 1)]


def test_facets_that_share_drivers_while_one_forms_branched_then_separated():
    result = neuron_results(1, 2, KEYS, synthetic_steps(shared_while_forming=True), V1_ROW, V2_ROW)
    pair = result["mlp"]["pairs"]["0-1"]
    assert pair["adjacent"] and pair["peak"] == 0.5 and pair["series"][-1] == 0.0
    assert pair["branched_then_separated"]
    assert result["mlp"]["label"] == RECURSIVE_GENERALIZATION
    assert result["heads"]["label"] == RECURSIVE_GENERALIZATION
    assert result["final_mlp_overlap_vs_v2"] == 0.0


def test_facets_that_never_share_drivers_stay_packing_like():
    result = neuron_results(1, 2, KEYS, synthetic_steps(shared_while_forming=False), V1_ROW, V2_ROW)
    assert not result["mlp"]["pairs"]["0-1"]["adjacent"]
    assert result["mlp"]["label"] == PACKING_LIKE


def test_evaluate_pair_ignores_overlap_outside_the_window():
    series = np.array([0.9] + [0.0] * (T - 1))
    null = [np.zeros(T)] * 20
    assert not evaluate_pair(series, null, (2, 4))["adjacent"]
    assert evaluate_pair(series, null, (0, 4))["adjacent"]


def test_exploratory_stats_on_one_hand_built_neuron():
    window = [STEPS[0], STEPS[4]]  # five checkpoints
    series = [0.2, 0.0, 0.2, 0.0, 0.0] + [0.0] * (T - 5)
    null_p95 = [0.1] * T
    results = [{
        "neuron": 1,
        "formation_windows": [[STEPS[0], STEPS[3]], [STEPS[1], STEPS[4]]],  # overlapping, facet 1 starts later
        "emergence_steps": [STEPS[3], STEPS[6]],
        "mlp": {"pairs": {"0-1": {"window": window, "series": series, "null_p95_series": null_p95,
                                  "v2_adjacent": False, "adjacent": False}}},
        "heads": {"pairs": {"0-1": {"window": window, "series": [1.0] * (T - 1) + [0.05], "null_p95_series": null_p95,
                                    "v2_adjacent": False, "adjacent": True}}},
    }]
    rise = {"mlp_neurons": 1.0, "attention_heads": 3.0, "embedding": 0.0}
    steps = [{"rises": {(1, 0): rise, (1, 1): rise}} for _ in range(T)]
    stats = exploratory_stats(results, steps, {1: {"facet_similarities": {"0-1": -0.1}}})
    assert stats["least_related_windows_overlapping"] == 1
    assert stats["least_related_formation_order_vs_v1"] == {"same order as v1": 1}
    assert stats["unrelated_in_window_checkpoints_above_step_null_p95"] == "2 of 5"
    assert stats["least_related_checkpoints_above_step_null_p95"] == {1: 2}
    assert stats["heads_adjacent_at_formation_not_at_end_by_heads"] == ["1 0-1"]
    assert stats["pooled_source_share_by_step"]["attention_heads"][str(STEPS[0])] == 0.75
