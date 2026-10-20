import numpy as np

from recgen.rubric_v7 import MIXES, N_POOLED, N_REAL
from recgen.rubric_v12 import quartiles
from recgen.rubric_v19 import (
    DOES_NOT_REPLICATE, MODELS, REPLICATES, RETEST, decide_a, difference, excess_over_models, h1, middle_layer,
    random_pairs, real_units, retest_indices, sampled_neurons,
)


def test_middle_layers_of_the_three_sizes():
    assert [middle_layer(n) for n in (12, 24, 36)] == [6, 12, 18]


def test_draws_are_seeded_per_model_and_purpose():
    a, b = real_units(2048, "gpt2"), real_units(2048, "gpt2")
    assert len(a) == N_REAL and np.array_equal(a, b) and not np.array_equal(a, real_units(2048, "gpt2-large"))
    pairs = {mix: random_pairs(2048, "gpt2-medium", mix) for mix in MIXES}
    assert all(len(p) == N_POOLED and all(x != y for x, y in p) for p in pairs.values())
    assert pairs["50/50"] != pairs["70/30"]
    assert sum(len(retest_indices(300, m)) for m in MODELS) == RETEST


def test_sampled_neurons_drop_dead_and_dense_neurons():
    rng = np.random.default_rng(0)
    window_max = rng.random((200, 6)) * 0.1
    window_max[0] = 5.0  # every neuron peaks at 5 in one window
    window_max[:, 1] = 0.1  # near-dead: peak far below the median peak
    window_max[:150, 2] = 4.0  # dense: over half the windows at half its peak
    assert list(sampled_neurons(window_max, "gpt2")) == [0, 3, 4, 5]


def test_excess_over_models_is_the_mean_difference_with_an_interval():
    rng = np.random.default_rng(0)
    per_model = [(np.r_[np.ones(30), np.zeros(20)], np.r_[np.ones(40), np.zeros(60)])] * 3  # 0.6 vs 0.4
    out = excess_over_models(per_model, rng)
    assert abs(out["excess"] - 0.2) < 1e-12 and out["interval"][0] < 0.2 < out["interval"][1]
    assert len(out["draws"]) == 2000
    same = difference(out, excess_over_models(per_model, rng))
    assert same["difference"] == 0 and same["interval"][0] < 0 < same["interval"][1]


def test_a_replicates_only_if_minilm_is_near_blind_and_both_other_readers_see_pools():
    assert decide_a({"minilm": [-0.01, 0.08], "network": [0.1, 0.3], "llm": [0.05, 0.2]})["verdict"] == REPLICATES
    assert decide_a({"minilm": [0.02, 0.09], "network": [0.1, 0.3], "llm": [0.05, 0.2]})["verdict"] == REPLICATES
    assert decide_a({"minilm": [-0.05, 0.15], "network": [0.1, 0.3], "llm": [0.05, 0.2]})["verdict"] == REPLICATES
    out = decide_a({"minilm": [0.05, 0.15], "network": [0.1, 0.3], "llm": [0.05, 0.2]})
    assert out["verdict"] == DOES_NOT_REPLICATE and not out["minilm_near_blind"]
    out = decide_a({"minilm": [-0.01, 0.08], "network": [0.1, 0.3], "llm": [-0.01, 0.2]})
    assert out["verdict"] == DOES_NOT_REPLICATE and not out["llm_above_zero"]


def test_h1_needs_a_negative_correlation_in_all_three_models():
    rng = np.random.default_rng(1)
    groups = []
    for _ in MODELS:
        peak = rng.normal(size=600)
        flag = (rng.random(600) < np.where(quartiles(peak) == 3, 0.2, 0.5)).astype(float)
        groups.append((peak, quartiles(peak), flag))
    assert h1(groups, rng)["holds"]
    flat = rng.normal(size=600)
    groups[2] = (flat, quartiles(flat), (flat > 0.5).astype(float))  # third model: positive correlation
    out = h1(groups, rng)
    assert out["models_negative"] == 2 and not out["holds"]
