import numpy as np
import torch
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

from recgen.activations import MIN_CONTEXT
from recgen.attribution import capture
from recgen.bridges import (
    bridge_set, later_facet_pairs, matched_control, pool_positions, pre_activations, window_steps,
)
from recgen.config import CONTEXT_LEN


def test_pool_positions_follow_the_flattened_order_of_recorded_windows():
    positions = pool_positions([7, 3])
    per_window = CONTEXT_LEN - MIN_CONTEXT
    assert len(positions) == 2 * per_window
    assert positions[0].tolist() == [7, MIN_CONTEXT]
    assert positions[per_window].tolist() == [3, MIN_CONTEXT]
    assert positions[-1].tolist() == [3, CONTEXT_LEN - 1]


def test_pairs_join_each_later_facet_to_every_facet_that_emerged_before_it():
    results = [{"neuron": 9, "emergence_steps": [5000, 3000, 5000, 8000]},
               {"neuron": 4, "emergence_steps": [3000, None]}]
    assert [(p["a"], p["b"]) for p in later_facet_pairs(results)] == [(1, 0), (1, 2), (0, 3), (1, 3), (2, 3)]


def test_window_runs_from_earlier_to_later_emergence_on_the_dense_grid():
    assert window_steps(3000, 8000) == [3000, 4000, 5000, 6000, 7000, 8000]


def test_bridge_is_most_similar_to_both_facets_with_one_text_per_window():
    cos_a = np.array([0.9, 0.5, 0.6, 0.4, 0.7])
    cos_b = np.array([0.1, 0.5, 0.6, 0.4, 0.5])
    windows = np.array([0, 1, 1, 2, 3])
    assert bridge_set(cos_a, cos_b, windows, size=3) == [2, 4, 3]  # 1 shares a window with 2


def test_control_matches_targets_among_texts_unlike_the_other_facet():
    match_on = np.array([0.50, 0.31, 0.29, 0.10, 0.30, 0.52])
    other = np.array([0.0, 0.0, 0.0, 0.0, 0.9, 0.0])  # index 4 is like the other facet
    windows = np.array([0, 1, 1, 2, 3, 4])
    picked = matched_control(np.array([0.3, 0.3, 0.5]), match_on, other, windows, excluded={0})
    assert picked == [1, 3, 5]  # 2 shares 1's window; 0 is excluded; 4 is too like the other facet


def test_recorded_pre_activations_match_the_attribution_capture():
    torch.manual_seed(0)
    config = GPTNeoXConfig(vocab_size=50, hidden_size=16, num_hidden_layers=3, num_attention_heads=2,
                           intermediate_size=32, max_position_embeddings=32, use_parallel_residual=True)
    model = GPTNeoXForCausalLM(config).eval()
    corpus = np.random.default_rng(0).integers(0, 50, size=(20, 24))
    pre = pre_activations(model, corpus, [5, 2], [3, 0], layer=2)
    captured = capture(model, corpus, [(5, 7), (2, 23)], [3, 0], layer=2).pre
    np.testing.assert_allclose(pre[[0, 1], [7, 23]], captured, atol=1e-2)  # float16 storage
