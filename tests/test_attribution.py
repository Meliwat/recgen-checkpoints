import numpy as np
import pytest
import torch
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

from recgen.attribution import background_examples, capture, components, mlp_coefficients

LAYER = 2
NEURONS = [3, 0, 17]


@pytest.fixture(scope="module")
def tiny_model():
    torch.manual_seed(0)
    config = GPTNeoXConfig(vocab_size=50, hidden_size=16, num_hidden_layers=3, num_attention_heads=2,
                           intermediate_size=32, max_position_embeddings=32, use_parallel_residual=True)
    model = GPTNeoXForCausalLM(config).eval()
    with torch.no_grad():
        for p in model.parameters():  # random LayerNorm gains and biases too, so none of them can hide an error
            p.normal_(0, 0.5)
    return model


@pytest.fixture(scope="module")
def corpus():
    return np.random.default_rng(0).integers(0, 50, size=(20, 24))


def test_components_sum_to_the_actual_pre_activation(tiny_model, corpus):
    examples = [(3, 5), (0, 23), (3, 9), (19, 0), (7, 12)]  # two positions share a window
    cap = capture(tiny_model, corpus, examples, NEURONS, layer=LAYER)
    parts = components(tiny_model, cap, NEURONS, layer=LAYER)
    assert cap.pre.shape == (5, 3)
    np.testing.assert_allclose(sum(parts.values()), cap.pre, atol=1e-4)


def test_capture_keeps_example_order_across_batches(tiny_model, corpus):
    examples = [(w, 4) for w in (18, 2, 11, 2, 0)] + [(w, 9) for w in range(17)]  # more windows than one batch
    cap = capture(tiny_model, corpus, examples, NEURONS, layer=LAYER)
    one_by_one = np.concatenate([capture(tiny_model, corpus, [e], NEURONS, layer=LAYER).pre for e in examples])
    np.testing.assert_allclose(cap.pre, one_by_one, atol=1e-5)


def test_upstream_neuron_term_is_its_activation_times_its_coefficient(tiny_model, corpus):
    cap = capture(tiny_model, corpus, [(4, 10)], NEURONS, layer=LAYER)
    coefficients = mlp_coefficients(tiny_model, NEURONS, layer=LAYER)
    assert coefficients.shape == (LAYER, 32, 3)
    total = (cap.acts[0, :, :, None] * coefficients).sum(axis=(0, 1)) / cap.sigma[0]
    np.testing.assert_allclose(components(tiny_model, cap, NEURONS, layer=LAYER)["mlp neurons"][0], total, rtol=1e-5)


def test_background_avoids_facet_windows_and_early_positions():
    candidates = [{"facets": [[[1, 40, 2.0], [3, 50, 2.0]]]}, {"facets": [[[5, 60, 1.0]]]}]
    background = background_examples(candidates, n_corpus_windows=8, n=5, rng=np.random.default_rng(0))
    assert sorted(w for w, _ in background) == [0, 2, 4, 6, 7]
    assert all(32 <= p < 512 for _, p in background)
