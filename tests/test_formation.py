import numpy as np
import pytest
import torch
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

from recgen import attribution
from recgen.formation import capture

LAYER = 2
NEURONS = [3, 0, 17]


@pytest.fixture(scope="module")
def tiny_model():
    torch.manual_seed(0)
    config = GPTNeoXConfig(vocab_size=50, hidden_size=16, num_hidden_layers=3, num_attention_heads=4,
                           intermediate_size=32, max_position_embeddings=32, use_parallel_residual=True)
    model = GPTNeoXForCausalLM(config).eval()
    with torch.no_grad():
        for p in model.parameters():  # random LayerNorm gains and biases too, so none of them can hide an error
            p.normal_(0, 0.5)
    return model


@pytest.fixture(scope="module")
def corpus():
    return np.random.default_rng(0).integers(0, 50, size=(20, 24))


EXAMPLES = [(3, 5), (0, 23), (3, 9), (19, 0), (7, 12)] + [(w, 9) for w in range(8, 19)]  # shared window, >1 batch


def test_parts_sum_to_the_actual_pre_activation(tiny_model, corpus):
    cap = capture(tiny_model, corpus, EXAMPLES, NEURONS, layer=LAYER)
    parts = sum(v for k, v in cap.items() if k.startswith("part_"))
    np.testing.assert_allclose(parts, cap["pre"], atol=1e-4)


def test_heads_add_up_to_v2s_attention_term_minus_its_bias(tiny_model, corpus):
    cap = capture(tiny_model, corpus, EXAMPLES, NEURONS, layer=LAYER)
    v2 = attribution.components(tiny_model, attribution.capture(tiny_model, corpus, EXAMPLES, NEURONS, layer=LAYER),
                                NEURONS, layer=LAYER)
    assert cap["heads"].shape == (len(EXAMPLES), LAYER, 4, len(NEURONS))
    np.testing.assert_allclose(cap["part_attention_heads"] + cap["part_attention_biases"], v2["attention"], atol=1e-4)
    np.testing.assert_allclose(cap["part_mlp_neurons"], v2["mlp neurons"], atol=1e-4)


def test_a_single_head_term_matches_ablating_that_head_directly(tiny_model, corpus):
    """Zeroing head 2 of layer 1 before the output projection removes exactly its direct term
    from the next layer's input, so the term should equal the change in the linearized readout."""
    example = [(4, 10)]
    cap = capture(tiny_model, corpus, example, NEURONS, layer=LAYER)
    dense = tiny_model.gpt_neox.layers[1].attention.dense
    width = dense.weight.shape[1] // 4

    def zero_head(_module, inputs):
        z = inputs[0].clone()
        z[..., 2 * width:3 * width] = 0
        return (z,)

    handle = dense.register_forward_pre_hook(zero_head)
    try:
        ablated = attribution.capture(tiny_model, corpus, example, NEURONS, layer=LAYER)
    finally:
        handle.remove()
    # Hold sigma at the unablated value so the comparison is the linearized term itself.
    readout, _ = attribution._readout(tiny_model, NEURONS, LAYER)
    change = (attribution._center(torch.from_numpy(ablated.attn[:, 1] - attribution.capture(
        tiny_model, corpus, example, NEURONS, layer=LAYER).attn[:, 1])) @ readout).numpy() / cap["sigma"][:, None]
    np.testing.assert_allclose(-change[0], cap["heads"][0, 1, 2], atol=1e-4)
