import numpy as np
import pytest
import torch
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

from recgen.outputs import NeuronSetter, null_contexts, variant_logprobs

LAYER = 1


@pytest.fixture(scope="module")
def tiny_model():
    torch.manual_seed(0)
    config = GPTNeoXConfig(vocab_size=50, hidden_size=16, num_hidden_layers=3, num_attention_heads=2,
                           intermediate_size=32, max_position_embeddings=32, use_parallel_residual=True)
    model = GPTNeoXForCausalLM(config).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0, 0.5)
    return model


def full_forward(model, window, pos, chunk, change):
    """Log-probs at pos to pos + chunk - 1 from running the window with one neuron set at pos."""
    def hook(_module, _inputs, output):
        if change is not None:
            output[0, pos, change[0]] = change[1]
        return output

    handle = model.gpt_neox.layers[LAYER].mlp.act.register_forward_hook(hook)
    try:
        with torch.no_grad():
            logits = model(input_ids=torch.tensor(window[:pos + chunk])[None]).logits[0, pos:pos + chunk]
    finally:
        handle.remove()
    return torch.log_softmax(logits, dim=-1)


def test_cached_variants_equal_full_forward_passes(tiny_model):
    window = np.random.default_rng(0).integers(0, 50, size=24)
    variants = [None, (3, 0.0), (3, 2.5), (17, -1.0), None]  # more variants than one group
    setter = NeuronSetter(tiny_model, layer=LAYER)
    try:
        lp = variant_logprobs(tiny_model, setter, window, pos=9, variants=variants, chunk=4, group=2)
    finally:
        setter.handle.remove()
    assert lp.shape == (5, 4, 50)
    for got, change in zip(lp, variants):
        torch.testing.assert_close(got, full_forward(tiny_model, window, 9, 4, change), atol=1e-4, rtol=1e-4)
    assert not torch.allclose(lp[1], lp[2])  # the change reaches the output


def test_null_contexts_avoid_facet_windows_and_leave_room_for_the_next_tokens():
    candidates = [{"facets": [[[0, 40, 2.0], [3, 100, 1.9]]]}, {"facets": [[[5, 200, 1.0]]]}]
    contexts = null_contexts(candidates, n_windows=10, n=7, rng=np.random.default_rng(0))
    assert sorted(w for w, _ in contexts) == [1, 2, 4, 6, 7, 8, 9]
    assert all(32 <= p and p + 5 < 512 for _, p in contexts)


def test_prediction_change_measures_moved_probability_and_sharpening():
    from recgen.outputs import prediction_change
    as_is = torch.log_softmax(torch.tensor([2.0, 1.0, 0.0, -1.0]), dim=0)
    flatter = torch.log_softmax(torch.tensor([1.0, 0.5, 0.0, -0.5]), dim=0)  # the same logits, halved
    change = prediction_change(torch.stack([as_is, flatter]), vocab=4)
    p, q = as_is.exp(), flatter.exp()
    torch.testing.assert_close(change["tv"], 0.5 * (p - q).abs().sum())
    torch.testing.assert_close(change["kl"], (p * (as_is - flatter)).sum())
    torch.testing.assert_close(change["sharpen"], torch.tensor(1.0))  # the neuron only sharpens
