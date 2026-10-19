import numpy as np
import torch
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

from recgen.activations import MIN_CONTEXT
from recgen.sweep import window_maxima


def test_window_maxima_match_a_full_forward_pass_at_every_requested_layer():
    torch.manual_seed(0)
    config = GPTNeoXConfig(vocab_size=50, hidden_size=16, num_hidden_layers=4, num_attention_heads=2,
                           intermediate_size=32, max_position_embeddings=64, use_parallel_residual=True)
    model = GPTNeoXForCausalLM(config).eval()
    corpus = np.random.default_rng(0).integers(0, 50, size=(20, 48))  # more windows than one batch
    layers = [1, 2]

    full = {}
    handles = [model.gpt_neox.layers[l].mlp.act.register_forward_hook(
        lambda _m, _i, out, l=l: full.setdefault(l, []).append(out)) for l in layers]
    with torch.no_grad():
        model(input_ids=torch.from_numpy(corpus).long())
    for h in handles:
        h.remove()

    maxima = window_maxima(model, corpus, layers)
    for l in layers:
        acts = full[l][0][:, MIN_CONTEXT:]
        values, positions = maxima[l]
        assert values.shape == (20, 32) and positions.min() >= MIN_CONTEXT
        np.testing.assert_allclose(values, acts.max(dim=1).values.numpy(), atol=1e-5)
        np.testing.assert_array_equal(positions, acts.argmax(dim=1).numpy() + MIN_CONTEXT)
