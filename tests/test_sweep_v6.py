import numpy as np
import torch
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

from recgen.sweep_v6 import input_weights, normalized_residuals, reconstructed_activations


def tiny_model():
    torch.manual_seed(0)
    config = GPTNeoXConfig(vocab_size=50, hidden_size=16, num_hidden_layers=4, num_attention_heads=2,
                           intermediate_size=32, max_position_embeddings=64, use_parallel_residual=True)
    return GPTNeoXForCausalLM(config).eval()


def test_residuals_and_weights_rebuild_a_full_forward_pass():
    model = tiny_model()
    corpus = np.random.default_rng(0).integers(0, 50, size=(40, 48))  # more windows than one batch
    rng = np.random.default_rng(1)
    keys_by_layer = {}
    for layer in (1, 2):
        keys = np.stack([rng.choice(40, size=30), rng.integers(32, 48, size=30)], axis=1)
        keys_by_layer[layer] = np.unique(keys, axis=0)  # sorted by window, as the recorder requires

    pre = {}
    handles = [model.gpt_neox.layers[l].mlp.dense_h_to_4h.register_forward_hook(
        lambda _m, _i, out, l=l: pre.setdefault(l, out)) for l in keys_by_layer]
    with torch.no_grad():
        hidden = model(input_ids=torch.from_numpy(corpus).long(), output_hidden_states=True).hidden_states
    for h in handles:
        h.remove()

    residuals = normalized_residuals(model, corpus, keys_by_layer)
    neurons = np.array([3, 7, 30])
    for layer, keys in keys_by_layer.items():
        h = hidden[layer][keys[:, 0], keys[:, 1]]  # the residual stream entering the block
        expected = torch.nn.functional.layer_norm(h, (h.shape[-1],), eps=model.config.layer_norm_eps)
        np.testing.assert_allclose(residuals[layer], expected.numpy(), atol=2e-3)

        weights, gain, constant = input_weights(model, layer, neurons)
        rows = np.tile(np.arange(len(keys)), (len(neurons), 1))
        rebuilt = reconstructed_activations(model, layer, residuals[layer], rows, weights, gain, constant)
        actual = model.gpt_neox.layers[layer].mlp.act(pre[layer][keys[:, 0], keys[:, 1]][:, neurons]).T
        np.testing.assert_allclose(rebuilt, actual.detach().numpy(), atol=5e-3)
