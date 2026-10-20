import numpy as np
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from recgen.activations import MIN_CONTEXT
from recgen.gpt2_v19 import input_weights, normalized_residuals, rebuilt_activations, retokenize, window_maxima


def tiny_model():
    torch.manual_seed(0)
    config = GPT2Config(vocab_size=50, n_embd=16, n_layer=4, n_head=2, n_positions=64, bos_token_id=0, eos_token_id=0)
    model = GPT2LMHeadModel(config).eval()
    with torch.no_grad():  # initialized at gain 1 and bias 0, which would hide a gain or bias left out
        for block in model.transformer.h:
            for p in (block.ln_2.weight, block.ln_2.bias, block.mlp.c_fc.bias):
                p.copy_(torch.randn_like(p) * 0.5 + (1.0 if p is block.ln_2.weight else 0.0))
    return model


def full_pass(model, corpus, layer):
    """Hidden states, and the layer's ln_2 input, c_fc output and activation, from one uninterrupted pass."""
    block, seen = model.transformer.h[layer], {}
    handles = [block.ln_2.register_forward_pre_hook(lambda _m, inputs: seen.setdefault("ln_2_in", inputs[0])),
               block.mlp.c_fc.register_forward_hook(lambda _m, _i, out: seen.setdefault("pre", out)),
               block.mlp.act.register_forward_hook(lambda _m, _i, out: seen.setdefault("act", out))]
    with torch.no_grad():
        hidden = model(input_ids=torch.from_numpy(corpus).long(), output_hidden_states=True).hidden_states
    for h in handles:
        h.remove()
    return hidden, seen


def test_window_maxima_match_a_full_forward_pass():
    model = tiny_model()
    corpus = np.random.default_rng(0).integers(0, 50, size=(20, 48))  # more windows than one batch
    _, seen = full_pass(model, corpus, 2)
    values, positions = window_maxima(model, corpus, 2)
    acts = seen["act"][:, MIN_CONTEXT:]
    assert values.shape == (20, 64) and positions.min() >= MIN_CONTEXT
    np.testing.assert_allclose(values, acts.max(dim=1).values.numpy(), atol=1e-5)
    np.testing.assert_array_equal(positions, acts.argmax(dim=1).numpy() + MIN_CONTEXT)


def test_the_mlp_reads_the_residual_after_attention_and_the_weights_rebuild_it():
    model = tiny_model()
    corpus = np.random.default_rng(1).integers(0, 50, size=(40, 48))
    layer = 1
    rng = np.random.default_rng(2)
    keys = np.unique(np.stack([rng.choice(40, size=30), rng.integers(32, 48, size=30)], axis=1), axis=0)
    hidden, seen = full_pass(model, corpus, layer)

    residuals = normalized_residuals(model, corpus, keys, layer)
    h = seen["ln_2_in"][keys[:, 0], keys[:, 1]]
    expected = torch.nn.functional.layer_norm(h, (h.shape[-1],), eps=model.config.layer_norm_epsilon)
    np.testing.assert_allclose(residuals, expected.numpy(), atol=2e-3)
    block_input = hidden[layer][keys[:, 0], keys[:, 1]]  # sequential residual: not what the MLP reads
    assert not torch.allclose(h, block_input, atol=1e-3)

    neurons = np.array([3, 7, 60])
    weights, gain, constant = input_weights(model, layer, neurons)
    rows = np.tile(np.arange(len(keys)), (len(neurons), 1))
    rebuilt = rebuilt_activations(model, layer, residuals, rows, weights, gain, constant)
    actual = seen["act"][keys[:, 0], keys[:, 1]][:, neurons].T
    np.testing.assert_allclose(rebuilt, actual.numpy(), atol=5e-3)
    pre = (residuals.astype(np.float32) @ (weights * gain).T + constant).T
    np.testing.assert_allclose(pre, seen["pre"][keys[:, 0], keys[:, 1]][:, neurons].T.numpy(), atol=5e-3)


class CharTokenizer:
    """A stand-in tokenizer: one token per character, ids are code points."""

    def decode(self, ids, clean_up_tokenization_spaces=False):
        return "".join(chr(int(i)) for i in ids)

    def __call__(self, texts):
        return {"input_ids": [[ord(c) for c in t] for t in texts]}


def test_retokenize_takes_the_text_from_each_start_and_cuts_it_to_a_window(monkeypatch):
    from recgen import gpt2_v19

    monkeypatch.setattr(gpt2_v19, "CONTEXT_LEN", 4)
    monkeypatch.setattr(gpt2_v19, "STREAM_SPAN", 6)
    stream = np.array([ord(c) for c in "abcdefghijklmnop"])
    out = retokenize(stream, np.array([0, 8]), CharTokenizer(), CharTokenizer())
    assert ["".join(map(chr, row)) for row in out] == ["abcd", "ijkl"]
