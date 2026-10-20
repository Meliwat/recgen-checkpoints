"""v19's GPT-2 adapters: the corpus re-tokenized for GPT-2, and the recording at the middle layer of
each size. One model at a time, deleted after use. Float16, with a float32 check on the CPU.

For each model, data/v19/ gets:
- {model}_maxima.npz: every neuron's strongest activation in each window and where (as recgen/sweep.py);
- {model}_network.npz: in sweep_v6's format, so analyze_v6.Layer reads it unchanged, the residual
  stream entering ln_2 (LayerNorm-normalized, no gain or bias) at the sampled neurons' top-50 examples
  and at rubric_v6's random positions, and the sampled neurons' input weights.
"""

import shutil
from functools import cache

import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, GPT2LMHeadModel

from recgen.activations import BATCH_SIZE, DEVICE, MIN_CONTEXT
from recgen.config import CONTEXT_LEN, DATA_DIR, MODEL, WEIGHTS_DIR
from recgen.corpus import load_corpus, packed_stream, window_starts
from recgen.facets import top_examples
from recgen.rubric_v6 import random_positions
from recgen.rubric_v19 import FP32_CHECK_WINDOWS, HUB, MODELS, STREAM_SPAN, middle_layer, sampled_neurons

V19_DIR = DATA_DIR / "v19"
CORPUS_PATH = V19_DIR / "corpus_gpt2.npy"
CONTEXT_BEFORE = MIN_CONTEXT  # tokens embedded by MiniLM before the firing token, as in recgen.facets


@cache
def tokenizer():
    return AutoTokenizer.from_pretrained(f"{HUB}/gpt2")


def retokenize(stream: np.ndarray, starts: np.ndarray, source, target) -> np.ndarray:
    """For each start, the first CONTEXT_LEN target tokens of the text of stream[start:start + STREAM_SPAN]."""
    texts = [source.decode(stream[s:s + STREAM_SPAN], clean_up_tokenization_spaces=False) for s in starts]
    ids = target(texts)["input_ids"]
    assert min(len(x) for x in ids) >= CONTEXT_LEN, "a window's text is too short"
    return np.array([x[:CONTEXT_LEN] for x in ids], dtype=np.int32)


def load_gpt2_corpus() -> np.ndarray:
    """The repo's 4,096 windows as GPT-2 tokens, built once from v1's packed Pythia stream."""
    if not CORPUS_PATH.exists():
        pythia = AutoTokenizer.from_pretrained(MODEL)
        stream = packed_stream(pythia)
        starts = window_starts(len(stream))
        assert np.array_equal(np.stack([stream[s:s + CONTEXT_LEN] for s in starts]), load_corpus()), "not v1's windows"
        V19_DIR.mkdir(exist_ok=True)
        np.save(CORPUS_PATH, retokenize(stream, starts, pythia, tokenizer()))
    return np.load(CORPUS_PATH)


def context_text(corpus: np.ndarray, window: int, pos: int) -> str:
    """What MiniLM embeds: CONTEXT_BEFORE tokens up to and including the firing token."""
    return tokenizer().decode(corpus[window, pos - CONTEXT_BEFORE:pos + 1])


def display_text(corpus: np.ndarray, window: int, pos: int, before: int = 24, after: int = 4) -> str:
    """Context with the firing token marked as [[token]], as recgen.facets.display_text."""
    tok = tokenizer()
    left = tok.decode(corpus[window, max(0, pos - before):pos])
    token = tok.decode(corpus[window, pos:pos + 1])
    right = tok.decode(corpus[window, pos + 1:pos + 1 + after])
    return f"{left}[[{token}]]{right}".replace("\n", "⏎")


class _StopForward(Exception):
    pass


def window_maxima(model, corpus: np.ndarray, layer: int) -> tuple[np.ndarray, np.ndarray]:
    """Each neuron's strongest activation in each window [W, N] and its position [W, N], ignoring the
    first MIN_CONTEXT positions. The forward pass stops at the layer's MLP activation."""
    device = next(model.parameters()).device
    captured = {}

    def hook(_module, _inputs, output):
        captured["acts"] = output
        raise _StopForward

    handle = model.transformer.h[layer].mlp.act.register_forward_hook(hook)
    values, positions = [], []
    try:
        with torch.no_grad():
            for start in range(0, len(corpus), BATCH_SIZE):
                try:
                    model(input_ids=torch.from_numpy(corpus[start:start + BATCH_SIZE]).long().to(device))
                except _StopForward:
                    pass
                v, p = captured.pop("acts")[:, MIN_CONTEXT:].max(dim=1)
                values.append(v.float().cpu().numpy())
                positions.append((p + MIN_CONTEXT).to(torch.int16).cpu().numpy())
    finally:
        handle.remove()
    return np.concatenate(values), np.concatenate(positions)


def normalized_residuals(model, corpus: np.ndarray, keys: np.ndarray, layer: int) -> np.ndarray:
    """The residual stream entering the layer's ln_2 (after the block's attention) at each (window,
    position) key, LayerNorm-normalized without gain or bias: float16 [K, d_model]. Keys must be sorted
    by window. The forward pass stops there."""
    assert np.all(np.diff(keys[:, 0]) >= 0), "keys must be sorted by window"
    norm = model.transformer.h[layer].ln_2
    device = next(model.parameters()).device
    windows = np.unique(keys[:, 0])
    batch, out = {}, []

    def hook(_module, inputs):
        h = inputs[0][batch["rows"], batch["pos"]].float()
        mean, var = h.mean(dim=-1, keepdim=True), h.var(dim=-1, unbiased=False, keepdim=True)
        out.append(((h - mean) / torch.sqrt(var + norm.eps)).half().cpu())
        raise _StopForward

    handle = norm.register_forward_pre_hook(hook)
    try:
        with torch.no_grad():
            for start in range(0, len(windows), BATCH_SIZE):
                chunk = windows[start:start + BATCH_SIZE]
                lo = np.searchsorted(keys[:, 0], chunk[0], side="left")
                hi = np.searchsorted(keys[:, 0], chunk[-1], side="right")
                batch["rows"] = torch.from_numpy(np.searchsorted(chunk, keys[lo:hi, 0])).to(device)
                batch["pos"] = torch.from_numpy(keys[lo:hi, 1].astype(np.int64)).to(device)
                try:
                    model(input_ids=torch.from_numpy(corpus[chunk]).long().to(device))
                except _StopForward:
                    pass
    finally:
        handle.remove()
    return torch.cat(out).numpy()


def input_weights(model, layer: int, neurons: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The neurons' raw input weights [S, d], ln_2's gain [d], and each neuron's constant term [S], so that
    pre-activation = (weights * gain) . z + constant for a normalized residual z. c_fc is a Conv1D, whose
    weight is [d_model, d_mlp], so a neuron's weights are a column."""
    norm, fc = model.transformer.h[layer].ln_2, model.transformer.h[layer].mlp.c_fc
    index = torch.from_numpy(np.asarray(neurons)).long().to(fc.weight.device)
    w = fc.weight.index_select(1, index).detach().float().cpu().T.contiguous()
    constant = w @ norm.bias.detach().float().cpu() + fc.bias.index_select(0, index).detach().float().cpu()
    return w.numpy(), norm.weight.detach().float().cpu().numpy(), constant.numpy()


def rebuilt_activations(model, layer: int, z: np.ndarray, rows: np.ndarray, weights: np.ndarray,
                        gain: np.ndarray, constant: np.ndarray) -> np.ndarray:
    """Each neuron's activation [S, 50] at its examples (rows into z), rebuilt from the stored residuals."""
    folded = weights * gain
    pre = np.stack([z[r].astype(np.float32) @ folded[s] for s, r in enumerate(rows)]) + constant[:, None]
    with torch.no_grad():
        return model.transformer.h[layer].mlp.act(torch.from_numpy(pre)).numpy()


def maxima_path(model: str):
    return V19_DIR / f"{model}_maxima.npz"


def network_path(model: str):
    return V19_DIR / f"{model}_network.npz"


def sampled_top_examples(model: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The sampled neurons [S], their top examples as (window, position) [S, 50, 2], strongest first, and
    activations [S, 50], from the stored window maxima."""
    data = np.load(maxima_path(model))
    window_max, window_pos = data["max"].astype(np.float32), data["pos"]
    neurons = sampled_neurons(window_max, model)
    examples = [top_examples(window_max, window_pos, int(n)) for n in neurons]
    keys = np.array([[(w, p) for w, p, _ in ex] for ex in examples], dtype=np.int32)
    acts = np.array([[a for _, _, a in ex] for ex in examples], dtype=np.float32)
    return neurons, keys, acts


def float32_error(local_dir, corpus: np.ndarray, layer: int, neurons: np.ndarray, stored: np.ndarray,
                  device: str = "cpu") -> float:
    """The largest difference between float32 window maxima (on the CPU unless device says otherwise) and the stored
    float16 ones [W, N], over the sampled neurons in the first FP32_CHECK_WINDOWS windows."""
    model = GPT2LMHeadModel.from_pretrained(local_dir, dtype=torch.float32).to(device).eval()
    values, _ = window_maxima(model, corpus[:FP32_CHECK_WINDOWS], layer)
    return float(np.abs(values[:, neurons] - stored[:FP32_CHECK_WINDOWS, neurons].astype(np.float32)).max())


def record(name: str, corpus: np.ndarray) -> None:
    local_dir = WEIGHTS_DIR / name
    snapshot_download(f"{HUB}/{name}", local_dir=local_dir, allow_patterns=["config.json", "model.safetensors"])
    try:
        model = GPT2LMHeadModel.from_pretrained(local_dir, dtype=torch.float16).to(DEVICE).eval()
        n_layers = model.config.n_layer
        layer = middle_layer(n_layers)
        values, positions = window_maxima(model, corpus, layer)
        assert np.isfinite(values).all(), "float16 overflow"
        V19_DIR.mkdir(exist_ok=True)
        np.savez(maxima_path(name), max=values.astype(np.float16), pos=positions, layer=layer, n_layers=n_layers)

        neurons, keys, acts = sampled_top_examples(name)
        random_keys = random_positions(len(corpus))
        unique = np.unique(np.concatenate([keys.reshape(-1, 2), random_keys]), axis=0)
        index = {(int(w), int(p)): i for i, (w, p) in enumerate(unique)}
        rows = np.array([[index[int(w), int(p)] for w, p in ex] for ex in keys], dtype=np.int32)
        random_rows = np.array([index[int(w), int(p)] for w, p in random_keys], dtype=np.int32)
        z = normalized_residuals(model, corpus, unique, layer)
        weights, gain, constant = input_weights(model, layer, neurons)
        error = float(np.abs(rebuilt_activations(model, layer, z, rows, weights, gain, constant) - acts).max())
        del model
        fp32 = float32_error(local_dir, corpus, layer, neurons, values.astype(np.float16))
        np.savez(network_path(name), keys=unique, z=z, neurons=neurons, example_rows=rows, random_rows=random_rows,
                 weights=weights, gain=gain, constant=constant, layer=layer, n_layers=n_layers,
                 activation_error=error, float32_error=fp32)
        print(f"{name}: layer {layer} of {n_layers}, {len(neurons)} sampled neurons, {len(unique)} positions; "
              f"activations rebuilt within {error:.4f}, float32 within {fp32:.4f} "
              f"(median top activation {np.median(acts):.2f})", flush=True)
    finally:
        shutil.rmtree(local_dir)


def main() -> None:
    corpus = load_gpt2_corpus()
    for name in MODELS:
        if network_path(name).exists():
            print(f"{name}: already recorded")
            continue
        record(name, corpus)


if __name__ == "__main__":
    main()
