"""v6 recording: at each depth of each Pythia size, the residual stream entering the layer,
LayerNorm-normalized, at every sampled neuron's top-50 example positions and at the random
positions of rubric_v6, plus the sampled neurons' input weights. One model at a time, deleted
after use.

The neurons and their examples come from v5's cached window maxima (data/v5) with v5's
eligibility rule and seeds, so v6 reads the same neurons and examples as v5. Float16, as in v5.
"""

import json
import shutil

import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import GPTNeoXForCausalLM

from recgen.activations import BATCH_SIZE, DEVICE
from recgen.analyze_v5 import rng_for
from recgen.config import DATA_DIR, RESULTS_DIR, WEIGHTS_DIR
from recgen.corpus import load_corpus
from recgen.facets import top_examples
from recgen.rubric_v5 import DEPTHS, MODELS, eligible, sample
from recgen.rubric_v6 import random_positions
from recgen.sweep import sweep_path

V6_DIR = DATA_DIR / "v6"


def v6_path(model: str, depth: str):
    return V6_DIR / f"{model}_{depth.replace(' ', '_')}.npz"


class _StopForward(Exception):
    pass


def sampled_top_examples(model_size: str, depth: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """v5's sampled neurons [S], their top examples as (window, position) [S, 50, 2], strongest first,
    and activations [S, 50]; the layer and the model's layer count."""
    data = np.load(sweep_path(model_size, depth))
    window_max, window_pos = data["max"].astype(np.float32), data["pos"]
    neurons = sample(eligible(window_max), rng_for(model_size, depth))
    examples = [top_examples(window_max, window_pos, int(n)) for n in neurons]
    keys = np.array([[(w, p) for w, p, _ in ex] for ex in examples], dtype=np.int32)
    acts = np.array([[a for _, _, a in ex] for ex in examples], dtype=np.float32)
    return neurons, keys, acts, int(data["layer"]), int(data["n_layers"])


def normalized_residuals(model, corpus: np.ndarray, keys_by_layer: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    """For each layer, the residual stream entering the block at each (window, position) key,
    LayerNorm-normalized without gain or bias: float16 [K, d_model]. Keys must be sorted by window.
    The forward pass stops at the deepest layer's MLP LayerNorm."""
    assert model.config.use_parallel_residual, "the MLP must read the residual stream entering the block"
    for keys in keys_by_layer.values():
        assert np.all(np.diff(keys[:, 0]) >= 0), "keys must be sorted by window"
    blocks = model.gpt_neox.layers
    device = next(model.parameters()).device
    deepest = max(keys_by_layer)
    windows = np.unique(np.concatenate([keys[:, 0] for keys in keys_by_layer.values()]))
    batch, out = {}, {layer: [] for layer in keys_by_layer}

    def hook_for(layer):
        eps = blocks[layer].post_attention_layernorm.eps

        def hook(_module, inputs):
            rows, pos = batch[layer]
            h = inputs[0][rows, pos].float()
            mean, var = h.mean(dim=-1, keepdim=True), h.var(dim=-1, unbiased=False, keepdim=True)
            out[layer].append(((h - mean) / torch.sqrt(var + eps)).half().cpu())
            if layer == deepest:
                raise _StopForward
        return hook

    handles = [blocks[layer].post_attention_layernorm.register_forward_pre_hook(hook_for(layer))
               for layer in keys_by_layer]
    try:
        with torch.no_grad():
            for start in range(0, len(windows), BATCH_SIZE):
                chunk = windows[start:start + BATCH_SIZE]
                for layer, keys in keys_by_layer.items():
                    lo = np.searchsorted(keys[:, 0], chunk[0], side="left")
                    hi = np.searchsorted(keys[:, 0], chunk[-1], side="right")
                    rows = np.searchsorted(chunk, keys[lo:hi, 0])
                    batch[layer] = (torch.from_numpy(rows).to(device), torch.from_numpy(keys[lo:hi, 1].astype(np.int64)).to(device))
                try:
                    model(input_ids=torch.from_numpy(corpus[chunk]).long().to(device))
                except _StopForward:
                    pass
    finally:
        for handle in handles:
            handle.remove()
    return {layer: torch.cat(parts).numpy() for layer, parts in out.items()}


def input_weights(model, layer: int, neurons: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The neurons' raw input weights [S, d], the MLP LayerNorm's gain [d], and each neuron's constant
    term [S], so that pre-activation = (weights * gain) . z + constant for a normalized residual z."""
    norm = model.gpt_neox.layers[layer].post_attention_layernorm
    fc = model.gpt_neox.layers[layer].mlp.dense_h_to_4h
    index = torch.from_numpy(np.asarray(neurons)).long().to(fc.weight.device)
    w = fc.weight.index_select(0, index).detach().float().cpu()
    constant = w @ norm.bias.detach().float().cpu() + fc.bias.index_select(0, index).detach().float().cpu()
    return w.numpy(), norm.weight.detach().float().cpu().numpy(), constant.numpy()


def reconstructed_activations(model, layer: int, z: np.ndarray, rows: np.ndarray, weights: np.ndarray,
                              gain: np.ndarray, constant: np.ndarray) -> np.ndarray:
    """Each neuron's activation [S, 50] at its examples (rows into z), rebuilt from the stored residuals."""
    folded = weights * gain
    pre = np.stack([z[r].astype(np.float32) @ folded[s] for s, r in enumerate(rows)]) + constant[:, None]
    with torch.no_grad():
        return model.gpt_neox.layers[layer].mlp.act(torch.from_numpy(pre)).numpy()


def record(model_size: str, corpus: np.ndarray) -> None:
    sampled = {depth: sampled_top_examples(model_size, depth) for depth in DEPTHS}
    v5_neurons = json.loads((RESULTS_DIR / "classification_v5.json").read_text())["neurons"]
    for depth, (neurons, *_) in sampled.items():
        assert neurons.tolist() == sorted(map(int, v5_neurons[f"{model_size} {depth}"])), "not v5's neurons"
    random_keys = random_positions(len(corpus))
    keys_by_layer = {}
    for neurons, keys, _, layer, _ in sampled.values():
        keys_by_layer[layer] = np.unique(np.concatenate([keys.reshape(-1, 2), random_keys]), axis=0)
    local_dir = WEIGHTS_DIR / f"pythia-{model_size}"
    snapshot_download(f"EleutherAI/pythia-{model_size}", local_dir=local_dir,
                      allow_patterns=["config.json", "model.safetensors"])
    try:
        model = GPTNeoXForCausalLM.from_pretrained(local_dir, torch_dtype=torch.float16).to(DEVICE).eval()
        residuals = normalized_residuals(model, corpus, keys_by_layer)
        for depth, (neurons, keys, acts, layer, n_layers) in sampled.items():
            unique = keys_by_layer[layer]
            index = {(int(w), int(p)): i for i, (w, p) in enumerate(unique)}
            rows = np.array([[index[int(w), int(p)] for w, p in ex] for ex in keys], dtype=np.int32)
            random_rows = np.array([index[int(w), int(p)] for w, p in random_keys], dtype=np.int32)
            weights, gain, constant = input_weights(model, layer, neurons)
            rebuilt = reconstructed_activations(model, layer, residuals[layer], rows, weights, gain, constant)
            error = np.abs(rebuilt - acts)
            np.savez(v6_path(model_size, depth), keys=unique, z=residuals[layer], neurons=neurons,
                     example_rows=rows, random_rows=random_rows, weights=weights, gain=gain, constant=constant,
                     layer=layer, n_layers=n_layers, activation_error=error.max())
            print(f"pythia-{model_size} {depth}: layer {layer}, {len(neurons)} neurons, {len(unique)} positions; "
                  f"activations rebuilt within {error.max():.3f} (median top activation {np.median(acts):.2f})",
                  flush=True)
    finally:
        shutil.rmtree(local_dir)


def main() -> None:
    corpus = load_corpus()
    V6_DIR.mkdir(exist_ok=True)
    for model_size in MODELS:
        if all(v6_path(model_size, depth).exists() for depth in DEPTHS):
            print(f"pythia-{model_size}: already recorded")
            continue
        record(model_size, corpus)


if __name__ == "__main__":
    main()
