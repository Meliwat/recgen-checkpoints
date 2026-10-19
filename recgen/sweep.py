"""v5 recording: at three depths of each Pythia size, every neuron's strongest activation in
each corpus window and where it happened. One model at a time, deleted after use.

All sizes run in float16, the precision Pythia was trained and released in, so they're
comparable with each other (v1 to v4 ran 410m in float32).
"""

import shutil

import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import GPTNeoXForCausalLM

from recgen.activations import BATCH_SIZE, DEVICE, MIN_CONTEXT
from recgen.config import DATA_DIR, WEIGHTS_DIR
from recgen.corpus import load_corpus
from recgen.rubric_v5 import DEPTHS, MODELS, layer_index

SWEEP_DIR = DATA_DIR / "v5"


def sweep_path(model: str, depth: str):
    return SWEEP_DIR / f"{model}_{depth.replace(' ', '_')}.npz"


class _StopForward(Exception):
    pass


def window_maxima(model, corpus: np.ndarray, layers: list[int]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """For each layer: each neuron's strongest activation in each window [W, N] and its position
    [W, N], ignoring the first MIN_CONTEXT positions. The forward pass stops after the deepest layer."""
    blocks = model.gpt_neox.layers
    device = next(model.parameters()).device
    deepest = max(layers)
    captured = {}

    def hook_for(layer):
        def hook(_module, _inputs, output):
            captured[layer] = output
            if layer == deepest:
                raise _StopForward
        return hook

    handles = [blocks[layer].mlp.act.register_forward_hook(hook_for(layer)) for layer in layers]
    out = {layer: ([], []) for layer in layers}
    try:
        with torch.no_grad():
            for start in range(0, len(corpus), BATCH_SIZE):
                try:
                    model(input_ids=torch.from_numpy(corpus[start:start + BATCH_SIZE]).long().to(device))
                except _StopForward:
                    pass
                for layer in layers:
                    values, positions = captured.pop(layer)[:, MIN_CONTEXT:].max(dim=1)
                    out[layer][0].append(values.float().cpu().numpy())
                    out[layer][1].append((positions + MIN_CONTEXT).to(torch.int16).cpu().numpy())
    finally:
        for handle in handles:
            handle.remove()
    return {layer: (np.concatenate(v), np.concatenate(p)) for layer, (v, p) in out.items()}


def record(model_size: str, corpus: np.ndarray) -> None:
    local_dir = WEIGHTS_DIR / f"pythia-{model_size}"
    snapshot_download(f"EleutherAI/pythia-{model_size}", local_dir=local_dir,
                      allow_patterns=["config.json", "model.safetensors"])
    try:
        model = GPTNeoXForCausalLM.from_pretrained(local_dir, torch_dtype=torch.float16).to(DEVICE).eval()
        n_layers = model.config.num_hidden_layers
        layers = {depth: layer_index(n_layers, fraction) for depth, fraction in DEPTHS.items()}
        maxima = window_maxima(model, corpus, sorted(set(layers.values())))
        for depth, layer in layers.items():
            values, positions = maxima[layer]
            np.savez(sweep_path(model_size, depth), max=values.astype(np.float16), pos=positions,
                     layer=layer, n_layers=n_layers)
        print(f"pythia-{model_size}: recorded layers {layers}", flush=True)
    finally:
        shutil.rmtree(local_dir)


def main() -> None:
    corpus = load_corpus()
    SWEEP_DIR.mkdir(exist_ok=True)
    for model_size in MODELS:
        if all(sweep_path(model_size, depth).exists() for depth in DEPTHS):
            print(f"pythia-{model_size}: already recorded")
            continue
        record(model_size, corpus)


if __name__ == "__main__":
    main()
