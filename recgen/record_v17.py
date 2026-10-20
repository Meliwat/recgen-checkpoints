"""v17 recording: at the middle layer of each Pythia size, the residual stream entering the layer,
LayerNorm-normalized, at every sampled neuron's band example positions (recgen/band_v17.py). One
model at a time, float16 as in v5 and v6, deleted after use; models already in data/v17/ are skipped.

As a check, each neuron's activation at its band examples is rebuilt from the stored residuals and its
input weights (as sweep_v6 does) and compared with v5's window maxima there.
"""

import shutil
import time

import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import GPTNeoXForCausalLM

from recgen.activations import DEVICE
from recgen.band_v17 import band_examples
from recgen.config import CONTEXT_LEN, DATA_DIR, WEIGHTS_DIR
from recgen.corpus import load_corpus
from recgen.rubric_v5 import MODELS
from recgen.rubric_v17 import DEPTH
from recgen.sweep import sweep_path
from recgen.sweep_v6 import input_weights, normalized_residuals, reconstructed_activations

V17_DIR = DATA_DIR / "v17"


def v17_path(model: str):
    return V17_DIR / f"{model}_{DEPTH}.npz"


def band_rows(keys: np.ndarray, unique: np.ndarray) -> np.ndarray:
    """Each band example's row in unique (sorted (window, position) pairs), -1 where keys are -1."""
    codes = unique[:, 0].astype(np.int64) * CONTEXT_LEN + unique[:, 1]
    wanted = keys[..., 0].astype(np.int64) * CONTEXT_LEN + keys[..., 1]
    rows = np.searchsorted(codes, wanted).astype(np.int32)
    included = keys[..., 0] >= 0
    assert np.array_equal(codes[rows[included]], wanted[included])
    return np.where(included, rows, -1)


def record(model_size: str, corpus: np.ndarray) -> None:
    started = time.time()
    band = band_examples(model_size)
    included = band["included"]
    unique = np.unique(band["keys"][included].reshape(-1, 2), axis=0)
    rows = band_rows(band["keys"], unique)
    data = np.load(sweep_path(model_size, DEPTH))
    layer, n_layers = int(data["layer"]), int(data["n_layers"])
    neurons = band["neurons"]
    expected = data["max"][band["keys"][included][..., 0], neurons[included][:, None]].astype(np.float32)
    local_dir = WEIGHTS_DIR / f"pythia-{model_size}"
    snapshot_download(f"EleutherAI/pythia-{model_size}", local_dir=local_dir,
                      allow_patterns=["config.json", "model.safetensors"])
    try:
        model = GPTNeoXForCausalLM.from_pretrained(local_dir, torch_dtype=torch.float16).to(DEVICE).eval()
        z = normalized_residuals(model, corpus, {layer: unique})[layer]
        weights, gain, constant = input_weights(model, layer, neurons[included])
        rebuilt = reconstructed_activations(model, layer, z, rows[included], weights, gain, constant)
        error = float(np.abs(rebuilt - expected).max())
        np.savez(v17_path(model_size), keys=unique, z=z, neurons=neurons, band_rows=rows, band_counts=band["counts"],
                 layer=layer, n_layers=n_layers, activation_error=error,
                 median_band_activation=float(np.median(expected)), seconds=time.time() - started)
        print(f"pythia-{model_size}: layer {layer}, {included.sum()} of {len(neurons)} neurons with a band sample, "
              f"{len(unique)} positions; activations rebuilt within {error:.3f} "
              f"(median band activation {np.median(expected):.2f}); {time.time() - started:.0f} s", flush=True)
        del model
    finally:
        shutil.rmtree(local_dir)
        if DEVICE == "mps":
            torch.mps.empty_cache()


def main() -> None:
    corpus = load_corpus()
    V17_DIR.mkdir(exist_ok=True)
    for model_size in MODELS:
        if v17_path(model_size).exists():
            print(f"pythia-{model_size}: already recorded")
            continue
        record(model_size, corpus)


if __name__ == "__main__":
    main()
