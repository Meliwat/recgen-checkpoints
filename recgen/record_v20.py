"""v20 recording: at the middle layer of each Pythia size, the residual stream entering the layer,
LayerNorm-normalized, at every sampled neuron's examples in both rank bands (recgen/band_v20.py), in one forward
pass. The primary band's residuals are stored as v17 stored its band. The second band's are read by the network
reader in memory once the model is freed, and only those readings are stored, since data/v20 is capped at 2 GB.
One model at a time, float16 as in v5 and v6, weights deleted after use; models already recorded are skipped.

As a check, each neuron's activation at every example of both bands is rebuilt from the residuals and its input
weights (as sweep_v6 does) and compared with v5's window maxima there.

Usage: python -m recgen.record_v20 [model ...]  (all six sizes, smallest first, if none are given)
"""

import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import GPTNeoXForCausalLM

from recgen.activations import DEVICE
from recgen.analyze_v6 import PRIMARY, Layer
from recgen.analyze_v17 import THREADS, read_examples
from recgen.band_v20 import rank_band_examples
from recgen.config import DATA_DIR, WEIGHTS_DIR
from recgen.corpus import load_corpus
from recgen.record_v17 import band_rows
from recgen.rubric_v5 import MODELS
from recgen.rubric_v6 import embed
from recgen.rubric_v20 import DEPTH, PRIMARY_BAND, RANKS
from recgen.sweep import sweep_path
from recgen.sweep_v6 import input_weights, normalized_residuals, reconstructed_activations

V20_DIR = DATA_DIR / "v20"
SECOND_BAND = "band2"
MEASURES = ("flag", "n_facets", "coherence")


def v20_path(model: str):
    return V20_DIR / f"{model}_{DEPTH}.npz"


def second_band_path(model: str):
    return V20_DIR / f"{model}_{DEPTH}_{SECOND_BAND}_readings.npz"


def band_readings(layer: Layer, z: np.ndarray, rows: np.ndarray) -> list[dict | None]:
    """The network reader on each sampled neuron's examples at rows [S, 50] into z (None where rows are -1),
    embedded and calibrated exactly as analyze_v17.neuron_readings reads a band."""
    mean, _, directions = layer.embeddings[PRIMARY]

    def one(s: int) -> dict | None:
        if rows[s][0] < 0:
            return None
        direction = directions[[s]]
        return read_examples(embed(z[rows[s]], mean, direction), layer.nulls[PRIMARY].similarities(direction))

    with ThreadPoolExecutor(THREADS) as pool:
        return list(pool.map(one, range(len(rows))))


def record(model_size: str, corpus: np.ndarray) -> None:
    started = time.time()
    bands = {band: rank_band_examples(model_size, band) for band in RANKS}
    neurons = bands[PRIMARY_BAND]["neurons"]
    unique = np.unique(np.concatenate([b["keys"][b["included"]].reshape(-1, 2) for b in bands.values()]), axis=0)
    rows = {band: band_rows(b["keys"], unique) for band, b in bands.items()}
    data = np.load(sweep_path(model_size, DEPTH))
    layer, n_layers = int(data["layer"]), int(data["n_layers"])
    expected = {band: data["max"][b["keys"][b["included"]][..., 0], neurons[b["included"]][:, None]].astype(np.float32)
                for band, b in bands.items()}
    local_dir = WEIGHTS_DIR / f"pythia-{model_size}"
    snapshot_download(f"EleutherAI/pythia-{model_size}", local_dir=local_dir,
                      allow_patterns=["config.json", "model.safetensors"])
    try:
        model = GPTNeoXForCausalLM.from_pretrained(local_dir, torch_dtype=torch.float16).to(DEVICE).eval()
        z = normalized_residuals(model, corpus, {layer: unique})[layer]
        weights, gain, constant = input_weights(model, layer, neurons)
        errors = {}
        for band, b in bands.items():
            included = b["included"]
            rebuilt = reconstructed_activations(model, layer, z, rows[band][included], weights[included], gain,
                                                constant[included])
            errors[band] = float(np.abs(rebuilt - expected[band]).max())
        del model
    finally:
        shutil.rmtree(local_dir)
        if DEVICE == "mps":
            torch.mps.empty_cache()

    primary = bands[PRIMARY_BAND]
    keep = np.unique(rows[PRIMARY_BAND][primary["included"]])
    np.savez(v20_path(model_size), keys=unique[keep], z=z[keep], neurons=neurons,
             band_rows=band_rows(primary["keys"], unique[keep]), band_counts=primary["counts"],
             fractions=primary["fractions"], layer=layer, n_layers=n_layers, activation_error=errors[PRIMARY_BAND],
             median_band_activation=float(np.median(expected[PRIMARY_BAND])), seconds=time.time() - started)

    second = bands[SECOND_BAND]
    readings = band_readings(Layer(model_size, DEPTH), z, rows[SECOND_BAND])
    np.savez(second_band_path(model_size), neurons=neurons, keys=second["keys"], fractions=second["fractions"],
             counts=second["counts"], included=second["included"],
             **{m: np.array([np.nan if r is None else r[m] for r in readings], float) for m in MEASURES},
             activation_error=errors[SECOND_BAND], median_band_activation=float(np.median(expected[SECOND_BAND])),
             seconds=time.time() - started)
    print(f"pythia-{model_size}: layer {layer}, " + ", ".join(
        f"{band} {b['included'].sum()} of {len(neurons)} neurons (activations rebuilt within {errors[band]:.4f}, "
        f"median {np.median(expected[band]):.2f})" for band, b in bands.items())
          + f"; {len(unique)} positions, {len(keep)} stored; {time.time() - started:.0f} s", flush=True)


def recorded(model: str) -> bool:
    return v20_path(model).exists() and second_band_path(model).exists()


def main() -> None:
    models = sys.argv[1:] or MODELS
    if unknown := set(models) - set(MODELS):
        raise SystemExit(f"unknown model sizes: {sorted(unknown)}; choose from {MODELS}")
    corpus = load_corpus()
    V20_DIR.mkdir(exist_ok=True)
    for model_size in models:
        if recorded(model_size):
            print(f"pythia-{model_size}: already recorded")
            continue
        record(model_size, corpus)


if __name__ == "__main__":
    main()
