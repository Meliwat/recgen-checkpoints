"""v22's corpus and recording: v1's window construction on an OpenWebText sample with GPT-2's tokenizer, and v19's
recording of v19's sampled neurons on it (rubric_v22). One model at a time, deleted after use.

data/v22/ gets the raw download in its own directory (RAW_DIR, never executed, only read as parquet), the corpus,
and for each model files in v19's formats, so v19's units and analysis read them unchanged:
- {model}_maxima.npz: every neuron's strongest activation in each window and where;
- {model}_network.npz: the residual stream entering ln_2 at v19's sampled neurons' 50 strongest OpenWebText examples
  and at rubric_v6's random positions, the neurons' input weights, and the checks.

Run with model names to record only those (the pipeline checks free disk space between models).
"""

import hashlib
import shutil
import sys

import numpy as np
import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import GPT2LMHeadModel

from recgen.activations import DEVICE
from recgen.config import CONTEXT_LEN, DATA_DIR, N_WINDOWS, WEIGHTS_DIR
from recgen.corpus import packed_stream, window_starts
from recgen.facets import top_examples
from recgen.gpt2_v19 import (
    float32_error, input_weights, normalized_residuals, rebuilt_activations, tokenizer, window_maxima,
)
from recgen.gpt2_v19 import network_path as v19_network_path
from recgen.rubric_v6 import random_positions
from recgen.rubric_v19 import HUB, MODELS, middle_layer
from recgen.rubric_v22 import (
    CORPUS_FILE, CORPUS_REPO, CORPUS_REVISION, CORPUS_SHA256, FP32_DEVICE, MIN_WINDOWS, N_DOCUMENTS,
)

V22_DIR = DATA_DIR / "v22"
RAW_DIR = V22_DIR / "openwebtext-10k"
CORPUS_PATH = V22_DIR / "corpus_owt.npy"


def documents() -> list[str]:
    """The sample's documents in the file's order, downloaded once and checked against the pre-registered hash."""
    path = RAW_DIR / CORPUS_FILE
    if not path.exists():
        hf_hub_download(CORPUS_REPO, CORPUS_FILE, repo_type="dataset", revision=CORPUS_REVISION, local_dir=RAW_DIR)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == CORPUS_SHA256, "not the pre-registered file"
    docs = pq.read_table(path, columns=["text"]).column("text").to_pylist()
    assert len(docs) == N_DOCUMENTS
    return docs


def windows(stream: np.ndarray) -> np.ndarray:
    """v1's windows: N_WINDOWS seeded full windows of the packed stream, in stream order."""
    n_full = len(stream) // CONTEXT_LEN
    assert n_full >= max(N_WINDOWS, MIN_WINDOWS), f"only {n_full} full windows"
    return np.stack([stream[s:s + CONTEXT_LEN] for s in window_starts(len(stream))])


def load_owt_corpus() -> np.ndarray:
    """The 4,096 OpenWebText windows as GPT-2 tokens [N_WINDOWS, CONTEXT_LEN], built once."""
    if not CORPUS_PATH.exists():
        V22_DIR.mkdir(exist_ok=True)
        np.save(CORPUS_PATH, windows(packed_stream(tokenizer(), documents())).astype(np.int32))
    return np.load(CORPUS_PATH)


def maxima_path(model: str):
    return V22_DIR / f"{model}_maxima.npz"


def network_path(model: str):
    return V22_DIR / f"{model}_network.npz"


def v19_neurons(model: str) -> np.ndarray:
    """v19's sampled neurons of one model, which v22 keeps."""
    return np.load(v19_network_path(model))["neurons"]


def sampled_top_examples(model: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """v19's sampled neurons [S], their 50 strongest OpenWebText examples as (window, position) [S, 50, 2], strongest
    first, and activations [S, 50], from v22's window maxima."""
    data = np.load(maxima_path(model))
    window_max, window_pos = data["max"].astype(np.float32), data["pos"]
    neurons = v19_neurons(model)
    examples = [top_examples(window_max, window_pos, int(n)) for n in neurons]
    keys = np.array([[(w, p) for w, p, _ in ex] for ex in examples], dtype=np.int32)
    acts = np.array([[a for _, _, a in ex] for ex in examples], dtype=np.float32)
    return neurons, keys, acts


def record(name: str, corpus: np.ndarray) -> None:
    """gpt2_v19.record on OpenWebText, with v19's neurons and the float32 check on FP32_DEVICE."""
    local_dir = WEIGHTS_DIR / name
    snapshot_download(f"{HUB}/{name}", local_dir=local_dir, allow_patterns=["config.json", "model.safetensors"])
    try:
        model = GPT2LMHeadModel.from_pretrained(local_dir, dtype=torch.float16).to(DEVICE).eval()
        n_layers = model.config.n_layer
        layer = middle_layer(n_layers)
        values, positions = window_maxima(model, corpus, layer)
        assert np.isfinite(values).all(), "float16 overflow"
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
        if DEVICE == "mps":
            torch.mps.empty_cache()
        fp32 = float32_error(local_dir, corpus, layer, neurons, values.astype(np.float16), FP32_DEVICE)
        np.savez(network_path(name), keys=unique, z=z, neurons=neurons, example_rows=rows, random_rows=random_rows,
                 weights=weights, gain=gain, constant=constant, layer=layer, n_layers=n_layers,
                 activation_error=error, float32_error=fp32)
        print(f"{name}: layer {layer} of {n_layers}, v19's {len(neurons)} sampled neurons, {len(unique)} positions; "
              f"activations rebuilt within {error:.4f}, float32 ({FP32_DEVICE}) within {fp32:.4f} "
              f"(median top activation {np.median(acts):.2f})", flush=True)
    finally:
        shutil.rmtree(local_dir)


def main(names: list[str]) -> None:
    corpus = load_owt_corpus()
    for name in names or MODELS:
        if network_path(name).exists():
            print(f"{name}: already recorded")
            continue
        record(name, corpus)


if __name__ == "__main__":
    main(sys.argv[1:])
