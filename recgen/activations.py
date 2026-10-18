"""Record MLP neuron activations at one layer of a Pythia training checkpoint."""

import shutil
from contextlib import contextmanager

import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import GPTNeoXForCausalLM

from recgen.config import LAYER, MODEL, WEIGHTS_DIR

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BATCH_SIZE = 16
# Activations at the first positions of a window come with almost no context
# (and attention-sink effects), so they never count as a neuron's examples.
MIN_CONTEXT = 32


@contextmanager
def checkpoint(step: int):
    """Download one checkpoint, yield the model, and delete the weights after."""
    local_dir = WEIGHTS_DIR / f"step{step}"
    snapshot_download(MODEL, revision=f"step{step}", local_dir=local_dir,
                      allow_patterns=["config.json", "model.safetensors"])
    model = GPTNeoXForCausalLM.from_pretrained(local_dir, torch_dtype=torch.float32).to(DEVICE).eval()
    try:
        yield model
    finally:
        del model
        shutil.rmtree(local_dir)


class _StopForward(Exception):
    pass


def layer_activations(model, corpus: np.ndarray):
    """Yield (start_window, activations[B, T, d_mlp]) for each batch, stopping the
    forward pass at LAYER since later layers can't affect it."""
    captured = {}

    def hook(_module, _inputs, output):
        captured["acts"] = output
        raise _StopForward

    handle = model.gpt_neox.layers[LAYER].mlp.act.register_forward_hook(hook)
    try:
        with torch.no_grad():
            for start in range(0, len(corpus), BATCH_SIZE):
                ids = torch.from_numpy(corpus[start:start + BATCH_SIZE]).long().to(DEVICE)
                try:
                    model(input_ids=ids)
                except _StopForward:
                    pass
                acts = captured.pop("acts")
                acts[:, :MIN_CONTEXT] = float("-inf")
                yield start, acts
    finally:
        handle.remove()


def window_maxima(model, corpus: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each neuron's strongest activation in every window, and where it happened.
    Returns (max[W, N] float32, position[W, N] int16)."""
    maxima, positions = [], []
    for _, acts in layer_activations(model, corpus):
        values, pos = acts.max(dim=1)
        maxima.append(values.float().cpu().numpy())
        positions.append(pos.to(torch.int16).cpu().numpy())
    return np.concatenate(maxima), np.concatenate(positions)


def neuron_activations(model, corpus: np.ndarray, neurons: list[int]) -> np.ndarray:
    """Full activations of a few neurons at every token. Returns [W, T, len(neurons)] float16;
    the first MIN_CONTEXT positions are -inf."""
    index = torch.tensor(neurons, device=DEVICE)
    chunks = [acts.index_select(2, index).half().cpu().numpy() for _, acts in layer_activations(model, corpus)]
    return np.concatenate(chunks)
