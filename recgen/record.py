"""Record activations for one or more checkpoints.

For every neuron: its max activation in each corpus window and where it happened.
For the neurons in results/selected_neurons.json (if present): every activation.
"""

import argparse
import json

import numpy as np

from recgen.activations import checkpoint, neuron_activations, window_maxima
from recgen.config import CHECKPOINT_STEPS, DATA_DIR, RESULTS_DIR
from recgen.corpus import load_corpus

SELECTED_PATH = RESULTS_DIR / "selected_neurons.json"


def window_maxima_path(step: int):
    return DATA_DIR / f"step{step}_window_max.npz"


def neuron_activations_path(step: int):
    return DATA_DIR / f"step{step}_neuron_acts.npz"


def selected_neurons() -> list[int]:
    if not SELECTED_PATH.exists():
        return []
    return json.loads(SELECTED_PATH.read_text())["traced"]


def record(step: int, corpus: np.ndarray, neurons: list[int]) -> None:
    need_maxima = not window_maxima_path(step).exists()
    need_acts = bool(neurons) and not neuron_activations_path(step).exists()
    if not (need_maxima or need_acts):
        print(f"step {step}: already recorded")
        return
    with checkpoint(step) as model:
        if need_maxima:
            maxima, positions = window_maxima(model, corpus)
            np.savez(window_maxima_path(step), max=maxima, pos=positions)
        if need_acts:
            acts = neuron_activations(model, corpus, neurons)
            np.savez(neuron_activations_path(step), acts=acts, neurons=np.array(neurons))
    print(f"step {step}: recorded", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, nargs="*", default=CHECKPOINT_STEPS)
    args = parser.parse_args()
    corpus = load_corpus()
    neurons = selected_neurons()
    for step in args.steps:
        record(step, corpus, neurons)


if __name__ == "__main__":
    main()
