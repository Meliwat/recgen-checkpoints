"""v23 recording: v4's attribution at the final checkpoint on v23's facet examples and v13's background.

Examples are captured in chunks within one checkpoint load, and head terms are stored in float16 like the MLP
activations, to bound memory and keep data/v23/ under 2 GB. The file has the keys formation.record writes, so
analyze_v4.step_summary reads it unchanged.
"""

import shutil

import numpy as np

from recgen.activations import checkpoint
from recgen.attribution import N_BACKGROUND, background_examples, mlp_coefficients
from recgen.candidates_v13 import load_candidates as load_v13_candidates
from recgen.candidates_v13 import v13_path
from recgen.candidates_v23 import RECORDING_PATH, V23_DIR, load_candidates
from recgen.config import SEED
from recgen.corpus import load_corpus
from recgen.formation import capture
from recgen.rubric_v23 import STEP

CHUNK = 1500  # examples per capture call
HALF_PRECISION = ("acts", "heads")
MIN_FREE_GB = 8  # the checkpoint is about 0.9 GB; at least 7 GB must remain


def v13_background(n_corpus_windows: int) -> list[tuple[int, int]]:
    """v13's background positions: v4's rule applied to v13's candidates with the same seed."""
    background = background_examples(load_v13_candidates(), n_corpus_windows, N_BACKGROUND, np.random.default_rng(SEED))
    v13 = np.load(v13_path(STEP))
    assert np.array_equal(v13["examples"][int(v13["n_facet_examples"]):], np.array(background)), "v13's background should reproduce"
    return background


def record(corpus: np.ndarray, examples: list[tuple[int, int]], n_facet: int, neurons: list[int], path) -> None:
    free_gb = shutil.disk_usage(V23_DIR).free / 1e9
    assert free_gb >= MIN_FREE_GB, f"only {free_gb:.1f} GB free"
    chunks, error = [], 0.0
    with checkpoint(STEP) as model:
        for start in range(0, len(examples), CHUNK):
            cap = capture(model, corpus, examples[start:start + CHUNK], neurons)
            parts = [v for k, v in cap.items() if k.startswith("part_")]
            error = max(error, float(np.abs(sum(parts) - cap["pre"]).max()))
            chunks.append({k: v.astype(np.float16) if k in HALF_PRECISION else np.ascontiguousarray(v) for k, v in cap.items()})
            print(f"  examples {start} to {start + len(chunks[-1]['pre'])} of {len(examples)}", flush=True)
        coefficients = mlp_coefficients(model, neurons)
    np.savez(path, examples=np.array(examples), n_facet_examples=n_facet, neurons=np.array(neurons),
             coefficients=coefficients, reconstruction_error=error,
             **{k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]})
    print(f"step {STEP}: recorded, largest reconstruction error {error:.1e}", flush=True)


def main() -> None:
    corpus = load_corpus()
    candidates = load_candidates()
    neurons = [c["neuron"] for c in candidates]
    facet_examples = sorted({(w, p) for c in candidates for facet in c["facets"] for w, p, _ in facet})
    background = v13_background(len(corpus))
    V23_DIR.mkdir(exist_ok=True)
    print(f"{len(neurons)} neurons, {sum(len(c['facets']) for c in candidates)} facets, {len(facet_examples)} facet examples, "
          f"{len(set(background) & set(facet_examples))} background positions that are also facet examples", flush=True)
    if RECORDING_PATH.exists():
        print(f"step {STEP}: already recorded")
        return
    record(corpus, facet_examples + background, len(facet_examples), neurons, RECORDING_PATH)


if __name__ == "__main__":
    main()
