"""Bridge search: while a later facet rises, does the neuron first respond to text that sits
between it and an earlier facet? The rule is pre-registered in recgen/rubric_v2.py.

select: embed every position of the background windows (the pool) and pick, for each pair of
        an earlier and a later facet, the bridge set and its two matched control sets.
record: the traced neurons' pre-activations on the pool and on every facet example, at each
        checkpoint (v1's and the denser ones) inside some pair's window.
"""

import argparse
import json

import numpy as np
import torch

from recgen.activations import BATCH_SIZE, MIN_CONTEXT, checkpoint
from recgen.attribution import N_BACKGROUND, background_examples
from recgen.config import CHECKPOINT_STEPS, CONTEXT_LEN, DATA_DIR, LAYER, RESULTS_DIR, SEED
from recgen.corpus import load_corpus
from recgen.facets import centroid, context_text, display_text, embed
from recgen.record import selected_neurons
from recgen.rubric_v2 import BRIDGE_SIZE, dense_grid

POOL_EMBEDDINGS_PATH = DATA_DIR / "pool_embeddings.npy"
DENSE_DIR = DATA_DIR / "dense"
BRIDGES_PATH = RESULTS_DIR / "bridges.json"
STEPS = dense_grid(CHECKPOINT_STEPS)
EXAMPLES_SHOWN = 3


def pool_windows(candidates: list[dict], n_corpus_windows: int) -> list[int]:
    """The background windows used as the attribution baseline: random, holding no facet example."""
    return sorted(w for w, _ in background_examples(candidates, n_corpus_windows, N_BACKGROUND,
                                                    np.random.default_rng(SEED)))


def pool_positions(windows: list[int]) -> np.ndarray:
    """Every position with full context in the windows, window by window: [n, 2] of (window, pos)."""
    pos = np.arange(MIN_CONTEXT, CONTEXT_LEN)
    return np.stack([np.repeat(windows, len(pos)), np.tile(pos, len(windows))], axis=1)


def facet_examples(candidates: list[dict]) -> list[tuple[int, int]]:
    return sorted({(w, p) for c in candidates for facet in c["facets"] for w, p, _ in facet})


def later_facet_pairs(results: list[dict]) -> list[dict]:
    """Every (earlier facet A, later facet B) pair within a neuron, from v1's emergence steps."""
    pairs = []
    for r in results:
        steps = r["emergence_steps"]
        if None in steps:
            continue
        for b, step_b in enumerate(steps):
            pairs += [{"neuron": r["neuron"], "a": a, "b": b, "step_a": step_a, "step_b": step_b}
                      for a, step_a in enumerate(steps) if step_a < step_b]
    return pairs


def window_steps(step_a: int, step_b: int) -> list[int]:
    return [s for s in STEPS if step_a <= s <= step_b]


def one_per_window(order: np.ndarray, windows: np.ndarray, size: int, excluded=()) -> list[int]:
    """The first `size` indices in order whose window hasn't been used yet."""
    picked, used = [], set(excluded)
    for i in order:
        if windows[i] not in used:
            picked.append(int(i))
            used.add(windows[i])
            if len(picked) == size:
                break
    return picked


def bridge_set(cos_a: np.ndarray, cos_b: np.ndarray, windows: np.ndarray, size: int = BRIDGE_SIZE) -> list[int]:
    """Pool texts most similar to both facets: highest smaller similarity, one per window."""
    return one_per_window(np.argsort(-np.minimum(cos_a, cos_b), kind="stable"), windows, size)


def matched_control(targets: np.ndarray, match_on: np.ndarray, other: np.ndarray, windows: np.ndarray,
                    excluded=()) -> list[int]:
    """For each target, the pool text closest to it on `match_on` among texts at or below the
    pool's median on `other`, one per window and none from excluded windows."""
    available = (other <= np.median(other)) & ~np.isin(windows, list(excluded))
    picked = []
    for t in targets:
        i = int(np.argmin(np.where(available, np.abs(match_on - t), np.inf)))
        picked.append(i)
        available &= windows != windows[i]
    return picked


def select(corpus: np.ndarray, candidates: list[dict], results: list[dict]) -> list[dict]:
    positions = pool_positions(pool_windows(candidates, len(corpus)))
    if not POOL_EMBEDDINGS_PATH.exists():
        print(f"embedding {len(positions):,} pool texts", flush=True)
        np.save(POOL_EMBEDDINGS_PATH, embed([context_text(corpus, w, p) for w, p in positions]).astype(np.float16))
    pool = np.load(POOL_EMBEDDINGS_PATH).astype(np.float32)
    by_neuron = {c["neuron"]: c for c in candidates}
    centroids = {}
    pairs = later_facet_pairs(results)
    for pair in pairs:
        facets = by_neuron[pair["neuron"]]["facets"]
        for f in (pair["a"], pair["b"]):
            if (pair["neuron"], f) not in centroids:
                centroids[pair["neuron"], f] = centroid(embed([context_text(corpus, w, p) for w, p, _ in facets[f]]))
        cos_a, cos_b = pool @ centroids[pair["neuron"], pair["a"]], pool @ centroids[pair["neuron"], pair["b"]]
        windows = positions[:, 0]
        bridge = bridge_set(cos_a, cos_b, windows)
        used = set(windows[bridge])
        a_control = matched_control(cos_a[bridge], cos_a, cos_b, windows, used)
        b_control = matched_control(cos_b[bridge], cos_b, cos_a, windows, used)
        pair.update({
            "bridge": bridge, "a_control": a_control, "b_control": b_control,
            "similarity": {name: {"to_a": round(float(np.median(cos_a[idx])), 3),
                                  "to_b": round(float(np.median(cos_b[idx])), 3)}
                           for name, idx in (("bridge", bridge), ("a_control", a_control), ("b_control", b_control))},
            "bridge_examples": [display_text(corpus, *positions[i]) for i in bridge[:EXAMPLES_SHOWN]],
        })
    BRIDGES_PATH.write_text(json.dumps(pairs, indent=1))
    print(f"{len(pairs)} facet pairs; checkpoints needed: {needed_steps(pairs)}")
    return pairs


def needed_steps(pairs: list[dict]) -> list[int]:
    return sorted({s for p in pairs for s in window_steps(p["step_a"], p["step_b"])})


class _StopForward(Exception):
    pass


def pre_activations(model, corpus: np.ndarray, windows: list[int], neurons: list[int],
                    layer: int = LAYER) -> np.ndarray:
    """The neurons' pre-activations at every position of the windows: [len(windows), T, N] float16."""
    device = next(model.parameters()).device
    index = torch.tensor(neurons, device=device)
    chunks = []

    def hook(_module, _inputs, output):
        chunks.append(output.index_select(-1, index).half().cpu().numpy())
        raise _StopForward

    handle = model.gpt_neox.layers[layer].mlp.dense_h_to_4h.register_forward_hook(hook)
    try:
        with torch.no_grad():
            for start in range(0, len(windows), BATCH_SIZE):
                try:
                    model(input_ids=torch.from_numpy(corpus[windows[start:start + BATCH_SIZE]]).long().to(device))
                except _StopForward:
                    pass
    finally:
        handle.remove()
    return np.concatenate(chunks)


def dense_path(step: int):
    return DENSE_DIR / f"step{step}_pre.npz"


def record(corpus: np.ndarray, candidates: list[dict], steps: list[int]) -> None:
    """pool: [n_pool, N] in pool_positions order. facet: [n_facet_examples, N] in facet_examples order."""
    DENSE_DIR.mkdir(exist_ok=True)
    neurons = selected_neurons()
    windows = pool_windows(candidates, len(corpus))
    examples = facet_examples(candidates)
    facet_windows = sorted({w for w, _ in examples})
    row = {w: i for i, w in enumerate(facet_windows)}
    for step in steps:
        if dense_path(step).exists():
            print(f"step {step}: already recorded")
            continue
        with checkpoint(step) as model:
            pool = pre_activations(model, corpus, windows, neurons)[:, MIN_CONTEXT:].reshape(-1, len(neurons))
            facet = pre_activations(model, corpus, facet_windows, neurons)
        facet = facet[[row[w] for w, _ in examples], [p for _, p in examples]]
        np.savez(dense_path(step), pool=pool, facet=facet, neurons=np.array(neurons))
        print(f"step {step}: recorded", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["select", "record"])
    args = parser.parse_args()
    corpus = load_corpus()
    candidates = json.loads((RESULTS_DIR / "candidates.json").read_text())
    if args.stage == "select":
        results = json.loads((RESULTS_DIR / "classification.json").read_text())["neurons"]
        select(corpus, candidates, results)
    else:
        record(corpus, candidates, needed_steps(json.loads(BRIDGES_PATH.read_text())))


if __name__ == "__main__":
    main()
