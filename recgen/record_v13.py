"""v13 recording: v4's attribution at every v4 checkpoint, on v13's facet examples and background text."""

import numpy as np

from recgen.attribution import N_BACKGROUND, background_examples
from recgen.candidates_v13 import V13_DIR, load_candidates, v13_path
from recgen.config import SEED
from recgen.corpus import load_corpus
from recgen.formation import record
from recgen.rubric_v4 import STEPS


def main() -> None:
    corpus = load_corpus()
    candidates = load_candidates()
    neurons = [c["neuron"] for c in candidates]
    facet_examples = sorted({(w, p) for c in candidates for facet in c["facets"] for w, p, _ in facet})
    background = background_examples(candidates, len(corpus), N_BACKGROUND, np.random.default_rng(SEED))
    V13_DIR.mkdir(exist_ok=True)
    print(f"{len(neurons)} neurons, {sum(len(c['facets']) for c in candidates)} facets, "
          f"{len(facet_examples)} facet examples", flush=True)
    for step in STEPS:
        if v13_path(step).exists():
            print(f"step {step}: already recorded")
            continue
        record(step, corpus, facet_examples + background, len(facet_examples), neurons, path=v13_path(step))


if __name__ == "__main__":
    main()
