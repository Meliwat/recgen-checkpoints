"""v12's fresh Haiku readings: at the middle layer of each size, HAIKU_PER_QUARTILE sampled neurons from
each peak quartile, none among v7's 100 real neurons, each shown as its own 50 strongest examples with
v7's prompt. Answers are cached under data/v12/ and collected in results/v12_llm_answers.json."""

import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from recgen.config import DATA_DIR, RESULTS_DIR
from recgen.corpus import load_corpus
from recgen.llm_reader import read, unit_prompt
from recgen.rubric_v5 import MODELS
from recgen.rubric_v12 import HAIKU_DEPTH, HAIKU_PER_QUARTILE, quartiles, rng_for
from recgen.sweep import sweep_path
from recgen.sweep_v6 import v6_path
from recgen.units_v7 import build_units

LLM_DIR = DATA_DIR / "v12" / "llm"
ANSWERS_PATH = RESULTS_DIR / "v12_llm_answers.json"
WORKERS = 4


def fresh_units(model: str) -> list[dict]:
    """The fresh neurons for one model, each with its peak quartile among all sampled neurons."""
    v6 = np.load(v6_path(model, HAIKU_DEPTH))
    neurons, rows, keys = v6["neurons"], v6["example_rows"], v6["keys"]
    peak = np.load(sweep_path(model, HAIKU_DEPTH))["max"][:, neurons].astype(np.float32).max(axis=0)
    quartile = quartiles(peak)
    used = {u["neurons"][0] for u in build_units(model) if u["kind"] == "real"}
    rng = rng_for(model, 0)
    units = []
    for q in range(4):
        pool = np.array([s for s in np.flatnonzero(quartile == q) if s not in used])
        for s in np.sort(rng.choice(pool, size=HAIKU_PER_QUARTILE, replace=False)):
            units.append({"id": f"{model}:fresh::{int(s)}", "model": model, "neurons": (int(s),),
                          "neuron_id": int(neurons[s]), "quartile": int(q), "rows": rows[s], "keys": keys[rows[s]]})
    return units


def main() -> None:
    corpus = load_corpus()
    LLM_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [(u["id"], unit_prompt(u, corpus)) for m in MODELS for u in fresh_units(m)]
    print(f"{len(jobs)} readings", flush=True)
    with ThreadPoolExecutor(WORKERS) as pool:
        records = list(pool.map(lambda job: read(job[0], job[1], LLM_DIR), jobs))
    answers = {r["id"]: {"label": r["label"], "error": r["error"], "models": r["models"],
                         "prompt_sha256": r["prompt_sha256"], "reply": r["reply"]} for r in records}
    ANSWERS_PATH.write_text(json.dumps(answers, indent=1))
    print(f"done: {sum(a['label'] is None for a in answers.values())} without a label", flush=True)


if __name__ == "__main__":
    main()
