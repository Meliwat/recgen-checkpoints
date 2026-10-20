"""Post hoc v20 readings, added after an independent review of v20; not pre-registered (recgen/rubric_v20.py has
no part in them). Three small experiments, each read by Haiku with v7's prompt and CLI through the cached reader
(recgen.llm_reader), from the same working directory as v20's readings:

A. Same-round top 50: v12's 600 fresh neurons' top-50 prompts, byte-identical to v12's, read again, so Haiku's
   within-neuron comparison (T1) and T2's paired difference can use rank-band and top-50 readings of one round.
B. Random-text floor: RANDOM_UNITS units of BAND_SIZE random positions each, drawn from v6's random positions at
   the middle layer (one per window), spread over the six sizes: how often Haiku calls random text unrelated jobs.
C. Positive control below the top: per size and peak quartile, POOLS_PER_QUARTILE pools of two of v12's fresh
   neurons from that quartile, each giving POOL_PART of its 50 examples from ranks 201 to 500: whether the readers
   tell two neurons' rank-band examples from one neuron's. analyze_v20 reads the same pools with the network reader.

At most WORKERS calls at once; reading stops once these readings' CLI-reported cost reaches SPEND_LIMIT_USD.
Answers are cached under data/v20/ and collected in results/v20_posthoc_llm_answers.json.
"""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from recgen.band_v20 import rank_band_examples
from recgen.config import RESULTS_DIR
from recgen.corpus import load_corpus
from recgen.llm_reader import unit_prompt
from recgen.read_v12 import ANSWERS_PATH as V12_ANSWERS_PATH
from recgen.read_v12 import fresh_units
from recgen.read_v17 import SpendGuard
from recgen.read_v20 import V20_DIR
from recgen.rubric_v5 import MODELS
from recgen.rubric_v20 import BAND_SIZE, DEPTH, PRIMARY_BAND, WORKERS, rng_for
from recgen.sweep_v6 import v6_path

ANSWERS_PATH = RESULTS_DIR / "v20_posthoc_llm_answers.json"
KINDS = {"top_same_round": V20_DIR / "llm_top_same_round", "random": V20_DIR / "llm_random",
         "pool": V20_DIR / "llm_pool"}
SPEND_LIMIT_USD = 5.0
RANDOM_UNITS = 100
POOLS_PER_QUARTILE = 10
POOL_PART = 25  # examples from each neuron of a pool
RANDOM_PURPOSE, POOL_PURPOSE = 101, 102  # rng_for purposes, apart from v20's


def random_units(model: str) -> list[dict]:
    """This model's share of RANDOM_UNITS units, each BAND_SIZE of v6's random positions at DEPTH (one per window,
    the same text for every model), drawn without replacement within a unit."""
    v6 = np.load(v6_path(model, DEPTH))
    keys = v6["keys"][v6["random_rows"]]
    n_units = RANDOM_UNITS // len(MODELS) + (MODELS.index(model) < RANDOM_UNITS % len(MODELS))
    rng = rng_for(model, RANDOM_PURPOSE)
    return [{"id": f"{model}:random::{i}", "model": model,
             "keys": keys[np.sort(rng.choice(len(keys), size=BAND_SIZE, replace=False))]} for i in range(n_units)]


def pool_picks(keys_a: np.ndarray, keys_b: np.ndarray, rng) -> tuple[np.ndarray, np.ndarray]:
    """POOL_PART of each neuron's BAND_SIZE rank-band examples (indices), drawn at random, skipping the second
    neuron's examples at a position the first already gives, as v7's pools skip shared examples."""
    from_a = np.sort(rng.choice(BAND_SIZE, size=POOL_PART, replace=False))
    taken = {tuple(k) for k in keys_a[from_a].tolist()}
    from_b = [int(i) for i in rng.permutation(BAND_SIZE) if tuple(keys_b[i].tolist()) not in taken][:POOL_PART]
    return from_a, np.sort(np.array(from_b))


def pool_units(model: str) -> list[dict]:
    """POOLS_PER_QUARTILE pools per peak quartile, each of two of v12's fresh neurons from that quartile (no neuron
    in two pools), with POOL_PART of each one's primary-band examples; picks index each neuron's examples."""
    band = rank_band_examples(model, PRIMARY_BAND)
    fresh = fresh_units(model)
    rng = rng_for(model, POOL_PURPOSE)
    units = []
    for q in range(4):
        candidates = [u["neurons"][0] for u in fresh if u["quartile"] == q]
        for a, b in rng.choice(candidates, size=2 * POOLS_PER_QUARTILE, replace=False).reshape(-1, 2).tolist():
            assert band["included"][a] and band["included"][b]
            picks = pool_picks(band["keys"][a], band["keys"][b], rng)
            units.append({"id": f"{model}:pool-rank201-500::{a}-{b}", "model": model, "quartile": q, "neurons": (a, b),
                          "picks": picks, "keys": np.concatenate([band["keys"][a][picks[0]], band["keys"][b][picks[1]]])})
    return units


def jobs_for(model: str, corpus, v12_answers: dict) -> list[tuple[str, str, str]]:
    """(kind, unit id, prompt) for one model's same-round top-50 readings, random units and pools."""
    jobs = []
    for u in fresh_units(model):
        text = unit_prompt(u, corpus)
        assert hashlib.sha256(text.encode()).hexdigest() == v12_answers[u["id"]]["prompt_sha256"]
        jobs.append(("top_same_round", u["id"], text))
    jobs += [("random", u["id"], unit_prompt(u, corpus)) for u in random_units(model)]
    jobs += [("pool", u["id"], unit_prompt(u, corpus)) for u in pool_units(model)]
    return jobs


def main() -> None:
    corpus = load_corpus()
    v12_answers = json.loads(V12_ANSWERS_PATH.read_text())
    for directory in KINDS.values():
        directory.mkdir(parents=True, exist_ok=True)
    jobs = [job for model in MODELS for job in jobs_for(model, corpus, v12_answers)]
    print(f"{len(jobs)} readings", flush=True)
    guard = SpendGuard(SPEND_LIMIT_USD)
    with ThreadPoolExecutor(WORKERS) as pool:
        records = []
        for i, record in enumerate(pool.map(lambda job: guard.read(job[1], job[2], KINDS[job[0]]), jobs), 1):
            records.append(record)
            if i % 100 == 0:
                print(f"{i} of {len(jobs)} read, ${guard.spent:.2f} so far", flush=True)
    answers = {kind: {} for kind in KINDS}
    for (kind, uid, _), record in zip(jobs, records):
        if record is None:
            answers[kind][uid] = {"label": None, "error": "spend limit reached"}
            continue
        answers[kind][uid] = {
            "label": record["label"], "error": record["error"], "models": record["models"],
            "prompt_sha256": record["prompt_sha256"], "cost_usd": record["cost_usd"], "reply": record["reply"],
            "input_tokens": sum(record["usage"].get(k) or 0 for k in
                                ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))}
    answers["total_cost_usd"] = round(sum((r or {}).get("cost_usd") or 0 for r in records), 4)
    ANSWERS_PATH.write_text(json.dumps(answers, indent=1))
    missing = sum(a["label"] is None for kind in KINDS for a in answers[kind].values())
    print(f"done: {missing} without a label, ${answers['total_cost_usd']:.2f} CLI-reported", flush=True)


if __name__ == "__main__":
    main()
