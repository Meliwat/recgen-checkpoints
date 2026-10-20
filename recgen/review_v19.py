"""Post hoc, added after an independent review of v19; not pre-registered. Checks on why Haiku flags far fewer
GPT-2 neurons than Pythia neurons, on what its retest agreement measures, and on two recording details.

- Pythia at matched sizes: Haiku's real-neuron flag rate on v7's Pythia sizes nearest each GPT-2 size.
- Pythia read near in time: Haiku's flag rate on Pythia real neurons read the same night as v19 (v11's readings
  at a quarter and three quarters of depth, from its cache in data/v11/, and v12's fresh middle-layer neurons),
  and v11's retest kappa. If Haiku itself had drifted since v7, these would be low too.
- The examples Haiku sees: for real neurons, the share of examples whose 9-token context (8 tokens before and
  the firing token) occurs more than once in the unit, the median share of the unit's most common firing token,
  and the prompt's length, GPT-2 against v7's Pythia units.
- The retests: how many retested units are real neurons, and the share flagged at the first reading.
- Prompts: whether every cached reading's prompt hash matches the prompt rebuilt from the code.
- The largest coordinate of the normalized residual (which the precision check's relative error is relative to).
- Token purity's ties: v12's covariates take a neuron's top 50 windows with a stable sort, the readers with
  numpy's default sort, so at a float16 tie they can hold different windows.
"""

import hashlib
import json
from collections import Counter

import numpy as np

from recgen.analyze_v7 import retest_stats
from recgen.config import DATA_DIR, RESULTS_DIR
from recgen.corpus import load_corpus
from recgen.facets import tokenizer as pythia_tokenizer
from recgen.gpt2_v19 import maxima_path, sampled_top_examples
from recgen.gpt2_v19 import tokenizer as gpt2_tokenizer
from recgen.llm_reader import unit_prompt as pythia_prompt
from recgen.read_v19 import unit_prompt as gpt2_prompt
from recgen.rubric_v5 import MODELS as PYTHIA_MODELS
from recgen.rubric_v6 import cohen_kappa
from recgen.rubric_v7 import UNRELATED_JOBS
from recgen.rubric_v19 import MODELS
from recgen.units_v7 import build_units as pythia_units

# each GPT-2 size (124M, 355M, 774M) with the Pythia size nearest it in parameters
SIZE_MATCHED = {"gpt2": "160m", "gpt2-medium": "410m", "gpt2-large": "1b"}
V11_DIR = DATA_DIR / "v11"
V11_DEPTHS = ("quarter", "three_quarters")
CONTEXT_BEFORE = 8  # tokens before the firing token in a duplicate check's context


def flag_rate(labels) -> float:
    """The share of readings labeled unrelated jobs, over readings with a label."""
    labels = [label for label in labels if label is not None]
    return float(np.mean([label == UNRELATED_JOBS for label in labels]))


def real_rates(answers: dict, models) -> dict[str, float]:
    """Haiku's flag rate on each model's real neurons, from first readings keyed by unit id."""
    return {m: flag_rate([a["label"] for k, a in answers.items() if k.startswith(f"{m}:real:")]) for m in models}


def size_matched(v19_answers: dict, v7_answers: dict) -> dict:
    gpt2, pythia = real_rates(v19_answers["first"], MODELS), real_rates(v7_answers["first"], PYTHIA_MODELS)
    matched = [pythia[SIZE_MATCHED[m]] for m in MODELS]
    return {"gpt2": gpt2, "pythia_v7": pythia, "pythia_size_matched": dict(zip(SIZE_MATCHED.values(), matched)),
            "mean": {"gpt2": float(np.mean(list(gpt2.values()))), "pythia_size_matched": float(np.mean(matched))}}


def cached_labels(directory, pattern: str) -> dict[str, dict]:
    """Cached readings in one directory, by file name."""
    return {f.name: json.loads(f.read_text()) for f in sorted(directory.glob(pattern))}


def retest_kappa(first: dict[str, dict], second: dict[str, dict]) -> dict:
    """analyze_v7's retest statistics over readings cached under the same name, with the prompt checked."""
    assert all(first[k]["prompt_sha256"] == v["prompt_sha256"] for k, v in second.items()), "a retest saw another prompt"
    return retest_stats({"first": first, "retest": second})


def near_in_time(v12_answers: dict) -> dict:
    """Haiku on Pythia's real neurons in v11 (its cache, if present) and v12, and v11's retest."""
    out = {"v12_fresh_middle": {"readings": len(v12_answers), "flagged": flag_rate([a["label"] for a in v12_answers.values()])}}
    first = cached_labels(V11_DIR / "llm", "*.json")
    if not first:
        return {**out, "v11": None}
    out["v11"] = {}
    for depth in V11_DEPTHS:
        real = [a["label"] for name, a in first.items() if f"_{depth}_real__" in name]
        out["v11"][depth] = {"readings": len(real), "flagged": flag_rate(real)}
    out["v11"]["retest"] = retest_kappa(first, cached_labels(V11_DIR / "llm_retest", "*.json"))
    return out


def retest_composition(answers: dict) -> dict:
    """How many retested units are real neurons, and the share flagged at their first reading."""
    ids = [k for k, v in answers["retest"].items() if k in answers["first"] and v["label"] is not None]
    return {"units": len(ids), "real_neurons": sum(":real:" in k for k in ids),
            "flagged_at_first_reading": flag_rate([answers["first"][k]["label"] for k in ids])}


def example_stats(units: list[dict], corpus: np.ndarray, decode) -> dict:
    """For real units: mean share of examples whose 9-token context recurs in the unit, and the median share of
    the unit's most common firing token (decoded, stripped, case-folded)."""
    duplicated, top_token = [], []
    for unit in (u for u in units if u["kind"] == "real"):
        contexts = Counter(tuple(corpus[w, p - CONTEXT_BEFORE:p + 1]) for w, p in unit["keys"])
        duplicated.append(sum(n for n in contexts.values() if n > 1) / len(unit["keys"]))
        tokens = Counter(decode(corpus[w, p:p + 1]).strip().lower() for w, p in unit["keys"])
        top_token.append(tokens.most_common(1)[0][1] / len(unit["keys"]))
    return {"duplicated_context_share": float(np.mean(duplicated)), "median_top_token_share": float(np.median(top_token))}


def prompt_stats(units: list[dict], corpus: np.ndarray, prompt, answers: dict) -> dict:
    """For real units: mean prompt length in characters and mean input tokens the CLI reported."""
    real = [u for u in units if u["kind"] == "real"]
    return {"mean_prompt_characters": float(np.mean([len(prompt(u, corpus)) for u in real])),
            "mean_input_tokens": float(np.mean([answers["first"][u["id"]]["input_tokens"] for u in real]))}


def examples_seen(gpt2_units: dict[str, list[dict]], gpt2_corpus: np.ndarray, v19_answers: dict, v7_answers: dict) -> dict:
    out = {"gpt2": {}, "pythia_v7": {}}
    for model, units in gpt2_units.items():
        out["gpt2"][model] = {**example_stats(units, gpt2_corpus, gpt2_tokenizer().decode),
                              **prompt_stats(units, gpt2_corpus, gpt2_prompt, v19_answers)}
    corpus, decode = load_corpus(), pythia_tokenizer().decode
    for model in PYTHIA_MODELS:
        units = pythia_units(model)
        out["pythia_v7"][model] = {**example_stats(units, corpus, decode), **prompt_stats(units, corpus, pythia_prompt, v7_answers)}
    return out


def prompts_reproduce(units: dict[str, list[dict]], corpus: np.ndarray, answers: dict) -> dict:
    """How many of v19's readings were of the prompt the code builds for their unit today."""
    prompts = {u["id"]: gpt2_prompt(u, corpus) for model_units in units.values() for u in model_units}
    readings = [(k, a) for part in ("first", "retest") for k, a in answers[part].items()]
    matching = sum(hashlib.sha256(prompts[k].encode()).hexdigest() == a["prompt_sha256"] for k, a in readings)
    return {"readings": len(readings), "matching_prompt_hash": matching}


def largest_coordinate(vectors: np.ndarray) -> dict:
    """Where each normalized residual [n, d] is largest in absolute value: the dimension that most often holds it,
    how often, the median largest absolute value, and the median absolute value over all coordinates."""
    magnitude = np.abs(vectors)
    dims, counts = np.unique(magnitude.argmax(axis=1), return_counts=True)
    return {"dimension": int(dims[counts.argmax()]), "share_of_positions": float(counts.max() / len(vectors)),
            "median_largest": float(np.median(magnitude.max(axis=1))), "median_coordinate": float(np.median(magnitude))}


def purity_ties(model: str, neurons: list[dict], corpus: np.ndarray) -> dict:
    """The share of sampled neurons whose stable-sort top 50 windows (v12's token purity) differ from the readers'
    top 50, and the largest change in token purity if it were taken over the readers' 50."""
    data = np.load(maxima_path(model))
    window_max, window_pos = data["max"], data["pos"]
    ids, keys, _ = sampled_top_examples(model)
    differ, change = 0, 0.0
    for s, n in enumerate(map(int, ids)):
        stable = np.argsort(-window_max[:, n].astype(np.float32), kind="stable")[:50]
        if set(stable.tolist()) != set(keys[s, :, 0].tolist()):
            differ += 1
            tokens = [int(corpus[w, p]) for w, p in keys[s]]
            purity = max(tokens.count(t) for t in set(tokens)) / len(tokens)
            change = max(change, abs(purity - neurons[s]["token_purity"]))
    return {"neurons_differing": differ / len(ids), "largest_purity_change": change}


def haiku_checks(gpt2_units: dict[str, list[dict]], gpt2_corpus: np.ndarray, v19_answers: dict) -> dict:
    """The checks above on Haiku's readings, from v19's units, GPT-2 corpus and answers, and Pythia's results."""
    v7 = json.loads((RESULTS_DIR / "v7_llm_answers.json").read_text())
    v12 = json.loads((RESULTS_DIR / "v12_llm_answers.json").read_text())
    return {"size_matched": size_matched(v19_answers, v7), "pythia_near_in_time": near_in_time(v12),
            "retest_composition": {"gpt2": retest_composition(v19_answers), "pythia_v7": retest_composition(v7)},
            "examples_seen": examples_seen(gpt2_units, gpt2_corpus, v19_answers, v7),
            "prompts_reproduce": prompts_reproduce(gpt2_units, gpt2_corpus, v19_answers)}
