"""v20's Haiku readings: v12's 600 fresh middle-layer neurons, each shown as its 50 examples from each rank band
(recgen/band_v20.py) through v7's prompt and cached CLI reader. RETEST of the primary band's units are read a
second time, and the same neurons' top-50 prompts (byte-identical to v12's) are read again as a drift check.
Answers are cached under data/v20/ and collected in results/v20_llm_answers.json."""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor

from recgen.band_v20 import rank_band_examples
from recgen.config import DATA_DIR, RESULTS_DIR
from recgen.corpus import load_corpus
from recgen.llm_reader import unit_prompt
from recgen.read_v12 import ANSWERS_PATH as V12_ANSWERS_PATH
from recgen.read_v12 import fresh_units
from recgen.read_v17 import SpendGuard
from recgen.rubric_v5 import MODELS
from recgen.rubric_v20 import PRIMARY_BAND, RANKS, RETEST, SPEND_LIMIT_USD, WORKERS, rng_for

V20_DIR = DATA_DIR / "v20"
ANSWERS_PATH = RESULTS_DIR / "v20_llm_answers.json"
KINDS = {"band": V20_DIR / "llm", "band2": V20_DIR / "llm_band2", "retest": V20_DIR / "llm_retest",
         "top_reread": V20_DIR / "llm_top_reread"}


def unit_id(model: str, band: str, s: int) -> str:
    first, last = RANKS[band]
    return f"{model}:rank{first}-{last}::{s}"


def rank_units(model: str, band: str) -> list[dict]:
    """v12's fresh neurons of one model that have a sample in band, each with its band examples as keys and its
    v12 top-50 unit as "top"."""
    examples = rank_band_examples(model, band)
    units = []
    for top in fresh_units(model):
        s = top["neurons"][0]
        if examples["included"][s]:
            units.append({"id": unit_id(model, band, s), "model": model, "neurons": (s,), "neuron_id": top["neuron_id"],
                          "quartile": top["quartile"], "keys": examples["keys"][s], "top": top})
    return units


def retest_units(units: list[dict], model: str) -> list[dict]:
    chosen = rng_for(model, 9).choice(len(units), size=RETEST // len(MODELS), replace=False)
    return [units[i] for i in sorted(chosen)]


def jobs_for(model: str, corpus, v12_answers: dict) -> list[tuple[str, str, str]]:
    """(kind, unit id, prompt) for one model's readings of both bands, retests and top-50 re-readings."""
    jobs = []
    for band in RANKS:
        units = rank_units(model, band)
        jobs += [(band, u["id"], unit_prompt(u, corpus)) for u in units]
        if band != PRIMARY_BAND:
            continue
        for u in retest_units(units, model):
            jobs.append(("retest", u["id"], unit_prompt(u, corpus)))
            top_text = unit_prompt(u["top"], corpus)
            assert hashlib.sha256(top_text.encode()).hexdigest() == v12_answers[u["top"]["id"]]["prompt_sha256"]
            jobs.append(("top_reread", u["top"]["id"], top_text))
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
