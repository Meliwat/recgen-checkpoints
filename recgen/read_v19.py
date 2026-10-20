"""v19's Haiku readings: every unit of each GPT-2 size, shown blind with v7's prompt, system prompt and CLI
(recgen.llm_reader). Answers are cached in data/v19/llm/ (second readings in data/v19/llm_retest/) and
collected in results/v19_llm_answers.json with the cost the CLI reported. No new reading starts once the
reported cost, cached readings included, reaches the spend guard."""

import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from recgen.config import RESULTS_DIR, SEED
from recgen.gpt2_v19 import V19_DIR, display_text, load_gpt2_corpus
from recgen.llm_reader import call_cli, read
from recgen.rubric_v7 import SNIPPET_AFTER, SNIPPET_BEFORE, prompt
from recgen.rubric_v19 import MODELS, SPEND_GUARD_USD
from recgen.units_v19 import build_units, retest_ids

LLM_DIR = V19_DIR / "llm"
RETEST_DIR = V19_DIR / "llm_retest"
ANSWERS_PATH = RESULTS_DIR / "v19_llm_answers.json"
WORKERS = 6


def unit_prompt(unit: dict, corpus: np.ndarray) -> str:
    """As llm_reader.unit_prompt, with GPT-2's tokens: the unit's examples in a fixed random order (seeded by
    its id), with no activation values."""
    seed = int(hashlib.sha256(unit["id"].encode()).hexdigest()[:8], 16)
    order = np.random.default_rng([SEED, 7, seed]).permutation(len(unit["keys"]))
    return prompt([display_text(corpus, int(w), int(p), SNIPPET_BEFORE, SNIPPET_AFTER) for w, p in unit["keys"][order]])


class SpendGuardReached(Exception):
    pass


class Budget:
    """Wraps the CLI so that no call starts once the reported spend reaches the limit."""

    def __init__(self, limit: float, spent: float):
        self.limit, self.spent, self.lock = limit, spent, threading.Lock()

    def runner(self, text: str) -> dict:
        with self.lock:
            if self.spent >= self.limit:
                raise SpendGuardReached
        answer = call_cli(text)
        with self.lock:
            self.spent += answer["cost_usd"] or 0
        return answer


def cached_cost(directories) -> float:
    return sum(json.loads(f.read_text()).get("cost_usd") or 0 for d in directories for f in d.glob("*.json"))


def read_all(units: dict[str, list[dict]], corpus: np.ndarray, llm_dir, retest_dir, guard: float) -> dict:
    """Every unit of each model, and the retested ones again, read or taken from the caches in llm_dir and retest_dir,
    with no new reading once the reported spend reaches guard. The answers, in v19_llm_answers.json's format."""
    llm_dir.mkdir(parents=True, exist_ok=True)
    retest_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for model, model_units in units.items():
        again = set(retest_ids(model_units, model))
        for unit in model_units:
            text = unit_prompt(unit, corpus)
            jobs.append((unit["id"], text, llm_dir))
            if unit["id"] in again:
                jobs.append((unit["id"], text, retest_dir))
    budget = Budget(guard, cached_cost((llm_dir, retest_dir)))
    print(f"{len(jobs)} readings ({sum(d == retest_dir for _, _, d in jobs)} retests); "
          f"${budget.spent:.2f} already spent", flush=True)

    def job(args):
        try:
            return read(*args, runner=budget.runner)
        except SpendGuardReached:
            return None

    records = []
    with ThreadPoolExecutor(WORKERS) as pool:
        for i, record in enumerate(pool.map(job, jobs), 1):
            records.append(record)
            if i % 100 == 0:
                print(f"{i} of {len(jobs)}: ${budget.spent:.2f} spent, "
                      f"{sum(r is None or r['label'] is None for r in records)} without a label", flush=True)
    answers = {"first": {}, "retest": {}, "not_read": []}
    for (unit_id, _, directory), record in zip(jobs, records):
        part = "first" if directory == llm_dir else "retest"
        if record is None:
            answers["not_read"].append(f"{part} {unit_id}")
            continue
        answers[part][unit_id] = {
            "label": record["label"], "error": record["error"], "models": record["models"],
            "prompt_sha256": record["prompt_sha256"], "cost_usd": record["cost_usd"],
            "input_tokens": sum(record["usage"].get(k) or 0 for k in
                                ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")),
            "reply": record["reply"]}
    answers["cost_usd"] = round(sum(a["cost_usd"] or 0 for part in ("first", "retest") for a in answers[part].values()), 4)
    return answers


def main() -> None:
    corpus = load_gpt2_corpus()
    answers = read_all({model: build_units(model) for model in MODELS}, corpus, LLM_DIR, RETEST_DIR, SPEND_GUARD_USD)
    ANSWERS_PATH.write_text(json.dumps(answers, indent=1))
    print(f"done: ${answers['cost_usd']:.2f}; {len(answers['not_read'])} not read (spend guard), "
          f"{sum(a['label'] is None for a in answers['first'].values())} first readings without a label", flush=True)


if __name__ == "__main__":
    main()
