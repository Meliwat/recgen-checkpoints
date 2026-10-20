"""v17's Haiku readings: v12's 600 fresh middle-layer neurons, each shown as its 50 band examples
(recgen/band_v17.py) through v7's prompt and cached CLI reader. RETEST of them are read a second time,
and the same neurons' top-50 prompts (byte-identical to v12's) are read again as a drift check.
Answers are cached under data/v17/ and collected in results/v17_llm_answers.json."""

import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor

from recgen.band_v17 import band_examples
from recgen.config import DATA_DIR, RESULTS_DIR
from recgen.corpus import load_corpus
from recgen.llm_reader import call_cli, read, unit_prompt
from recgen.read_v12 import ANSWERS_PATH as V12_ANSWERS_PATH
from recgen.read_v12 import fresh_units
from recgen.rubric_v5 import MODELS
from recgen.rubric_v17 import RETEST, SPEND_LIMIT_USD, rng_for

LLM_DIR = DATA_DIR / "v17" / "llm"
RETEST_DIR = DATA_DIR / "v17" / "llm_retest"
TOP_REREAD_DIR = DATA_DIR / "v17" / "llm_top_reread"
ANSWERS_PATH = RESULTS_DIR / "v17_llm_answers.json"
WORKERS = 4  # other sessions share the CLI tonight; at most 6 calls at once in all
KINDS = {"band": LLM_DIR, "retest": RETEST_DIR, "top_reread": TOP_REREAD_DIR}


def band_units(model: str) -> list[dict]:
    """v12's fresh neurons of one model that have a band sample, each with its band examples as keys and
    its v12 top-50 unit as "top"."""
    band = band_examples(model)
    units = []
    for top in fresh_units(model):
        s = top["neurons"][0]
        if band["included"][s]:
            units.append({"id": f"{model}:band::{s}", "model": model, "neurons": (s,), "neuron_id": top["neuron_id"],
                          "quartile": top["quartile"], "keys": band["keys"][s], "top": top})
    return units


def retest_units(units: list[dict], model: str) -> list[dict]:
    chosen = rng_for(model, 9).choice(len(units), size=RETEST // len(MODELS), replace=False)
    return [units[i] for i in sorted(chosen)]


class SpendGuard:
    """llm_reader.read, which stops calling the CLI once the CLI-reported cost of the readings so far
    (cached ones included) reaches the limit. Calls already running may overshoot it slightly."""

    def __init__(self, limit: float, runner=call_cli):
        self.limit, self.runner, self.spent, self.lock = limit, runner, 0.0, threading.Lock()

    def read(self, unit_id: str, text: str, directory) -> dict | None:
        with self.lock:
            if self.spent >= self.limit:
                return None
        record = read(unit_id, text, directory, self.runner)
        with self.lock:
            self.spent += record.get("cost_usd") or 0
        return record


def jobs_for(model: str, corpus, v12_answers: dict) -> list[tuple[str, str, str]]:
    """(kind, unit id, prompt) for one model's band readings, retests and top-50 re-readings."""
    units = band_units(model)
    jobs = [("band", u["id"], unit_prompt(u, corpus)) for u in units]
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
    for (kind, unit_id, _), record in zip(jobs, records):
        if record is None:
            answers[kind][unit_id] = {"label": None, "error": "spend limit reached"}
            continue
        answers[kind][unit_id] = {
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
