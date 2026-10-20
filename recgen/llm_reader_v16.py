"""v16's readers: v7's reader (recgen/llm_reader.py) with the essay's rubrics (recgen/rubric_v16).

Each arm's answers are cached in data/v16/<arm>/llm/ (and llm_retest/ for second readings), keyed
by the prompt's SHA-256 as in v7, so reruns never call the model again. A reading that failed with a
CLI error is read again on the next run; a reply with no valid label after every attempt is final.
Every CLI call's reported cost goes to data/v16/spend.jsonl, which the spend guard sums.
results/v16_answers.json collects every reading, the pilot's included, for the analysis.

    uv run python -m recgen.llm_reader_v16 [--pilot] [--max-new N]
"""

import argparse
import hashlib
import json
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from recgen import llm_reader
from recgen.analyze_v7 import group
from recgen.config import DATA_DIR, RESULTS_DIR, SEED
from recgen.corpus import load_corpus
from recgen.rubric_v5 import MODELS
from recgen.rubric_v16 import (
    ARMS, PILOT_DEPTH, PILOT_MODEL, PILOT_UNITS, RESERVE_USD, SPEND_CAP_USD, parse_label, prompt, reading_order,
)
from recgen.units_v7 import build_units, retest_ids

V16_DIR = DATA_DIR / "v16"
PILOT_DIR = V16_DIR / "pilot"
SPEND_LOG = V16_DIR / "spend.jsonl"
PRECHECK_LOG = V16_DIR / "precheck_caching.jsonl"
ANSWERS_PATH = RESULTS_DIR / "v16_answers.json"
WORKERS = 6
MAX_FAILED = 8  # readings in one run that end in a CLI error before no new reading starts
TIMEOUT_SECONDS = 600
UNPARSEABLE = "unparseable reply"  # llm_reader.read's error when every reply lacked a label


def cli_for(model: str) -> list[str]:
    """v7's CLI flags with only the model changed."""
    cli = list(llm_reader.CLI)
    cli[cli.index("--model") + 1] = model
    return cli


def child_env() -> dict:
    """This environment without effort settings (the CLI's default effort, as in v7) and with the
    CLI's prompt caching off (rubric_v16)."""
    env = {k: v for k, v in os.environ.items() if "EFFORT" not in k.upper()}
    env["DISABLE_PROMPT_CACHING"] = "1"
    return env


def call_cli(model: str, text: str) -> dict:
    """One reading through the claude CLI: the reply, the resolved model, usage and cost."""
    run = subprocess.run(cli_for(model), input=text, capture_output=True, text=True, timeout=TIMEOUT_SECONDS,
                         env=child_env())
    out = json.loads(run.stdout)
    if out.get("is_error"):
        raise RuntimeError(out.get("result", "CLI error"))
    return {"reply": out.get("result", ""), "models": list((out.get("modelUsage") or {}).keys()),
            "usage": out.get("usage", {}), "cost_usd": out.get("total_cost_usd")}


class SpendGuard:
    """Lets a reading start only if the spend so far, plus a reserve for it and each reading in
    flight, stays within the cap. Every CLI call's cost is added as it returns, retries included."""

    def __init__(self, cap: float, spent: float, log_path=None):
        self.cap, self.spent, self.in_flight, self.log_path = cap, spent, 0, log_path
        self.lock = threading.Lock()

    def try_start(self) -> bool:
        with self.lock:
            if self.spent + (self.in_flight + 1) * RESERVE_USD > self.cap:
                return False
            self.in_flight += 1
            return True

    def finish(self) -> None:
        with self.lock:
            self.in_flight -= 1

    def add(self, cost: float, note: dict) -> None:
        with self.lock:
            self.spent += cost
            if self.log_path is not None:
                with open(self.log_path, "a") as f:
                    f.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "cost_usd": cost, **note}) + "\n")


def logged_spend() -> float:
    """Every v16 CLI call's reported cost so far: the caching check and the spend log."""
    total = 0.0
    for path, key in ((PRECHECK_LOG, "cost"), (SPEND_LOG, "cost_usd")):
        if path.exists():
            total += sum(json.loads(line).get(key) or 0.0 for line in path.read_text().splitlines() if line.strip())
    return total


def final_record(job: dict) -> dict | None:
    """The cached answer to this job's prompt if it is final: a label, or a reply with none."""
    path = llm_reader.cache_path(job["directory"], job["id"])
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    if record["prompt_sha256"] != hashlib.sha256(job["text"].encode()).hexdigest():
        return None
    if record["label"] is None and record["error"] != UNPARSEABLE:
        return None
    return record


def job(arm: str, unit: dict, corpus, directory, retest: bool, unit_group: str) -> dict:
    _, rubric = ARMS[arm]
    return {"arm": arm, "id": unit["id"], "group": unit_group, "retest": retest, "directory": directory,
            "text": llm_reader.unit_prompt(unit, corpus, partial(prompt, rubric))}


def main_jobs(corpus) -> list[dict]:
    """Every v16 reading in reading order (rubric_v16.reading_order): Haiku's arms, then Sonnet's,
    each unit's E3 and E4 readings next to each other."""
    units, retests = [], set()
    for size in MODELS:
        size_units = build_units(size)
        units += size_units
        retests |= set(retest_ids(size_units, size))
    readings = [(u, False) for u in units] + [(u, True) for u in units if u["id"] in retests]
    order = reading_order([(group(u), again) for u, again in readings], SEED)
    jobs = []
    for reader in dict.fromkeys(r for r, _ in ARMS.values()):
        for i in order:
            unit, again = readings[i]
            for arm, (r, _) in ARMS.items():
                if r == reader:
                    directory = V16_DIR / arm / ("llm_retest" if again else "llm")
                    jobs.append(job(arm, unit, corpus, directory, again, group(unit)))
    return jobs


def pilot_units() -> list[dict]:
    """PILOT_UNITS units outside v7's sample: the first of each group at PILOT_MODEL's PILOT_DEPTH."""
    chosen = []
    units = build_units(PILOT_MODEL, PILOT_DEPTH)
    for unit_group, n in PILOT_UNITS.items():
        chosen += [u for u in units if group(u) == unit_group][:n]
    tag = f"{PILOT_MODEL}-{PILOT_DEPTH.replace(' ', '-')}"
    return [{**u, "id": tag + u["id"][len(PILOT_MODEL):]} for u in chosen]


def pilot_jobs(corpus) -> list[dict]:
    return [job(arm, unit, corpus, PILOT_DIR / arm, False, group(unit)) for unit in pilot_units() for arm in ARMS]


def read_all(jobs: list[dict], guard: SpendGuard, runner=call_cli, workers: int = WORKERS) -> list[dict | None]:
    """Read jobs in order, while the spend guard allows and until MAX_FAILED of them end in a CLI
    error; None for jobs not started."""
    failed, lock = 0, threading.Lock()

    def one(j: dict) -> dict | None:
        nonlocal failed
        cached = final_record(j)
        if cached is not None:
            return cached
        with lock:
            if failed >= MAX_FAILED:
                return None
        if not guard.try_start():
            return None
        model, rubric = ARMS[j["arm"]]

        def counted(text: str) -> dict:
            out = runner(model, text)
            kind = "pilot" if PILOT_DIR in j["directory"].parents else "retest" if j["retest"] else "first"
            guard.add(out.get("cost_usd") or 0.0, {"arm": j["arm"], "id": j["id"], "kind": kind})
            return out

        try:
            llm_reader.cache_path(j["directory"], j["id"]).unlink(missing_ok=True)  # a failed reading, if any
            record = llm_reader.read(j["id"], j["text"], j["directory"], counted, partial(parse_label, rubric=rubric))
        finally:
            guard.finish()
        if record["label"] is None and record["error"] != UNPARSEABLE:
            with lock:
                failed += 1
        return record

    records = []
    with ThreadPoolExecutor(workers) as pool:
        for i, record in enumerate(pool.map(one, jobs), 1):
            records.append(record)
            if i % 50 == 0:
                print(f"{i} of {len(jobs)}: {sum(r is None for r in records)} not started, "
                      f"{sum(r is not None and r['label'] is None for r in records)} without a label, "
                      f"${guard.spent:.2f} spent", flush=True)
    return records


def answer(record: dict) -> dict:
    usage = record["usage"]
    return {"label": record["label"], "error": record["error"], "models": record["models"],
            "prompt_sha256": record["prompt_sha256"],
            "input_tokens": sum(usage.get(k) or 0 for k in
                                ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")),
            "output_tokens": usage.get("output_tokens") or 0,
            "thinking_tokens": (usage.get("output_tokens_details") or {}).get("thinking_tokens") or 0,
            "cost_usd": record["cost_usd"], "reply": record["reply"]}


def collect(jobs: list[dict], pilot: list[dict]) -> dict:
    """Every final answer, per arm (first readings and retests) and for the pilot."""
    out = {"arms": {arm: {"first": {}, "retest": {}} for arm in ARMS}, "pilot": {arm: {} for arm in ARMS},
           "spend_usd": round(logged_spend(), 4)}
    for j in jobs:
        record = final_record(j)
        if record is not None:
            out["arms"][j["arm"]]["retest" if j["retest"] else "first"][j["id"]] = answer(record)
    for j in pilot:
        record = final_record(j)
        if record is not None:
            out["pilot"][j["arm"]][j["id"]] = answer(record)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true", help="read only the pilot units")
    parser.add_argument("--max-new", type=int, default=None, help="read at most this many uncached jobs")
    args = parser.parse_args()
    corpus = load_corpus()
    pilot = pilot_jobs(corpus)
    jobs = [] if args.pilot else main_jobs(corpus)
    for j in pilot + jobs:
        j["directory"].mkdir(parents=True, exist_ok=True)
    todo = [j for j in (pilot if args.pilot else jobs) if final_record(j) is None][:args.max_new]
    guard = SpendGuard(SPEND_CAP_USD, logged_spend(), SPEND_LOG)
    print(f"{len(pilot) if args.pilot else len(jobs)} readings, {len(todo)} to read now, "
          f"${guard.spent:.2f} of ${SPEND_CAP_USD:.0f} spent so far", flush=True)
    if todo:
        read_all(todo, guard)
    answers = collect(main_jobs(corpus) if args.pilot else jobs, pilot)
    ANSWERS_PATH.write_text(json.dumps(answers, indent=1))
    for arm, parts in answers["arms"].items():
        done = list(parts["first"].values()) + list(parts["retest"].values())
        print(f"{arm}: {len(parts['first'])} first readings and {len(parts['retest'])} retests final, "
              f"{sum(a['label'] is None for a in done)} without a label; models "
              f"{sorted({m for a in done for m in a['models']})}; ${sum(a['cost_usd'] or 0 for a in done):.2f}", flush=True)
    print(f"pilot: {sum(len(p) for p in answers['pilot'].values())} readings; total CLI-reported spend "
          f"${answers['spend_usd']:.2f}", flush=True)


if __name__ == "__main__":
    main()
