"""v10's language-model readers: v7's reader (recgen/llm_reader.py) with the model changed. The CLI flags
are otherwise v7's. The run differs only in how it is driven: four workers instead of six, a longer
timeout, CLAUDE_CODE_EFFORT_LEVEL removed from the CLI's environment, and failed readings retried.

Answers are cached in data/v10/<model>/llm/ (and llm_retest/ for second readings), keyed by the
prompt's SHA-256 as in v7, so reruns never call the model again. A reading that failed with a CLI
error (a rate limit, a timeout) is read again on the next run; a reply with no valid label after
every attempt is final. results/v10_<model>_answers.json collects the answers for the analysis.

    uv run python -m recgen.llm_reader_v10 sonnet [--max-new 300]
"""

import argparse
import hashlib
import json
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor

from recgen import llm_reader
from recgen.analyze_v7 import group
from recgen.config import DATA_DIR, RESULTS_DIR, SEED
from recgen.corpus import load_corpus
from recgen.rubric_v5 import MODELS
from recgen.rubric_v10 import READERS, reading_order
from recgen.units_v7 import build_units, retest_ids

V10_DIR = DATA_DIR / "v10"
WORKERS = 4
MAX_FAILED = 8  # readings in one run that end in a CLI error before no new reading starts
TIMEOUT_SECONDS = 600
UNPARSEABLE = "unparseable reply"  # llm_reader.read's error when every reply lacked a label


def answers_path(model: str):
    return RESULTS_DIR / f"v10_{model}_answers.json"


def cli_for(model: str) -> list[str]:
    """v7's CLI flags with only the model changed."""
    cli = list(llm_reader.CLI)
    cli[cli.index("--model") + 1] = model
    return cli


def child_env() -> dict:
    """This environment without an effort override, so the CLI uses its default effort, as in v7."""
    return {k: v for k, v in os.environ.items() if k != "CLAUDE_CODE_EFFORT_LEVEL"}


def call_cli(model: str, text: str) -> dict:
    """One reading through the claude CLI: the reply, the resolved model, and token usage."""
    run = subprocess.run(cli_for(model), input=text, capture_output=True, text=True, timeout=TIMEOUT_SECONDS,
                         env=child_env())
    out = json.loads(run.stdout)
    if out.get("is_error"):
        raise RuntimeError(out.get("result", "CLI error"))
    return {"reply": out.get("result", ""), "models": list((out.get("modelUsage") or {}).keys()),
            "usage": out.get("usage", {}), "cost_usd": out.get("total_cost_usd")}


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


def jobs_for(model: str, corpus) -> list[dict]:
    """Every reading of one reader, in v7's unit order: each unit once, and the retest units twice."""
    first, retest = V10_DIR / model / "llm", V10_DIR / model / "llm_retest"
    jobs = []
    for size in MODELS:
        units = build_units(size)
        again = set(retest_ids(units, size))
        for unit in units:
            text = llm_reader.unit_prompt(unit, corpus)
            jobs.append({"id": unit["id"], "group": group(unit), "text": text, "directory": first, "retest": False})
            if unit["id"] in again:
                jobs.append({"id": unit["id"], "group": group(unit), "text": text, "directory": retest, "retest": True})
    return jobs


def read_all(jobs: list[dict], runner, workers: int = WORKERS) -> list[dict | None]:
    """Read jobs in order, until MAX_FAILED of them end in a CLI error; None for jobs not started."""
    failed, lock = 0, threading.Lock()

    def one(job: dict) -> dict | None:
        nonlocal failed
        with lock:
            if failed >= MAX_FAILED:
                return None
        cached = final_record(job)
        if cached is not None:
            return cached
        llm_reader.cache_path(job["directory"], job["id"]).unlink(missing_ok=True)  # a failed reading, if any
        record = llm_reader.read(job["id"], job["text"], job["directory"], runner)
        if record["label"] is None and record["error"] != UNPARSEABLE:
            with lock:
                failed += 1
        return record

    records = []
    with ThreadPoolExecutor(workers) as pool:
        for i, record in enumerate(pool.map(one, jobs), 1):
            records.append(record)
            if i % 25 == 0:
                print(f"{i} of {len(jobs)} read, {sum(r is not None and r['label'] is None for r in records)} "
                      f"without a label", flush=True)
    return records


def collect(model: str, jobs: list[dict]) -> dict:
    """Every final answer of one reader, in v7's answers format plus output tokens and cost."""
    answers = {"first": {}, "retest": {}}
    for job in jobs:
        record = final_record(job)
        if record is None:
            continue
        usage = record["usage"]
        answers["retest" if job["retest"] else "first"][job["id"]] = {
            "label": record["label"], "error": record["error"], "models": record["models"],
            "prompt_sha256": record["prompt_sha256"],
            "input_tokens": sum(usage.get(k) or 0 for k in
                                ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")),
            "output_tokens": usage.get("output_tokens") or 0,
            "thinking_tokens": (usage.get("output_tokens_details") or {}).get("thinking_tokens") or 0,
            "cost_usd": record["cost_usd"], "reply": record["reply"]}
    return answers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=READERS)
    parser.add_argument("--max-new", type=int, default=None, help="read at most this many uncached jobs")
    args = parser.parse_args()
    jobs = jobs_for(args.model, load_corpus())
    for job in jobs:
        job["directory"].mkdir(parents=True, exist_ok=True)
    order = reading_order([(job["group"], job["retest"]) for job in jobs], SEED)
    todo = [jobs[i] for i in order if final_record(jobs[i]) is None]
    print(f"{args.model}: {len(jobs)} readings, {len(jobs) - len(todo)} cached, "
          f"reading {len(todo[:args.max_new])} now", flush=True)
    read_all(todo[:args.max_new], lambda text: call_cli(args.model, text))
    answers = collect(args.model, jobs)
    answers_path(args.model).write_text(json.dumps(answers, indent=1))
    done = list(answers["first"].values()) + list(answers["retest"].values())
    print(f"{args.model}: {len(answers['first'])} of {sum(not j['retest'] for j in jobs)} first readings and "
          f"{len(answers['retest'])} of {sum(j['retest'] for j in jobs)} retests final, "
          f"{sum(a['label'] is None for a in done)} without a label; models "
          f"{sorted({m for a in done for m in a['models']})}; {sum(a['input_tokens'] for a in done)} input and "
          f"{sum(a['output_tokens'] for a in done)} output tokens, "
          f"${sum(a['cost_usd'] or 0 for a in done):.2f} by the CLI's count", flush=True)


if __name__ == "__main__":
    main()
