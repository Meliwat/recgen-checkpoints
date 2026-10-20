"""v7's language-model reader: each unit's examples, shown blind, labeled through the claude CLI.

Every answer is cached in data/v7/llm/ (and data/v7/llm_retest/ for second readings), so reruns
never call the model again. results/v7_llm_answers.json collects them for the analysis.
"""

import hashlib
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from recgen.config import DATA_DIR, RESULTS_DIR, SEED
from recgen.corpus import load_corpus
from recgen.facets import display_text
from recgen.rubric_v5 import MODELS
from recgen.rubric_v7 import LLM_MODEL, SNIPPET_AFTER, SNIPPET_BEFORE, SYSTEM_PROMPT, parse_label, prompt
from recgen.units_v7 import build_units, retest_ids

LLM_DIR = DATA_DIR / "v7" / "llm"
RETEST_DIR = DATA_DIR / "v7" / "llm_retest"
ANSWERS_PATH = RESULTS_DIR / "v7_llm_answers.json"
WORKERS = 6
ATTEMPTS = 4
BACKOFF_SECONDS = (10, 30, 90)
CLI = ["claude", "-p", "--model", LLM_MODEL, "--system-prompt", SYSTEM_PROMPT, "--tools", "", "--strict-mcp-config",
       "--setting-sources", "", "--no-session-persistence", "--output-format", "json"]


def unit_prompt(unit: dict, corpus: np.ndarray) -> str:
    """The unit's examples in a fixed random order (seeded by its id), with no activation values."""
    seed = int(hashlib.sha256(unit["id"].encode()).hexdigest()[:8], 16)
    order = np.random.default_rng([SEED, 7, seed]).permutation(len(unit["keys"]))
    return prompt([display_text(corpus, int(w), int(p), SNIPPET_BEFORE, SNIPPET_AFTER) for w, p in unit["keys"][order]])


def call_cli(text: str) -> dict:
    """One reading through the claude CLI: the reply, the resolved model, and token usage."""
    run = subprocess.run(CLI, input=text, capture_output=True, text=True, timeout=300)
    out = json.loads(run.stdout)
    if out.get("is_error"):
        raise RuntimeError(out.get("result", "CLI error"))
    return {"reply": out.get("result", ""), "models": list((out.get("modelUsage") or {}).keys()),
            "usage": out.get("usage", {}), "cost_usd": out.get("total_cost_usd")}


def cache_path(directory, unit_id: str):
    return directory / (unit_id.replace(":", "_").replace("/", "-") + ".json")


def read(unit_id: str, text: str, directory, runner=call_cli) -> dict:
    """The cached answer for this prompt, or a new reading with retries."""
    path = cache_path(directory, unit_id)
    digest = hashlib.sha256(text.encode()).hexdigest()
    if path.exists():
        cached = json.loads(path.read_text())
        if cached["prompt_sha256"] == digest:
            return cached
    error = None
    for attempt in range(ATTEMPTS):
        try:
            answer = runner(text)
            label = parse_label(answer["reply"])
            if label is not None:
                break
            error = "unparseable reply"
        except (RuntimeError, json.JSONDecodeError, subprocess.TimeoutExpired) as e:
            answer, label, error = {"reply": "", "models": [], "usage": {}, "cost_usd": None}, None, str(e)
        if attempt < ATTEMPTS - 1:
            time.sleep(BACKOFF_SECONDS[min(attempt, len(BACKOFF_SECONDS) - 1)])
    record = {"id": unit_id, "prompt_sha256": digest, "label": label, "error": None if label else error, **answer}
    path.write_text(json.dumps(record))
    return record


def main() -> None:
    corpus = load_corpus()
    LLM_DIR.mkdir(parents=True, exist_ok=True)
    RETEST_DIR.mkdir(parents=True, exist_ok=True)
    jobs = []
    for model in MODELS:
        units = build_units(model)
        again = set(retest_ids(units, model))
        for unit in units:
            text = unit_prompt(unit, corpus)
            jobs.append((unit["id"], text, LLM_DIR))
            if unit["id"] in again:
                jobs.append((unit["id"], text, RETEST_DIR))
    print(f"{len(jobs)} readings ({sum(1 for _, _, d in jobs if d == RETEST_DIR)} retests)", flush=True)
    with ThreadPoolExecutor(WORKERS) as pool:
        records = []
        for i, record in enumerate(pool.map(lambda job: read(*job), jobs), 1):
            records.append(record)
            if i % 100 == 0:
                print(f"{i} of {len(jobs)} read, {sum(r['label'] is None for r in records)} without a label", flush=True)
    answers = {"first": {}, "retest": {}}
    for (unit_id, _, directory), record in zip(jobs, records):
        answers["first" if directory == LLM_DIR else "retest"][unit_id] = {
            "label": record["label"], "error": record["error"], "models": record["models"],
            "prompt_sha256": record["prompt_sha256"],
            "input_tokens": sum(record["usage"].get(k) or 0 for k in
                                ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")),
            "reply": record["reply"]}
    ANSWERS_PATH.write_text(json.dumps(answers, indent=1))
    print(f"done: {sum(a['label'] is None for a in answers['first'].values())} first readings without a label", flush=True)


if __name__ == "__main__":
    main()
