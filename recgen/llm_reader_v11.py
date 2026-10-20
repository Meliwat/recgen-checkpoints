"""v11's language-model reader: v7's reader (recgen/llm_reader.py, with the same prompt, CLI flags,
retries and prompt-hash cache) on v11's units at the quarter and three-quarters depths.

Every answer is cached in data/v11/llm/ (and data/v11/llm_retest/ for second readings), so reruns
never call the model again. results/v11_llm_answers.json collects them by depth for the analysis.
"""

import json
from concurrent.futures import ThreadPoolExecutor

from recgen.config import DATA_DIR, RESULTS_DIR
from recgen.corpus import load_corpus
from recgen.llm_reader import read, unit_prompt
from recgen.rubric_v5 import MODELS
from recgen.rubric_v11 import DEPTHS
from recgen.units_v11 import build_units, retest_ids

LLM_DIR = DATA_DIR / "v11" / "llm"
RETEST_DIR = DATA_DIR / "v11" / "llm_retest"
ANSWERS_PATH = RESULTS_DIR / "v11_llm_answers.json"
WORKERS = 4


def reading_jobs(corpus) -> list[tuple[str, str, str, object]]:
    """(depth, unit id, prompt, cache directory) for every first reading and retest."""
    jobs = []
    for depth in DEPTHS:
        for model in MODELS:
            units = build_units(model, depth)
            again = set(retest_ids(units, model))
            for unit in units:
                text = unit_prompt(unit, corpus)
                jobs.append((depth, unit["id"], text, LLM_DIR))
                if unit["id"] in again:
                    jobs.append((depth, unit["id"], text, RETEST_DIR))
    return jobs


def collect(jobs: list[tuple], records: list[dict]) -> dict:
    """The answers by depth, then first reading or retest, then unit id, in v7's format."""
    answers = {depth: {"first": {}, "retest": {}} for depth in DEPTHS}
    for (depth, unit_id, _, directory), record in zip(jobs, records):
        answers[depth]["first" if directory == LLM_DIR else "retest"][unit_id] = {
            "label": record["label"], "error": record["error"], "models": record["models"],
            "prompt_sha256": record["prompt_sha256"],
            "input_tokens": sum(record["usage"].get(k) or 0 for k in
                                ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")),
            "reply": record["reply"]}
    return answers


def main() -> None:
    corpus = load_corpus()
    LLM_DIR.mkdir(parents=True, exist_ok=True)
    RETEST_DIR.mkdir(parents=True, exist_ok=True)
    jobs = reading_jobs(corpus)
    print(f"{len(jobs)} readings ({sum(1 for job in jobs if job[3] == RETEST_DIR)} retests)", flush=True)
    with ThreadPoolExecutor(WORKERS) as pool:
        records = []
        for i, record in enumerate(pool.map(lambda job: read(*job[1:]), jobs), 1):
            records.append(record)
            if i % 100 == 0:
                print(f"{i} of {len(jobs)} read, {sum(r['label'] is None for r in records)} without a label", flush=True)
    answers = collect(jobs, records)
    ANSWERS_PATH.write_text(json.dumps(answers, indent=1))
    missing = sum(a["label"] is None for part in answers.values() for a in part["first"].values())
    print(f"done: {missing} first readings without a label", flush=True)


if __name__ == "__main__":
    main()
