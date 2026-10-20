"""v9's language-model readings: Haiku on every primary union, through v7's cached reader and prompt.
Real neurons reuse v7's readings. Answers are cached under data/v9/ and collected in
results/v9_llm_answers.json."""

import json
from concurrent.futures import ThreadPoolExecutor

from recgen.config import DATA_DIR, RESULTS_DIR
from recgen.corpus import load_corpus
from recgen.llm_reader import read, unit_prompt
from recgen.rubric_v5 import MODELS
from recgen.rubric_v9 import RETEST_PER_MODEL, rng_for
from recgen.units_v9 import build_unions

LLM_DIR = DATA_DIR / "v9" / "llm"
RETEST_DIR = DATA_DIR / "v9" / "llm_retest"
ANSWERS_PATH = RESULTS_DIR / "v9_llm_answers.json"
WORKERS = 4  # other sessions use the same CLI quota tonight


def main() -> None:
    corpus = load_corpus()
    LLM_DIR.mkdir(parents=True, exist_ok=True)
    RETEST_DIR.mkdir(parents=True, exist_ok=True)
    jobs = []
    for model in MODELS:
        units, _ = build_unions(model)
        again = {units[i]["id"] for i in rng_for(model, 9).choice(len(units), size=RETEST_PER_MODEL, replace=False)}
        for unit in units:
            text = unit_prompt(unit, corpus)
            jobs.append((unit["id"], text, LLM_DIR))
            if unit["id"] in again:
                jobs.append((unit["id"], text, RETEST_DIR))
    print(f"{len(jobs)} readings", flush=True)
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
            "prompt_sha256": record["prompt_sha256"], "reply": record["reply"],
            "input_tokens": sum(record["usage"].get(k) or 0 for k in
                                ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))}
    ANSWERS_PATH.write_text(json.dumps(answers, indent=1))
    print(f"done: {sum(a['label'] is None for a in answers['first'].values())} without a label", flush=True)


if __name__ == "__main__":
    main()
