"""Runs v16's concurrent addendum (recgen/rubric_v16_addendum): v7's rubric and E3, both readers, on
v7's real neurons and 50/50 pools, in one seeded order and with v16's main-run settings.

The CLI calls, cache, retries and spend guard are v16's (recgen/llm_reader_v16). Answers are cached in
data/v16/concurrent/<arm>/llm/, every CLI call's reported cost goes to data/v16/concurrent/spend.jsonl,
and results/v16_concurrent_answers.json collects the final answers for the analysis.

    uv run python -m recgen.llm_reader_v16_addendum [--max-new N]
"""

import argparse
import json

from recgen import llm_reader
from recgen.analyze_v7 import group
from recgen.config import RESULTS_DIR, SEED
from recgen.corpus import load_corpus
from recgen.llm_reader_v16 import V16_DIR, SpendGuard, answer, final_record, read_all
from recgen.rubric_v5 import MODELS
from recgen.rubric_v16_addendum import ARMS, GROUPS, SPEND_CAP_USD, make_prompt, reading_order
from recgen.units_v7 import build_units

CONCURRENT_DIR = V16_DIR / "concurrent"
SPEND_LOG = CONCURRENT_DIR / "spend.jsonl"
ANSWERS_PATH = RESULTS_DIR / "v16_concurrent_answers.json"


def jobs_for(units: list[dict], corpus) -> list[dict]:
    """Every addendum reading of these units, in the pre-registered order."""
    jobs = [{"arm": arm, "id": unit["id"], "group": group(unit), "retest": False,
             "directory": CONCURRENT_DIR / arm / "llm", "text": llm_reader.unit_prompt(unit, corpus, make_prompt(rubric))}
            for unit in units for arm, (_, rubric) in ARMS.items()]
    return [jobs[i] for i in reading_order(len(jobs), SEED)]


def addendum_units() -> list[dict]:
    return [u for size in MODELS for u in build_units(size) if group(u) in GROUPS]


def logged_spend() -> float:
    if not SPEND_LOG.exists():
        return 0.0
    return sum(json.loads(line)["cost_usd"] or 0.0 for line in SPEND_LOG.read_text().splitlines() if line.strip())


def collect(jobs: list[dict]) -> dict:
    """Every final answer, per arm, and the addendum's CLI-reported spend."""
    out = {"arms": {arm: {} for arm in ARMS}, "spend_usd": round(logged_spend(), 4)}
    for j in sorted(jobs, key=lambda j: (j["arm"], j["id"])):
        record = final_record(j)
        if record is not None:
            out["arms"][j["arm"]][j["id"]] = answer(record)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new", type=int, default=None, help="read at most this many uncached jobs")
    args = parser.parse_args()
    jobs = jobs_for(addendum_units(), load_corpus())
    for arm in ARMS:
        (CONCURRENT_DIR / arm / "llm").mkdir(parents=True, exist_ok=True)
    todo = [j for j in jobs if final_record(j) is None][:args.max_new]
    guard = SpendGuard(SPEND_CAP_USD, logged_spend(), SPEND_LOG)
    print(f"{len(jobs)} readings, {len(todo)} to read now, ${guard.spent:.2f} of ${SPEND_CAP_USD:.0f} spent so far",
          flush=True)
    if todo:
        read_all(todo, guard, arms=ARMS)
    answers = collect(jobs)
    ANSWERS_PATH.write_text(json.dumps(answers, indent=1))
    for arm, done in answers["arms"].items():
        print(f"{arm}: {len(done)} final, {sum(a['label'] is None for a in done.values())} without a label; models "
              f"{sorted({m for a in done.values() for m in a['models']})}; "
              f"${sum(a['cost_usd'] or 0 for a in done.values()):.2f}", flush=True)
    print(f"total CLI-reported addendum spend ${answers['spend_usd']:.2f}", flush=True)


if __name__ == "__main__":
    main()
