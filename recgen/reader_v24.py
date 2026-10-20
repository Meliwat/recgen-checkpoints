"""v24's readings (recgen/rubric_v24): Haiku, Sonnet and Opus, one shot under v7's rubric and E3, on v20's
100 random-text units and 100 real neurons and 100 50/50 pools drawn from v7's units, through v16's reader.

Caches in data/v24/: <series>/llm/ and llm_retest/, pilot/<series>/, and every CLI call's reported cost in
spend.jsonl, which the spend guard sums. Cached readings are never read again. results/v24_answers.json
collects every reading, the pilot's included.

    uv run python -m recgen.reader_v24 [--pilot]
"""

import argparse
import hashlib
import json
import subprocess

import numpy as np

from recgen import llm_reader
from recgen.analyze_v7 import group as v7_group
from recgen.config import DATA_DIR, RESULTS_DIR
from recgen.corpus import load_corpus
from recgen.llm_reader_v16 import SpendGuard, answer, final_record, pilot_units, read_all
from recgen.reader_v21 import prompt_for
from recgen.rubric_v5 import MODELS
from recgen.rubric_v24 import (
    ARMS, DEPTH, GROUPS, PILOT_DEPTH, PILOT_MODEL, PILOT_RANDOM_PURPOSE, POOL, RANDOM, READERS, REAL, SPEND_CAP_USD,
    V7_RUBRIC, WORKERS, anchor_ids, per_size_counts, random_rng, random_unit_keys, reading_order, retest_ids,
)
from recgen.sweep_v6 import v6_path
from recgen.units_v7 import build_units

V24_DIR = DATA_DIR / "v24"
PILOT_DIR = V24_DIR / "pilot"
SPEND_LOG = V24_DIR / "spend.jsonl"
ANSWERS_PATH = RESULTS_DIR / "v24_answers.json"
V20_ANSWERS = "v20:results/v20_posthoc_llm_answers.json"  # read from git, branch v20


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def spent() -> float:
    """Every v24 CLI call's reported cost so far, from the spend log."""
    if not SPEND_LOG.exists():
        return 0.0
    return sum(json.loads(line).get("cost_usd") or 0.0 for line in SPEND_LOG.read_text().splitlines() if line.strip())


def v20_random_answers() -> dict:
    """v20's post hoc Haiku readings of its random-text units (unit id -> record), from branch v20."""
    text = subprocess.run(["git", "show", V20_ANSWERS], capture_output=True, text=True, check=True).stdout
    return json.loads(text)["random"]


def random_units(depth: str = DEPTH) -> list[dict]:
    """v20's 100 random-text units, rebuilt: per size, 50 of v6's random positions each, with v20's seed."""
    units = []
    for size, n in per_size_counts().items():
        v6 = np.load(v6_path(size, depth))
        positions = v6["keys"][v6["random_rows"]]
        units += [{"id": f"{size}:random::{i}", "group": RANDOM, "keys": keys}
                  for i, keys in enumerate(random_unit_keys(positions, n, random_rng(size)))]
    return units


def anchor_units() -> list[dict]:
    """100 of v7's real neurons and 100 of its 50/50 pools, the same per-size counts as the random units."""
    units = []
    for size, n in per_size_counts().items():
        size_units = build_units(size)
        for g in (REAL, POOL):
            in_group = [u for u in size_units if v7_group(u) == g]
            chosen = set(anchor_ids([u["id"] for u in in_group], size, g, n))
            units += [{**u, "group": g} for u in in_group if u["id"] in chosen]
    return units


def check_random_prompts(units: list[dict], corpus, v20: dict) -> int:
    """Raise unless every random-text unit's v7-rubric prompt hashes to v20's for the same unit."""
    for u in units:
        if sha256(llm_reader.unit_prompt(u, corpus, prompt_for(V7_RUBRIC))) != v20[u["id"]]["prompt_sha256"]:
            raise RuntimeError(f"{u['id']}: the rebuilt random-text unit isn't v20's")
    return len(units)


def job(arm: str, unit: dict, corpus, retest: bool, directory=None) -> dict:
    _, rubric = ARMS[arm]
    return {"arm": arm, "id": unit["id"], "group": unit["group"], "retest": retest,
            "directory": directory or V24_DIR / arm / ("llm_retest" if retest else "llm"),
            "text": llm_reader.unit_prompt(unit, corpus, prompt_for(rubric))}


def main_jobs(units: list[dict], corpus) -> list[dict]:
    """Every first reading in one seeded random order over all six series, then every retest the same way."""
    first = [(u, arm) for u in units for arm in ARMS]
    again_ids = set(retest_ids({g: [u["id"] for u in units if u["group"] == g] for g in GROUPS}))
    again = [(u, arm) for u in units if u["id"] in again_ids for arm in ARMS]
    jobs = [job(arm, u, corpus, False) for u, arm in (first[i] for i in reading_order(len(first)))]
    return jobs + [job(arm, u, corpus, True) for u, arm in (again[i] for i in reading_order(len(again), 1))]


def pilot_unit_list() -> list[dict]:
    """Two units outside the sample: a random-text unit from pythia-160m's quarter-depth random positions
    with another seed, and v16's first pilot real neuron (pythia-160m, quarter depth)."""
    v6 = np.load(v6_path(PILOT_MODEL, PILOT_DEPTH))
    keys = random_unit_keys(v6["keys"][v6["random_rows"]], 1, random_rng(PILOT_MODEL, PILOT_RANDOM_PURPOSE))[0]
    tag = f"{PILOT_MODEL}-{PILOT_DEPTH}"
    real = next(u for u in pilot_units() if v7_group(u) == REAL)
    return [{"id": f"{tag}:random::0", "group": RANDOM, "keys": keys}, {**real, "group": REAL}]


def pilot_jobs(corpus) -> list[dict]:
    return [job(arm, u, corpus, False, PILOT_DIR / arm) for u in pilot_unit_list() for arm in ARMS]


def collect(jobs: list[dict], pilot: list[dict], checked: int) -> dict:
    out = {"readers": READERS, "series": {arm: {"first": {}, "retest": {}} for arm in ARMS},
           "pilot": {arm: {} for arm in ARMS}, "random_prompts_matching_v20": checked, "spend_usd": round(spent(), 4)}
    for j in jobs:
        if (record := final_record(j)) is not None:
            out["series"][j["arm"]]["retest" if j["retest"] else "first"][j["id"]] = answer(record)
    for j in pilot:
        if (record := final_record(j)) is not None:
            out["pilot"][j["arm"]][j["id"]] = answer(record)
    return out


def report(answers: dict) -> None:
    for arm, parts in answers["series"].items():
        done = list(parts["first"].values()) + list(parts["retest"].values())
        print(f"{arm}: {len(parts['first'])} first readings and {len(parts['retest'])} retests final, "
              f"{sum(a['label'] is None for a in done)} without a label; models "
              f"{sorted({m for a in done for m in a['models']})}; ${sum(a['cost_usd'] or 0 for a in done):.2f}", flush=True)
    print(f"pilot: {sum(len(p) for p in answers['pilot'].values())} readings; total CLI-reported spend "
          f"${answers['spend_usd']:.2f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true", help="read only the pilot")
    args = parser.parse_args()
    corpus = load_corpus()
    randoms = random_units()
    checked = check_random_prompts(randoms, corpus, v20_random_answers())
    print(f"{checked} random-text units rebuilt, their v7-rubric prompts all v20's", flush=True)
    pilot, jobs = pilot_jobs(corpus), main_jobs(randoms + anchor_units(), corpus)
    for j in pilot + jobs:
        j["directory"].mkdir(parents=True, exist_ok=True)
    todo = [j for j in (pilot if args.pilot else jobs) if final_record(j) is None]
    guard = SpendGuard(SPEND_CAP_USD, spent(), SPEND_LOG)
    print(f"{len(pilot) if args.pilot else len(jobs)} readings, {len(todo)} to read now, "
          f"${guard.spent:.2f} of ${SPEND_CAP_USD:.0f} spent so far", flush=True)
    if todo:
        read_all(todo, guard, workers=WORKERS, arms=ARMS, pilot_dir=PILOT_DIR)
    answers = collect(jobs, pilot, checked)
    ANSWERS_PATH.write_text(json.dumps(answers, indent=1))
    report(answers)


if __name__ == "__main__":
    main()
