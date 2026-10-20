"""v21's readings (recgen/rubric_v21): arm A, one-shot Opus under v7's rubric and E3 through v16's
reader, and arm B, the Opus agent under E3 through v8's runner, as v14 ran it.

Caches in data/v21/: arm A in a/<series>/llm/ and llm_retest/ (a/pilot/ for its pilot), with every
arm A CLI call's reported cost in spend.jsonl; arm B in b/agent/, b/agent_retest/, b/pilot/ and
b/requeued/, each reading with its own cost, as in v14. Cached readings are never read again. The
spend guard counts both. results/v21_answers.json collects every reading, both pilots included.

    uv run python -m recgen.reader_v21 [--pilot]

Without --pilot it reads arm A, then arm B (each reading only what isn't cached), then writes the
answers; with --pilot it reads only the pilots.
"""

import argparse
import hashlib
import json
import os
import shutil
import time
from functools import partial
from pathlib import Path

import numpy as np

from recgen import llm_reader
from recgen.agent_v8 import SpendGuard as AgentSpendGuard
from recgen.agent_v8 import read_model
from recgen.analyze_v7 import group
from recgen.config import DATA_DIR, RESULTS_DIR, SEED, WEIGHTS_DIR
from recgen.corpus import load_corpus
from recgen.llm_reader_v16 import SpendGuard, answer, final_record, pilot_units, read_all
from recgen.probe_v8 import UnitView
from recgen.rubric_v5 import MODELS
from recgen.rubric_v7 import DEPTH
from recgen.rubric_v7 import prompt as v7_prompt
from recgen.rubric_v8 import MODELS as AGENT_MODELS
from recgen.rubric_v8 import choose, retest_ids
from recgen.rubric_v16 import prompt as essay_prompt
from recgen.rubric_v21 import (
    ARMS_A, GROUPS_A, PILOT_READINGS_A, PILOT_UNITS_B, READER, RERUNS, SPEND_CAP_USD, V7_RUBRIC, agent_prompt,
    as_v8_prompt, infrastructure_failure, reading_order_a, retest_ids_a,
)
from recgen.sweep import sweep_path
from recgen.sweep_v6 import v6_path
from recgen.units_v7 import build_units

V21_DIR = DATA_DIR / "v21"
A_DIR, B_DIR = V21_DIR / "a", V21_DIR / "b"
A_PILOT_DIR, B_REQUEUED_DIR = A_DIR / "pilot", B_DIR / "requeued"
SPEND_LOG = V21_DIR / "spend.jsonl"
ANSWERS_PATH = RESULTS_DIR / "v21_answers.json"
V8_ANSWERS = (RESULTS_DIR / "v8_agent_answers.json", RESULTS_DIR / "v8_pilot_sonnet_answers.json")
PILOT_MODEL = "160m"  # v8's and v14's arm B pilot size
LOAD_TRIES, LOAD_WAIT_SECONDS = 12, 300  # waiting for memory another job may hold


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def spent() -> float:
    """Every v21 CLI call's reported cost so far: arm A's spend log and every arm B reading on disk."""
    total = 0.0
    if SPEND_LOG.exists():
        total += sum(json.loads(line).get("cost_usd") or 0.0 for line in SPEND_LOG.read_text().splitlines() if line.strip())
    if B_DIR.exists():
        total += sum(json.loads(f.read_text())["cost_usd"] for f in B_DIR.rglob("*.json"))
    return total


def prompt_for(rubric: str):
    """A rubric's one-shot prompt from snippets: v7's exact prompt, or v16's for the essay's rubric."""
    return v7_prompt if rubric == V7_RUBRIC else partial(essay_prompt, rubric)


def job_a(unit: dict, rubric: str, corpus, retest: bool, directory: Path | None = None) -> dict:
    arm = f"opus-{rubric}"
    return {"arm": arm, "id": unit["id"], "group": group(unit), "retest": retest,
            "directory": directory or A_DIR / arm / ("llm_retest" if retest else "llm"),
            "text": llm_reader.unit_prompt(unit, corpus, prompt_for(rubric))}


def arm_a_jobs(corpus) -> list[dict]:
    """Every arm A reading in reading order (rubric_v21.reading_order_a): first readings, then retests."""
    units, again = [], []
    for size in MODELS:
        size_units = [u for u in build_units(size) if group(u) in GROUPS_A]
        units += size_units
        again += retest_ids_a(size_units, size)
    by_id = {u["id"]: u for u in units}
    jobs = []
    for chosen, retest, purpose in ((units, False, 0), ([by_id[k] for k in again], True, 1)):
        jobs += [job_a(chosen[i], rubric, corpus, retest) for i, rubric in reading_order_a(len(chosen), SEED, purpose)]
    return jobs


def pilot_jobs_a(corpus) -> list[dict]:
    """v16's pilot units (pythia-160m, quarter depth), each read once under its rubric_v21 rubric."""
    units = pilot_units()
    return [job_a(next(u for u in units if group(u) == unit_group), rubric, corpus, False, A_PILOT_DIR / f"opus-{rubric}")
            for unit_group, rubric in PILOT_READINGS_A]


def run_a(jobs: list[dict]) -> None:
    todo = [j for j in jobs if final_record(j) is None]
    guard = SpendGuard(SPEND_CAP_USD, spent(), SPEND_LOG)
    print(f"arm A: {len(jobs)} readings, {len(todo)} to read now, ${guard.spent:.2f} of ${SPEND_CAP_USD:.0f} spent "
          "so far", flush=True)
    if todo:
        read_all(todo, guard, arms=ARMS_A, pilot_dir=A_PILOT_DIR)


def requeue(directory: Path, requeued: Path = B_REQUEUED_DIR) -> list[str]:
    """v14's rule: move each cached arm B reading that failed for an infrastructure reason out of the
    cache, unless it was already moved RERUNS times, so it is read again. Returns the moved names."""
    moved = []
    for path in sorted(directory.glob("*.json")):
        if not infrastructure_failure(json.loads(path.read_text())):
            continue
        earlier = list((requeued / directory.name).glob(f"{path.stem}.*.json"))
        if len(earlier) >= RERUNS:
            continue
        target = requeued / directory.name / f"{path.stem}.{len(earlier) + 1}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        path.rename(target)
        moved.append(path.name)
    return moved


def v8_prompt_hashes() -> dict[str, str]:
    """v8's prompt SHA-256 per unit (the same for both of its agents, v14's and the pilot's)."""
    hashes = {}
    for path in V8_ANSWERS:
        answers = json.loads(path.read_text())
        parts = answers["sonnet"].values() if "sonnet" in answers else answers.values()
        hashes |= {k: r["prompt_sha256"] for part in parts for k, r in part.items()}
    return hashes


def check_prompts(size: str, units: list[dict], corpus) -> int:
    """Raise unless every arm B prompt, with E3's decision paragraph swapped back, is v8's prompt for the
    same unit (by hash). Returns the number checked."""
    v6, v5, hashes = dict(np.load(v6_path(size, DEPTH))), np.load(sweep_path(size, DEPTH)), v8_prompt_hashes()
    for u in units:
        text = UnitView(u, v6, v5["max"], v5["pos"], corpus, agent_prompt).prompt()
        if sha256(as_v8_prompt(text)) != hashes[u["id"]]:
            raise RuntimeError(f"{u['id']}: arm B's prompt isn't v8's with E3's decision paragraph")
    return len(units)


def read_size_b(size: str, units: list[dict], jobs: list, folders: dict, guard, corpus) -> list[dict]:
    """read_model for one size, waiting and retrying while Pythia can't get the memory to load. The
    weights are deleted at the end either way (read_model keeps them if loading failed)."""
    try:
        for attempt in range(LOAD_TRIES):
            try:
                return read_model(size, units, jobs, folders, {READER: guard}, corpus, agent_prompt)
            except RuntimeError as e:
                if "out of memory" not in str(e).lower() or attempt == LOAD_TRIES - 1:
                    raise
                print(f"pythia-{size}: out of memory, waiting {LOAD_WAIT_SECONDS} s ({e})", flush=True)
                time.sleep(LOAD_WAIT_SECONDS)
    finally:
        shutil.rmtree(WEIGHTS_DIR / f"pythia-{size}", ignore_errors=True)
    raise AssertionError("unreachable")


def run_b(corpus, pilot: bool) -> dict:
    """Arm B's readings (or its pilot's), as {"first", "retest"}: unit id -> record, in v14's format."""
    folders = ({(READER, reading): B_DIR / "pilot" for reading in ("first", "retest")} if pilot
               else {(READER, "first"): B_DIR / "agent", (READER, "retest"): B_DIR / "agent_retest"})
    for folder in set(folders.values()):
        if folder.exists() and (moved := requeue(folder)):
            print(f"requeued {len(moved)} infrastructure failures from {folder.name}", flush=True)
    guard = AgentSpendGuard(SPEND_CAP_USD, spent=spent())
    print(f"arm B{' pilot' if pilot else ''}: ${guard.spent:.2f} of ${SPEND_CAP_USD:.0f} spent so far", flush=True)
    out = {"first": {}, "retest": {}}
    for size in [PILOT_MODEL] if pilot else AGENT_MODELS:
        units = choose(build_units(size), size)
        if pilot:
            units = [next(u for u in units if (u["kind"], u["mix"]) == kind) for kind in PILOT_UNITS_B]
        print(f"pythia-{size}: {check_prompts(size, units, corpus)} prompts are v8's with E3's decision paragraph",
              flush=True)
        again = set() if pilot else set(retest_ids(units, size))
        jobs = [(u["id"], READER, "first") for u in units]
        jobs += [(u["id"], READER, "retest") for u in units if u["id"] in again]
        for (unit_id, _, reading), record in zip(jobs, read_size_b(size, units, jobs, folders, guard, corpus)):
            out[reading][unit_id] = record
    return out


def cached_pilot_b() -> dict:
    """Arm B's pilot readings from its cache, as {unit id: record}."""
    records = [json.loads(f.read_text()) for f in sorted((B_DIR / "pilot").glob("*.json"))]
    return {r["id"]: r for r in records}


def collect(jobs_a: list[dict], pilot_a: list[dict], arm_b: dict, pilot_b: dict) -> dict:
    out = {"reader": READER, "arm_a": {arm: {"first": {}, "retest": {}} for arm in ARMS_A}, "arm_b": arm_b,
           "pilot": {"arm_a": {arm: {} for arm in ARMS_A}, "arm_b": pilot_b}, "spend_usd": round(spent(), 4)}
    for j in jobs_a:
        if (record := final_record(j)) is not None:
            out["arm_a"][j["arm"]]["retest" if j["retest"] else "first"][j["id"]] = answer(record)
    for j in pilot_a:
        if (record := final_record(j)) is not None:
            out["pilot"]["arm_a"][j["arm"]][j["id"]] = answer(record)
    return out


def report(answers: dict) -> None:
    for arm, parts in answers["arm_a"].items():
        done = list(parts["first"].values()) + list(parts["retest"].values())
        print(f"{arm}: {len(parts['first'])} first readings and {len(parts['retest'])} retests final, "
              f"{sum(a['label'] is None for a in done)} without a label; models "
              f"{sorted({m for a in done for m in a['models']})}; ${sum(a['cost_usd'] or 0 for a in done):.2f}", flush=True)
    done = [r for part in answers["arm_b"].values() for r in part.values()]
    print(f"opus agent-E3: {len(answers['arm_b'].get('first', {}))} first readings and "
          f"{len(answers['arm_b'].get('retest', {}))} retests, {sum(r['label'] is None for r in done)} without a label "
          f"({sum(r['error'] == 'spend cap reached' for r in done)} stopped by the spend guard); "
          f"${sum(r['cost_usd'] for r in done):.2f}", flush=True)
    print(f"pilots: {sum(len(p) for p in answers['pilot']['arm_a'].values())} arm A and {len(answers['pilot']['arm_b'])} "
          f"arm B readings; total CLI-reported spend ${answers['spend_usd']:.2f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true", help="read only the two pilots")
    args = parser.parse_args()
    for key in [k for k in os.environ if "EFFORT" in k.upper()]:  # arm B's CLI inherits this environment
        del os.environ[key]
    corpus = load_corpus()
    pilot_a, jobs_a = pilot_jobs_a(corpus), arm_a_jobs(corpus)
    for j in pilot_a + jobs_a:
        j["directory"].mkdir(parents=True, exist_ok=True)
    if args.pilot:
        run_a(pilot_a)
        pilot_b, arm_b = run_b(corpus, pilot=True)["first"], {}
    else:
        run_a(jobs_a)
        pilot_b, arm_b = cached_pilot_b(), run_b(corpus, pilot=False)
    answers = collect(jobs_a, pilot_a, arm_b, pilot_b)
    ANSWERS_PATH.write_text(json.dumps(answers, indent=1))
    report(answers)


if __name__ == "__main__":
    main()
