"""v14's Opus agent: v8's runner (recgen/agent_v8.read_model) with only the model changed
(recgen/rubric_v14).

Readings are cached in data/v14/ (agent_opus/, agent_opus_retest/, and pilot/ for the pilot),
keyed by the prompt as in v8, so reruns never call Pythia or the CLI again. Before a run, each
cached reading that failed for an infrastructure reason is moved to data/v14/requeued/, so it is
read again (at most RERUNS times). results/v14_agent_answers.json collects the readings in v8's
format, and results/v14_pilot_answers.json the pilot's.

    uv run python -m recgen.agent_v14 [--pilot]
"""

import argparse
import json
from pathlib import Path

from recgen.agent_v8 import SpendGuard, read_model
from recgen.config import DATA_DIR, RESULTS_DIR
from recgen.corpus import load_corpus
from recgen.rubric_v8 import MODELS, choose, retest_ids
from recgen.rubric_v14 import AGENT_MODEL, PILOT_MODEL, RERUNS, SPEND_CAP_USD, infrastructure_failure, pilot_units
from recgen.units_v7 import build_units

V14_DIR = DATA_DIR / "v14"
REQUEUED_DIR = V14_DIR / "requeued"
ANSWERS_PATH = RESULTS_DIR / "v14_agent_answers.json"
PILOT_ANSWERS_PATH = RESULTS_DIR / "v14_pilot_answers.json"


def directories(pilot: bool) -> dict[tuple[str, str], Path]:
    if pilot:  # the pilot has no retests; v8 mapped both readings to one pilot folder too
        return {(AGENT_MODEL, reading): V14_DIR / "pilot" for reading in ("first", "retest")}
    return {(AGENT_MODEL, "first"): V14_DIR / "agent_opus", (AGENT_MODEL, "retest"): V14_DIR / "agent_opus_retest"}


def requeue(directory: Path, requeued: Path = REQUEUED_DIR) -> list[str]:
    """Move each cached reading that failed for an infrastructure reason out of the cache, unless it
    was already moved RERUNS times, so the next run reads it again. Returns the moved files' names."""
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


def spent(root: Path = V14_DIR) -> float:
    """The CLI-reported cost of every v14 reading so far, the pilot's and requeued ones included."""
    return sum(json.loads(f.read_text())["cost_usd"] for f in root.rglob("*.json")) if root.exists() else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true", help="read v14's pilot units on pythia-160m")
    args = parser.parse_args()
    folders = directories(args.pilot)
    for folder in set(folders.values()):
        if folder.exists() and (moved := requeue(folder)):
            print(f"requeued {len(moved)} infrastructure failures from {folder.name}", flush=True)
    guard = SpendGuard(SPEND_CAP_USD, spent=spent())
    print(f"spent so far: ${guard.spent:.2f} of ${SPEND_CAP_USD:.0f}", flush=True)
    corpus = load_corpus()
    out = {"first": {}, "retest": {}}
    for model in [PILOT_MODEL] if args.pilot else MODELS:
        units = choose(build_units(model), model)
        if args.pilot:
            units = pilot_units(units)
        again = set() if args.pilot else set(retest_ids(units, model))
        jobs = [(u["id"], AGENT_MODEL, "first") for u in units]
        jobs += [(u["id"], AGENT_MODEL, "retest") for u in units if u["id"] in again]
        records = read_model(model, units, jobs, folders, {AGENT_MODEL: guard}, corpus)
        for (unit_id, _, reading), record in zip(jobs, records):
            out[reading][unit_id] = record
    readings = [r for part in out.values() for r in part.values()]
    print(f"{AGENT_MODEL}: {len(readings)} readings, {sum(r['label'] is None for r in readings)} without a label "
          f"({sum(infrastructure_failure(r) for r in readings)} infrastructure failures, "
          f"{sum(r['error'] == 'spend cap reached' for r in readings)} stopped by the spend guard), "
          f"${sum(r['cost_usd'] for r in readings):.2f} for these, ${spent():.2f} for all v14 readings", flush=True)
    (PILOT_ANSWERS_PATH if args.pilot else ANSWERS_PATH).write_text(json.dumps({AGENT_MODEL: out}, indent=1))


if __name__ == "__main__":
    main()
