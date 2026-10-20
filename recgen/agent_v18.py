"""v18's falsification-driven Opus agent (recgen/rubric_v18): v8's runner (recgen/agent_v8.read_model)
with v18's prompt, tool budget, CLI caps and parser, and an examples tool without v15's held-out
windows; then v15's Sonnet simulator on each final description, on v15's held-out sets, and v15's judge.

Readings are cached in data/v18/ (agent_opus/, agent_opus_retest/, simulate_sonnet/,
simulate_sonnet_retest/, judge_haiku/; pilot/ for the pilot), keyed by the prompt, so reruns call
neither Pythia nor the CLI. Before a run, each cached reading that failed for an infrastructure reason
is moved to data/v18/requeued/ (at most RERUNS times), so it is read again. results/v18_agent_answers.json
collects everything for the analysis, and results/v18_pilot_answers.json the pilot's.

    uv run python -m recgen.agent_v18 [--pilot]
"""

import argparse
import json
import subprocess
from functools import partial
from pathlib import Path

from recgen.agent_v8 import SpendGuard, cli_command, read_model, run_cli
from recgen.config import DATA_DIR, RESULTS_DIR
from recgen.corpus import load_corpus
from recgen.probe_v8 import UnitView
from recgen.rubric_v8 import MODELS, choose, retest_ids
from recgen.rubric_v15 import (
    JUDGE, PRIMARY_SIMULATOR, SIMULATORS, held_out_ranks, judge_prompt, parse_judgment, parse_predictions, simulator_prompt,
)
from recgen.rubric_v18 import (
    AGENT_MODEL, BUDGET_PER_ATTEMPT_USD, MAX_TOOL_CALLS, MAX_TURNS, PILOT_MODEL, PILOT_UNITS, RERUNS, SIMULATION_RESERVE_USD,
    SPEND_CAP_USD, TIMEOUT_SECONDS, TOOLS, infrastructure_failure, parse_answer, prompt, round_robin,
)
from recgen.simulate_v15 import SPEND_CAP_REACHED, WORKDIR as SIMULATOR_WORKDIR, run_jobs, shown_lines, summary
from recgen.units_v7 import build_units

V18_DIR = DATA_DIR / "v18"
PILOT_DIR = V18_DIR / "pilot"
REQUEUED_DIR = V18_DIR / "requeued"
VERSIONS_PATH = V18_DIR / "cli_versions.txt"
ANSWERS_PATH = RESULTS_DIR / "v18_agent_answers.json"
PILOT_ANSWERS_PATH = RESULTS_DIR / "v18_pilot_answers.json"
V15_ANSWERS_PATH = RESULTS_DIR / "v15_answers.json"
V15_PILOT_ANSWERS_PATH = RESULTS_DIR / "v15_pilot_answers.json"
SIMULATOR = SIMULATORS[PRIMARY_SIMULATOR]
RUNNER = partial(run_cli, cli=cli_command(MAX_TURNS, BUDGET_PER_ATTEMPT_USD), timeout=TIMEOUT_SECONDS)
BACKEND_OPTIONS = {"max_calls": MAX_TOOL_CALLS, "tools": TOOLS}
READINGS = ("first", "retest")
SUFFIX = {"first": "", "retest": "_retest"}


class AuditView(UnitView):
    """v8's view of a unit with v18's prompt, and with v15's held-out windows taken out of the ranked
    list the examples tool reads (the rest renumbered; ranks 1 to 50 are unchanged)."""

    def __init__(self, unit: dict, v6: dict, window_max, window_pos, corpus):
        super().__init__(unit, v6, window_max, window_pos, corpus)
        self.held_out = sorted(rank for rank, _ in held_out_ranks(unit["id"], len(self.ranked)))
        held = set(self.held_out)
        self.ranked = [example for rank, example in enumerate(self.ranked, 1) if rank not in held]

    def prompt(self) -> str:
        return prompt(shown_lines(self))


def folders(root: Path) -> dict[str, Path]:
    return {**{f"agent {r}": root / f"agent_opus{SUFFIX[r]}" for r in READINGS},
            **{f"simulate {r}": root / f"simulate_sonnet{SUFFIX[r]}" for r in READINGS}, "judge": root / "judge_haiku"}


def failed_outside_the_model(record: dict) -> bool:
    """infrastructure_failure for agent readings; for a simulation or judgment, a reading without a valid
    reply whose tries ended in a CLI error rather than in a reply that didn't parse."""
    if "label" in record:
        return infrastructure_failure(record)
    return record["parsed"] is None and any(a["error"] not in (None, "invalid reply", SPEND_CAP_REACHED)
                                            for a in record["attempts"])


def requeue(directory: Path, requeued: Path = REQUEUED_DIR) -> list[str]:
    """Move each cached reading that failed for an infrastructure reason out of the cache, unless it was
    already moved RERUNS times, so the next run reads it again. Returns the moved files' names."""
    moved = []
    for path in sorted(directory.glob("*.json")):
        if not failed_outside_the_model(json.loads(path.read_text())):
            continue
        earlier = list((requeued / directory.name).glob(f"{path.stem}.*.json"))
        if len(earlier) >= RERUNS:
            continue
        target = requeued / directory.name / f"{path.stem}.{len(earlier) + 1}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        path.rename(target)
        moved.append(path.name)
    return moved


def spent(root: Path = V18_DIR) -> float:
    """The CLI-reported cost of every v18 reading so far, the pilot's and requeued ones included."""
    return sum(json.loads(f.read_text()).get("cost_usd") or 0.0 for f in root.rglob("*.json")) if root.exists() else 0.0


def cli_version() -> str:
    return subprocess.run(["claude", "--version"], capture_output=True, text=True).stdout.strip()


def readings_on_disk(root: Path = V18_DIR) -> int:
    return sum(1 for _ in root.rglob("*.json")) if root.exists() else 0


def record_cli_version(made_readings: bool, path: Path = VERSIONS_PATH) -> list[str]:
    """The CLI versions that made v18's readings, kept in data/v18 so a rerun from the cache, which
    makes no readings, writes the same answers whatever CLI is installed then."""
    versions = json.loads(path.read_text()) if path.exists() else []
    if made_readings and (version := cli_version()) not in versions:
        versions.append(version)
        path.write_text(json.dumps(versions))
    return versions


def read_agents(units: dict[str, list[dict]], wanted: dict[str, set[str]], reading: str, folder: Path, corpus) -> dict:
    """Each wanted unit's agent reading, size by size, with the agent spend guard."""
    guard = SpendGuard(SPEND_CAP_USD - SIMULATION_RESERVE_USD, spent=spent())
    out = {}
    for model, model_units in units.items():
        jobs = [(u["id"], AGENT_MODEL, reading) for u in model_units if u["id"] in wanted[model]]
        if not jobs:
            continue
        records = read_model(model, model_units, jobs, {(AGENT_MODEL, reading): folder}, {AGENT_MODEL: guard}, corpus,
                             view_class=AuditView, backend_options=BACKEND_OPTIONS, runner=RUNNER, parse=parse_answer)
        out.update({unit_id: record for (unit_id, *_), record in zip(jobs, records)})
    print(f"agent, {reading}: {len(out)} readings, {sum(r['label'] is None for r in out.values())} without an answer, "
          f"${spent():.2f} spent on v18 in all", flush=True)
    return out


def simulate(agent: dict, held_out: dict, folder: Path, name: str) -> dict:
    """v15's Sonnet simulator on each answered unit's description, on v15's held-out snippets."""
    answered = [k for k, r in agent.items() if r["label"] is not None]
    jobs = [(k, simulator_prompt(agent[k]["description"], [h["snippet"] for h in held_out[k]]), folder, SIMULATOR,
             parse_predictions) for k in answered]
    records = run_jobs(jobs, SpendGuard(SPEND_CAP_USD, spent=spent()), name)
    return dict(zip(answered, records))


def judge(agent: dict, folder: Path) -> dict:
    answered = [k for k, r in agent.items() if r["label"] is not None]
    jobs = [(k, judge_prompt(agent[k]["description"]), folder, JUDGE, parse_judgment) for k in answered]
    return dict(zip(answered, run_jobs(jobs, SpendGuard(SPEND_CAP_USD, spent=spent()), "judge")))


def check_held_out(v15_units: dict, unit_ids) -> None:
    """The held-out ranks the examples tool hides must be the ones v15 scored."""
    for unit_id in unit_ids:
        ours = sorted(rank for rank, _ in held_out_ranks(unit_id))
        if ours != sorted(h["rank"] for h in v15_units[unit_id]["held_out"]):
            raise ValueError(f"held-out ranks of {unit_id} differ from v15's")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true", help="read v18's two pilot units on pythia-160m")
    args = parser.parse_args()
    root = PILOT_DIR if args.pilot else V18_DIR
    paths = folders(root)
    for folder in paths.values():
        if folder.exists() and (moved := requeue(folder)):
            print(f"requeued {len(moved)} infrastructure failures from {folder.name}", flush=True)
    print(f"claude CLI {cli_version()}; spent so far: ${spent():.2f} of ${SPEND_CAP_USD:.0f}", flush=True)
    before = readings_on_disk()
    SIMULATOR_WORKDIR.mkdir(parents=True, exist_ok=True)
    corpus = load_corpus()
    v15_units = json.loads((V15_PILOT_ANSWERS_PATH if args.pilot else V15_ANSWERS_PATH).read_text())["units"]
    units, retested = {}, {}
    for model in [PILOT_MODEL] if args.pilot else MODELS:
        chosen = choose(build_units(model), model)
        if args.pilot:
            chosen = [next(u for u in chosen if (u["kind"], u["mix"]) == kind) for kind in PILOT_UNITS]
            retested[model] = set()
        else:
            retested[model] = set(retest_ids(chosen, model))
        units[model] = round_robin(chosen)
    unit_ids = [u["id"] for model_units in units.values() for u in model_units]
    check_held_out(v15_units, unit_ids)
    held_out = {k: v15_units[k]["held_out"] for k in unit_ids}

    agent = {"first": read_agents(units, {m: {u["id"] for u in us} for m, us in units.items()}, "first",
                                  paths["agent first"], corpus)}
    simulation = {"first": simulate(agent["first"], held_out, paths["simulate first"], "simulator, first")}
    judgment = judge(agent["first"], paths["judge"])
    if any(retested.values()):
        agent["retest"] = read_agents(units, retested, "retest", paths["agent retest"], corpus)
        simulation["retest"] = simulate(agent["retest"], held_out, paths["simulate retest"], "simulator, retest")

    versions = record_cli_version(readings_on_disk() > before)
    out = {"agent": AGENT_MODEL, "cli_versions": versions, "simulator": SIMULATOR, "judge": JUDGE, "units": {}}
    for model, model_units in units.items():
        for u in model_units:
            entry = {"id": u["id"], "model": model, "kind": u["kind"],
                     "group": "pooled 50/50" if u["kind"] == "pooled" else u["kind"],
                     "retested": u["id"] in retested[model], "held_out_ranks": sorted(h["rank"] for h in held_out[u["id"]])}
            for reading in agent:
                if u["id"] in agent[reading]:
                    entry[reading] = {"agent": agent[reading][u["id"]],
                                      "simulation": summary(simulation[reading].get(u["id"]))}
            entry["first"]["judge"] = summary(judgment.get(u["id"]))
            out["units"][u["id"]] = entry
    records = [r for part in agent.values() for r in part.values()]
    print(f"{len(records)} agent readings, {sum(r['label'] is None for r in records)} without an answer, "
          f"${sum(r['cost_usd'] for r in records):.2f} for them; ${spent():.2f} for all v18 readings", flush=True)
    (PILOT_ANSWERS_PATH if args.pilot else ANSWERS_PATH).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
