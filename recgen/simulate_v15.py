"""v15's readings: an explainer writes an explanation from a unit's 50 shown examples, two
simulators predict the unit's activations on 60 held-out examples from the explanation alone, and
a judge says whether the explanation names unrelated patterns, all through the claude CLI with the
rules of recgen/rubric_v15.

No Pythia model is loaded: units, values and held-out examples come from v8's UnitView (v5's window
maxima and v6's stored residuals). Every reading is cached in data/v15/<role>/, keyed by its prompt,
so reruns make no CLI calls. results/v15_answers.json collects every explanation and prediction with
the held-out ground truth, for the analysis. `--pilot` runs v8's six pilot units (pythia-160m, not in
the sample) into data/v15/pilot/ and results/v15_pilot_answers.json.
"""

import argparse
import hashlib
import json
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from recgen.agent_v8 import PILOT_MODEL, PILOT_PER_KIND, SpendGuard, cache_path
from recgen.config import DATA_DIR, RESULTS_DIR
from recgen.corpus import load_corpus
from recgen.probe_v8 import UnitView, v7_order
from recgen.rubric_v7 import DEPTH
from recgen.rubric_v8 import MODELS, SCALE, choose, retest_ids, snippet_line
from recgen.rubric_v15 import (
    ATTEMPTS, EXPLAINER, JUDGE, SIMULATORS, SPEND_CAP_USD, SYSTEM_PROMPT, explainer_prompt, held_out_ranks, judge_prompt,
    parse_explanation, parse_judgment, parse_predictions, simulator_prompt,
)
from recgen.sweep import sweep_path
from recgen.sweep_v6 import v6_path
from recgen.units_v7 import build_units

V15_DIR = DATA_DIR / "v15"
PILOT_DIR = V15_DIR / "pilot"
WORKDIR = Path(tempfile.gettempdir()) / "recgen-v15-workdir"  # empty and outside the repo, so the CLI sees none of it
ANSWERS_PATH = RESULTS_DIR / "v15_answers.json"
PILOT_ANSWERS_PATH = RESULTS_DIR / "v15_pilot_answers.json"
WORKERS = 6  # concurrent CLI calls
BUDGET_USD = "1"  # per call, a safety cap
TIMEOUT_SECONDS = 300
BACKOFF_SECONDS = (10, 30)
SPEND_CAP_REACHED = "spend cap reached"
CLI = ["claude", "-p", "--system-prompt", SYSTEM_PROMPT, "--tools", "", "--strict-mcp-config", "--setting-sources", "",
       "--no-session-persistence", "--output-format", "json", "--max-budget-usd", BUDGET_USD]
READINGS = ("first", "retest")


def shown_lines(view: UnitView) -> list[str]:
    """The unit's 50 shown examples in v7's order with their values: the snippet lines of v8's agent prompt."""
    order = v7_order(view.unit["id"], len(view.first))
    return [snippet_line(i + 1, view.first[j][1], view.snippet(view.first[j][0])) for i, j in enumerate(order)]


def held_out(view: UnitView, window_max: np.ndarray) -> list[dict]:
    """The unit's held-out examples with their truth: the unit's value, and each of its neurons' values on
    the same scale. The neuron with the larger one drives the example."""
    out = []
    for rank, stratum in held_out_ranks(view.unit["id"], len(view.ranked)):
        (window, position), value = view.ranked[rank - 1]
        neuron_values = SCALE * window_max[window, view.neuron_ids].astype(np.float64) / view.norms / view.top
        out.append({"rank": rank, "stratum": stratum, "window": int(window), "position": int(position),
                    "value": float(value), "neuron_values": [float(v) for v in neuron_values],
                    "driver": int(np.argmax(neuron_values)), "snippet": view.snippet((window, position))})
    return out


def call_cli(text: str, model: str) -> dict:
    """One reading through the claude CLI: the reply, the models used, usage, cost, and any CLI error."""
    run = subprocess.run(CLI + ["--model", model], input=text, capture_output=True, text=True, timeout=TIMEOUT_SECONDS,
                         cwd=WORKDIR)
    out = json.loads(run.stdout)
    return {"reply": out.get("result") or "", "error": (out.get("result") or "CLI error") if out.get("is_error") else None,
            "models": list((out.get("modelUsage") or {}).keys()), "usage": out.get("usage") or {},
            "cost_usd": out.get("total_cost_usd") or 0.0}


def read(unit_id: str, text: str, directory: Path, model: str, parse, runner=call_cli, guard: SpendGuard | None = None
         ) -> dict:
    """The cached reading for this prompt, or a new one with up to ATTEMPTS tries until parse accepts the
    reply. A reading the spend guard stops is returned unparsed and isn't cached."""
    path, digest = cache_path(directory, unit_id), hashlib.sha256(text.encode()).hexdigest()
    if path.exists():
        cached = json.loads(path.read_text())
        if cached["prompt_sha256"] == digest:
            return cached
    attempts, out, parsed = [], {}, None
    if guard and not guard.allows():
        attempts.append({"error": SPEND_CAP_REACHED, "cost_usd": 0.0, "reply": ""})
    else:
        for attempt in range(ATTEMPTS):
            try:
                out = runner(text, model)
                error = out["error"]
            except (json.JSONDecodeError, subprocess.TimeoutExpired, OSError) as e:
                out, error = {}, f"{type(e).__name__}: {e}"
            parsed = None if error else parse(out["reply"])
            attempts.append({"error": error or (None if parsed is not None else "invalid reply"),
                             "cost_usd": out.get("cost_usd", 0.0), "reply": out.get("reply", "")})
            if guard:
                guard.add(attempts[-1]["cost_usd"])
            if parsed is not None:
                break
            if error and attempt < ATTEMPTS - 1:
                time.sleep(BACKOFF_SECONDS[min(attempt, len(BACKOFF_SECONDS) - 1)])
    usage = out.get("usage", {})
    record = {"id": unit_id, "model": model, "prompt_sha256": digest, "parsed": parsed, "error": attempts[-1]["error"],
              "attempts": attempts, "cost_usd": sum(a["cost_usd"] for a in attempts),
              "input_tokens": sum(usage.get(k) or 0 for k in ("input_tokens", "cache_creation_input_tokens",
                                                              "cache_read_input_tokens")),
              "output_tokens": usage.get("output_tokens") or 0, "models": out.get("models", []),
              "reply": out.get("reply", "")}
    if record["error"] != SPEND_CAP_REACHED:
        path.write_text(json.dumps(record))
    return record


def units_of(model: str, corpus: np.ndarray, pilot: bool = False) -> list[dict]:
    """Every unit of one model as the readers need it: ids, neurons, log peak (v12's measure), the
    explainer's prompt and the held-out examples."""
    v6 = dict(np.load(v6_path(model, DEPTH)))
    v5 = np.load(sweep_path(model, DEPTH))
    window_max, window_pos = v5["max"], v5["pos"]
    units = choose(build_units(model), model)
    if pilot:
        units = [u for kind in ("real", "pooled", "related") for u in [x for x in units if x["kind"] == kind][:PILOT_PER_KIND]]
    again = set() if pilot else set(retest_ids(units, model))
    median_peak = float(np.median(window_max[:, v6["neurons"]].astype(np.float64).max(axis=0)))
    out = []
    for unit in units:
        view = UnitView(unit, v6, window_max, window_pos, corpus)
        peaks = window_max[:, view.neuron_ids].astype(np.float64).max(axis=0)
        out.append({"id": unit["id"], "model": model, "kind": unit["kind"],
                    "group": "pooled 50/50" if unit["kind"] == "pooled" else unit["kind"],
                    "neurons": [int(s) for s in unit["neurons"]], "neuron_ids": view.neuron_ids,
                    "log_peak": [float(np.log(p / median_peak)) for p in peaks], "retest": unit["id"] in again,
                    "explainer_prompt": explainer_prompt(shown_lines(view)), "held_out": held_out(view, window_max)})
    return out


def run_jobs(jobs: list[tuple], guard: SpendGuard, name: str) -> list[dict]:
    """Each (unit id, prompt, directory, model, parser) job's record, WORKERS at a time."""
    for directory in {job[2] for job in jobs}:
        directory.mkdir(parents=True, exist_ok=True)
    pending = sum(not (cache_path(d, u).exists() and json.loads(cache_path(d, u).read_text())["prompt_sha256"]
                       == hashlib.sha256(t.encode()).hexdigest()) for u, t, d, *_ in jobs)
    print(f"{name}: {len(jobs)} readings, {pending} to run", flush=True)
    records = []
    with ThreadPoolExecutor(WORKERS) as pool:
        for record in pool.map(lambda job: read(*job, guard=guard), jobs):
            records.append(record)
            if pending and len(records) % 50 == 0:
                print(f"  {len(records)} of {len(jobs)}, {sum(r['parsed'] is None for r in records)} invalid, "
                      f"${guard.spent:.2f} spent in all", flush=True)
    return records


def summary(record: dict | None) -> dict | None:
    return None if record is None else {k: record[k] for k in (
        "parsed", "error", "attempts", "cost_usd", "input_tokens", "output_tokens", "models", "prompt_sha256", "reply")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true", help="read v8's six pilot units (pythia-160m), outside the sample")
    args = parser.parse_args()
    root = PILOT_DIR if args.pilot else V15_DIR
    WORKDIR.mkdir(parents=True, exist_ok=True)
    guard = SpendGuard(SPEND_CAP_USD, spent=sum(json.loads(f.read_text()).get("cost_usd") or 0.0
                                                for f in V15_DIR.rglob("*.json")) if V15_DIR.exists() else 0.0)
    corpus = load_corpus()
    units = [u for model in ([PILOT_MODEL] if args.pilot else MODELS) for u in units_of(model, corpus, args.pilot)]
    readings = [(u, reading) for reading in READINGS for u in units if reading == "first" or u["retest"]]
    suffix = {"first": "", "retest": "_retest"}

    explained = run_jobs([(u["id"], u["explainer_prompt"], root / f"explain_sonnet{suffix[r]}", EXPLAINER, parse_explanation)
                          for u, r in readings], guard, "explainer")
    explanation = {(u["id"], r): rec for (u, r), rec in zip(readings, explained)}
    simulated = [(u, r, arm) for u, r in readings if explanation[u["id"], r]["parsed"] for arm in SIMULATORS]
    simulations = run_jobs([(u["id"], simulator_prompt(explanation[u["id"], r]["parsed"], [h["snippet"] for h in u["held_out"]]),
                             root / f"simulate_{arm}{suffix[r]}", SIMULATORS[arm], parse_predictions) for u, r, arm in simulated],
                           guard, "simulators")
    simulation = {(u["id"], r, arm): rec for (u, r, arm), rec in zip(simulated, simulations)}
    judged = [u for u in units if explanation[u["id"], "first"]["parsed"]]
    judgments = run_jobs([(u["id"], judge_prompt(explanation[u["id"], "first"]["parsed"]), root / "judge_haiku", JUDGE,
                           parse_judgment) for u in judged], guard, "judge")
    judgment = {u["id"]: rec for u, rec in zip(judged, judgments)}

    out = {"explainer": EXPLAINER, "simulators": SIMULATORS, "judge": JUDGE, "units": {}}
    for u in units:
        entry = {k: v for k, v in u.items() if k != "explainer_prompt"}
        entry["explainer_prompt_sha256"] = hashlib.sha256(u["explainer_prompt"].encode()).hexdigest()
        for r in READINGS:
            if (u["id"], r) in explanation:
                entry[r] = {"explanation": summary(explanation[u["id"], r]),
                            "simulations": {arm: summary(simulation.get((u["id"], r, arm))) for arm in SIMULATORS}}
        entry["first"]["judge"] = summary(judgment.get(u["id"]))
        out["units"][u["id"]] = entry
    records = explained + simulations + judgments
    print(f"{len(records)} readings, {sum(r['parsed'] is None for r in records)} invalid, "
          f"{sum(len(r['attempts']) > 1 for r in records)} retried, ${sum(r['cost_usd'] for r in records):.2f} of quota; "
          f"models used: {sorted({m for r in records for m in r['models']})}", flush=True)
    (PILOT_ANSWERS_PATH if args.pilot else ANSWERS_PATH).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
