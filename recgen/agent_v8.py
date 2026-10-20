"""v8's investigating agent: each unit read by the claude CLI with the probe tools of recgen/probe_v8.

Per model, Pythia is downloaded, kept in memory while that model's units are read (WORKERS at a
time), then deleted. Every reading is cached in data/v8/ with its tool calls, their answers and
the agent's messages, so reruns never call Pythia or the CLI again. results/v8_agent_answers.json
collects them for the analysis. `--pilot` reads a few pythia-160m units, which aren't in v8's
sample, to check the tools and the cost (recgen/rubric_v8).
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import GPTNeoXForCausalLM

from recgen.activations import DEVICE
from recgen.config import DATA_DIR, RESULTS_DIR, WEIGHTS_DIR
from recgen.corpus import load_corpus
from recgen.probe_v8 import Backend, TextReader, UnitView
from recgen.rubric_v7 import DEPTH
from recgen.rubric_v8 import AGENT_MODEL, ATTEMPTS, MAX_TOOL_CALLS, MODELS, SYSTEM_PROMPT, choose, parse_answer, retest_ids
from recgen.sweep import sweep_path
from recgen.sweep_v6 import v6_path
from recgen.units_v7 import build_units

V8_DIR = DATA_DIR / "v8"
WORKDIR = V8_DIR / "workdir"  # an empty folder, so the CLI sees nothing of the repo
ANSWERS_PATH = RESULTS_DIR / "v8_agent_answers.json"
PILOT_MODEL, PILOT_PER_KIND = "160m", 2
WORKERS = 6
MAX_TURNS = 2 * MAX_TOOL_CALLS + 2  # a hard stop; the tool budget is enforced by the backend
BUDGET_USD = "1"  # per attempt, a safety cap
TIMEOUT_SECONDS = 900
FORWARDER = Path(__file__).with_name("mcp_v8.py")
PROBE_TOOLS = ["mcp__probe__examples", "mcp__probe__activation"]
CLI = ["claude", "-p", "--system-prompt", SYSTEM_PROMPT, "--tools", "", "--strict-mcp-config",
       "--setting-sources", "", "--no-session-persistence", "--output-format", "stream-json", "--verbose",
       "--allowedTools", ",".join(PROBE_TOOLS), "--max-turns", str(MAX_TURNS),
       "--max-budget-usd", BUDGET_USD]


def run_cli(text: str, url: str, handle: str, model: str = AGENT_MODEL) -> dict:
    """One agent reading: its final reply, its messages and tool requests, and usage."""
    # -I keeps recgen/ off the forwarder's path, where recgen/select.py would shadow the standard library.
    config = {"mcpServers": {"probe": {"command": sys.executable, "args": ["-I", str(FORWARDER), url, handle]}}}
    run = subprocess.run(CLI + ["--model", model, "--mcp-config", json.dumps(config)], input=text, capture_output=True, text=True,
                         timeout=TIMEOUT_SECONDS, cwd=WORKDIR)
    events = [json.loads(line) for line in run.stdout.splitlines() if line.startswith("{")]
    result = next((e for e in reversed(events) if e.get("type") == "result"), None)
    if result is None:
        raise RuntimeError(run.stderr[-500:] or "no result from the CLI")
    init = next((e for e in events if e.get("type") == "system" and e.get("subtype") == "init"), {})
    messages = []
    for event in events:
        if event.get("type") == "assistant":
            for block in event["message"]["content"]:
                if block.get("type") == "text":
                    messages.append({"text": block["text"]})
                elif block.get("type") == "tool_use":
                    messages.append({"tool": block["name"].removeprefix("mcp__probe__"), "input": block["input"]})
    return {"reply": result.get("result") or "", "error": result.get("subtype") if result.get("is_error") else None,
            "tools_offered": init.get("tools", []), "messages": messages, "num_turns": result.get("num_turns"),
            "models": list((result.get("modelUsage") or {}).keys()), "usage": result.get("usage") or {},
            "cost_usd": result.get("total_cost_usd") or 0.0}


def cache_path(directory: Path, unit_id: str) -> Path:
    return directory / (unit_id.replace(":", "_").replace("/", "-") + ".json")


def read(view: UnitView, directory: Path, backend: Backend | None, url: str | None, runner=run_cli) -> dict:
    """The cached reading for this prompt, or a new one with up to ATTEMPTS tries."""
    text = view.prompt()
    path, digest = cache_path(directory, view.unit["id"]), hashlib.sha256(text.encode()).hexdigest()
    if path.exists():
        cached = json.loads(path.read_text())
        if cached["prompt_sha256"] == digest:
            return cached
    attempts, out, calls, answer = [], {}, [], None
    for _ in range(ATTEMPTS):
        handle = uuid.uuid4().hex
        backend.open(handle, view.unit["id"])
        try:
            out = runner(text, url, handle)
            error = out["error"] or (None if set(PROBE_TOOLS) <= set(out["tools_offered"]) else "probe tools not offered")
        except (RuntimeError, json.JSONDecodeError, subprocess.TimeoutExpired) as e:
            out, error = {}, str(e)
        calls = backend.close(handle)
        answer = None if error else parse_answer(out["reply"])
        attempts.append({"error": error or (None if answer else "no valid label"), "cost_usd": out.get("cost_usd", 0.0),
                         "tool_calls": len(calls)})
        if answer:
            break
    usage = out.get("usage", {})
    record = {"id": view.unit["id"], "prompt_sha256": digest, **(answer or {"label": None, "patterns": [], "confidence": None}),
              "error": attempts[-1]["error"], "attempts": attempts, "cost_usd": sum(a["cost_usd"] for a in attempts),
              "input_tokens": sum(usage.get(k) or 0 for k in ("input_tokens", "cache_creation_input_tokens",
                                                              "cache_read_input_tokens")),
              "output_tokens": usage.get("output_tokens") or 0, "num_turns": out.get("num_turns"),
              "models": out.get("models", []), "tools_offered": out.get("tools_offered", []),
              "reply": out.get("reply", ""), "messages": out.get("messages", []), "tool_calls": calls}
    path.write_text(json.dumps(record))
    return record


def load_model(model: str):
    local_dir = WEIGHTS_DIR / f"pythia-{model}"
    snapshot_download(f"EleutherAI/pythia-{model}", local_dir=local_dir, allow_patterns=["config.json", "model.safetensors"])
    return GPTNeoXForCausalLM.from_pretrained(local_dir, torch_dtype=torch.float16).to(DEVICE).eval(), local_dir


def read_model(model: str, units: list[dict], again: set[str], directories: dict[str, Path], corpus: np.ndarray,
               agent_model: str = AGENT_MODEL) -> list:
    """Every reading of one model's units, as (unit id, reading, record), loading Pythia only if some
    reading isn't cached."""
    v6 = dict(np.load(v6_path(model, DEPTH)))
    v5 = np.load(sweep_path(model, DEPTH))
    window_max, window_pos = v5["max"], v5["pos"]
    views = {u["id"]: UnitView(u, v6, window_max, window_pos, corpus) for u in units}
    jobs = [(u["id"], "first") for u in units] + [(u["id"], "retest") for u in units if u["id"] in again]
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    def cached(job):
        unit_id, reading = job
        path = cache_path(directories[reading], unit_id)
        return path.exists() and json.loads(path.read_text())["prompt_sha256"] == hashlib.sha256(
            views[unit_id].prompt().encode()).hexdigest()

    pending = [job for job in jobs if not cached(job)]
    backend, url, local_dir = None, None, None
    if pending:
        print(f"pythia-{model}: {len(pending)} of {len(jobs)} readings to run", flush=True)
        WORKDIR.mkdir(parents=True, exist_ok=True)
        pythia, local_dir = load_model(model)
        backend = Backend(views, TextReader(pythia, int(v6["layer"])))
        url = backend.serve()
    try:
        with ThreadPoolExecutor(WORKERS) as pool:
            records = list(pool.map(lambda job: read(views[job[0]], directories[job[1]], backend, url,
                                                     partial(run_cli, model=agent_model)), jobs))
    finally:
        if backend:
            backend.stop()
        if local_dir:
            shutil.rmtree(local_dir)
    for (unit_id, reading), record in zip(jobs, records):
        if pending:
            print(f"  {unit_id} {reading}: {record['label']} ({len(record['tool_calls'])} tool calls, "
                  f"${record['cost_usd']:.3f})", flush=True)
    return [(unit_id, reading, record) for (unit_id, reading), record in zip(jobs, records)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true", help="read a few pythia-160m units outside v8's sample")
    parser.add_argument("--pilot-agent", default=AGENT_MODEL, help="the agent's model in the pilot")
    args = parser.parse_args()
    pilot, agent_model = args.pilot, args.pilot_agent if args.pilot else AGENT_MODEL
    corpus = load_corpus()
    out = {"first": {}, "retest": {}}
    for model in [PILOT_MODEL] if pilot else MODELS:
        units = choose(build_units(model), model)
        if pilot:
            units = [u for kind in ("real", "pooled", "related") for u in [x for x in units if x["kind"] == kind][:PILOT_PER_KIND]]
            again, directories = set(), {"first": V8_DIR / f"pilot_{agent_model}"}
        else:
            again, directories = set(retest_ids(units, model)), {"first": V8_DIR / "agent", "retest": V8_DIR / "agent_retest"}
        for unit_id, reading, record in read_model(model, units, again, directories, corpus, agent_model):
            out[reading][unit_id] = record
    readings = [r for part in out.values() for r in part.values()]
    print(f"{len(readings)} readings, {sum(r['label'] is None for r in readings)} without a label, "
          f"${sum(r['cost_usd'] for r in readings):.2f} of quota", flush=True)
    path = RESULTS_DIR / f"v8_pilot_{agent_model}_answers.json" if pilot else ANSWERS_PATH
    path.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
