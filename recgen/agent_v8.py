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
import tempfile
import threading
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
from recgen.rubric_v8 import (
    AGENT_MODEL, ATTEMPTS, MAX_TOOL_CALLS, MODELS, SECOND_AGENT_MODEL, SYSTEM_PROMPT, choose, parse_answer, retest_ids,
)
from recgen.sweep import sweep_path
from recgen.sweep_v6 import v6_path
from recgen.units_v7 import build_units

V8_DIR = DATA_DIR / "v8"
WORKDIR = Path(tempfile.gettempdir()) / "recgen-v8-workdir"  # empty and outside the repo, so the CLI sees none of it
ANSWERS_PATH = RESULTS_DIR / "v8_agent_answers.json"
PILOT_MODEL, PILOT_PER_KIND = "160m", 2
WORKERS = 6
MAX_TURNS = 2 * MAX_TOOL_CALLS + 2  # a hard stop; the tool budget is enforced by the backend
BUDGET_USD = "1"  # per attempt, a safety cap
SPEND_CAP_USD = {AGENT_MODEL: 3.0, SECOND_AGENT_MODEL: 20.0}  # per arm, well above the pilot's estimates
TIMEOUT_SECONDS = 900
FORWARDER = Path(__file__).with_name("mcp_v8.py")
PROBE_TOOLS = ["mcp__probe__examples", "mcp__probe__activation"]


def cli_command(max_turns: int = MAX_TURNS, budget_usd: str = BUDGET_USD) -> list[str]:
    return ["claude", "-p", "--system-prompt", SYSTEM_PROMPT, "--tools", "", "--strict-mcp-config",
            "--setting-sources", "", "--no-session-persistence", "--output-format", "stream-json", "--verbose",
            "--allowedTools", ",".join(PROBE_TOOLS), "--max-turns", str(max_turns),
            "--max-budget-usd", budget_usd]


CLI = cli_command()


def run_cli(text: str, url: str, handle: str, model: str = AGENT_MODEL, cli: list[str] = CLI,
            timeout: int = TIMEOUT_SECONDS) -> dict:
    """One agent reading: its final reply, its messages and tool requests, and usage."""
    # -I keeps recgen/ off the forwarder's path, where recgen/select.py would shadow the standard library.
    config = {"mcpServers": {"probe": {"command": sys.executable, "args": ["-I", str(FORWARDER), url, handle]}}}
    run = subprocess.run(cli + ["--model", model, "--mcp-config", json.dumps(config)], input=text, capture_output=True, text=True,
                         timeout=timeout, cwd=WORKDIR)
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


class SpendGuard:
    """Stops new readings of an arm once it has spent its cap (CLI-reported), across threads."""

    def __init__(self, cap: float, spent: float = 0.0):
        self.cap, self.spent, self.lock = cap, spent, threading.Lock()

    def allows(self) -> bool:
        with self.lock:
            return self.spent < self.cap

    def add(self, cost: float) -> None:
        with self.lock:
            self.spent += cost


def read(view: UnitView, directory: Path, backend: Backend | None, url: str | None, runner=run_cli,
         guard: SpendGuard | None = None, parse=parse_answer) -> dict:
    """The cached reading for this prompt, or a new one with up to ATTEMPTS tries. A reading the
    spend guard stops is returned without a label and isn't cached. The cache key is the unit's prompt;
    the directory names the agent's model, so a change to SYSTEM_PROMPT, TOOLS or MAX_TOOL_CALLS needs
    the cache cleared."""
    text = view.prompt()
    path, digest = cache_path(directory, view.unit["id"]), hashlib.sha256(text.encode()).hexdigest()
    if path.exists():
        cached = json.loads(path.read_text())
        if cached["prompt_sha256"] == digest:
            return cached
    attempts, out, calls, answer = [], {}, [], None
    if guard and not guard.allows():
        attempts.append({"error": "spend cap reached", "cost_usd": 0.0, "tool_calls": 0})
    else:
        for _ in range(ATTEMPTS):
            handle = uuid.uuid4().hex
            backend.open(handle, view.unit["id"])
            try:
                out = runner(text, url, handle)
                error = out["error"] or (None if set(PROBE_TOOLS) <= set(out["tools_offered"]) else "probe tools not offered")
            except (RuntimeError, json.JSONDecodeError, subprocess.TimeoutExpired) as e:
                out, error = {}, str(e)
            calls = backend.close(handle)
            answer = None if error else parse(out["reply"])
            attempts.append({"error": error or (None if answer else "no valid label"),
                             "cost_usd": out.get("cost_usd", 0.0), "tool_calls": len(calls), "reply": out.get("reply", "")})
            if guard:
                guard.add(attempts[-1]["cost_usd"])
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
    if record["error"] != "spend cap reached":
        path.write_text(json.dumps(record))
    return record


def load_model(model: str):
    local_dir = WEIGHTS_DIR / f"pythia-{model}"
    snapshot_download(f"EleutherAI/pythia-{model}", local_dir=local_dir, allow_patterns=["config.json", "model.safetensors"])
    return GPTNeoXForCausalLM.from_pretrained(local_dir, torch_dtype=torch.float16).to(DEVICE).eval(), local_dir


def read_model(model: str, units: list[dict], jobs: list[tuple[str, str, str]], directories: dict[tuple[str, str], Path],
               guards: dict[str, SpendGuard], corpus: np.ndarray, view_class=UnitView, backend_options: dict | None = None,
               runner=run_cli, parse=parse_answer) -> list[dict]:
    """The record of each (unit id, agent, reading) job of one model, loading Pythia only if some
    reading isn't cached. The keyword arguments let a later version change the view, the backend's
    budget and tools, the runner and the parser; their defaults are v8's."""
    v6 = dict(np.load(v6_path(model, DEPTH)))
    v5 = np.load(sweep_path(model, DEPTH))
    views = {u["id"]: view_class(u, v6, v5["max"], v5["pos"], corpus) for u in units}
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    def cached(job):
        unit_id, agent, reading = job
        path = cache_path(directories[agent, reading], unit_id)
        return path.exists() and json.loads(path.read_text())["prompt_sha256"] == hashlib.sha256(
            views[unit_id].prompt().encode()).hexdigest()

    pending = [job for job in jobs if not cached(job)]
    backend, url, local_dir = None, None, None
    if pending:
        print(f"pythia-{model}: {len(pending)} of {len(jobs)} readings to run", flush=True)
        WORKDIR.mkdir(parents=True, exist_ok=True)
        pythia, local_dir = load_model(model)
        backend = Backend(views, TextReader(pythia, int(v6["layer"])), **(backend_options or {}))
        url = backend.serve()

    def run(job):
        unit_id, agent, reading = job
        record = read(views[unit_id], directories[agent, reading], backend, url, partial(runner, model=agent),
                      guards[agent], parse)
        if job in pending:
            print(f"  {unit_id} {agent} {reading}: {record['label']} ({len(record['tool_calls'])} tool calls, "
                  f"${record['cost_usd']:.3f})", flush=True)
        return record

    try:
        with ThreadPoolExecutor(WORKERS) as pool:
            return list(pool.map(run, jobs))
    finally:
        if backend:
            backend.stop()
        if local_dir:
            shutil.rmtree(local_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true", help="read a few pythia-160m units outside v8's sample")
    parser.add_argument("--pilot-agent", default=AGENT_MODEL, help="the agent's model in the pilot")
    args = parser.parse_args()
    agents = [args.pilot_agent] if args.pilot else [AGENT_MODEL, SECOND_AGENT_MODEL]
    directories = {(agent, reading): V8_DIR / (f"pilot_{agent}" if args.pilot else f"agent_{agent}{suffix}")
                   for agent in agents for reading, suffix in (("first", ""), ("retest", "_retest"))}
    guards = {agent: SpendGuard(SPEND_CAP_USD.get(agent, 1.0), spent=sum(
        json.loads(f.read_text())["cost_usd"] for (a, _), d in directories.items() if a == agent and d.exists()
        for f in d.glob("*.json"))) for agent in agents}
    corpus = load_corpus()
    out = {agent: {"first": {}, "retest": {}} for agent in agents}
    for model in [PILOT_MODEL] if args.pilot else MODELS:
        units = choose(build_units(model), model)
        if args.pilot:
            units = [u for kind in ("real", "pooled", "related") for u in [x for x in units if x["kind"] == kind][:PILOT_PER_KIND]]
        again = set() if args.pilot else set(retest_ids(units, model))
        jobs = [(u["id"], agent, "first") for agent in agents for u in units]
        jobs += [(u["id"], agent, "retest") for agent in agents for u in units if u["id"] in again]
        for (unit_id, agent, reading), record in zip(jobs, read_model(model, units, jobs, directories, guards, corpus)):
            out[agent][reading][unit_id] = record
    for agent, parts in out.items():
        readings = [r for part in parts.values() for r in part.values()]
        print(f"{agent}: {len(readings)} readings, {sum(r['label'] is None for r in readings)} without a label, "
              f"${sum(r['cost_usd'] for r in readings):.2f} of quota", flush=True)
    if args.pilot:
        (RESULTS_DIR / f"v8_pilot_{args.pilot_agent}_answers.json").write_text(json.dumps(out[args.pilot_agent], indent=1))
    else:
        ANSWERS_PATH.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
