"""v8's tool backend: what the agent can learn about one unit, on the scale of recgen/rubric_v8.

A unit's ranked list comes from v5's window maxima (data/v5): one example per window, at the token
where the unit fires most. Its first 50 examples are v7's, with values rebuilt from v6's stored
residuals (data/v6), since a pool's other neuron was never recorded at those tokens. Text the
agent writes goes through Pythia itself. A local HTTP server answers each agent's tool calls
(recgen/mcp_v8.py forwards them) and logs every call and answer.
"""

import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch

from recgen.config import SEED
from recgen.facets import display_text, tokenizer
from recgen.rubric_v7 import SNIPPET_AFTER, SNIPPET_BEFORE
from recgen.rubric_v8 import (
    EXAMPLES_PER_CALL, MARK_THRESHOLD, MAX_TEXT_TOKENS, MAX_TOOL_CALLS, NO_CALLS_LEFT, SCALE, TEXTS_PER_CALL, TOOLS,
    normalizers, prompt, ranked_rest, snippet_line, unit_values,
)


def v7_order(unit_id: str, n: int) -> np.ndarray:
    """The order v7 showed a unit's examples in (recgen.llm_reader.unit_prompt)."""
    seed = int(hashlib.sha256(unit_id.encode()).hexdigest()[:8], 16)
    return np.random.default_rng([SEED, 7, seed]).permutation(n)


def rebuilt_activations(v6: dict, rows: np.ndarray, samples: list[int]) -> np.ndarray:
    """The sampled neurons' activations [len(rows), k] at v6's stored positions, from the residuals."""
    directions = v6["weights"][samples].astype(np.float32) * v6["gain"].astype(np.float32)
    pre = v6["z"][rows].astype(np.float32) @ directions.T + v6["constant"][samples]
    return torch.nn.functional.gelu(torch.from_numpy(pre)).numpy()


class UnitView:
    """One unit as the agent sees it: its ranked examples and values on the 0-10 scale."""

    def __init__(self, unit: dict, v6: dict, window_max: np.ndarray, window_pos: np.ndarray, corpus: np.ndarray):
        self.unit, self.corpus = unit, corpus
        self.neuron_ids = [int(v6["neurons"][s]) for s in unit["neurons"]]
        maxima = window_max[:, self.neuron_ids].astype(np.float64)
        self.norms = normalizers(maxima)
        per_window = maxima / self.norms
        window_values = per_window.max(axis=1)
        self.top = float(window_values.max())
        first_keys = unit["keys"].astype(int)
        first_values = unit_values(rebuilt_activations(v6, unit["rows"], list(unit["neurons"])), self.norms)
        self.first = [(tuple(k), SCALE * v / self.top) for k, v in zip(first_keys, first_values)]
        rest = ranked_rest(window_values, first_keys[:, 0])
        strongest = window_pos[:, self.neuron_ids][rest, per_window[rest].argmax(axis=1)]
        self.ranked = sorted(self.first, key=lambda kv: -kv[1]) + [
            ((int(w), int(p)), SCALE * float(window_values[w]) / self.top) for w, p in zip(rest, strongest)]

    def snippet(self, key: tuple[int, int]) -> str:
        return display_text(self.corpus, key[0], key[1], SNIPPET_BEFORE, SNIPPET_AFTER)

    def prompt(self) -> str:
        """v7's prompt for this unit, with each snippet's activation in front of it."""
        order = v7_order(self.unit["id"], len(self.first))
        return prompt([snippet_line(i + 1, self.first[j][1], self.snippet(self.first[j][0])) for i, j in enumerate(order)])

    def examples(self, start: int, count: int) -> str:
        start, count = max(1, int(start)), max(1, min(int(count), EXAMPLES_PER_CALL))
        lines = [f"rank {r} ({value:.1f}): {self.snippet(key)}"
                 for r, (key, value) in enumerate(self.ranked[start - 1:start - 1 + count], start)]
        return "\n".join(lines) if lines else f"No examples at rank {start}; ranks go from 1 to {len(self.ranked)}."


class TextReader:
    """Pythia's activations of chosen neurons at the probed layer on text the agent writes."""

    def __init__(self, model, layer: int):
        self.model, self.layer, self.tok = model, layer, tokenizer()
        self.device = next(model.parameters()).device
        self.lock = threading.Lock()

    def activations(self, text: str, neuron_ids: list[int]) -> tuple[list[int], np.ndarray]:
        """The text's tokens (at most MAX_TEXT_TOKENS) and the neurons' activations at each [T, k]. An
        end-of-text token goes first, as between documents in training, and isn't returned."""
        ids = self.tok(text)["input_ids"][:MAX_TEXT_TOKENS]
        if not ids:
            return [], np.zeros((0, len(neuron_ids)))
        captured = {}

        class Stop(Exception):
            pass

        def hook(_module, _inputs, output):
            captured["acts"] = output[0, 1:, neuron_ids].float().cpu().numpy()
            raise Stop

        with self.lock:
            handle = self.model.gpt_neox.layers[self.layer].mlp.act.register_forward_hook(hook)
            try:
                with torch.no_grad():
                    self.model(input_ids=torch.tensor([[self.tok.eos_token_id] + ids], device=self.device))
            except Stop:
                pass
            finally:
                handle.remove()
        return ids, captured["acts"]

    def describe(self, view: UnitView, text: str) -> str:
        ids, acts = self.activations(text, view.neuron_ids)
        if not ids:
            return "(empty text)"
        values = SCALE * unit_values(acts, view.norms) / view.top
        pieces = [self.tok.decode([i]) for i in ids]
        marked = "".join(f"[[{p}|{v:.1f}]]" if v >= MARK_THRESHOLD else p for p, v in zip(pieces, values))
        best = int(np.argmax(values))
        return f"strongest {values[best]:.1f} at [[{pieces[best]}]]: {marked}".replace("\n", "⏎")


class Backend:
    """Answers tool calls for the units being read, one session per reading, and logs them all."""

    def __init__(self, views: dict[str, UnitView], reader: TextReader | None):
        self.views, self.reader = views, reader
        self.sessions: dict[str, dict] = {}
        self.lock = threading.Lock()
        self.server = None

    def open(self, handle: str, unit_id: str) -> None:
        with self.lock:
            self.sessions[handle] = {"unit": unit_id, "calls": []}

    def close(self, handle: str) -> list[dict]:
        with self.lock:
            return self.sessions.pop(handle)["calls"]

    def call(self, handle: str, tool: str, arguments: dict) -> str:
        with self.lock:
            session = self.sessions[handle]
            number = len(session["calls"]) + 1
            session["calls"].append({"tool": tool, "arguments": arguments, "answer": None})
        if number > MAX_TOOL_CALLS:
            answer = NO_CALLS_LEFT
        else:
            answer = self.answer(self.views[session["unit"]], tool, arguments)
            left = MAX_TOOL_CALLS - number
            answer += f"\n\n({left} tool call{'s' if left != 1 else ''} left)" if left else f"\n\n({NO_CALLS_LEFT})"
        session["calls"][number - 1]["answer"] = answer
        return answer

    def answer(self, view: UnitView, tool: str, arguments: dict) -> str:
        try:
            if tool == "examples":
                return view.examples(int(arguments["start"]), int(arguments["count"]))
            if tool == "activation":
                texts = arguments["texts"]
                if isinstance(texts, str):
                    texts = [texts]
                if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
                    return "texts must be a list of strings."
                notes = [f"Text {i + 1}, {self.reader.describe(view, t)}" for i, t in enumerate(texts[:TEXTS_PER_CALL])]
                if len(texts) > TEXTS_PER_CALL:
                    notes.append(f"(Only the first {TEXTS_PER_CALL} texts were read.)")
                return "\n".join(notes)
        except (KeyError, TypeError, ValueError) as e:
            return f"Invalid arguments: {e}"
        return f"Unknown tool {tool}."

    def serve(self) -> str:
        """Start the HTTP server on a free local port; returns its URL."""
        backend = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.reply({"tools": TOOLS})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                self.reply({"text": backend.call(body["handle"], body["tool"], body.get("arguments") or {})})

            def reply(self, payload):
                data = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *_args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def stop(self) -> None:
        if self.server:
            self.server.shutdown()
            self.server.server_close()
