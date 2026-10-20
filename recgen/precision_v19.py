"""Post hoc check of v19's recording, not pre-registered. The pre-registered check (a float32 forward pass on the
CPU against the stored float16 window maxima) came out at 0.013 for GPT-2 small and 0.009 for large, but 3.97 for
medium. This asks which computation is wrong, for one size: next-token loss on LOSS_WINDOWS windows, and each
hidden state's largest relative difference from a float64 CPU reference up to the recorded layer, for float32 on
the CPU and float32 and float16 on MPS (the recording's device); then float32 window maxima on MPS against the
stored float16 ones over MAXIMA_WINDOWS windows.

Writes results/v19_precision_check.json. It downloads the model and deletes it after, so it isn't part of
run_v19.sh: uv run python -m recgen.precision_v19 gpt2-medium
"""

import json
import shutil
import sys

import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import GPT2LMHeadModel

from recgen.config import RESULTS_DIR, WEIGHTS_DIR
from recgen.gpt2_v19 import load_gpt2_corpus, maxima_path, sampled_top_examples, window_maxima
from recgen.rubric_v19 import HUB

LOSS_WINDOWS = 4
MAXIMA_WINDOWS = 64
RUNS = {"cpu float64": (torch.float64, "cpu"), "cpu float32": (torch.float32, "cpu"),
        "mps float32": (torch.float32, "mps"), "mps float16": (torch.float16, "mps")}


def check(name: str) -> dict:
    corpus = load_gpt2_corpus()
    stored = np.load(maxima_path(name))
    layer = int(stored["layer"])
    neurons = sampled_top_examples(name)[0]
    ids = torch.from_numpy(corpus[:LOSS_WINDOWS]).long()
    local_dir = WEIGHTS_DIR / name
    snapshot_download(f"{HUB}/{name}", local_dir=local_dir, allow_patterns=["config.json", "model.safetensors"])
    loss, hidden = {}, {}
    try:
        for label, (dtype, device) in RUNS.items():
            model = GPT2LMHeadModel.from_pretrained(local_dir, dtype=dtype).to(device).eval()
            with torch.no_grad():
                out = model(input_ids=ids.to(device), labels=ids.to(device), output_hidden_states=True)
            loss[label] = float(out.loss)
            hidden[label] = [h.float().cpu().double() for h in out.hidden_states[:layer + 1]]
            if label == "mps float32":
                maxima = window_maxima(model, corpus[:MAXIMA_WINDOWS], layer)[0]
            del model, out
    finally:
        shutil.rmtree(local_dir)
    reference = hidden["cpu float64"]
    differences = {label: [float((h - g)[:, 1:].abs().max() / g[:, 1:].abs().max()) for h, g in zip(hidden[label], reference)]
                   for label in RUNS if label != "cpu float64"}  # position 0 left out: its residual norm is huge in GPT-2
    stored_max = stored["max"][:MAXIMA_WINDOWS, neurons].astype(np.float32)
    return {"model": name, "layer": layer, "next_token_loss": loss,
            "largest_relative_difference_from_cpu_float64_by_hidden_state": differences,
            "mps_float32_vs_stored_float16_window_maxima": {
                "windows": MAXIMA_WINDOWS, "largest_difference": float(np.abs(maxima[:, neurons] - stored_max).max())}}


def main() -> None:
    result = check(sys.argv[1])
    (RESULTS_DIR / "v19_precision_check.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
