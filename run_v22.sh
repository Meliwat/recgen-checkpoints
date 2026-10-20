#!/usr/bin/env bash
# v22 pipeline: v19's GPT-2 neurons on OpenWebText. Needs v19's recordings (data/v19/*_network.npz, for the sampled
# neurons), results/classification_v19.json, results/v7_llm_answers.json and the claude CLI signed in. The corpus is
# downloaded once into data/v22/openwebtext-10k/. Recording skips models already cached in data/v22/, and Haiku is
# called only for units not cached in data/v22/llm/.
set -euo pipefail
uv run python -m recgen.gpt2_v22
uv run python -m recgen.read_v22
uv run python -m recgen.analyze_v22
uv run python -m recgen.plots_v22
