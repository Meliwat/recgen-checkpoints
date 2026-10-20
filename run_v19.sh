#!/usr/bin/env bash
# v19 pipeline: GPT-2 small, medium and large. Needs data/corpus.npy, plus the pile-10k dataset the first time
# (to rebuild v1's stream for GPT-2's corpus). Recording skips models already cached in data/v19/, and Haiku
# is called only for units not cached in data/v19/llm/.
set -euo pipefail
uv run python -m recgen.gpt2_v19
uv run python -m recgen.read_v19
uv run python -m recgen.analyze_v19
uv run python -m recgen.plots_v19
