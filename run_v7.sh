#!/usr/bin/env bash
# v7 pipeline. Needs v6's data/v6/ recordings and v5's data/v5/embeddings.npz.
# The language-model reader calls the claude CLI only for units not already cached in data/v7/.
set -euo pipefail
uv run python -m recgen.llm_reader
uv run python -m recgen.analyze_v7
uv run python -m recgen.plots_v7
