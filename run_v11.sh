#!/usr/bin/env bash
# v11 pipeline. Needs v6's data/v6/ recordings, v5's data/v5/embeddings.npz, and v6's and v7's results.
# The language-model reader calls the claude CLI only for units not already cached in data/v11/.
set -euo pipefail
uv run python -m recgen.llm_reader_v11
uv run python -m recgen.analyze_v11
uv run python -m recgen.plots_v11
