#!/usr/bin/env bash
# v10 pipeline. Needs v6's data/v6/ recordings, v5's data/v5/embeddings.npz and v7's Haiku answers
# (results/v7_llm_answers.json). The readers call the claude CLI only for units not already cached in data/v10/.
set -euo pipefail
uv run python -m recgen.llm_reader_v10 sonnet
uv run python -m recgen.llm_reader_v10 opus
uv run python -m recgen.analyze_v10
uv run python -m recgen.plots_v10
