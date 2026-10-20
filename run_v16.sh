#!/usr/bin/env bash
# v16 pipeline. Needs v6's data/v6/ recordings, results/v7_llm_answers.json and v10's Sonnet cache in data/v10/sonnet/.
# The readers call the claude CLI only for readings not already cached in data/v16/.
set -euo pipefail
uv run python -m recgen.llm_reader_v16
uv run python -m recgen.analyze_v16
uv run python -m recgen.plots_v16
