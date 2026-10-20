#!/usr/bin/env bash
# v16 pipeline. Needs v6's data/v6/ recordings, results/v7_llm_answers.json, v7's Haiku cache in data/v7/llm/ (for the
# post-hoc thinking split) and v10's Sonnet cache in data/v10/sonnet/.
# The readers (the main run, then the concurrent addendum) call the claude CLI only for readings not already
# cached in data/v16/.
set -euo pipefail
uv run python -m recgen.llm_reader_v16
uv run python -m recgen.llm_reader_v16_addendum
uv run python -m recgen.analyze_v16
uv run python -m recgen.plots_v16
