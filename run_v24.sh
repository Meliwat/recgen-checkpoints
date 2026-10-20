#!/usr/bin/env bash
# v24 pipeline. Needs v6's data/v6/ recordings, the v20 branch (v20's random-text answers, read through git),
# results/v7_llm_answers.json, results/v16_answers.json, results/v21_answers.json, v10's Sonnet cache in
# data/v10/sonnet/ (read only) and the claude CLI signed in.
# The reader calls the claude CLI only for readings not already cached in data/v24/.
set -euo pipefail
uv run python -m recgen.reader_v24
uv run python -m recgen.analyze_v24
uv run python -m recgen.plots_v24
