#!/usr/bin/env bash
# v21 pipeline. Needs v5's data/v5/ window maxima, v6's data/v6/ recordings, results/v7_llm_answers.json,
# results/v8_agent_answers.json, results/v16_answers.json, v10's Sonnet cache in data/v10/sonnet/, v14's Opus
# agent cache in data/v14/ (read only) and the claude CLI signed in.
# The readers call the claude CLI (and load Pythia) only for readings not already cached in data/v21/.
set -euo pipefail
uv run python -m recgen.reader_v21
uv run python -m recgen.analyze_v21
uv run python -m recgen.plots_v21
