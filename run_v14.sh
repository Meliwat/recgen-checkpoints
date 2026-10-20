#!/usr/bin/env bash
# v14 pipeline. Needs v5's data/v5/ window maxima, v6's data/v6/ recordings, results/classification_v7.json
# and results/v8_agent_answers.json. The Opus agent calls the claude CLI (and loads Pythia) only for readings
# not already cached in data/v14/.
set -euo pipefail
uv run python -m recgen.agent_v14
uv run python -m recgen.analyze_v14
uv run python -m recgen.plots_v14
