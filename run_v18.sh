#!/usr/bin/env bash
# v18 pipeline. Needs v5's data/v5/ window maxima, v6's data/v6/ recordings, results/v15_answers.json,
# results/classification_v8.json and results/v14_opus_extract.json. Calls the claude CLI (and loads Pythia)
# only for readings not already cached in data/v18/.
set -euo pipefail
uv run python -m recgen.agent_v18
uv run python -m recgen.analyze_v18
uv run python -m recgen.plots_v18
