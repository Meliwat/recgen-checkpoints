#!/usr/bin/env bash
# v8 pipeline. Needs v5's data/v5/ window maxima, v6's data/v6/ recordings and results/classification_v7.json.
# The agents call the claude CLI (and load Pythia) only for readings not already cached in data/v8/.
set -euo pipefail
uv run python -m recgen.agent_v8
uv run python -m recgen.analyze_v8
uv run python -m recgen.plots_v8
