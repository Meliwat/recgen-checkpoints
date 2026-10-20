#!/usr/bin/env bash
# v15 pipeline. Needs v5's data/v5/ window maxima, v6's data/v6/ recordings and results/classification_v8.json.
# Loads no Pythia model. Calls the claude CLI only for readings not already cached in data/v15/.
set -euo pipefail
uv run python -m recgen.simulate_v15
uv run python -m recgen.analyze_v15
uv run python -m recgen.plots_v15
