#!/usr/bin/env bash
# v6 pipeline. Needs data/corpus.npy, v5's cache (data/v5/) and results/classification_v5.json.
# Recording skips models already cached in data/v6/.
set -euo pipefail
uv run python -m recgen.sweep_v6
uv run python -m recgen.analyze_v6
uv run python -m recgen.plots_v6
