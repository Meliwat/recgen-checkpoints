#!/usr/bin/env bash
# v5 pipeline. Needs data/corpus.npy. Recording skips models already cached in data/v5/.
set -euo pipefail
uv run python -m recgen.sweep
uv run python -m recgen.analyze_v5
uv run python -m recgen.plots_v5
