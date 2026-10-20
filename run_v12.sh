#!/usr/bin/env bash
# v12 pipeline. Needs v5's and v6's caches and results. Haiku is called only for neurons not cached in data/v12/.
set -euo pipefail
uv run python -m recgen.read_v12
uv run python -m recgen.analyze_v12
uv run python -m recgen.plots_v12
