#!/usr/bin/env bash
# v9 pipeline. Needs v5's data/v5/ cache, v6's data/v6/ recordings and v7's results.
# Haiku is called only for unions not already cached in data/v9/.
set -euo pipefail
uv run python -m recgen.read_v9
uv run python -m recgen.analyze_v9
uv run python -m recgen.plots_v9
