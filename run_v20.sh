#!/usr/bin/env bash
# v20 pipeline. Needs data/corpus.npy, v5's and v6's caches and results, v12's and v17's Haiku answers and
# v17's results. Recording skips models already in data/v20/; Haiku is called only for readings not cached there.
set -euo pipefail
uv run python -m recgen.record_v20
uv run python -m recgen.read_v20
uv run python -m recgen.analyze_v20
uv run python -m recgen.plots_v20
