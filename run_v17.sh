#!/usr/bin/env bash
# v17 pipeline. Needs data/corpus.npy, v5's and v6's caches and results, and v12's Haiku answers.
# Recording skips models already in data/v17/; Haiku is called only for readings not cached there.
set -euo pipefail
uv run python -m recgen.record_v17
uv run python -m recgen.read_v17
uv run python -m recgen.analyze_v17
uv run python -m recgen.plots_v17
