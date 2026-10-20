#!/usr/bin/env bash
# v13 pipeline (record_v13 builds the candidates on first run). Needs v6's data/v6/ recordings and results, and v4's results. Recording skips checkpoints cached in data/v13/.
set -euo pipefail
uv run python -m recgen.record_v13
uv run python -m recgen.analyze_v13
uv run python -m recgen.plots_v13
