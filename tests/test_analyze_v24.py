from functools import partial

import numpy as np
import pytest

from recgen import analyze_v24
from recgen.rubric_v5 import MODELS
from recgen.rubric_v16 import bootstrap
from recgen.rubric_v24 import (
    ARMS, NO_SIZABLE_FLOOR, POOL, RANDOM, REAL, REAL_BEYOND_POOLS, SIZABLE_FLOOR, UNDEFINED_POSITION,
)


def unit_id(size: str, group: str, i: int) -> str:
    return {RANDOM: f"{size}:random::{i}", REAL: f"{size}:real::{i}", POOL: f"{size}:pooled:50/50:{i}-{i + 1}"}[group]


def reading(label: str) -> dict:
    return {"label": label, "reply": f'{{"label": "{label}", "patterns": ["x", "y"]}}', "models": ["m"],
            "cost_usd": 0.01, "input_tokens": 10, "output_tokens": 5, "thinking_tokens": 2, "prompt_sha256": "h"}


def labels_for(series: str, group: str, i: int) -> str:
    """Random text: all C, or a fifth A for Haiku under E3; real neurons: all A; pools: half A, half C."""
    if group == RANDOM:
        return "A" if series == "haiku-E3" and i % 5 == 0 else "C"
    if group == REAL:
        return "A"
    return "A" if i % 2 else "C"


@pytest.fixture
def result(monkeypatch):
    monkeypatch.setattr(analyze_v24, "bootstrap", partial(bootstrap, draws=50))
    ids = [(size, g, i) for size in MODELS for g in (RANDOM, REAL, POOL) for i in range(10)]
    series = {s: {"first": {unit_id(*k): reading(labels_for(s, k[1], k[2])) for k in ids},
                  "retest": {unit_id(size, g, 0): reading(labels_for(s, g, 0)) for size in MODELS[:2] for g in (RANDOM, REAL)}}
              for s in ARMS}
    answers = {"series": series, "pilot": {s: {} for s in ARMS}, "spend_usd": 1.0, "random_prompts_matching_v20": 60}
    v20 = {unit_id(size, RANDOM, i): {"label": "C", "prompt_sha256": "h"} for size in MODELS for i in range(10)}
    v7 = {"first": {unit_id(size, REAL, i): {"label": "B", "prompt_sha256": "h"} for size in MODELS for i in range(10)}}
    v16 = {"arms": {a: {"first": {}} for a in ("haiku-E3", "sonnet-E3")}}
    v21 = {"arm_a": {a: {"first": {}} for a in ("opus-v7", "opus-E3")}}
    return analyze_v24.analyze(answers, v20, v7, None, v16, v21, np.random.default_rng(0))


def test_unit_place_reads_all_three_groups():
    assert analyze_v24.unit_place("2.8b:random::3") == ("2.8b", RANDOM)
    assert analyze_v24.unit_place("70m:real::12") == ("70m", REAL)
    assert analyze_v24.unit_place("1b:pooled:50/50:3-9") == ("1b", POOL)


def test_floor_shares_and_verdicts(result):
    series = result["summary"]["series"]
    assert series["opus-v7"]["shares"][RANDOM]["A"]["value"] == 0 and series["opus-v7"]["floor"]["verdict"] == NO_SIZABLE_FLOOR
    assert series["haiku-E3"]["floor"]["value"] == pytest.approx(0.2) and series["haiku-E3"]["floor"]["verdict"] == SIZABLE_FLOOR
    assert series["opus-v7"]["shares"][POOL]["C"]["value"] == pytest.approx(0.5)
    assert series["opus-v7"]["shares"][REAL]["units"] == 60


def test_position_and_aucs(result):
    st = result["summary"]["series"]["opus-v7"]
    a_share = st["position"]["one (main) job (A)"]
    assert a_share["p"]["value"] == pytest.approx(2.0) and a_share["reading"] == REAL_BEYOND_POOLS
    assert st["gaps"]["one (main) job (A)"]["real minus random"]["value"] == pytest.approx(1.0)
    assert st["label_auc"]["random over real"]["value"] == pytest.approx(1.0)
    assert st["label_auc"]["random over pools"]["value"] == pytest.approx(0.75)
    assert st["label_auc"]["pools over real"]["value"] == pytest.approx(0.75)
    assert st["floor_corrected_real_one_job"]["value"] == pytest.approx(1.0)


def test_position_is_undefined_when_pools_look_like_random_text(monkeypatch):
    monkeypatch.setattr(analyze_v24, "bootstrap", partial(bootstrap, draws=50))
    rows = [{"id": f"{s}:{g}", "size": s, "group": g, "labels": {"x-v7": "A" if g == REAL else "C"}}
            for s in MODELS for g in (RANDOM, REAL, POOL) for _ in range(5)]
    st = analyze_v24.series_stats(rows, "x-v7", np.random.default_rng(0))
    assert st["position"]["one (main) job (A)"] == {"p": None, "reading": UNDEFINED_POSITION}


def test_e3_minus_v7_and_earlier_rounds(result):
    s = result["summary"]
    assert s["e3_minus_v7"]["haiku"][RANDOM]["value"] == pytest.approx(0.2)
    assert s["e3_minus_v7"]["opus"][REAL]["value"] == 0
    v20 = s["haiku_v7_random_against_v20"]
    assert v20["units"] == 60 and v20["same label"] == 1.0
    v7 = s["anchor_against_earlier_rounds"]["haiku-v7"]
    assert v7["units"] == 60 and v7["same label"] == 0.0
    assert v7["A share, now minus earlier"][REAL]["value"] == pytest.approx(1.0)
    assert "sonnet-v7" not in s["anchor_against_earlier_rounds"]  # no v10 cache given


def test_retest_usage_and_checks(result):
    assert result["summary"]["retest"]["opus-E3"]["units"] == 4
    assert result["usage"]["opus-E3"]["first readings"] == 180 and result["usage"]["opus-E3"]["retests"] == 4
    assert result["checks"]["haiku-v7"]["same prompt"] == 120  # v7's real neurons and v20's random text
    assert result["exploratory"]["one_job_answers"]["opus-v7"][REAL]["mean patterns"] == 2
