import numpy as np
import pytest

from recgen.rubric_v5 import MODELS
from recgen.rubric_v24 import (
    ARMS, GROUPS, N_PER_GROUP, NO_SIZABLE_FLOOR, RANDOM_UNIT_SIZE, REAL_BELOW_RANDOM, REAL_BETWEEN, REAL_BEYOND_POOLS,
    REAL_LIKE_POOLS, RETEST_PER_GROUP, SIZABLE_FLOOR, UNCLEAR_FLOOR, UNCLEAR_POSITION, UNDEFINED_POSITION, anchor_ids,
    floor_verdict, per_size_counts, position, position_reading, random_rng, random_unit_keys, reading_order, retest_ids,
)


def test_six_series_by_full_model_id():
    assert len(ARMS) == 6 and {m for m, _ in ARMS.values()} == {"claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"}
    assert ARMS["opus-E3"] == ("claude-opus-5-5", "E3") and ARMS["haiku-v7"] == ("claude-haiku-5-5", "v7")


def test_per_size_counts_spread_100_units_as_v20_does():
    counts = per_size_counts()
    assert list(counts.values()) == [17, 17, 17, 17, 16, 16] and sum(counts.values()) == N_PER_GROUP
    assert list(counts) == list(MODELS)


def test_random_units_follow_v20s_draw():
    positions = np.arange(1024 * 2).reshape(1024, 2)
    rng = np.random.default_rng([0, 20, 2, MODELS.index("70m"), 101])
    expected = positions[np.sort(rng.choice(1024, size=50, replace=False))]
    units = random_unit_keys(positions, 3, random_rng("70m"))
    assert len(units) == 3 and np.array_equal(units[0], expected)
    assert all(len(u) == RANDOM_UNIT_SIZE == len({tuple(k) for k in u.tolist()}) for u in units)


def test_anchor_and_retest_draws_are_seeded_and_distinct():
    ids = [f"70m:real::{i}" for i in range(100)]
    drawn = anchor_ids(ids, "70m", "real", 17)
    assert len(set(drawn)) == 17 and drawn == anchor_ids(ids, "70m", "real", 17)
    assert drawn != anchor_ids(ids, "160m", "real", 17) and drawn == sorted(drawn, key=ids.index)
    by_group = {g: [f"{g}{i}" for i in range(100)] for g in GROUPS}
    again = retest_ids(by_group)
    assert len(again) == 3 * RETEST_PER_GROUP == len(set(again)) and again == retest_ids(by_group)


def test_reading_order_is_a_seeded_permutation():
    order = reading_order(1800)
    assert sorted(order) == list(range(1800)) and order == reading_order(1800) and order != reading_order(1800, 1)


def test_floor_verdict():
    assert floor_verdict([0.12, 0.3]) == SIZABLE_FLOOR
    assert floor_verdict([0.0, 0.08]) == NO_SIZABLE_FLOOR
    assert floor_verdict([0.05, 0.15]) == UNCLEAR_FLOOR


def test_position_and_its_readings():
    assert position(real=0.5, random=0.1, pools=0.3) == pytest.approx(2.0)
    assert position_reading([-0.05, 0.1], [1.5, 3.0]) == UNDEFINED_POSITION
    assert position_reading([0.1, 0.3], None) == UNDEFINED_POSITION
    assert position_reading([0.1, 0.3], [0.8, 1.4]) == REAL_LIKE_POOLS
    assert position_reading([0.1, 0.3], [1.2, 4.0]) == REAL_BEYOND_POOLS
    assert position_reading([0.1, 0.3], [0.1, 0.7]) == REAL_BETWEEN
    assert position_reading([0.1, 0.3], [-0.9, -0.2]) == REAL_BELOW_RANDOM
    assert position_reading([0.1, 0.3], [-0.2, 0.6]) == UNCLEAR_POSITION
