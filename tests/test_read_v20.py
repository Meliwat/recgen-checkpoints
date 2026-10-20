import numpy as np

import recgen.read_v20 as read_v20
from recgen.read_v20 import rank_units, retest_units, unit_id
from recgen.rubric_v5 import MODELS
from recgen.rubric_v20 import RETEST


def test_unit_ids_name_the_ranks_and_differ_from_v17s():
    assert unit_id("70m", "band", 7) == "70m:rank201-500::7"
    assert unit_id("70m", "band2", 7) == "70m:rank1001-2000::7"
    assert unit_id("70m", "band", 7) != "70m:band::7"


def test_rank_units_pair_each_fresh_neuron_with_its_band_sample(monkeypatch):
    fresh = [{"id": f"70m:fresh::{s}", "neurons": (s,), "neuron_id": 100 + s, "quartile": s % 4} for s in range(4)]
    keys = np.arange(4 * 50 * 2).reshape(4, 50, 2)
    monkeypatch.setattr(read_v20, "fresh_units", lambda model: fresh)
    monkeypatch.setattr(read_v20, "rank_band_examples",
                        lambda model, band: {"keys": keys, "included": np.array([True, False, True, True])})
    units = rank_units("70m", "band")
    assert [u["neuron_id"] for u in units] == [100, 102, 103] and units[1]["top"] is fresh[2]
    assert np.array_equal(units[1]["keys"], keys[2]) and units[1]["quartile"] == 2


def test_retest_units_are_a_fixed_draw_of_five_per_model():
    units = [{"id": i} for i in range(100)]
    picked = retest_units(units, "70m")
    assert len(picked) == RETEST // len(MODELS) == 5 and picked == retest_units(units, "70m")
    assert [u["id"] for u in picked] == sorted(u["id"] for u in picked) and picked != retest_units(units, "160m")
