"""These read v6's recordings (data/v6/, not committed) and are skipped without them."""

import numpy as np
import pytest

from recgen import units_v7
from recgen.rubric_v5 import MODELS
from recgen.rubric_v7 import MIXES, N_POOLED, N_REAL, N_RELATED
from recgen.rubric_v11 import DEPTHS, RETEST_PER_MODEL
from recgen.sweep_v6 import v6_path
from recgen.units_v11 import build_units, retest_ids

needs_v6 = pytest.mark.skipif(not all(v6_path(m, d).exists() for m in MODELS for d in ("quarter", "middle", "three quarters")),
                              reason="needs v6's recordings in data/v6/")


@needs_v6
@pytest.mark.parametrize("model", MODELS)
def test_the_middle_depth_rebuilds_v7s_units_and_retests(model):
    v7, v11 = units_v7.build_units(model), build_units(model, "middle")
    assert [(u["kind"], u["mix"], u["neurons"]) for u in v7] == [(u["kind"], u["mix"], u["neurons"]) for u in v11]
    assert all(np.array_equal(a["rows"], b["rows"]) and np.array_equal(a["keys"], b["keys"]) for a, b in zip(v7, v11))
    v7_positions = [[u["id"] for u in v7].index(i) for i in units_v7.retest_ids(v7, model)]
    v11_positions = [[u["id"] for u in v11].index(i) for i in retest_ids(v11, model)]
    assert v7_positions == v11_positions


@needs_v6
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("depth", DEPTHS)
def test_each_depth_has_v7s_unit_counts_with_depth_in_every_id(model, depth):
    units = build_units(model, depth)
    kinds = [(u["kind"], u["mix"]) for u in units]
    assert kinds.count(("real", None)) == N_REAL and kinds.count(("related", "50/50")) == N_RELATED
    assert all(kinds.count(("pooled", mix)) == N_POOLED for mix in MIXES)
    ids = [u["id"] for u in units]
    assert len(set(ids)) == len(ids) and all(i.startswith(f"{model}:{depth.replace(' ', '_')}:") for i in ids)
    assert len(retest_ids(units, model)) == RETEST_PER_MODEL
    # Real units take one example per window, as in v7.
    assert all(len({int(w) for w, _ in u["keys"]}) == len(u["keys"]) for u in units if u["kind"] == "real")
