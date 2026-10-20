import numpy as np

from recgen.rubric_v13 import N_NEURONS, select


def test_select_draws_only_flagged_neurons_with_a_fixed_seed():
    flagged = np.zeros(2048, bool)
    flagged[::3] = True
    a, b = select(2048, flagged), select(2048, flagged)
    assert len(a) == N_NEURONS and np.array_equal(a, b) and flagged[a].all()
    assert list(a) == sorted(a)


def test_select_takes_all_when_few_are_flagged():
    flagged = np.zeros(100, bool)
    flagged[[3, 7]] = True
    assert list(select(100, flagged)) == [3, 7]
