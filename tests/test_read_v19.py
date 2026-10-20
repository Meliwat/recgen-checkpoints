import pytest

from recgen import read_v19
from recgen.read_v19 import Budget, SpendGuardReached


def test_budget_starts_no_call_once_the_reported_spend_reaches_the_limit(monkeypatch):
    calls = []

    def fake_cli(text):
        calls.append(text)
        return {"reply": '{"label": "A"}', "models": [], "usage": {}, "cost_usd": 0.6}

    monkeypatch.setattr(read_v19, "call_cli", fake_cli)
    budget = Budget(limit=1.0, spent=0.5)
    budget.runner("first")
    assert abs(budget.spent - 1.1) < 1e-12
    with pytest.raises(SpendGuardReached):
        budget.runner("second")
    assert calls == ["first"]
