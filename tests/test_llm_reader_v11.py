from recgen.llm_reader_v11 import LLM_DIR, RETEST_DIR, collect
from recgen.rubric_v11 import DEPTHS


def record(label, tokens=10):
    return {"label": label, "error": None if label else "rate limited", "models": ["claude-haiku-5-5"],
            "prompt_sha256": "abc", "usage": {"input_tokens": tokens, "cache_read_input_tokens": 5}, "reply": "{}"}


def test_collect_groups_answers_by_depth_and_reading():
    jobs = [("quarter", "70m:quarter:real::3", "p", LLM_DIR), ("quarter", "70m:quarter:real::3", "p", RETEST_DIR),
            ("three quarters", "70m:three_quarters:real::3", "q", LLM_DIR)]
    answers = collect(jobs, [record("C"), record("A"), record(None)])
    assert set(answers) == set(DEPTHS)
    assert answers["quarter"]["first"]["70m:quarter:real::3"]["label"] == "C"
    assert answers["quarter"]["retest"]["70m:quarter:real::3"]["label"] == "A"
    entry = answers["three quarters"]["first"]["70m:three_quarters:real::3"]
    assert entry["label"] is None and entry["error"] == "rate limited" and entry["input_tokens"] == 15
    assert answers["three quarters"]["retest"] == {}
