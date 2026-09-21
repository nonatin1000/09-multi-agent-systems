from app.obs.request_stats import (
    estimate_prompt_cost_usd,
    is_free_model,
    parse_since_duration,
    percentile,
)


def test_is_free_model():
    assert is_free_model("meta-llama/llama-3:free")
    assert not is_free_model("openai/gpt-4o-mini")
    assert not is_free_model(None)


def test_estimate_prompt_cost_usd_free_and_zero_tokens():
    assert estimate_prompt_cost_usd("x:free", 1000) == 0
    assert estimate_prompt_cost_usd("openai/gpt-4o-mini", 0) == 0


def test_estimate_prompt_cost_usd_default_rate():
    cost = estimate_prompt_cost_usd("openai/gpt-4o-mini", 1_000_000)
    assert cost == 0.15


def test_parse_since_duration():
    assert parse_since_duration("24h") == 24 * 3_600_000
    assert parse_since_duration("7d") == 7 * 86_400_000
    assert parse_since_duration("30m") == 30 * 60_000
    assert parse_since_duration("90s") == 90_000
    assert parse_since_duration("bogus") is None
    assert parse_since_duration("0h") is None


def test_percentile_nearest_rank():
    values = [10, 20, 30, 40, 50]
    assert percentile(values, 50) == 30
    assert percentile([], 50) is None
