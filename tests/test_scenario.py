import pytest

from dssential.scenario import ScenarioAdjustment, apply_scenario


def test_scenario_is_transparent_arithmetic() -> None:
    result = apply_scenario(ScenarioAdjustment("price", 100, "percent", 10, "USD", 0, 200))
    assert result.adjusted_value == pytest.approx(110)
    assert result.percent_change == pytest.approx(10)
    assert "not a predicted business outcome" in result.explanation


def test_bounds_are_enforced() -> None:
    with pytest.raises(ValueError, match="above"):
        apply_scenario(ScenarioAdjustment("price", 100, "absolute", 101, "USD", maximum=200))


def test_zero_base_has_no_misleading_percentage() -> None:
    assert apply_scenario(ScenarioAdjustment("users", 0, "absolute", 5, "people")).percent_change is None
