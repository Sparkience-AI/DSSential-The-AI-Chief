from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from importlib.metadata import version

import numpy as np
import pandas as pd
import pytest

from dssential.forecasting import ForecastSpec, forecast


def _frame(values: np.ndarray, frequency: str = "D") -> pd.DataFrame:
    return pd.DataFrame({"when": pd.date_range("2025-01-01", periods=len(values), freq=frequency), "revenue": values})


def test_linear_trend_beats_naive_and_forecasts_in_original_units() -> None:
    result = forecast(
        _frame(np.arange(1.0, 41.0) * 100),
        ForecastSpec("when", "revenue", "D", 3, "USD", validation_points=30),
        input_fingerprint="abc",
    )
    assert result.validation.selected_model == "linear_trend"
    assert result.validation.evidence_status == "validated_improvement"
    assert [point.prediction for point in result.points] == pytest.approx([4100, 4200, 4300])
    assert result.manifest.input_fingerprint == "abc"
    assert len(result.manifest.result_hash) == 64
    assert result.narration_summary()["unit"] == "USD"


def test_baseline_only_when_trend_does_not_materially_improve() -> None:
    result = forecast(
        _frame(np.repeat(7.0, 30)),
        ForecastSpec("when", "revenue", "D", 2, "items", validation_points=18),
        input_fingerprint="abc",
    )
    assert result.validation.selected_model == "last_value"
    assert result.validation.evidence_status == "baseline_only"
    assert any("insufficient" in warning for warning in result.warnings)


def test_seasonal_naive_is_an_explicit_baseline() -> None:
    result = forecast(
        _frame(np.tile(np.array([10.0, 20.0, 30.0, 40.0]), 10)),
        ForecastSpec("when", "revenue", "D", 4, "items", validation_points=18, seasonal_period=4),
        input_fingerprint="abc",
    )
    assert result.validation.selected_model == "seasonal_naive"
    assert [point.prediction for point in result.points] == [10, 20, 30, 40]


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda frame: frame.drop(columns="when"), "Missing required"),
        (lambda frame: frame.assign(when="not-a-date"), "invalid"),
        (
            lambda frame: frame.assign(when=lambda value: value["when"].mask(value.index == 2, value["when"].iloc[1])),
            "unique",
        ),
        (lambda frame: frame.drop(index=2), "regular frequency"),
        (lambda frame: frame.iloc[::-1].reset_index(drop=True), "ordered"),
        (lambda frame: frame.assign(revenue=np.inf), "finite"),
    ],
)
def test_invalid_time_series_are_refused(mutator: Callable[[pd.DataFrame], pd.DataFrame], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        forecast(
            mutator(_frame(np.arange(30.0))),
            ForecastSpec("when", "revenue", "D", 2, "items", validation_points=18),
            input_fingerprint="abc",
        )


def test_result_is_serializable_and_intervals_are_ordered() -> None:
    result = forecast(
        _frame(np.arange(30.0)),
        ForecastSpec("when", "revenue", "D", 2, "items", validation_points=18),
        input_fingerprint="abc",
    )
    assert asdict(result)
    assert all(point.lower <= point.prediction <= point.upper for point in result.points)
    assert (
        result.validation.selection_points + result.validation.calibration_points + result.validation.test_points == 18
    )


def test_run_identity_is_unique_but_reproducibility_hash_is_stable() -> None:
    frame = _frame(np.arange(30.0))
    spec = ForecastSpec("when", "revenue", "D", 2, "items", validation_points=18)
    first = forecast(frame, spec, input_fingerprint="abc")
    second = forecast(frame, spec, input_fingerprint="abc")
    assert first.manifest.run_id != second.manifest.run_id
    assert first.manifest.reproducibility_hash == second.manifest.reproducibility_hash
    assert first.manifest.source_revision == "unavailable"
    assert any("Source revision is unavailable" in warning for warning in first.warnings)


def test_manifest_records_execution_identity_and_source_revision() -> None:
    result = forecast(
        _frame(np.arange(40.0)),
        ForecastSpec("when", "revenue", "D", 2, "items"),
        input_fingerprint="abc",
        source_revision="git:0123456789abcdef",
    )
    assert result.manifest.method_version == "transparent-baselines-v2"
    assert result.manifest.code_version == version("dssential")
    assert result.manifest.source_revision == "git:0123456789abcdef"
    assert set(result.manifest.runtime_versions) == {"python", "implementation"}
    assert set(result.manifest.dependency_versions) == {"dssential", "numpy", "pandas"}


def test_source_revision_is_part_of_reproducibility_identity() -> None:
    frame = _frame(np.arange(40.0))
    spec = ForecastSpec("when", "revenue", "D", 2, "items")
    first = forecast(frame, spec, input_fingerprint="abc", source_revision="git:first")
    second = forecast(frame, spec, input_fingerprint="abc", source_revision="git:second")
    assert first.manifest.reproducibility_hash != second.manifest.reproducibility_hash

    with pytest.raises(ValueError, match="Source revision cannot be blank"):
        forecast(frame, spec, input_fingerprint="abc", source_revision="")


def test_minimum_validation_policy_and_limited_evidence_warning() -> None:
    with pytest.raises(ValueError, match="At least 18 rolling validation points"):
        forecast(
            _frame(np.arange(40.0)),
            ForecastSpec("when", "revenue", "D", 2, "items", validation_points=17),
            input_fingerprint="abc",
        )
    result = forecast(
        _frame(np.arange(40.0)),
        ForecastSpec("when", "revenue", "D", 2, "items", validation_points=18),
        input_fingerprint="abc",
    )
    assert result.validation.selection_points == 6
    assert result.validation.calibration_points == 6
    assert result.validation.test_points == 6
    assert any("Limited validation evidence" in warning for warning in result.warnings)
