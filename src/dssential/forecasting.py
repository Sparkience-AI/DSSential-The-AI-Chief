"""Transparent forecasting with baseline comparison and rolling validation."""

from __future__ import annotations

import json
import os
import platform
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from typing import Literal, cast
from uuid import uuid4

import numpy as np
import pandas as pd

ModelName = Literal["last_value", "seasonal_naive", "linear_trend"]
METHOD_VERSION = "transparent-baselines-v2"
MIN_VALIDATION_POINTS = 18
RECOMMENDED_VALIDATION_POINTS = 30


@dataclass(frozen=True)
class ForecastSpec:
    timestamp_column: str
    target_column: str
    frequency: str
    horizon: int
    unit: str
    confidence: float = 0.9
    validation_points: int = RECOMMENDED_VALIDATION_POINTS
    seasonal_period: int | None = None


@dataclass(frozen=True)
class MetricResult:
    model: ModelName
    mae: float
    rmse: float


@dataclass(frozen=True)
class ValidationResult:
    metrics: tuple[MetricResult, ...]
    selected_model: ModelName
    interval_coverage: float
    selection_points: int
    calibration_points: int
    test_points: int
    evidence_status: Literal["validated_improvement", "baseline_only"]


@dataclass(frozen=True)
class ForecastPoint:
    timestamp: str
    prediction: float
    lower: float
    upper: float


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    reproducibility_hash: str
    created_at: str
    input_fingerprint: str
    method_version: str
    code_version: str
    source_revision: str
    specification: dict[str, object]
    runtime_versions: dict[str, str]
    dependency_versions: dict[str, str]
    random_seed: None
    warnings: tuple[str, ...]
    limitations: tuple[str, ...]
    result_hash: str


@dataclass(frozen=True)
class ForecastResult:
    points: tuple[ForecastPoint, ...]
    validation: ValidationResult
    manifest: RunManifest
    warnings: tuple[str, ...]
    limitations: tuple[str, ...]

    def narration_summary(self) -> dict[str, object]:
        """Return the only payload permitted to reach an optional narrator."""
        return {
            "method": self.validation.selected_model,
            "forecast": [asdict(point) for point in self.points],
            "validation": asdict(self.validation),
            "unit": self.manifest.specification["unit"],
            "warnings": list(self.warnings),
            "limitations": list(self.limitations),
        }


def _validate_spec(spec: ForecastSpec) -> None:
    if not spec.timestamp_column or not spec.target_column:
        raise ValueError("Timestamp and target columns are required.")
    if not spec.frequency:
        raise ValueError("A confirmed pandas frequency (for example D, W, MS) is required.")
    if not spec.unit.strip():
        raise ValueError("The target unit is required.")
    if spec.horizon < 1 or spec.horizon > 365:
        raise ValueError("Forecast horizon must be between 1 and 365 periods.")
    if not 0.5 <= spec.confidence <= 0.99:
        raise ValueError("Confidence must be between 0.50 and 0.99.")
    if spec.validation_points < MIN_VALIDATION_POINTS:
        raise ValueError(
            f"At least {MIN_VALIDATION_POINTS} rolling validation points are required so selection, "
            "calibration, and test splits each contain at least 6 observations."
        )
    if spec.seasonal_period is not None and spec.seasonal_period < 2:
        raise ValueError("Seasonal period must be at least 2 observations.")


def _prepare(frame: pd.DataFrame, spec: ForecastSpec) -> pd.Series:
    _validate_spec(spec)
    missing = {spec.timestamp_column, spec.target_column} - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}.")
    timestamps = pd.to_datetime(frame[spec.timestamp_column], errors="coerce", utc=True, format="mixed")
    values = pd.to_numeric(frame[spec.target_column], errors="coerce")
    if timestamps.isna().any() or values.isna().any():
        raise ValueError("Timestamp and target columns cannot contain missing or invalid values.")
    timestamp_index = pd.DatetimeIndex(timestamps)
    if not timestamp_index.is_monotonic_increasing:
        raise ValueError(
            "Timestamps must already be ordered from earliest to latest; DSSential will not sort implicitly."
        )
    series = pd.Series(values.to_numpy(dtype=float), index=timestamp_index)
    if series.index.has_duplicates:
        raise ValueError("Timestamps must be unique.")
    if not np.isfinite(series.to_numpy()).all():
        raise ValueError("Target values must be finite.")
    try:
        expected = pd.date_range(series.index[0], series.index[-1], freq=spec.frequency)
    except ValueError as exc:
        raise ValueError(f"Unsupported frequency: {spec.frequency}.") from exc
    if not series.index.equals(expected):
        raise ValueError(
            "Timestamps do not form the confirmed regular frequency. "
            "Resolve gaps or choose the correct frequency before forecasting."
        )
    seasonal_minimum = (spec.seasonal_period or 1) * 2 + spec.validation_points
    minimum = max(12, spec.validation_points + 4, seasonal_minimum)
    if len(series) < minimum:
        raise ValueError(f"At least {minimum} regular observations are required.")
    return series


def _predict(history: np.ndarray, steps: int, model: ModelName, seasonal_period: int | None = None) -> np.ndarray:
    if model == "last_value":
        return np.repeat(history[-1], steps)
    if model == "seasonal_naive":
        if seasonal_period is None or len(history) < seasonal_period:
            raise ValueError("Seasonal-naive forecasting requires sufficient seasonal history.")
        season = history[-seasonal_period:]
        return cast(np.ndarray, np.resize(season, steps))
    x = np.arange(len(history), dtype=float)
    slope, intercept = np.polyfit(x, history, 1)
    return cast(np.ndarray, intercept + slope * np.arange(len(history), len(history) + steps, dtype=float))


def _rolling_predictions(
    values: np.ndarray,
    points: int,
    model: ModelName,
    seasonal_period: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    start = len(values) - points
    actual: list[float] = []
    predicted: list[float] = []
    for origin in range(start, len(values)):
        actual.append(float(values[origin]))
        predicted.append(float(_predict(values[:origin], 1, model, seasonal_period)[0]))
    return np.asarray(actual), np.asarray(predicted)


def _metric(actual: np.ndarray, predicted: np.ndarray, model: ModelName) -> MetricResult:
    errors = actual - predicted
    return MetricResult(
        model=model,
        mae=float(np.mean(np.abs(errors))),
        rmse=float(np.sqrt(np.mean(np.square(errors)))),
    )


def _package_version() -> str:
    try:
        return version("dssential")
    except PackageNotFoundError:
        return "uninstalled"


def _execution_identity(source_revision: str | None) -> tuple[str, str, dict[str, str], dict[str, str]]:
    code_version = _package_version()
    resolved_revision = source_revision if source_revision is not None else os.environ.get("DSSENTIAL_SOURCE_REVISION")
    if resolved_revision is None:
        resolved_revision = f"package:dssential@{code_version}"
    if not resolved_revision.strip():
        raise ValueError("Source revision cannot be blank.")
    runtime_versions = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
    }
    dependency_versions = {
        "dssential": code_version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    return code_version, resolved_revision, runtime_versions, dependency_versions


def forecast(
    frame: pd.DataFrame,
    spec: ForecastSpec,
    *,
    input_fingerprint: str,
    source_revision: str | None = None,
) -> ForecastResult:
    """Validate, backtest, select, and fit a small transparent model family."""
    series = _prepare(frame.copy(deep=True), spec)
    values = series.to_numpy(dtype=float)
    models: tuple[ModelName, ...] = (
        ("last_value", "seasonal_naive", "linear_trend")
        if spec.seasonal_period is not None
        else ("last_value", "linear_trend")
    )
    predictions = {
        model: _rolling_predictions(values, spec.validation_points, model, spec.seasonal_period) for model in models
    }
    selection_count = max(3, spec.validation_points // 3)
    calibration_count = max(3, (spec.validation_points - selection_count) // 2)
    test_count = spec.validation_points - selection_count - calibration_count
    if test_count < 3:
        calibration_count -= 3 - test_count
        test_count = 3
    metrics = tuple(
        _metric(
            predictions[model][0][:selection_count],
            predictions[model][1][:selection_count],
            model,
        )
        for model in models
    )
    baseline_models = {"last_value", "seasonal_naive"}
    best_baseline = min(
        (metric for metric in metrics if metric.model in baseline_models),
        key=lambda result: (result.mae, result.model),
    )
    trend_metric = next(metric for metric in metrics if metric.model == "linear_trend")
    improves_baseline = trend_metric.mae < best_baseline.mae * 0.95
    selected: ModelName = "linear_trend" if improves_baseline else best_baseline.model

    actual, selected_predicted = predictions[selected]
    errors = np.abs(actual - selected_predicted)
    calibration_end = selection_count + calibration_count
    calibration = errors[selection_count:calibration_end]
    test_errors = errors[calibration_end:]
    quantile = float(np.quantile(calibration, spec.confidence, method="higher"))
    coverage = float(np.mean(test_errors <= quantile)) if len(test_errors) else float("nan")

    predicted = _predict(values, spec.horizon, selected, spec.seasonal_period)
    future_index = pd.date_range(series.index[-1], periods=spec.horizon + 1, freq=spec.frequency)[1:]
    points = tuple(
        ForecastPoint(
            timestamp=timestamp.isoformat(),
            prediction=float(value),
            lower=float(value - quantile),
            upper=float(value + quantile),
        )
        for timestamp, value in zip(future_index, predicted, strict=True)
    )
    spec_payload = asdict(spec)
    code_version, resolved_revision, runtime_versions, dependency_versions = _execution_identity(source_revision)
    reproducibility_identity = {
        "input": input_fingerprint,
        "specification": spec_payload,
        "method_version": METHOD_VERSION,
        "code_version": code_version,
        "source_revision": resolved_revision,
        "runtime_versions": runtime_versions,
        "dependency_versions": dependency_versions,
    }
    reproducibility_hash = sha256(json.dumps(reproducibility_identity, sort_keys=True).encode()).hexdigest()
    warnings: list[str] = []
    if min(selection_count, calibration_count, test_count) < 10:
        warnings.append(
            "Limited validation evidence: at least one selection, calibration, or test split has fewer than "
            f"10 observations. Use {RECOMMENDED_VALIDATION_POINTS} or more validation points when the series allows."
        )
    if not improves_baseline:
        warnings.append(
            "Evidence is insufficient to prefer linear trend: it did not improve rolling MAE "
            "over the best naive baseline by at least 5%."
        )
    if np.isfinite(coverage) and coverage + 0.1 < spec.confidence:
        warnings.append("Observed holdout interval coverage is materially below the requested confidence.")
    limitations = (
        "Intervals are empirical error bands calibrated on early rolling holdouts and checked on "
        "later holdouts; they are not guarantees.",
        "Only transparent naive baselines and a linear trend are evaluated in this release.",
        "Structural breaks, causal effects, and business outcomes are not inferred.",
    )
    validation = ValidationResult(
        metrics=metrics,
        selected_model=selected,
        interval_coverage=coverage,
        selection_points=selection_count,
        calibration_points=calibration_count,
        test_points=len(test_errors),
        evidence_status="validated_improvement" if improves_baseline else "baseline_only",
    )
    result_hash = sha256(
        json.dumps(
            {
                "points": [asdict(point) for point in points],
                "validation": asdict(validation),
                "specification": spec_payload,
                "warnings": warnings,
                "limitations": limitations,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    return ForecastResult(
        points=points,
        validation=validation,
        manifest=RunManifest(
            run_id=str(uuid4()),
            reproducibility_hash=reproducibility_hash,
            created_at=datetime.now(UTC).isoformat(),
            input_fingerprint=input_fingerprint,
            method_version=METHOD_VERSION,
            code_version=code_version,
            source_revision=resolved_revision,
            specification=spec_payload,
            runtime_versions=runtime_versions,
            dependency_versions=dependency_versions,
            random_seed=None,
            warnings=tuple(warnings),
            limitations=limitations,
            result_hash=result_hash,
        ),
        warnings=tuple(warnings),
        limitations=limitations,
    )
