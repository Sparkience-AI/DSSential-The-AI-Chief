"""Streamlit interface for DSSential's validated deterministic core."""

from __future__ import annotations

from dataclasses import asdict

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dssential.forecasting import ForecastResult, ForecastSpec, forecast
from dssential.ingestion import DatasetSnapshot, load_csv
from dssential.narration import DEFAULT_MODELS, NarrationConfig, NarrationError, Provider, generate_narration
from dssential.scenario import ScenarioAdjustment, apply_scenario

st.set_page_config(page_title="DSSential", page_icon="📊", layout="wide")


def _provider_controls() -> NarrationConfig:
    st.sidebar.header("Optional AI narration")
    provider: Provider = st.sidebar.selectbox("Provider", tuple(DEFAULT_MODELS))
    model = st.sidebar.text_input("Model ID", value=DEFAULT_MODELS[provider])
    api_key = st.sidebar.text_input(f"{provider} API key", type="password", help="Held only in this browser session.")
    consent = st.sidebar.checkbox(
        f"Send the validated summary to {provider}",
        help="No raw rows are sent. Review the displayed summary before consenting.",
    )
    identity = (provider, model)
    if st.session_state.get("narrator_identity") != identity:
        st.session_state["narrator_identity"] = identity
        st.session_state.pop("narration", None)
    return NarrationConfig(provider, model, api_key, consent)


def _load_snapshot() -> DatasetSnapshot | None:
    upload = st.file_uploader("Upload a CSV", type=["csv"])
    if upload is None:
        return None
    try:
        snapshot = load_csv(upload.getvalue(), name=upload.name)
    except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        st.error(str(exc))
        return None
    if st.session_state.get("dataset_hash") != snapshot.sha256:
        st.session_state["dataset_hash"] = snapshot.sha256
        st.session_state.pop("forecast_result", None)
        st.session_state.pop("narration", None)
    return snapshot


def _forecast_chart(frame: pd.DataFrame, spec: ForecastSpec, result: ForecastResult) -> go.Figure:
    forecast_frame = pd.DataFrame([asdict(point) for point in result.points])
    forecast_frame["timestamp"] = pd.to_datetime(forecast_frame["timestamp"], utc=True)
    figure = go.Figure()
    figure.add_scatter(
        x=pd.to_datetime(frame[spec.timestamp_column], utc=True),
        y=pd.to_numeric(frame[spec.target_column]),
        name="Observed",
        mode="lines",
    )
    figure.add_scatter(
        x=forecast_frame["timestamp"], y=forecast_frame["upper"], name="Upper bound", mode="lines", line={"width": 0}
    )
    figure.add_scatter(
        x=forecast_frame["timestamp"],
        y=forecast_frame["lower"],
        name="Empirical interval",
        mode="lines",
        fill="tonexty",
        line={"width": 0},
    )
    figure.add_scatter(
        x=forecast_frame["timestamp"],
        y=forecast_frame["prediction"],
        name=f"Forecast ({result.validation.selected_model})",
        mode="lines+markers",
    )
    figure.update_layout(yaxis_title=spec.unit, xaxis_title="Timestamp", hovermode="x unified")
    return figure


def _forecast_page(snapshot: DatasetSnapshot, config: NarrationConfig) -> None:
    frame = snapshot.frame
    st.subheader("Validated forecast")
    st.caption("DSSential never invents dates. Irregular, duplicate, missing, or invalid values are refused.")
    c1, c2, c3 = st.columns(3)
    timestamp = c1.selectbox("Timestamp column", list(frame.columns))
    numeric = list(frame.select_dtypes(include="number").columns)
    if not numeric:
        st.warning("The dataset has no numeric target column.")
        return
    target = c2.selectbox("Numeric target", numeric)
    frequency = c3.text_input("Confirmed frequency", "D", help="Examples: D, W, MS, QS")
    c4, c5, c6 = st.columns(3)
    unit = c4.text_input("Target unit", "units")
    horizon = c5.number_input("Forecast periods", 1, 365, 12)
    confidence = c6.slider("Empirical interval confidence", 0.50, 0.99, 0.90, 0.01)
    seasonal_enabled = st.checkbox("Add an explicit seasonal-naive baseline")
    seasonal_period = st.number_input("Season length in observations", 2, 365, 7) if seasonal_enabled else None
    validation_points = st.slider("Rolling validation points", 12, 60, 15)
    if st.button("Run validated forecast", type="primary"):
        spec = ForecastSpec(
            timestamp,
            target,
            frequency,
            int(horizon),
            unit,
            confidence,
            validation_points,
            int(seasonal_period) if seasonal_period else None,
        )
        try:
            result = forecast(frame, spec, input_fingerprint=snapshot.sha256)
        except ValueError as exc:
            st.error(str(exc))
        else:
            st.session_state["forecast_result"] = (spec, result)
            st.session_state.pop("narration", None)

    stored = st.session_state.get("forecast_result")
    if not stored:
        return
    spec, result = stored
    st.plotly_chart(_forecast_chart(frame, spec, result), use_container_width=True)
    st.dataframe(
        pd.DataFrame([asdict(metric) for metric in result.validation.metrics]),
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        f"Model metrics use the first {result.validation.selection_points} rolling origins. "
        f"The next {result.validation.calibration_points} calibrate intervals; the final "
        f"{result.validation.test_points} evaluate interval coverage."
    )
    st.write(f"Evidence status: **{result.validation.evidence_status.replace('_', ' ')}**")
    if result.validation.test_points:
        st.write(
            f"Later-holdout interval coverage: **{result.validation.interval_coverage:.1%}** "
            f"({result.validation.test_points} points)"
        )
    for warning in result.warnings:
        st.warning(warning)
    with st.expander("Limitations and reproducibility manifest", expanded=True):
        for limitation in result.limitations:
            st.write(f"- {limitation}")
        st.json(asdict(result.manifest))
    summary = result.narration_summary()
    with st.expander("Exact summary eligible for AI narration"):
        st.json(summary)
    if st.button("Generate optional narration"):
        try:
            st.session_state["narration"] = generate_narration(summary, config)
        except (ValueError, NarrationError) as exc:
            st.error(str(exc))
    if "narration" in st.session_state:
        st.subheader("Advisory narration")
        st.write(st.session_state["narration"])


def _scenario_page(snapshot: DatasetSnapshot) -> None:
    frame = snapshot.frame
    numeric = list(frame.select_dtypes(include="number").columns)
    st.subheader("Transparent scenario preview")
    st.info(
        "This applies visible arithmetic to one input. It is not Monte Carlo simulation, risk analysis, "
        "sensitivity analysis, optimization, or a predicted outcome."
    )
    if not numeric:
        st.warning("The dataset has no numeric columns.")
        return
    variable = st.selectbox("Variable", numeric)
    base = st.number_input("Base value", value=float(frame[variable].mean()))
    operation = st.selectbox("Operation", ("percent", "absolute", "set"))
    amount = st.number_input("Adjustment", value=0.0)
    unit = st.text_input("Unit", "units", key="scenario_unit")
    bounded = st.checkbox("Enforce allowed bounds")
    minimum = st.number_input("Minimum", value=float(frame[variable].min())) if bounded else None
    maximum = st.number_input("Maximum", value=float(frame[variable].max())) if bounded else None
    if st.button("Apply scenario"):
        try:
            result = apply_scenario(ScenarioAdjustment(variable, base, operation, amount, unit, minimum, maximum))
        except ValueError as exc:
            st.error(str(exc))
        else:
            st.json(asdict(result))
            st.caption(result.explanation)


def main() -> None:
    st.title("DSSential: validated decision analysis")
    st.write(
        "A local-first workbench for reproducible forecasting and transparent scenario previews. "
        "Deterministic calculations are authoritative; AI narration is optional."
    )
    config = _provider_controls()
    snapshot = _load_snapshot()
    if snapshot is None:
        st.info("Upload a CSV to begin. Your data stays in this application session.")
        return
    st.success(
        f"Loaded {snapshot.name}: {snapshot.rows:,} rows x {snapshot.columns} columns; "
        f"SHA-256 {snapshot.sha256[:12]}..."
    )
    with st.expander("Data overview"):
        st.dataframe(snapshot.frame.head(50), use_container_width=True)
        st.dataframe(snapshot.frame.isna().sum().rename("missing"), use_container_width=True)
    page = st.radio("Workflow", ("Forecast", "Scenario preview"), horizontal=True)
    _forecast_page(snapshot, config) if page == "Forecast" else _scenario_page(snapshot)


if __name__ == "__main__":
    main()
