"""Local stdio MCP adapter over DSSential's deterministic analysis core."""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any, Literal

from .forecasting import ForecastSpec, forecast
from .ingestion import load_csv
from .scenario import ScenarioAdjustment, apply_scenario

MAX_MCP_CSV_BYTES = 2 * 1024 * 1024
MAX_MCP_LABEL_CHARS = 256


def _bounded_label(value: str, field: str) -> None:
    if not value.strip():
        raise ValueError(f"{field} is required.")
    if len(value) > MAX_MCP_LABEL_CHARS:
        raise ValueError(f"{field} exceeds the {MAX_MCP_LABEL_CHARS}-character limit.")


def capabilities() -> dict[str, object]:
    """Describe the supported analytical boundary without making a model call."""
    return {
        "product": "DSSential",
        "transport": "local stdio",
        "tools": ["validated_forecast_csv", "preview_scenario"],
        "requires_cloud": False,
        "unsupported": [
            "causal inference",
            "Monte Carlo risk",
            "VaR or CVaR",
            "Sobol sensitivity",
            "optimization",
            "autonomous business decisions",
        ],
    }


def validated_forecast_csv(
    csv_text: str,
    timestamp_column: str,
    target_column: str,
    frequency: str,
    horizon: int,
    unit: str,
    confidence: float = 0.9,
    validation_points: int = 30,
    seasonal_period: int | None = None,
) -> dict[str, object]:
    """Run the validated forecast workflow on a bounded UTF-8 CSV payload."""
    _bounded_label(timestamp_column, "Timestamp column")
    _bounded_label(target_column, "Target column")
    _bounded_label(frequency, "Frequency")
    _bounded_label(unit, "Unit")
    payload = csv_text.encode("utf-8")
    if len(payload) > MAX_MCP_CSV_BYTES:
        raise ValueError("MCP CSV payload exceeds the 2 MiB limit.")
    snapshot = load_csv(payload, name="mcp-input.csv")
    result = forecast(
        snapshot.frame,
        ForecastSpec(
            timestamp_column=timestamp_column,
            target_column=target_column,
            frequency=frequency,
            horizon=horizon,
            unit=unit,
            confidence=confidence,
            validation_points=validation_points,
            seasonal_period=seasonal_period,
        ),
        input_fingerprint=snapshot.sha256,
    )
    return asdict(result)


def preview_scenario(
    variable: str,
    base_value: float,
    operation: Literal["percent", "absolute", "set"],
    amount: float,
    unit: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> dict[str, object]:
    """Apply transparent bounded arithmetic; this never predicts an outcome."""
    _bounded_label(variable, "Variable")
    _bounded_label(unit, "Unit")
    values = (base_value, amount, minimum, maximum)
    if any(value is not None and not math.isfinite(value) for value in values):
        raise ValueError("Scenario numbers must be finite.")
    return asdict(
        apply_scenario(
            ScenarioAdjustment(
                variable=variable,
                base_value=base_value,
                operation=operation,
                amount=amount,
                unit=unit,
                minimum=minimum,
                maximum=maximum,
            )
        )
    )


def build_server() -> Any:
    """Create the MCP server lazily so the deterministic library stays standalone."""
    from mcp.server.fastmcp import FastMCP

    server = FastMCP(
        "DSSential",
        instructions=(
            "Use only for validated univariate forecasts and transparent arithmetic scenario previews. "
            "Never describe results as causal, guaranteed, optimized, or autonomous decisions."
        ),
    )
    server.tool()(capabilities)
    server.tool()(validated_forecast_csv)
    server.tool()(preview_scenario)
    return server


def main() -> None:
    """Run the local server using MCP stdio transport."""
    build_server().run(transport="stdio")


if __name__ == "__main__":
    main()
