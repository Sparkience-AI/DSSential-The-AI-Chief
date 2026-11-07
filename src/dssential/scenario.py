"""Transparent arithmetic scenario previews without invented outcome models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Operation = Literal["percent", "absolute", "set"]


@dataclass(frozen=True)
class ScenarioAdjustment:
    variable: str
    base_value: float
    operation: Operation
    amount: float
    unit: str
    minimum: float | None = None
    maximum: float | None = None


@dataclass(frozen=True)
class ScenarioResult:
    variable: str
    base_value: float
    adjusted_value: float
    absolute_change: float
    percent_change: float | None
    unit: str
    explanation: str


def apply_scenario(adjustment: ScenarioAdjustment) -> ScenarioResult:
    if not adjustment.variable.strip() or not adjustment.unit.strip():
        raise ValueError("Variable and unit are required.")
    if adjustment.minimum is not None and adjustment.maximum is not None and adjustment.minimum > adjustment.maximum:
        raise ValueError("Minimum cannot exceed maximum.")
    if adjustment.operation == "percent":
        adjusted = adjustment.base_value * (1 + adjustment.amount / 100)
        explanation = f"Applied a {adjustment.amount:g}% arithmetic adjustment."
    elif adjustment.operation == "absolute":
        adjusted = adjustment.base_value + adjustment.amount
        explanation = f"Added {adjustment.amount:g} {adjustment.unit}."
    elif adjustment.operation == "set":
        adjusted = adjustment.amount
        explanation = f"Set the value to {adjustment.amount:g} {adjustment.unit}."
    else:
        raise ValueError(f"Unsupported operation: {adjustment.operation}.")
    if adjustment.minimum is not None and adjusted < adjustment.minimum:
        raise ValueError("Adjusted value is below the declared minimum.")
    if adjustment.maximum is not None and adjusted > adjustment.maximum:
        raise ValueError("Adjusted value is above the declared maximum.")
    change = adjusted - adjustment.base_value
    percent = None if adjustment.base_value == 0 else change / adjustment.base_value * 100
    return ScenarioResult(
        variable=adjustment.variable,
        base_value=adjustment.base_value,
        adjusted_value=adjusted,
        absolute_change=change,
        percent_change=percent,
        unit=adjustment.unit,
        explanation=explanation + " This is an input preview, not a predicted business outcome.",
    )
