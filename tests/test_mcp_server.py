from __future__ import annotations

import json
from pathlib import Path

import pytest

from dssential.mcp_server import MAX_MCP_CSV_BYTES, capabilities, preview_scenario, validated_forecast_csv


def _csv() -> str:
    rows = ["date,revenue"]
    rows.extend(f"2025-01-{index:02d},{index * 100}" for index in range(1, 32))
    rows.extend(f"2025-02-{index:02d},{(index + 31) * 100}" for index in range(1, 10))
    return "\n".join(rows) + "\n"


def test_mcp_forecast_returns_validated_evidence_without_source_rows() -> None:
    result = validated_forecast_csv(_csv(), "date", "revenue", "D", 2, "USD")
    assert result["validation"]["selected_model"] == "linear_trend"
    assert result["validation"]["evidence_status"] == "validated_improvement"
    assert result["manifest"]["input_fingerprint"]
    assert "date,revenue" not in json.dumps(result)
    assert result["limitations"]


def test_mcp_payload_and_scenario_boundaries() -> None:
    with pytest.raises(ValueError, match="2 MiB"):
        validated_forecast_csv("x" * (MAX_MCP_CSV_BYTES + 1), "date", "value", "D", 1, "units")
    scenario = preview_scenario("price", 100, "percent", 10, "USD", 0, 120)
    assert scenario["adjusted_value"] == pytest.approx(110)
    assert "not a predicted business outcome" in str(scenario["explanation"])
    assert "optimization" in capabilities()["unsupported"]


def test_mcp_rejects_unbounded_or_non_finite_metadata() -> None:
    with pytest.raises(ValueError, match="256-character"):
        preview_scenario("x" * 257, 100, "percent", 10, "USD")
    with pytest.raises(ValueError, match="finite"):
        preview_scenario("price", float("nan"), "percent", 10, "USD")
    with pytest.raises(ValueError, match="256-character"):
        validated_forecast_csv(_csv(), "date", "revenue", "D", 2, "x" * 257)


def test_skill_package_has_routing_and_negative_evals() -> None:
    root = Path(__file__).parents[1] / ".agents" / "skills" / "dssential-decision-analysis"
    skill = (root / "SKILL.md").read_text(encoding="utf-8")
    cases = json.loads((root / "evals" / "cases.json").read_text(encoding="utf-8"))["cases"]
    assert "TODO" not in skill
    assert any(case["should_trigger"] is False for case in cases)
    assert any("no tool call" in " ".join(case["expected"]) for case in cases)
