---
name: dssential-decision-analysis
description: Run and explain DSSential validated forecasts or transparent bounded scenario previews. Use when a user has a real regular time series or asks for arithmetic what-if adjustments; do not use for causal, risk, sensitivity, or optimization claims.
---

# DSSential decision analysis

Use the local DSSential MCP tools. Calculations and returned limitations are authoritative; do not replace them with model arithmetic.

## Forecast workflow

1. Confirm the user supplied a real timestamp column, target, frequency, horizon, and unit. Ask for missing fields before calling a tool.
2. Confirm the timestamps are already ordered, unique, and regular. Never invent, sort, impute, or rescale data silently.
3. Call `capabilities` when scope is uncertain, then call `validated_forecast_csv` once with the confirmed specification.
4. Report the selected model, every baseline metric, interval coverage, evidence status, warnings, and limitations. If evidence is `baseline_only`, say that a trend model was not justified.
5. Treat empirical intervals as error bands, not probabilities or guarantees. Do not make business decisions for the user.

## Scenario workflow

Confirm the variable, base value, operation, amount, unit, and any hard bounds. Call `preview_scenario` and describe the result only as transparent input arithmetic. It is not a predicted outcome.

## Refuse or redirect

Do not present DSSential as performing Monte Carlo risk, VaR/CVaR, Sobol sensitivity, causal impact, optimization, autonomous action, or professional advice. Explain the boundary and ask whether the supported forecast or arithmetic preview would help instead.

Never send dataset contents or results to a cloud provider unless the user separately chooses a provider in the application and explicitly consents there.
