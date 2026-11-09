# DSSential: validated decision analysis

DSSential is a local-first, open-source workbench for transparent forecasting and scenario
exploration. It is designed for analysts who need reproducible evidence rather than autonomous
decisions. Deterministic calculations are authoritative; Gemini, OpenAI, or Anthropic can
optionally narrate the displayed validated summary with your own API key.

## Supported workflows

### Validated forecasting

- Requires a real timestamp column, confirmed frequency, target column, and unit.
- Refuses missing, duplicate, unsorted, irregular, non-finite, or fabricated time semantics.
- Preserves the original target scale and never silently imputes or scales source data.
- Compares linear trend with last-value and optional seasonal-naive baselines.
- Uses separate time-ordered rolling origins for model selection, interval calibration, and final
  empirical coverage checks.
- Selects linear trend only when it improves selection-set MAE over the best naive baseline by at
  least 5%; otherwise the result is explicitly `baseline_only`.
- Uses 30 rolling validation points by default (10 each for model selection, interval calibration,
  and later testing). Smaller datasets may use 18 or more, with an explicit limited-evidence warning
  whenever any split contains fewer than 10 observations.
- Records the input fingerprint, specification, method version, package version, source revision,
  Python runtime, numerical dependency versions, warnings, limitations, result hash, unique run ID,
  and stable reproducibility hash. Set `DSSENTIAL_SOURCE_REVISION` to a commit or build identifier;
  packaged runs otherwise identify the installed DSSential version.

### Transparent scenario preview

Apply a percent, absolute, or set-value adjustment with explicit units and optional bounds. This is
an arithmetic preview of an input—not a predicted outcome. DSSential does not currently claim
Monte Carlo outcome simulation, VaR/CVaR, Sobol sensitivity, causal impact, or optimization.

## Run locally

Python 3.11–3.13 is supported.

```bash
git clone https://github.com/Sparkience-AI/DSSential-The-AI-Chief.git
cd DSSential-The-AI-Chief
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -e ".[cloud]"
streamlit run app.py
```

Cloud packages are optional. Install `-e .` instead if you only need local deterministic analysis.
No application key is required for forecasts or scenario previews.

## Local MCP and agent skill

Install the project, then run `dssential-mcp`. The server uses local stdio only and exposes capability discovery, a bounded inline-CSV validated forecast, and transparent scenario arithmetic. It opens no network port and never calls a model provider. The repo-local `dssential-decision-analysis` skill under `.agents/skills/` teaches compatible agents when to ask for missing time-series semantics, how to report baseline evidence, and when to refuse unsupported claims.

The MCP CSV payload limit is 2 MiB. Use the Streamlit application for larger supported local files.

## Optional BYOK narration

Choose Gemini, OpenAI, or Anthropic in the sidebar, edit the model ID if needed, enter a session-only
API key, inspect the exact JSON summary, and check the provider-specific transmission consent box.
Only that bounded summary is transmitted. Raw rows and keys are excluded. Changing provider, model,
dataset, or forecast invalidates stale narration. OpenAI calls use the Responses API with
`store=False`.

## Validation and development

```bash
uv sync --group dev
ruff check .
ruff format --check .
mypy
pytest
python -m build
```

CI runs quality checks and wheel smoke tests on Python 3.11, 3.12, and 3.13.

## Boundaries

Forecasts and intervals are evidence for review, not guarantees or professional advice. The first
validated model family is deliberately small and transparent. Structural breaks, causal effects,
and business actions require domain review outside this tool. See [PRIVACY.md](PRIVACY.md) and
[SECURITY.md](SECURITY.md), the [threat model](docs/threat-model.md), and the
[release-readiness contract](docs/release-readiness.md).

## License

MIT. See [LICENSE](LICENSE).
