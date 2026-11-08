# Threat model

## Assets and trust boundary

DSSential protects uploaded rows, provider keys, model settings, validated result integrity, and the user's interpretation of evidence. The Streamlit process and optional local stdio MCP server are trusted local processes. CSV contents, MCP arguments, model narration, and copied reports are untrusted data.

## Primary threats and controls

- **Fabricated time semantics or hidden transformation:** forecasting rejects missing, unordered, duplicate, irregular, non-finite, or undersized series. It never sorts, imputes, scales, or creates dates.
- **Analytical overclaim:** the deterministic result always includes baselines, evidence status, warnings, and limitations. The MCP does not expose risk, sensitivity, optimization, causal, or action-taking tools.
- **Resource exhaustion:** uploads are limited to 10 MiB/250,000 rows/250 columns; MCP inline CSV is limited to 2 MiB, labels to 256 characters, and scenario inputs to finite numbers; forecast horizon is at most 365.
- **Prompt injection:** CSV strings are treated as data and are never executed. MCP tools do not call a provider. Optional application narration receives only the allowlisted validated summary after explicit consent.
- **Credential disclosure:** BYOK keys remain in process memory and are excluded from MCP, manifests, logs, and results.
- **Protocol misuse:** MCP uses local stdio, has no network listener or authentication surface, and exposes only three typed tools.

## Residual risk

The local process inherits the invoking user's OS permissions. Resource limits reduce accidental exhaustion but are not a hostile multi-tenant sandbox. Forecast validity still depends on the user's domain, units, frequency, and data-generating process.
