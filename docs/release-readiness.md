# Release readiness

## Verification gates

- Tests, Ruff, strict mypy, lock verification, source distribution, wheel build, and installed-wheel import pass on the supported matrix.
- The exact README source-run workflow is exercised on a clean virtual environment.
- Forecast golden fixtures cover baseline selection, minimum evidence policy, selection/calibration/test separation,
  interval ordering, execution identity and reproducibility hashes, malformed timestamps, and resource limits.
- MCP tests cover payload and metadata bounds, finite scenario arithmetic, no raw-row echo, unsupported capability disclosure, and skill negative routing.
- Streamlit labels, controls, errors, charts, keyboard flow, contrast, and narrow layout receive a manual accessibility pass before a tagged release.

## Performance envelope

Supported inputs are at most 10 MiB, 250,000 rows, 250 columns, 365 forecast periods, and 2 MiB per MCP CSV call. The release target is under 5 seconds for the deterministic 10,000-row daily forecast fixture on a typical developer laptop; CI records correctness rather than a hardware-sensitive hard timing assertion.

## Release and rollback

The manual release-evidence workflow creates wheel/source artifacts, CycloneDX SBOM, SHA-256 checksums, and GitHub provenance attestations. It never creates an installer. Verify artifacts in a clean environment, publish known limitations with the tag, and retain the previous release. Rollback means withdrawing the affected tag/artifacts and directing users to the last verified version; local user data is not migrated or deleted.
