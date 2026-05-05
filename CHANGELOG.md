# Changelog

MHX uses semantic-versioned pre-releases while the rebuilt solver API matures.

## Unreleased

- Declared the rebuilt public API as `v1` with `MHX_API_VERSION` compatibility checks.
- Added stable schema metadata for trajectory NPZ files, run manifests, artifact manifests, and validation suites.
- Added a CI legacy-import guard so active package paths cannot depend on archived scripts.
- Added release, migration, and API-compatibility documentation.

## 0.1.0a0

- Rebuilt MHX under `src/mhx/` with a validation-first reduced-MHD core.
- Added TOML-driven runs, deterministic FAST examples, figures, reports, and artifact manifests.
- Added spectral operators, reduced-MHD RHS/JVP checks, diagnostics, physics plugins, and validation benchmarks.
- Archived the original exploratory scripts under `legacy/old_mhx/`.
