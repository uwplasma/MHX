# Changelog

MHX uses semantic-versioned pre-releases while the rebuilt solver API matures.

## Unreleased

- Declared the rebuilt public API as `v1` with `MHX_API_VERSION` compatibility checks.
- Added stable schema metadata for trajectory NPZ files, run manifests, artifact manifests, and validation suites.
- Added a CI legacy-import guard so active package paths cannot depend on archived scripts.
- Added release, migration, and API-compatibility documentation.
- Added separate Docs, Benchmark Smoke, and Publish GitHub Actions workflows.
- Expanded release readiness to require every validation-suite case, including
  Orszag--Tang, turbulence, seed-QI, neural-ODE, and Rutherford executor lanes.
- Added a documentation figure manifest mapping key movies/figures to commands,
  claim levels, source paths, and tests.
- Added top-level `mhx.load_config` and `mhx.run` convenience APIs.
- Expanded installation/tutorial/example docs and synchronized diagnostics,
  output-schema, benchmark, validation, and publication-checklist pages.
- Hardened docs/tests around X/O critical points, turbulence helper branches,
  duration-policy failures, readiness-loader failures, and workflow presence.
- Added Rutherford production-promotion gates, current-sheet geometry and X/O
  histories, and a CLI promotion report that blocks production claims until
  convergence, seed-QI, fixed-scale media, and tolerance evidence are attached.

## 0.1.0a0

- Rebuilt MHX under `src/mhx/` with a validation-first reduced-MHD core.
- Added TOML-driven runs, deterministic FAST examples, figures, reports, and artifact manifests.
- Added spectral operators, reduced-MHD RHS/JVP checks, diagnostics, physics plugins, and validation benchmarks.
- Archived the original exploratory scripts under `legacy/old_mhx/`.
