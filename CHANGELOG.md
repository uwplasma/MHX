# Changelog

MHX uses semantic-versioned pre-releases while the rebuilt solver API matures.

## Unreleased

- Rebuilt the documentation: sphinx-book-theme with dark mode, a landing
  page with an autoplaying hero movie and section cards, and a Diataxis
  structure with getting-started, tutorial, how-to, physics, validation,
  reference, and development sections. Internal evidence pages moved to a
  collapsed project-records group.
- Documented the reduced-MHD model end to end: equations as implemented,
  derivation from visco-resistive MHD, normalization, conservation laws,
  equilibrium formulas, and model limits, with BibTeX citations rendered on
  a bibliography page.
- Added six executed tutorial notebooks (tearing mode, reconnection
  topology, gradients, inverse problem, ensembles and devices, implicit
  stepping) with committed outputs, plus a monthly workflow that
  re-executes them against the current solver.
- Added `examples/gallery/08_gradient.py` with a float64 finite-difference
  gradient check, and documented that implicit runs require
  `JAX_ENABLE_X64=1` to reach their Newton tolerances.
- Replaced GIF embeds in documentation pages with H.264 movies, added a
  movie gallery page with claim boundaries, and added a motion quality gate
  that rejects visually static movies.
- `SimulationResult.plot` now shows the island flux in its second panel,
  and the quickstart runs to `t_end=40` so the first figure shows a grown
  tearing mode.
- Split the docs toolchain into a `docs` extra, made `dev` include it, and
  turned on `fail_on_warning` for ReadTheDocs.

## 0.1.0a1 - 2026-05-25

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
- Added optional sub-cell Newton refinement and deterministic frame-to-frame
  association for magnetic-flux X/O critical points.
- Added `mhx validate release-candidate`, a static public-release gate that
  checks packaging, CI, ReadTheDocs, citation/version metadata, examples, docs,
  README claim boundaries, active legacy-import hygiene, and figure hashes.
- Updated README/release double-Harris media policy to the `128×128`,
  `t_end=160` residual-field GPU validation bundle and aligned duration labels.
- Closed the Rutherford production-promotion blocker with the `adcc714`
  periodic double-Harris GPU campaign: `128×128`, `t_end=240`, 123 history
  samples, reconnecting-flux amplification `8.3593`, Rutherford-width
  amplification `2.8912`, and passing duration/convergence/seed-QI/X-O/media
  promotion gates.

## 0.1.0a0

- Rebuilt MHX under `src/mhx/` with a validation-first reduced-MHD core.
- Added TOML-driven runs, deterministic FAST examples, figures, reports, and artifact manifests.
- Added spectral operators, reduced-MHD RHS/JVP checks, diagnostics, physics plugins, and validation benchmarks.
- Archived the original exploratory scripts under `legacy/old_mhx/`.
