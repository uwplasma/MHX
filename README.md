# MHX

[![CI](https://github.com/uwplasma/MHX/actions/workflows/ci.yml/badge.svg)](https://github.com/uwplasma/MHX/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/mhx/badge/?version=latest)](https://mhx.readthedocs.io/)

**MHX is being rebuilt as a JAX-native, differentiable plasma and magnetohydrodynamics framework for magnetic reconnection, tearing modes, validation, extension experiments, and eventually inverse design.**

The previous reduced-MHD tearing/plasmoid code has been preserved under
`legacy/old_mhx/`. The active package is a clean rebuild under `src/mhx/`.

## Current status

This rebuild is intentionally starting from a small, tested core:

- Python 3.10+ package with `src/` layout.
- `mhx` CLI with `mhx version`, `mhx init`, and `mhx run`.
- TOML configuration loading.
- Periodic Cartesian grids.
- JAX spectral derivative, inverse-Laplacian, and reduced-MHD RHS operators.
- Sphinx/MyST documentation skeleton.
- CI for linting, tests, coverage, and docs.

The current FAST reduced-MHD runs are smoke tests, not nonlinear reconnection
claims. The strongest current results are exact linear validation gates,
the direct Harris-sheet tearing eigenvalue gate for one published reference
case, finite-domain dispersion and time-domain eigenmode replay gates,
eigenfunction-localization gates, matrix-free operator checks, and extension APIs. Full FKR/Coppi
dispersion scans, nonlinear plasmoid dynamics, and neural-ODE inverse-design
claims remain roadmap items.

See `docs/audit.md` for the current skeptical validation audit and maturity
table.

The active public API is `v1`. Check it with:

```bash
mhx api status
MHX_API_VERSION=v1 mhx api status --json
```

## Install

```bash
git clone https://github.com/uwplasma/MHX.git
cd MHX
python -m pip install -e ".[dev,docs]"
```

JAX accelerator wheels are platform-specific. For GPU/TPU installs, follow the
official JAX instructions and then install MHX.

## Quickstart

Create a starter config:

```bash
mhx init examples/linear_tearing.toml
```

Run the deterministic reduced-MHD smoke workflow:

```bash
python tools/check_legacy_imports.py
mhx run examples/linear_tearing.toml --outdir outputs/smoke
```

Expected files:

- `outputs/smoke/config_effective.json`
- `outputs/smoke/diagnostics.json`
- `outputs/smoke/trajectory.npz`
- `outputs/smoke/manifest.json`

Regenerate figures:

```bash
mhx figures outputs/smoke --gif
```

This writes `energy_history.png`, `flux_final.png`, and `mode_amplitude.png`
under `outputs/smoke/figures/`, plus `flux_movie.gif` when `--gif` is passed.

Create a compact run report:

```bash
mhx report outputs/smoke
mhx artifact-manifest outputs/smoke
```

Run the same workflow as a benchmark:

```bash
mhx benchmark run --config examples/linear_tearing.toml --outdir outputs/benchmarks/linear_tearing_fast --gif
mhx benchmark validate outputs/benchmarks/linear_tearing_fast
mhx benchmark decay --outdir outputs/benchmarks/resistive_decay
mhx benchmark scaling --outdir outputs/benchmarks/reconnection_scaling
mhx benchmark fkr-window --outdir outputs/benchmarks/fkr_window
mhx benchmark fkr-growth --outdir outputs/benchmarks/fkr_growth_rate
mhx benchmark harris-delta-prime --outdir outputs/benchmarks/harris_delta_prime
mhx benchmark linear-tearing-eigenvalue --outdir outputs/benchmarks/linear_tearing_eigenvalue
mhx benchmark linear-tearing-dispersion --outdir outputs/benchmarks/linear_tearing_dispersion
mhx benchmark linear-tearing-layer --outdir outputs/benchmarks/linear_tearing_layer
mhx benchmark linear-tearing-timedomain --outdir outputs/benchmarks/linear_tearing_timedomain
mhx benchmark linearized-rhs --outdir outputs/benchmarks/linearized_rhs
mhx benchmark reduced-mhd-eigenmode --outdir outputs/benchmarks/reduced_mhd_eigenmode
mhx benchmark cosine-equilibrium-linearization --outdir outputs/benchmarks/cosine_equilibrium_linearization
mhx benchmark current-sheet-eigenvalue --outdir outputs/benchmarks/periodic_current_sheet_eigenvalue
mhx benchmark diffusion-eigenvalue --outdir outputs/benchmarks/diffusion_eigenvalue
mhx benchmark power-iteration --outdir outputs/benchmarks/power_iteration
mhx benchmark arnoldi --outdir outputs/benchmarks/arnoldi
mhx benchmark timing --outdir outputs/benchmarks/timing --repeats 3 --warmups 1
mhx benchmark catalog --outdir outputs/benchmarks/catalog
mhx validate all --outdir outputs/validation_suite
```

Current reviewer-facing validation figures include the direct Harris-sheet
tearing eigenvalue gate:

![Direct Harris-sheet tearing eigenvalue gate](docs/_static/validation/linear_tearing_eigenvalue/linear_tearing_eigenvalue.png)

and the finite-domain tearing dispersion gate:

![Finite-domain tearing dispersion gate](docs/_static/validation/linear_tearing_dispersion/linear_tearing_dispersion.png)

and the eigenfunction-layer localization gate:

![Harris tearing eigenfunction layer gate](docs/_static/validation/linear_tearing_layer/linear_tearing_layer.png)

and the time-domain eigenmode replay gate:

![Time-domain Harris tearing eigenmode replay](docs/_static/validation/linear_tearing_timedomain/linear_tearing_timedomain.png)

and the exact resistive-decay gate:

![Exact resistive-decay relative error](docs/_static/validation/exact_decay/decay_relative_error.png)

Inspect configurable physics terms:

```bash
mhx physics equilibria
mhx physics list
mhx physics lint hyper_resistivity
mhx diagnostics list
```

Try a reduced-state two-fluid toy extension:

```bash
mhx run examples/linear_tearing_twofluid_toy.toml --outdir outputs/twofluid_toy
mhx figures outputs/twofluid_toy --gif
```

Try a local plugin module that registers both a physics term and a diagnostic:

```bash
mhx run examples/linear_tearing_plugin_demo.toml --outdir outputs/plugin_demo
mhx figures outputs/plugin_demo --gif
mhx report outputs/plugin_demo
mhx diagnostics list-with-plugins --plugin-module examples.local_extension_plugin
mhx physics lint example_flux_drive --plugin-module examples.local_extension_plugin
mhx diagnostics lint final_flux_l2 --plugin-module examples.local_extension_plugin
```

Installed plugin packages can also use Python entry-point groups:

```bash
mhx physics list-with-plugins --entry-point-group mhx.physics
mhx diagnostics list-with-plugins --entry-point-group mhx.diagnostics
```

Use `examples/plugin_template/` as the starting layout for an external plugin
repository with entry points, source modules, and tests.
The local plugin also demonstrates diagnostic figure hooks:
`mhx figures outputs/plugin_demo` writes
`outputs/plugin_demo/figures/diagnostics/final_flux_l2_history.png` and records
the same hook outputs in `report.json` when `mhx report outputs/plugin_demo`
runs.

## Python API

```python
from mhx.config import load_config
from mhx.grids import CartesianGrid
from mhx.numerics.spectral import fft_derivative

cfg = load_config("examples/linear_tearing.toml")
grid = CartesianGrid.from_mesh_config(cfg.mesh)
x, _ = grid.mesh()
dfdx = fft_derivative(x * 0.0, axis=0, length=grid.lengths[0])
```

For a real time-dependent smoke run from Python:

```python
from mhx.benchmarks import run_linear_tearing_smoke
from mhx.config import load_config

cfg = load_config("examples/linear_tearing.toml")
trajectory, diagnostics = run_linear_tearing_smoke(cfg)
print(diagnostics["final_total_energy"])
```

## Roadmap

The full rebuild plan and execution log live in `plan.md`. Major milestones:

1. Clean package skeleton and validation-first numerics.
2. Spectral reduced-MHD tearing benchmark plus gradient checks.
3. Plugin-style physics terms and standardized diagnostics.
4. Finite-volume MHD, constrained transport, and external-code comparisons.
5. Neural ODE and differentiable inverse-design workflows.
6. Manuscript-grade docs, figures, movies, and reproducibility pipelines.

## Release and migration

- `CHANGELOG.md` records release-facing changes.
- `RELEASE.md` defines the release checklist and deprecation policy.
- `docs/api_policy.md` documents the v1 compatibility contract.
- `docs/migration.md` maps archived scripts to active CLI workflows.
- `CITATION.cff` is present for repository-level citation metadata; formal DOI
  metadata should be updated when the first release is tagged.

## Citation

MHX is not yet publication-release-citable. Until a tagged release and DOI are
created for the rebuilt package, cite the repository URL and commit SHA, or use
the provisional metadata in `CITATION.cff`.
