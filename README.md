# MHX

[![CI](https://github.com/uwplasma/MHX/actions/workflows/ci.yml/badge.svg)](https://github.com/uwplasma/MHX/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/uwplasma/MHX/main/badges/coverage.json)](https://github.com/uwplasma/MHX/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/mhx/badge/?version=latest)](https://mhx.readthedocs.io/)

**MHX is a JAX-native, differentiable plasma and magnetohydrodynamics toolkit
for magnetic reconnection, tearing-mode studies, reduced-MHD experiments, and
future inverse-design workflows.**

The active Python package lives under `src/mhx/` and exposes command-line tools,
benchmark runners, plotting utilities, artifact manifests, and a small public
API for reproducible reduced-MHD studies.

## MHD at a Glance

These previews keep the README visual and short. The first row shows longer
Harris-sheet and forced turbulent current-sheet replays with magnetic-flux
(`Az`/`ψ`) contours; the second row shows nonlinear reduced-MHD turbulence,
Orszag--Tang roll-up, and a Harris tearing layer sweep. See
[docs/media.md](docs/media.md) for source commands, visual QA, and claim
boundaries.

| Harris reconnection (`Az` contours) | Forced turbulent reconnection | Orszag--Tang current sheets |
| --- | --- | --- |
| ![Double-Harris reconnection replay](docs/_static/readme/double_harris_reconnection.gif) | ![Forced turbulent reconnection](docs/_static/readme/forced_turbulent_reconnection.gif) | ![Orszag-Tang current sheets](docs/_static/readme/orszag_tang_current.gif) |
| Single-sheet zoom from a `96×96`, `t=120` validation run: `j_z` background plus `Az` contours and detected X/O markers. | `64×64`, `t=80` forced-turbulence current sheet with reconnection-rate proxy diagnostics. | Solver-generated Orszag--Tang current-density cascade over a `96×96`, `t=10` validation run. |

| Decaying MHD turbulence | Orszag--Tang vorticity | Harris tearing layer |
| --- | --- | --- |
| ![Decaying MHD turbulence](docs/_static/readme/decaying_mhd_turbulence_current.gif) | ![Orszag-Tang vorticity roll-up](docs/_static/readme/orszag_tang_vorticity.gif) | ![Harris tearing layer sweep](docs/_static/readme/harris_layer_sweep.gif) |
| Solver-generated `64×64`, `t=8` decaying reduced-MHD turbulence with current filaments. | Vorticity roll-up and filamentation from the same nonlinear Orszag--Tang run. | Literature-anchored Harris eigenfunction localization sweep. |

## What Works Today

MHX currently supports deterministic reduced-MHD validation for spectral
operators, resistive decay, finite-domain Harris tearing checks, nonlinear
energy/dissipation budgets, Orszag--Tang vortex media, bounded double-Harris
and Rutherford execution-path checks, and seed-robust QI plus latent-ODE
workflow tests on small datasets.

Current results should be read at their manifest claim level. MHX does **not**
yet claim converged Rutherford island growth, Sweet-Parker plasmoid chains,
calibrated production surrogates, turbulence statistics, or inverse-design
results.

## Install

```bash
git clone https://github.com/uwplasma/MHX.git
cd MHX
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,docs]"
mhx version
```

JAX accelerator wheels are platform-specific. For GPU/TPU installs, follow the
official JAX instructions first, then install MHX.

## Quickstart

Run a deterministic reduced-MHD smoke workflow:

```bash
mhx init outputs/tutorial/linear_tearing.toml
mhx run examples/linear_tearing.toml --outdir outputs/smoke
mhx figures outputs/smoke --gif
mhx report outputs/smoke
mhx artifact-manifest outputs/smoke
```

Check the public API contract:

```bash
mhx api status
MHX_API_VERSION=v1 mhx api status --json
```

Use MHX from Python:

```python
import mhx

manifest = mhx.run("examples/linear_tearing.toml", outdir="outputs/python_api")
cfg = mhx.load_config("examples/linear_tearing.toml")
print(manifest, cfg.physics.model)
```

## Documentation

| Need | Start here |
| --- | --- |
| First run and plugin demo | [docs/quickstart.md](docs/quickstart.md) |
| Installation and environments | [docs/install.md](docs/install.md) |
| Guided tutorial | [docs/tutorial.md](docs/tutorial.md) |
| Media sources and claim boundaries | [docs/media.md](docs/media.md) |
| Physics validation details | [docs/validation.md](docs/validation.md) |
| Benchmark commands and expected artifacts | [docs/benchmarks.md](docs/benchmarks.md) |
| Diagnostics and output schemas | [docs/diagnostics.md](docs/diagnostics.md), [docs/output_schema.md](docs/output_schema.md) |
| Neural-ODE reproducibility | [docs/neural_ode_reproducibility.md](docs/neural_ode_reproducibility.md) |
| Performance and timing | [docs/performance.md](docs/performance.md) |
| Long-run duration evidence | [docs/long_run_evidence.md](docs/long_run_evidence.md) |
| Campaign planning and execution | [docs/campaign_runner.md](docs/campaign_runner.md) |
| API compatibility policy | [docs/api_policy.md](docs/api_policy.md) |

Common entry points:

```bash
mhx validate all --outdir outputs/validation_suite
mhx benchmark catalog --outdir outputs/benchmarks/catalog
mhx campaign rutherford-plan-production --outdir outputs/campaigns/rutherford_production_plan
# After target completion plus convergence/seed-QI evidence:
mhx campaign rutherford-promotion-check outputs/campaigns/rutherford_production_plan
mhx validate readiness --suite outputs/validation_suite --outdir outputs/validation_readiness
mhx api deprecations
mhx physics list
mhx diagnostics list
```

## Citation

MHX is not yet publication-release-citable. Until a tagged release and DOI are
created, cite the repository URL and commit SHA, or use the provisional metadata
in `CITATION.cff`.
