# MHX

[![CI](https://github.com/uwplasma/MHX/actions/workflows/ci.yml/badge.svg)](https://github.com/uwplasma/MHX/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/mhx/badge/?version=latest)](https://mhx.readthedocs.io/)

**MHX is a JAX-native, differentiable plasma and magnetohydrodynamics toolkit
for magnetic reconnection, tearing-mode studies, reduced-MHD experiments, and
future inverse-design workflows.**

The active Python package lives under `src/mhx/` and exposes command-line tools,
benchmark runners, plotting utilities, artifact manifests, and a small public
API for reproducible reduced-MHD studies.

## MHD at a Glance

These previews keep the README visual and short. The first two are compressed
from a longer seeded periodic double-Harris nonlinear run; the turbulence panel
is a labeled pedagogical schematic. See [docs/media.md](docs/media.md) for
source commands, visual QA, and claim boundaries.

| Seeded Harris-sheet response | Current filaments | Turbulent cascade guide |
| --- | --- | --- |
| ![Double-Harris reconnection replay](docs/_static/readme/double_harris_reconnection.gif) | ![Double-Harris current sheet](docs/_static/readme/double_harris_current_sheet.gif) | ![MHD turbulence cascade schematic](docs/_static/readme/mhd_turbulence_cascade.gif) |
| Seeded-minus-base flux perturbation over a `128×128`, `t=100` validation run. | Perturbation current filaments from the same long run. | Literature-inspired MHD cascade schematic, not solver output. |

## What Works Today

MHX currently supports deterministic reduced-MHD validation for spectral
operators, resistive decay, finite-domain Harris tearing checks, nonlinear
energy/dissipation budgets, bounded double-Harris and Rutherford execution-path
checks, and seed-robust QI plus latent-ODE workflow tests on small datasets.

Current results should be read at their manifest claim level. MHX does **not**
yet claim converged Rutherford island growth, Sweet-Parker plasmoid chains,
calibrated production surrogates, turbulence statistics, or inverse-design
results.

## Install

```bash
git clone https://github.com/uwplasma/MHX.git
cd MHX
python -m pip install -e ".[dev,docs]"
mhx version
```

JAX accelerator wheels are platform-specific. For GPU/TPU installs, follow the
official JAX instructions first, then install MHX.

## Quickstart

Run a deterministic reduced-MHD smoke workflow:

```bash
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
from mhx.benchmarks import run_linear_tearing_smoke
from mhx.config import load_config

cfg = load_config("examples/linear_tearing.toml")
trajectory, diagnostics = run_linear_tearing_smoke(cfg)
print(diagnostics["final_total_energy"])
```

## Documentation

| Need | Start here |
| --- | --- |
| First run and plugin demo | [docs/quickstart.md](docs/quickstart.md) |
| Media sources and claim boundaries | [docs/media.md](docs/media.md) |
| Physics validation details | [docs/validation.md](docs/validation.md) |
| Benchmark commands and expected artifacts | [docs/benchmarks.md](docs/benchmarks.md) |
| Long-run duration evidence | [docs/long_run_evidence.md](docs/long_run_evidence.md) |
| Campaign planning and execution | [docs/campaign_runner.md](docs/campaign_runner.md) |
| API compatibility policy | [docs/api_policy.md](docs/api_policy.md) |

Common entry points:

```bash
mhx validate all --outdir outputs/validation_suite
mhx benchmark catalog --outdir outputs/benchmarks/catalog
mhx campaign rutherford-plan-production --outdir outputs/campaigns/rutherford_production_plan
mhx physics list
mhx diagnostics list
```

## Citation

MHX is not yet publication-release-citable. Until a tagged release and DOI are
created, cite the repository URL and commit SHA, or use the provisional metadata
in `CITATION.cff`.
