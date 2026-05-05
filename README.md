# MHX

[![CI](https://github.com/uwplasma/MHX/actions/workflows/ci.yml/badge.svg)](https://github.com/uwplasma/MHX/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/mhx/badge/?version=latest)](https://mhx.readthedocs.io/)

**MHX is being rebuilt as a JAX-native, differentiable plasma and magnetohydrodynamics framework for magnetic reconnection, tearing modes, turbulence, validation, and inverse design.**

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

The first scientific target is a small spectral tearing benchmark with gradient
checks. Hall, radiative terms, 3D, finite-volume shock-capturing MHD, and neural
ODE workflows will be added after the core API is stable.

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
```

Inspect configurable physics terms:

```bash
mhx physics equilibria
mhx physics list
mhx physics lint hyper_resistivity
```

Try a reduced-state two-fluid toy extension:

```bash
mhx run examples/linear_tearing_twofluid_toy.toml --outdir outputs/twofluid_toy
mhx figures outputs/twofluid_toy --gif
```

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

## Citation

MHX is not yet release-citable. Until a tagged release and `CITATION.cff` are
created for the rebuilt package, cite the repository URL and commit SHA.
