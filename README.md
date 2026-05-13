# MHX

[![CI](https://github.com/uwplasma/MHX/actions/workflows/ci.yml/badge.svg)](https://github.com/uwplasma/MHX/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/mhx/badge/?version=latest)](https://mhx.readthedocs.io/)

**MHX is a validation-first rebuild of a JAX-native, differentiable plasma and
magnetohydrodynamics framework for magnetic reconnection, tearing modes,
extension experiments, and eventually inverse design.**

The active package lives under `src/mhx/`. The previous reduced-MHD
tearing/plasmoid code is archived under `legacy/old_mhx/`.

## MHD at a Glance

These compact animations are the README-facing overview. Detailed tests,
validation tables, still figures, and scaffold comparisons live in `docs/`.

| Reconnection replay | Current-sheet dynamics |
| --- | --- |
| ![Double-Harris reconnection replay](docs/_static/readme/double_harris_reconnection.gif) | ![Double-Harris current sheet](docs/_static/readme/double_harris_current_sheet.gif) |
| Real MHX seeded double-Harris validation output, compressed for the landing page. | Same replay through fixed-scale current density, useful for seeing sheet sharpening and filament structure. |

| Harris tearing | Plasmoid target | Turbulent cascade |
| --- | --- | --- |
| ![Harris tearing layer sweep](docs/_static/readme/harris_layer_sweep.gif) | ![Plasmoid scaling schematic](docs/_static/readme/plasmoid_scaling_schematic.gif) | ![MHD turbulence cascade schematic](docs/_static/readme/mhd_turbulence_cascade.gif) |
| Real finite-domain Harris eigenproblem validation. | Literature schematic for Sweet-Parker plasmoid scaling; not solver output. | Literature-inspired MHD turbulence schematic; not solver output. |

## Why MHX

- **Differentiable physics core:** JAX spectral operators, reduced-MHD RHS
  assembly, TOML-driven runs, diagnostics, figures, and artifact manifests.
- **Validation before storytelling:** every result is labeled as `smoke`,
  `validation`, `production_template`, or `production`.
- **Extension path:** plugin-style physics terms and diagnostics support local
  experiments without patching the core package.
- **Roadmap discipline:** long nonlinear Rutherford/plasmoid and neural-ODE
  inverse-design claims remain gated until duration, convergence, seed/QI, and
  manifest requirements are satisfied.

## What Is Reliable Today

MHX can currently defend deterministic FAST validation for:

- spectral derivative and inverse-Laplacian identities on periodic grids;
- exact resistive decay and linearized reduced-MHD operator checks;
- a published-reference Harris tearing eigenvalue gate plus finite-domain
  dispersion, eigenfunction-localization, and time-domain replay gates;
- nonlinear reduced-MHD energy/dissipation budget checks;
- short, bounded nonlinear double-Harris and Rutherford execution-path checks;
- seed-robust QI and fitted latent-ODE workflow validation on small datasets.

MHX does **not** yet claim converged Rutherford island growth, Sweet-Parker
plasmoid chains from the nonlinear solver, calibrated production surrogates, or
inverse-design results. Schematic assets are labeled as schematics; solver
figures and movies should be read at their manifest claim level.

## Quickstart

```bash
git clone https://github.com/uwplasma/MHX.git
cd MHX
python -m pip install -e ".[dev,docs]"
mhx version
```

JAX accelerator wheels are platform-specific. For GPU/TPU installs, follow the
official JAX instructions first, then install MHX.

Run the deterministic reduced-MHD smoke workflow:

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

## Reviewer Trail

Use the documentation for validation detail instead of treating the README as a
test log:

- [Quickstart](docs/quickstart.md): install, first run, figures, reports, and
  plugin demo.
- [Reviewer evidence map](docs/reviewer_evidence.md): claim levels, gates,
  source-code links, and reproduction commands.
- [Validation suite](docs/validation.md): physics gates, figures, tolerances,
  and limitations.
- [Publication checklist](docs/publication_checklist.md): what is ready for a
  paper figure and what remains validation-only.
- [Campaign runner operations](docs/campaign_runner.md): FAST artifacts versus
  long Rutherford/plasmoid production runs.
- [Skeptical audit](docs/audit.md): maturity table and explicit non-claims.
- [README media notes](docs/media.md): which landing-page assets are real
  validation output and which are schematics.

## Landing-Page Audit

The README intentionally avoids content that belongs in reviewer or developer
docs:

- exhaustive benchmark command catalogs;
- validation figure galleries and artifact inventories;
- CI/scaffold/test-output checklists;
- production-run chunking details and resume schemas;
- neural-ODE training artifact lists;
- plugin template walkthroughs beyond the first entry point.

Those details remain in `docs/`, where they can carry tolerances, commands,
claim boundaries, and maintenance context without burying the project overview.

## Common Workflows

| Goal | Entry point |
| --- | --- |
| Run the compact validation suite | `mhx validate all --outdir outputs/validation_suite` |
| Inspect benchmark commands | `mhx benchmark catalog --outdir outputs/benchmarks/catalog` and [docs/benchmarks.md](docs/benchmarks.md) |
| Generate a duration-guarded Rutherford plan | `mhx campaign rutherford-plan-production --outdir outputs/campaigns/rutherford_production_plan` |
| Exercise neural-ODE reproducibility | `mhx neural-ode dataset --outdir outputs/neural_ode/seed_qi_fast` and [docs/neural_ode_reproducibility.md](docs/neural_ode_reproducibility.md) |
| Inspect extension points | `mhx physics list`, `mhx diagnostics list`, and [docs/plugins.md](docs/plugins.md) |

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

## Release and Migration

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
