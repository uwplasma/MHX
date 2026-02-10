# MHX code map (current scripts → package refactor plan)

This repository currently consists of research scripts in the repo root. The goal
is to migrate the *reusable, tested* components into the installable `mhx/`
package, and keep `scripts/` as thin CLI wrappers.

Key guiding principles:

- Put the **physics + numerics** in `mhx/solver/` (pure functions where possible).
- Put **diagnostics/metrics** (e.g. `f_kin`, plasmoid complexity, growth fits) in
  `mhx/solver/diagnostics.py` so they are *defined once* and reused by scans,
  inverse design, and figure generators.
- Put **configuration** (including the *objective function definition*) in
  `mhx/config.py`, and persist configs alongside outputs.
- Put **I/O** (output directory layout + NPZ schema) in `mhx/io/`.

## Critical consistency issues to fix (known bugs)

1) **Objective mismatch (apples-to-oranges plots)**
   - `mhd_tearing_inverse_design_figures.py` historically used a different
     `(target_f_kin, target_complexity, lambda_complexity)` than the inverse-design
     training script, making “grid-search vs inverse-design” comparisons
     misleading.
   - Fix: define a single `Objective` dataclass in `mhx/config.py`, save it with
     each inverse-design run (e.g. `config.yaml` + `history.npz` fields), and make
     all comparison plots *load the objective used for training* unless
     explicitly overridden.

2) **Metrics API drift**
   - Multiple scripts wrap the solver in `_simulate_metrics(...)` helpers with
     slightly different return signatures.
   - Fix: standardize on a single return type, e.g.
     `simulate_metrics(...) -> (TearingMetrics, SimulationResult)`, and update all
     call sites.

## Script-by-script summary and migration target

### Core solver / shared utilities

- `mhd_tearing_solve.py`
  - Purpose: 2D/2.5D incompressible pseudo-spectral MHD tearing solver using
    Diffrax; includes equilibrium init, time stepping, diagnostics (energies,
    tearing amplitude, reconnection proxy, plasmoid complexity, growth fits),
    and `.npz` persistence.
  - Key reusable core: `_run_tearing_simulation_and_diagnostics`, spectral ops,
    equilibrium builders, diagnostic helpers (growth fit, complexity metric).
  - Migration:
    - `mhx/solver/tearing.py` (solver + time stepping)
    - `mhx/solver/equilibria.py` (equilibrium init)
    - `mhx/solver/diagnostics.py` (metrics: `f_kin`, `C_plasmoid`, `gamma_fit`,
      reconnection proxy, etc.)
    - `mhx/io/npz.py` (stable NPZ schema + load/save helpers)

- `run_MHD.py`
  - Purpose: older/alternate tearing driver + plotting/movie utilities; overlaps
    heavily with `mhd_tearing_solve.py`.
  - Migration: retire into `scripts/legacy/` after verifying no unique features.

- `run_MHD_box.py`
  - Purpose: 3D incompressible MHD box solver (ABC-type initial conditions)
    using Equinox + Diffrax.
  - Migration: `mhx/solver/box3d.py` (optional module) + `mhx` CLI subcommand
    later, or keep as an “examples/legacy” script if not in scope for tearing
    workflows.

### Diagnostics / postprocessing

- `mhd_tearing_postprocess.py`
  - Purpose: postprocess `.npz` outputs from tearing solver; produces
    publication-style plots + optional movies; includes growth-fit window logic.
  - Duplication: growth-fit window selection; tearing amplitude extraction;
    shell spectra.
  - Migration:
    - reusable pieces → `mhx/solver/diagnostics.py` and `mhx/plotting/` (later)
    - CLI wrapper → `mhx figures --run <outdir>`

- `mhd_reconnection_rate.py`
  - Purpose: reconnection diagnostics from saved solution files (time series,
    movie slices, growth fit from `rms(Bx)`).
  - Duplication: growth fit; reconnection proxy logic.
  - Migration: `mhx/solver/diagnostics.py` + plotting via `mhx figures`.

- `mhd_tearing_island_evolution.py`
  - Purpose: island width evolution and nonlinear tearing analysis from saved
    solutions; optional movies.
  - Migration: `mhx/solver/diagnostics.py` (island width) + `mhx figures`.

### Scans / benchmarks

- `mhd_tearing_scan.py`
  - Purpose: Loureiro-style scan over `(eta, nu)` (log-grid), runs solver,
    analyzes each run (linear fit windows, regimes), aggregates summary plots.
  - Migration:
    - scan engine → `mhx/scans/scan.py`
    - analysis helpers → `mhx/solver/diagnostics.py`
    - output management → `mhx/io/paths.py`, `mhx/io/npz.py`

- `mhd_linear_benchmarks.py`
  - Purpose: linear tearing benchmarks vs theory; resolution convergence; S,
    guide-field, viscosity scans.
  - Migration: `mhx/scans/benchmarks.py` + tests verifying benchmark fits in
    FAST mode.

### Optimization / inverse design

- `mhd_tearing_inverse_design.py`
  - Purpose: differentiable inverse design via Equinox+Optax; MLP outputs
    `(log10_eta, log10_nu)`; backprop through the MHD simulation; saves
    initial/mid/final solution NPZ + training history; makes diagnostic figures.
  - Critical issue: objective overrides in `main()` led to mismatches with the
    figure generator defaults.
  - Migration:
    - training loop → `mhx/inverse_design/train.py`
    - objective definition → `mhx/config.py:Objective`
    - persistence → `mhx/io/npz.py` + `mhx/io/paths.py`
    - figures → `mhx/inverse_design/figures.py` + `mhx figures`

- `mhd_tearing_inverse_design_figures.py`
  - Purpose: parameter scans in `(log10_eta, log10_nu)` to build reachable
    regions + paper figures; compares inverse-design history vs grid search.
  - Critical issue: must load the same `Objective` used during training when
    comparing “grid vs inverse”.
  - Migration: `mhx/inverse_design/figures.py` (should read objective/config
    from saved run artifacts).

- `mhd_tearing_ideal_tearing_opt.py`
  - Purpose: differentiable “ideal tearing” benchmark; optimizes sheet
    thickness parameter against theoretical scaling; produces figures.
  - Migration: `mhx/optim/ideal_tearing.py` (optional; after core config/io).

- `mhd_tearing_energy_plasmoid_opt.py`
  - Purpose: differentiable reconnection design using energy + plasmoid
    complexity objectives; saves history + figures.
  - Migration: `mhx/optim/energy_plasmoid.py` (optional; after Objective unify).

### ML surrogate models

- `mhd_tearing_ml.py`, `mhd_tearing_ml_v2.py`
  - Purpose: generate datasets from solver runs; train surrogate (MLP) and/or
    latent ODE reduced model; includes training and plotting utilities.
  - Migration:
    - dataset builder → `mhx/ml/datasets.py`
    - surrogates → `mhx/ml/surrogates.py`

- `mhd_tearing_postprocess_ml.py`, `mhd_tearing_postprocess_ml_v2.py`
  - Purpose: compare ML surrogates / latent ODE models to full simulations;
    produce paper-style plots.
  - Migration: `mhx/ml/figures.py` (later).

## Near-term milestones (first refactor slice)

1) Introduce an installable `mhx/` package + `mhx` CLI with `simulate --fast`.
2) Define and persist `Objective` so inverse-design and figure comparisons use
   the *same* targets/weights by default.
3) Add an end-to-end smoke test (FAST simulation → metrics → save → figures).

