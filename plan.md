# MHX Rebuild Plan and Agent Log

Generated: 2026-05-04

This file is both the implementation plan for the new MHX and the persistent execution log for coding agents. Keep this file in the repository root as `plan.md`. Every agent pass must update the log section at the bottom with what changed, what was tested, what failed, and what remains next.

---

## 0. Mission

Build the new **MHX** as a research-grade, open-source, differentiable JAX framework for magnetic reconnection, magnetohydrodynamics, plasma turbulence, and related instabilities. The new MHX will replace the current codebase. The old code must be moved into `legacy/` and preserved for reference, not incrementally patched.

The target product is not merely a solver. It is a **validated differentiable simulation framework** with:

- publication-anchored benchmark problems;
- reproducible comparison harnesses against established MHD, extended-MHD, spectral, JAX, and kinetic codes;
- differentiable research workflows that are scientifically honest;
- well-gated physics modules;
- complete documentation of equations, discretizations, assumptions, inputs, outputs, and diagnostics;
- publication-ready plotting and diagnostics out of the box.

The new MHX must be easy for a new graduate student to run, rigorous enough to survive peer review, and modular enough to become a long-lived research code.

---

## 1. Source-informed design facts

These source notes should guide architecture choices. They are included so the coding agent can reason from concrete references rather than inventing the design.

### 1.1 Current MHX state

The current public MHX repository describes itself as a “Differentiable pseudo-spectral reduced MHD tearing/plasmoid solver and analysis tools (JAX-based).” It already has some CLI commands, reduced-MHD tearing scripts, inverse-design examples, output conventions, and toy physics plugins for Hall, hyper-resistivity, and anisotropic pressure. It is too narrow for the new goal, but it contains material worth preserving in `legacy/`.

Reference: https://github.com/uwplasma/MHX

### 1.2 JAX ecosystem facts

JAX is designed for accelerator-oriented array computation and composable transformations for compilation, batching, automatic differentiation, and parallelization. It can run on CPU, GPU, and TPU. Official docs list Equinox, Optimistix, Lineax, and Diffrax as part of the broader JAX ecosystem for models and scientific solvers.

Reference: https://docs.jax.dev/en/latest/

JAX autodiff supports scalar-output gradients via `jax.grad`, higher-order derivatives by composing transformations, and forward/reverse differentiation of pure functions. This makes pure solver APIs mandatory.

Reference: https://docs.jax.dev/en/latest/automatic-differentiation.html

`jax.checkpoint`, also known as `jax.remat`, trades compute for memory during reverse-mode autodiff by recomputing intermediate linearization points rather than storing them. This is central for time-dependent PDE gradients.

Reference: https://docs.jax.dev/en/latest/_autosummary/jax.checkpoint.html

`jax.lax.custom_linear_solve` provides matrix-free linear solves with implicitly defined gradients, avoiding differentiation through the internal solve loop. This is important for implicit diffusion, projection, elliptic solves, and semi-implicit MHD steps.

Reference: https://docs.jax.dev/en/latest/_autosummary/jax.lax.custom_linear_solve.html

JAX defaults to 32-bit values unless `jax_enable_x64` is enabled. Scientific validation benchmarks should support and usually default to X64, while performance demos may offer X32.

Reference: https://docs.jax.dev/en/latest/default_dtypes.html

JAX provides profiling tools, including XProf/TensorBoard profiling and device-memory profiling. Timings must use `block_until_ready()` because execution is asynchronous.

References:
- https://docs.jax.dev/en/latest/profiling.html
- https://docs.jax.dev/en/latest/device_memory_profiling.html

Pallas is an experimental JAX kernel language for custom GPU/TPU kernels. It should not be a baseline dependency, but it may become useful for optimized stencil kernels, constrained-transport kernels, or high-order reconstruction kernels once bottlenecks are known.

Reference: https://docs.jax.dev/en/latest/pallas/index.html

### 1.3 Equinox facts

Equinox represents models as PyTrees and provides filtered transformations. `equinox.filter_jit` treats JAX/NumPy arrays dynamically and non-array leaves statically at the PyTree-leaf level. This is a good fit for MHX state, solver modules, reconstruction modules, physics closures, and differentiable parameter sets.

References:
- https://docs.kidger.site/equinox/all-of-equinox/
- https://docs.kidger.site/equinox/api/transformations/

### 1.4 Diffrax facts

Diffrax is a JAX-based differential-equation solver library supporting ODE/SDE/CDE solvers, PyTree states, dense solutions, vmappable solves, implicit solvers, and multiple adjoint methods.

Reference: https://docs.kidger.site/diffrax/

Diffrax’s default `RecursiveCheckpointAdjoint` differentiates through the numerical solution directly using online checkpointing, and its docs describe this as the preferred method for most problems. `BacksolveAdjoint` solves continuous adjoint equations backward in time; Diffrax warns that it is not recommended in practice because gradients are approximate, while checkpointed discrete adjoints also have low memory.

Reference: https://docs.kidger.site/diffrax/api/adjoints/

Interpretation for MHX: Diffrax should be used for prototypes, small research ODE/PDE experiments, adaptive solver experiments, and some inverse problems. The production PDE stepping path should also provide custom fixed-step `lax.scan` integrators because they are easier to benchmark, easier to compare code-to-code, easier to control under CFL constraints, and generally more transparent for PDE verification.

### 1.5 Lineax and implicit solves

Lineax is a JAX library for linear solves and linear least squares. It supports PyTree-valued matrices/vectors, general linear operators, Jacobian/transposes, stable gradients, structured matrices, complex inputs, autodiff, autoparallelism, and GPU/TPU support.

Reference: https://docs.kidger.site/lineax/

Lineax has iterative solvers such as CG, BiCGStab, GMRES, and LSMR, and can auto-select solvers based on operator structure.

Reference: https://docs.kidger.site/lineax/api/solvers/

Interpretation for MHX: use Lineax for Poisson/projection solves, implicit diffusion/resistivity/conduction, elliptic constraints, linearized MHD operators, Newton-Krylov prototypes, and Hessian-vector/least-squares workflows.

### 1.6 Optimistix, Optax, Orbax, jaxtyping

Optimistix provides JAX nonlinear solvers for root finding, minimization, fixed points, and least squares. It is useful for implicit steps, steady states, parameter inference, and nonlinear calibration.

Reference: https://docs.kidger.site/optimistix/

Optax is a gradient-processing and optimization library for JAX. It is useful for differentiable inverse-design experiments and parameter estimation.

Reference: https://optax.readthedocs.io/

Orbax provides checkpointing and persistence utilities for JAX users, including multi-host/multi-device settings.

Reference: https://orbax.readthedocs.io/

jaxtyping provides shape and dtype annotations for array-heavy code and should be used with runtime checks in development/test mode.

Reference: https://docs.kidger.site/jaxtyping/

### 1.7 Spectral adjoints and the arXiv:2506.14792 lesson

Skene and Burns, “Fast automated adjoints for spectral PDE solvers,” present automated reverse-mode adjoints for sparse spectral PDE solvers in Dedalus. Their approach applies AD to symbolic graph representations of PDEs, constructs adjoint solvers, and preserves the speed and memory efficiency of spectral methods.

References:
- https://arxiv.org/abs/2506.14792
- https://www.pnas.org/doi/10.1073/pnas.2530440123

Interpretation for MHX: do not rely only on naive reverse-mode AD through every spectral timestep for all use cases. Start with ordinary JAX reverse-mode through fixed-step spectral solvers, but design the spectral layer around explicit operator graphs so that custom transposes, custom VJPs, and automated discrete adjoints can be added later. The long-term spectral path should mimic the “operator graph -> adjoint operator graph” idea, but in JAX.

### 1.8 Existing codes to compare against

Athena++ is a C++ astrophysical MHD/GRMHD/AMR framework with improved coordinate/grid options, AMR, general relativity, performance, scalability, and modularity.

Reference: https://www.athena-astro.app/

The original Athena paper describes higher-order Godunov methods with constrained transport for divergence-free magnetic fields and provides a test suite for 1D, 2D, and 3D hydro/MHD comparisons.

Reference: https://arxiv.org/abs/0804.0402

PLUTO is a multiphysics, multialgorithm astrophysical code for 1D/2D/3D hypersonic flows, discontinuities, Newtonian/relativistic/MHD/RMHD fluids, and Godunov-type shock capturing. Its AMR paper describes finite-volume CTU, PPM/WENO/slope-limited reconstruction, GLM divergence cleaning, viscosity, resistivity, anisotropic thermal conduction, and stiff source treatment.

References:
- https://ui.adsabs.harvard.edu/abs/2007ApJS..170..228M/abstract
- https://arxiv.org/abs/1110.0740

MPI-AMRVAC is an open-source parallel AMR framework for hyperbolic PDEs, conservation laws, shock-dominated problems, and hydro/MHD modules, supporting 1D through 3D and Cartesian/cylindrical/spherical geometries.

Reference: https://amrvac.org/

FLASH is a public high-performance multiphysics code. Its current capabilities include MHD, full Braginskii extended-MHD terms in generalized Ohm’s law, radiation transfer, diffusion, anisotropic conduction, anisotropic resistivity, magnetized plasma transport coefficients, AMR, diagnostics, unit tests, and regression tests.

Reference: https://flash.rochester.edu/site/flashcode/

OpenMHD is a multidimensional finite-volume MHD code in modern Fortran and CUDA Fortran, parallelized with MPI/OpenMP/CUDA, originally developed for magnetic reconnection in space physics. It uses second-order RK, MUSCL, HLLD, resistive MHD terms, and hyperbolic divergence cleaning.

Reference: https://sci.nao.ac.jp/MEMBER/zenitani/openmhd-e.html

Dedalus is an open-source, Python, MPI-parallelized spectral framework for initial-value, boundary-value, and eigenvalue problems with flexible equation specification.

Reference: https://dedalus-project.org/

Gkeyll is a multiscale, multiphysics plasma framework covering fluid, multifluid, Vlasov, gyrokinetic, PKPM, space physics, plasma physics, GR, and high-energy astrophysics. Its ten-moment documentation notes that Hall, inertia, and tensor pressure effects are self-consistently embedded without explicitly solving generalized Ohm’s law, and that Orszag-Tang and GEM reconnection have been benchmarked.

References:
- https://gkeyll.readthedocs.io/
- https://heliophysics.princeton.edu/gkeyll/overview

JAX-Fluids is a fully differentiable JAX CFD solver for 3D compressible single- and two-phase flows. It provides a useful reference for JAX-native HPC, high-order finite-volume methods, differentiability, JAX primitive-based parallelization, positivity limiters, and performance benchmarking. JAX-Fluids 2.0 reports scaling on large GPU/TPU systems and stable AD gradients over extended trajectories.

References:
- https://github.com/tumaer/JAXFLUIDS
- https://www.sciencedirect.com/science/article/pii/S0010465524003564

astronomix is a recent differentiable hydro/MHD code in JAX focused on astrophysical applications. It supports 1D/2D/3D hydro/MHD, multi-GPU scaling, high-order finite-difference constrained-transport WENO MHD, a divergence-free/positivity-preserving finite-volume approach, adaptive time stepping, turbulent driving, stellar wind, and simple radiative cooling modules.

Reference: https://github.com/leo1200/astronomix

JAX-CFD is an older experimental JAX CFD project and is no longer maintained. It remains useful as a historical design reference but should not be treated as a state-of-the-art competitor.

Reference: https://github.com/google/jax-cfd

VPIC, WarpX, Smilei, and EPOCH are kinetic/PIC references, not direct MHD competitors. They are relevant for comparison at the physics-interface level, especially Hall, hybrid, generalized-MHD, and collisionless reconnection limits.

References:
- VPIC: https://github.com/lanl/vpic
- WarpX: https://github.com/BLAST-WarpX/warpx
- Smilei: https://smileipic.github.io/Smilei/
- EPOCH: https://www.york.ac.uk/physics-engineering-technology/ypi/research/computationalplasmaphysics/epoch/

---

## 2. Repository migration requirements

### 2.1 First coding action

The first coding agent pass must perform a clean migration:

1. Create `legacy/old_mhx/`.
2. Move the current MHX implementation, scripts, existing docs, examples, tests, benchmark files, and root-level legacy scripts into `legacy/old_mhx/`.
3. Preserve `.git/` and repository metadata.
4. Keep or recreate root-level `LICENSE`, `README.md`, `pyproject.toml`, `.github/`, and `plan.md` for the new project.
5. Add a root-level `legacy/README.md` explaining why the old code was moved and how to run it if needed.
6. Do not delete old MHX files unless they are exact generated artifacts or caches.

### 2.2 Migration acceptance test

After migration:

- `git status` clearly shows moved files, not silent deletions.
- `legacy/old_mhx/README.md` exists.
- Root `README.md` describes the new MHX mission.
- Root `plan.md` exists and contains an updated log entry.
- New package skeleton imports with `python -c "import mhx; print(mhx.__version__)"`.

---

## 3. Product requirements

### 3.1 Required product qualities

MHX must be:

- **JAX-native**: core numerics use JAX arrays and transformations.
- **Differentiable by design**: solvers expose pure functions suitable for `grad`, `value_and_grad`, `jvp`, `vjp`, `vmap`, `scan`, and sharding.
- **Validation-first**: every physics module has a benchmark and diagnostic.
- **Comparison-ready**: benchmark outputs can be compared to published results and external codes.
- **Documented**: docs include equations, nondimensionalization, discretization, assumptions, examples, and limitations.
- **Modular**: equations, numerical schemes, physics closures, diagnostics, plotting, and optimization workflows are separable.
- **Honest about differentiability**: long chaotic trajectories, shocks, limiters, and event counts must carry warnings and alternative smooth/statistical objectives.
- **Easy to use**: command-line and Python APIs must both work.
- **Performance-measurable**: benchmark harnesses report compile time, runtime, memory, and throughput.

### 3.2 Non-goals for the first release

The first release should not attempt to:

- reimplement PIC;
- compete with Athena++/FLASH/Gkeyll on all multiphysics capabilities;
- support every geometry and AMR from day one;
- guarantee meaningful gradients through discontinuous event counts or arbitrary adaptive-step chaotic trajectories;
- implement production-grade relativistic MHD immediately;
- use Pallas kernels before profiling identifies bottlenecks.

---

## 4. Recommended top-level package layout

Use `src/` layout.

```text
.
├── plan.md
├── README.md
├── LICENSE
├── pyproject.toml
├── legacy/
│   ├── README.md
│   └── old_mhx/
├── src/
│   └── mhx/
│       ├── __init__.py
│       ├── _version.py
│       ├── config/
│       ├── equations/
│       ├── state/
│       ├── grids/
│       ├── numerics/
│       │   ├── spectral/
│       │   ├── finite_volume/
│       │   ├── time_integrators/
│       │   ├── constrained_transport/
│       │   ├── divergence_cleaning/
│       │   ├── reconstruction/
│       │   ├── riemann/
│       │   └── linear_solvers/
│       ├── physics/
│       ├── diagnostics/
│       ├── benchmarks/
│       ├── compare/
│       ├── diff/
│       ├── plotting/
│       ├── io/
│       ├── cli/
│       └── utils/
├── tests/
├── examples/
├── benchmarks/
├── comparison_cases/
├── docs/
└── scripts/
```

### 4.1 Package layers

- `state/`: conservative, primitive, reduced-MHD, spectral-state, metadata PyTrees.
- `grids/`: Cartesian grids first; stretched and curvilinear later.
- `equations/`: RHS definitions and flux functions.
- `numerics/`: methods independent of physics where possible.
- `physics/`: resistivity, viscosity, cooling, conduction, Hall, hyper-resistivity, closures.
- `diagnostics/`: tested scientific diagnostics.
- `benchmarks/`: benchmark registry and runnable cases.
- `compare/`: external-code comparison adapters.
- `diff/`: differentiable objectives, gradient checks, inverse problems.
- `plotting/`: publication-ready figure recipes.
- `io/`: checkpointing, datasets, run manifests.
- `cli/`: command-line entrypoints.

---

## 5. Core design choices

### 5.1 State representation

Use Equinox modules or dataclasses registered as PyTrees for solver states. Separate array-valued dynamic leaves from static metadata.

Recommended primary arrays:

- Compressible conservative MHD state: `U[..., nvar]`, with variables last and spatial axes first.
- Primitive state: `W[..., nprim]`, produced as needed.
- Magnetic field for constrained transport: face-centered arrays when CT is enabled; cell-centered fallback for GLM.
- Reduced MHD state: flux function `psi`, vorticity `omega`, stream function `phi` when solved explicitly, guide-field variables as needed.
- Spectral state: complex Fourier coefficients with clear physical/spectral conversion utilities.

Use variable-last layout initially:

```text
U.shape == (*spatial_shape, nvar)
```

Reasoning:

- natural for `vmap` over variables and faces;
- simple `xarray` conversion;
- spatial axes are explicit for domain sharding;
- compatible with `jax.numpy` slicing and stencil utilities.

Do not make layout immutable: provide a layout abstraction so performance profiling can later justify variable-first or structure-of-arrays changes.

### 5.2 Configuration

Use a validated config schema, but keep solver state pure.

Recommended:

- `Config` as plain dataclass or Pydantic/msgspec model outside JIT;
- convert configs to lightweight static PyTree leaves before JIT;
- all numerical values that should be differentiable must live in a `Params` PyTree, not in static config.

Example separation:

```python
state: MHDState                 # arrays only
params: PhysicsParams           # differentiable arrays/scalars
cfg: SolverConfig               # mostly static choices
```

### 5.3 Pure stepping API

Every solver path must expose:

```python
def rhs(state, params, grid, cfg):
    ...

def step(state, params, grid, cfg, dt):
    ...

def evolve(state0, params, grid, cfg, t0, t1, dt_or_controller):
    ...
```

For fixed-step differentiable benchmarks, prefer:

```python
final_state, history = lax.scan(step_fn, state0, step_indices)
```

Adaptive stepping can be added, but benchmarked differentiability should start with fixed timesteps.

---

## 6. Numerical tracks

MHX should support two production tracks early. They solve different scientific problems and prevent the code from becoming either too fragile or too narrow.

### 6.1 Track A: pseudo-spectral reduced MHD / incompressible MHD

Purpose:

- tearing-mode growth rates;
- Rutherford island growth;
- plasmoid instability in periodic setups;
- guide-field turbulence;
- inverse design;
- differentiable reduced models;
- fast educational demos.

Methods:

- Fourier pseudo-spectral derivatives on periodic domains.
- 2/3 dealiasing option.
- `jax.numpy.fft` for transforms.
- Explicit RK4 and SSPRK3.
- IMEX option for diffusion/resistivity/hyper-resistivity.
- Optional projection using spectral Poisson solve.
- Spectral linear operators exposed as composable operator objects.

Differentiability strategy:

1. Baseline: direct reverse-mode through fixed-step `lax.scan` with `jax.checkpoint`/`eqx.filter_checkpoint` around step blocks.
2. Memory mode: chunked scans with rematerialization.
3. Long-term: custom spectral operator graph and custom VJP/transpose rules inspired by automated sparse-spectral adjoints.
4. Always validate gradients against finite differences on small grids.

Why keep this track:

- It preserves the useful core of legacy MHX.
- It gives clean differentiability demos.
- It is ideal for early benchmark success.
- It supports operator-level adjoint research.

### 6.2 Track B: finite-volume compressible MHD

Purpose:

- Brio-Wu shock tube;
- rotor/blast/Orszag-Tang;
- Kelvin-Helmholtz instability;
- Rayleigh-Taylor instability;
- Harris current sheet;
- Sweet-Parker reconnection;
- plasmoid reconnection with convergence;
- radiative source terms;
- comparison with Athena++, PLUTO, MPI-AMRVAC, FLASH, OpenMHD, and astronomix.

Baseline methods:

- Cartesian uniform grid first.
- Conservative finite-volume update.
- Reconstruction: piecewise constant, minmod/MC, WENO3/WENO5, optional smooth-WENO mode.
- Riemann solvers: Rusanov/Lax-Friedrichs first, HLL next, HLLD target.
- Time integrators: Euler, SSPRK2, SSPRK3.
- Divergence control: GLM cleaning first, constrained transport as priority milestone.
- Positivity preservation: density and pressure floors with explicit diagnostics; later true positivity-preserving limiting.
- Resistive terms: central differences, explicit first, IMEX/implicit later.
- Viscosity, conduction, cooling: source/operator-split modules.

Differentiability strategy:

- Provide `mode="research"` and `mode="smooth_ad"` variants.
- In `research` mode, use physically standard limiters/solvers even if nonsmooth.
- In `smooth_ad` mode, use smooth approximations for limiters/floors/objectives where appropriate.
- Do not hide nonsmoothness. Emit warnings when users differentiate through shock-capturing or event-count diagnostics.
- Use fixed timestep for gradient tests.
- Treat adaptive CFL timestep as nondifferentiable by default via `stop_gradient` or by freezing a schedule.

Why this track is mandatory:

- Reconnection applications often require compressibility, shocks, current sheets, cooling, conduction, and radiative terms.
- Serious code-to-code comparisons require finite-volume MHD.
- Spectral reduced MHD alone cannot validate against many research-grade astrophysical and space-plasma benchmarks.

### 6.3 Future Track C: DG / spectral element / AMR

Do not implement in the first release unless an immediate collaborator requires it. Keep interfaces compatible with later:

- discontinuous Galerkin;
- mapped grids;
- AMR or block-structured refinement;
- local time stepping.

---

## 7. JAX implementation rules

### 7.1 Mandatory rules

- No mutation of solver state inside JIT.
- No Python loops over timesteps in production paths; use `lax.scan` or Diffrax.
- No data-dependent Python branching inside JIT; use `lax.cond`, `lax.switch`, or static config.
- Shapes must be static inside compiled functions.
- Use `block_until_ready()` in every timing benchmark.
- Use `donate_argnums` or `equinox.filter_jit(..., donate=...)` in long simulations to reduce memory pressure.
- Keep all differentiable numerical parameters in PyTrees.
- Keep non-differentiable choices static.
- Enable X64 for validation unless a benchmark explicitly targets float32 performance.
- Record dtype in every run manifest.

### 7.2 Recommended dependencies

Core dependencies:

- `jax`
  - Do **not** list `jaxlib` as a project dependency. Keep `pyproject.toml` portable and let users install accelerator-specific JAX wheels from the official JAX instructions, for example `jax`, `jax[cuda13]`, or platform-specific instructions in the docs.
- `numpy`
- `scipy` only outside JIT/reference tests
- `equinox`
- `jaxtyping`
- `lineax`
- `diffrax`
- `optimistix`
- `optax`
- `matplotlib`
- `xarray`
- `h5py` or `zarr`
- `tomli; python_version < "3.11"` for Python 3.10 TOML reading
- `tomli-w` or `tomlkit` for writing/round-tripping example TOML files, if needed
- `pytest`

Optional extras:

- `orbax-checkpoint` for large checkpointing and multi-device runs.
- `rich` for CLI output.
- `typer` for CLI.
- `sphinx`, `myst-parser`, `sphinx-copybutton`, `sphinxcontrib-bibtex`, and `pydata-sphinx-theme` for ReadTheDocs documentation.
- `ruff`, `mypy`, `pre-commit` for development.
- `pytest-benchmark` for performance tests.
- `pandas` for comparison tables.

### 7.3 Diffrax use policy

Use Diffrax for:

- small ODE/RHS prototypes;
- adaptive time integration experiments;
- differentiable inverse-problem notebooks;
- stiff source term prototypes;
- comparing adjoint modes.

Do not make Diffrax the only production time integration path. A PDE code needs explicit control over CFL schedules, output cadence, histories, comparison with external codes, and fixed-step reproducibility.

### 7.4 Lineax use policy

Use Lineax for:

- Poisson solves;
- projection methods;
- implicit magnetic diffusion;
- implicit thermal conduction;
- hyper-resistive operators;
- linearized MHD/eigenvalue problems;
- Jacobian-vector and adjoint-vector products;
- least-squares calibration.

Wrap custom implicit solves with either Lineax’s differentiable API or `jax.lax.custom_linear_solve`. Always test transpose/adjoint consistency.

### 7.5 Spectral adjoint plan

Phase 1:

- Use ordinary JAX AD through FFT-based timesteps.
- Add checkpointed scan.
- Add finite-difference gradient tests.

Phase 2:

- Represent spectral equations as operator objects: derivative, Laplacian, inverse Laplacian, product, projection, dealiasing, linear diffusion, nonlinear bracket.
- Add custom transpose tests for each linear operator.
- Add custom VJP for common solve blocks.

Phase 3:

- Build an operator-graph adjoint path inspired by sparse-spectral automated adjoints.
- Compare memory/runtime of naive reverse-mode vs checkpointed vs custom adjoint.

---

## 8. Physics modules and gates

Physics modules must be opt-in and gated. A gate checks assumptions, nondimensional parameters, grid resolution, timestep constraints, and diagnostics.

### 8.1 Gate table

| Module | Gate checks | Required diagnostics |
|---|---|---|
| Ideal MHD | no explicit resistivity; warn that reconnection is numerical/ideal-topological only | energy, divB, flux conservation |
| Resistive MHD | Lundquist number `S`, resistive scale estimate, timestep constraint | Ohmic heating, reconnection rate, current layer width |
| Viscosity | Reynolds number, magnetic Prandtl number | viscous heating, energy budget |
| Hyper-resistivity | `S_H`, explicit `dt ~ dx^4/eta_H` limit, physical-vs-numerical label | effective dissipation scale, comparison to eta-only run |
| Hall MHD | `d_i/L`, grid resolves ion inertial length or warns | Hall quadrupole field, Ohm-law term budget |
| Electron inertia | `d_e/L`, grid resolves or declares reduced model | electron-inertia Ohm term |
| Anomalous resistivity | closure formula, activation threshold, comparison control | eta_eff maps, Ohm-law budget |
| Radiative cooling | cooling time vs timestep, positivity constraints, optically thin assumption | cooling losses, temperature floors, energy budget |
| Thermal conduction | parallel/perpendicular coefficients, timestep or implicit solve | conductive flux, heat budget |
| 3D topology | field-line tolerance, divB tolerance, resolution | field-line maps, E_parallel integral, QSL/null metrics |

### 8.2 Baseline equations

Documentation must include the conservative ideal/resistive MHD equations in nondimensional form.

Continuity:

```math
\partial_t \rho + \nabla\cdot(\rho \mathbf{v}) = 0.
```

Momentum:

```math
\partial_t(\rho\mathbf{v}) + \nabla\cdot\left[\rho\mathbf{v}\mathbf{v} + \left(p + \frac{B^2}{2}\right)I - \mathbf{B}\mathbf{B}\right]
= \nabla\cdot\Pi + \mathbf{S}_{m}.
```

Induction, schematic generalized form:

```math
\partial_t \mathbf{B}
= \nabla\times\left(
\mathbf{v}\times\mathbf{B}
- \eta \mathbf{J}
+ \eta_H \nabla^2\mathbf{J}
- \frac{\mathbf{J}\times\mathbf{B}}{ne}
+ \frac{\nabla\cdot\mathbf{P}_e}{ne}
+ \frac{m_e}{ne^2}\frac{d\mathbf{J}}{dt}
+ \cdots
\right),
```

with module-dependent terms enabled only when gates pass.

Energy:

```math
\partial_t E + \nabla\cdot F_E
= \eta J^2 + Q_{visc}
- n_e n_i \Lambda(T)
+ \nabla\cdot(\kappa_\parallel \hat{b}\hat{b}\cdot\nabla T)
+ H_{ext}.
```

Reduced-MHD equations must be documented separately, including normalization, guide-field assumptions, Poisson bracket definitions, and boundary conditions.

---

## 9. Diagnostics that must be first-class APIs

Diagnostics must be tested functions, not ad hoc plotting code.

### 9.1 Numerical diagnostics

- Mass, momentum, total energy.
- Magnetic, kinetic, internal, radiated, dissipated energy.
- `||div B||_1`, `||div B||_2`, `||div B||_inf`.
- CFL numbers and timestep limiting terms.
- Positivity/floor activation counts.
- Conservation residuals.
- Effective numerical diffusion estimates where feasible.

### 9.2 Reconnection diagnostics

- Current density `J = curl B`.
- Reconnection electric field.
- Flux function `psi` in 2D.
- X-point/O-point detection.
- Island width.
- Plasmoid count, with nondifferentiability warning.
- Current-sheet length/width.
- Inflow/outflow speeds.
- Sweet-Parker scaling metrics.
- Integrated `E_parallel` for 3D.
- Ohm’s-law term budget.
- `J · E` and energy conversion maps.

### 9.3 Turbulence diagnostics

- Magnetic and kinetic energy spectra.
- Current-density spectra.
- Structure functions.
- Intermittency proxies.
- Current-sheet statistics.
- Plasmoid/flux-rope size distributions.
- Reconnection-rate distribution over sites.

### 9.4 3D topology diagnostics

- Field-line tracing.
- Connectivity maps.
- Null finder.
- Spine/fan diagnostics.
- Separator/separatrix support where feasible.
- QSL squashing factor `Q`.
- Flux-rope identification.
- Magnetic helicity and relative helicity proxies.

### 9.5 Synthetic diagnostics

- Synthetic spacecraft cuts.
- Lineouts matching published figures.
- Synthetic emission/cooling maps.
- Synthetic radiative spectra proxies for radiative reconnection demos.

---

## 10. Benchmark registry

Every benchmark must include:

```toml
# benchmark registry entry, represented as TOML
name = ""
category = ""
equations = []
dimensions = 2
initial_conditions = ""
boundary_conditions = []
parameters = {}
resolution_suite = []
expected_result = ""
reference_sources = []
external_codes = []
metrics = []
plots = []
tolerances = {}
fast_ci = true
slow_ci = false
requires_gpu = false
requires_x64 = true
```

Each benchmark should provide:

- a TOML config;
- a Python example;
- a CLI command;
- a notebook where appropriate;
- a reference plot recipe;
- a test with tolerances;
- a documented failure mode.

---

## 11. Benchmark tiers

### 11.1 Tier 0: numerical sanity and convergence

Start here before claiming reconnection results.

1. Linear advection.
2. Diffusion equation.
3. Alfvén wave convergence.
4. Brio-Wu shock tube.
5. MHD rotor.
6. MHD blast.
7. Circularly polarized Alfvén wave.
8. Orszag-Tang vortex.
9. Divergence-cleaning/CT test.
10. Smooth manufactured solution for gradient validation.

### 11.2 Tier 1: hydrodynamic instability comparisons

Needed for comparison against JAX-Fluids, astronomix hydro mode, and classical CFD references.

1. Kelvin-Helmholtz instability.
2. Rayleigh-Taylor instability.
3. Sod shock tube.
4. Double shear layer.
5. Decaying turbulence.
6. Forced turbulence.

Differentiability examples:

- optimize KH initial perturbation amplitude or phase;
- infer viscosity from RT growth;
- find a minimal seed perturbation under norm constraint.

### 11.3 Tier 2: 2D reconnection

1. Linear tearing mode.
2. Rutherford island growth.
3. Sweet-Parker current sheet.
4. Plasmoid instability.
5. Ideal-tearing onset.
6. Harris-sheet reconnection.
7. Coalescing magnetic islands.
8. Hyper-resistive plasmoid benchmark.
9. Under-resolution caution benchmark.

Required plots:

- growth rate vs theory;
- reconnection rate vs Lundquist number;
- island width vs time;
- current sheet length/width;
- plasmoid count vs resolution;
- energy partition;
- Ohm’s-law budget.

### 11.4 Tier 3: Hall/generalized-MHD

1. GEM-like reconnection setup.
2. Hall quadrupole out-of-plane magnetic field.
3. Hall vs resistive-MHD reconnection-rate comparison.
4. Electron-inertia switch test.
5. Generalized Ohm’s-law term budget.
6. Anomalous-resistivity closure comparison.

Compare qualitatively and quantitatively with Gkeyll and literature where possible.

### 11.5 Tier 4: radiative and thermal reconnection

1. Optically thin cooling in a current sheet.
2. Bremsstrahlung-cooled double current sheet.
3. Radiative X-point collapse.
4. Plasmoid radiative collapse.
5. Conduction along magnetic field lines.
6. Cooling-law calibration inverse problem.

Required diagnostics:

- cooling time maps;
- radiated energy;
- temperature/density floors;
- Ohmic vs radiative budget;
- reconnection-rate changes vs cooling strength.

### 11.6 Tier 5: 3D reconnection and turbulence

1. 3D null collapse.
2. Separator reconnection.
3. QSL/slipping reconnection demo.
4. Flux-rope coalescence.
5. Turbulent current-sheet reconnection.
6. Guide-field turbulence.
7. Synthetic spacecraft crossing through 3D reconnection region.

Do not call 3D cases validated until field-line/topology diagnostics exist.

---

## 12. External comparison plan

### 12.1 Comparison principles

- Compare equations before comparing plots.
- Match nondimensional parameters, boundary conditions, reconstruction order, divergence strategy, and output cadence as closely as possible.
- Record unavoidable differences.
- Use multiple resolutions.
- Compare convergence trends, not just single images.
- Keep scripts reproducible and public.
- Do not claim superiority unless the benchmark harness supports it.

### 12.2 Comparison categories

#### Category A: highly cited/state-of-the-art MHD codes

- Athena++
- PLUTO
- MPI-AMRVAC
- FLASH
- OpenMHD
- Dedalus

#### Category B: extended-fluid / kinetic reference codes

- Gkeyll
- VPIC
- WarpX
- Smilei
- EPOCH

#### Category C: JAX-native and differentiable codes

- astronomix
- JAX-Fluids
- JAX-CFD as historical/unmaintained reference

### 12.3 Numerical comparison metrics

For each comparable run:

- L1/L2/Linf error against analytic or high-resolution reference.
- Conservation error.
- `divB` norm.
- Shock/current-sheet width.
- Reconnection rate.
- Growth rate.
- Energy partition.
- Spectra.
- Plasmoid statistics where applicable.

### 12.4 Performance comparison metrics

Collect:

- install/setup time;
- compile/JIT time;
- first-run wall time;
- steady-state wall time;
- steps per second;
- cell updates per second;
- memory peak;
- maximum resolution that fits on hardware;
- CPU/GPU utilization notes;
- strong scaling;
- weak scaling;
- output I/O time;
- gradient runtime and memory for differentiable benchmarks.

Timing rules:

- always call `block_until_ready()` before stopping timers;
- report hardware, driver, CUDA/JAX versions, dtype, backend, and device count;
- separate compile time from execution time;
- repeat enough times for stable medians;
- record exact command and config.

### 12.5 Ease-of-use comparison metrics

For each code and benchmark:

- installation steps;
- whether GPU setup was required;
- config length;
- lines of Python/user code;
- time to first figure;
- documentation clarity;
- whether equations and assumptions are documented;
- whether output can be loaded into Python easily;
- whether the benchmark is reproducible from one command.

### 12.6 Code-specific comparison targets

#### Athena++

Use for:

- Brio-Wu;
- rotor;
- blast;
- Orszag-Tang;
- Kelvin-Helmholtz;
- Rayleigh-Taylor;
- Harris/Sweet-Parker where configs are available;
- AMR comparison later.

MHX goal: match validation trends and provide differentiability/ease-of-use advantages, not necessarily raw HPC performance.

#### PLUTO

Use for:

- shock-capturing MHD;
- RMHD/non-ideal comparisons later;
- anisotropic conduction/resistivity comparisons where feasible;
- AMR comparisons through published data or local runs.

#### MPI-AMRVAC

Use for:

- solar/astrophysical MHD examples;
- AMR-driven current-sheet problems;
- radiative/cooling comparisons if applicable;
- 2D/3D shock-dominated MHD.

#### FLASH

Use for:

- extended-MHD term inventory comparison;
- Braginskii/Hall/Nernst/Biermann conceptual comparison;
- radiation-MHD and anisotropic conduction comparison;
- HED/radiative reconnection setup inspiration.

Access may require request/registration for full code. If full public reproducibility is not possible, use published benchmarks and documented capabilities.

#### OpenMHD

Use for:

- resistive MHD reconnection;
- HLLD/MUSCL/GLM finite-volume comparison;
- lightweight public code-to-code comparison.

This is likely the best direct finite-volume reconnection comparison in early phases.

#### Dedalus

Use for:

- spectral PDE comparisons;
- reduced/incompressible MHD;
- eigenvalue or linear stability problems;
- adjoint methodology comparison.

MHX should learn from Dedalus adjoints but remain JAX-native.

#### Gkeyll

Use for:

- GEM reconnection;
- ten-moment/two-fluid comparisons;
- generalized-MHD closure interpretation;
- Hall/inertia/pressure-tensor effect benchmarks.

MHX should not claim kinetic fidelity, but should show where resistive/Hall MHD agrees or fails.

#### VPIC / WarpX / Smilei / EPOCH

Use for:

- collisionless reconnection reference behavior;
- PIC-vs-MHD qualitative comparisons;
- synthetic diagnostics and scale-separation demonstrations;
- not for one-to-one MHD runtime comparisons.

#### astronomix

Use for:

- JAX-native MHD finite-difference/finite-volume comparison;
- Orszag-Tang;
- MHD blast;
- turbulence;
- radiative cooling if simple module overlaps;
- ease-of-use, performance, differentiability comparisons.

MHX should differentiate itself by being reconnection-centered, benchmark/gallery-heavy, physics-gated, topology-aware, and inverse-problem-ready.

#### JAX-Fluids

Use for:

- JAX-native performance methodology;
- hydrodynamic KH/RT/shock examples;
- differentiability and HPC scaling inspiration;
- not direct MHD comparison.

---

## 13. Differentiable research workflows

### 13.1 Required demos

1. **Inverse tearing perturbation**
   - Optimize initial perturbation to hit a target island width or reconnection rate.
   - Validate gradient against finite differences on small grids.

2. **Resistivity inference**
   - Infer `eta`, `eta_H`, or anomalous-resistivity parameters from target reconnection curves.
   - Include identifiability warnings.

3. **Guide-field sensitivity**
   - Differentiate reconnection rate and energy partition with respect to guide field.

4. **Cooling-law calibration**
   - Fit cooling-law parameters to target temperature/collapse/emission proxy.

5. **Sensor placement / synthetic spacecraft**
   - Optimize lineout/sampling locations to infer reconnection metrics.

6. **Minimal seed for instability**
   - Optimize perturbations for KH/RT/tearing onset under norm constraints.

7. **Turbulent ensemble objective**
   - Optimize statistical metrics over ensembles, not individual plasmoid event counts.

### 13.2 Gradient validation

Every differentiable demo must include:

- finite-difference check at tiny resolution;
- dtype sensitivity check;
- step-size sensitivity check;
- comparison of direct reverse-mode vs checkpointed mode where applicable;
- warning if objective is nonsmooth.

### 13.3 Smooth objective design

Prefer smooth objectives:

- integrated magnetic energy in regions;
- soft current-sheet indicators;
- smoothed reconnection flux;
- spectral slopes;
- time-integrated `J·E`;
- differentiable lineout loss;
- smoothed island-width proxy.

Avoid raw gradients through:

- plasmoid count;
- hard X/O point identity changes;
- hard thresholded current sheets;
- adaptive event times;
- discontinuous floors/limiters unless using a documented surrogate.

---

## 14. Documentation requirements

The documentation must be complete enough that a reviewer can reproduce the equations and methods without reading source code.

### 14.1 Required docs structure

```text
docs/
  index.md
  getting_started.md
  installation.md
  equations/
    ideal_mhd.md
    resistive_mhd.md
    reduced_mhd.md
    hall_mhd.md
    generalized_ohm.md
    radiative_terms.md
    nondimensionalization.md
  numerics/
    overview.md
    spectral.md
    finite_volume.md
    reconstruction.md
    riemann_solvers.md
    constrained_transport.md
    divergence_cleaning.md
    time_integration.md
    implicit_solvers.md
    differentiability.md
  physics_gates/
    overview.md
    resistivity.md
    hyper_resistivity.md
    hall.md
    radiative_cooling.md
    conduction.md
  diagnostics/
    reconnection.md
    turbulence.md
    topology_3d.md
    energy_budgets.md
  benchmarks/
    index.md
    brio_wu.md
    orszag_tang.md
    tearing.md
    sweet_parker.md
    plasmoid.md
    gem.md
    radiative_reconnection.md
  comparisons/
    overview.md
    athena.md
    pluto.md
    openmhd.md
    astronomix.md
    jax_fluids.md
    gkeyll.md
  differentiable_workflows/
    inverse_tearing.md
    resistivity_inference.md
    sensor_placement.md
    cooling_calibration.md
  api/
  contributing.md
  roadmap.md
```

### 14.2 Each physics-doc page must include

- governing equations;
- nondimensional parameters;
- assumptions;
- valid regimes;
- invalid regimes;
- numerical implementation;
- gate checks;
- diagnostics;
- benchmark cases;
- references;
- example config;
- example Python snippet.

### 14.3 Each benchmark-doc page must include

- purpose;
- equations solved;
- IC/BC;
- parameters;
- expected result;
- reference source;
- command line;
- Python API;
- output files;
- plots;
- comparison notes;
- tolerance definition;
- known failure modes.

---

## 15. CLI and Python API

### 15.1 CLI target

```bash
mhx init tearing --out configs/tearing.toml
mhx run configs/tearing.toml
mhx plot outputs/runs/<run_id>
mhx benchmark run tearing --resolution 64,128,256
mhx benchmark report outputs/benchmarks/<id>
mhx compare run openmhd sweet_parker
mhx diff inverse-tearing configs/inverse_tearing.toml
mhx docs doctor
mhx physics check configs/run.toml
```

### 15.2 Python API target

```python
import mhx

cfg = mhx.load_config("configs/tearing.toml")
grid = mhx.make_grid(cfg.grid)
state0, params = mhx.problems.tearing.initial_state(cfg, grid)
result = mhx.evolve(state0, params, grid, cfg)
fig = mhx.plotting.reconnection_summary(result)
```

Differentiable example:

```python
def loss(params):
    result = mhx.evolve(state0, params, grid, cfg)
    return mhx.diff.objectives.target_island_width(result, target=0.1)

value, grad = mhx.value_and_grad(loss)(params)
```

### 15.3 Output schema

Every run writes:

```text
outputs/runs/<run_id>/
  manifest.toml
  config.toml
  metrics.json
  history.zarr or history.h5
  snapshots/
  figures/
  logs/
  environment.txt
```

Manifest must include:

- git commit;
- date/time;
- hardware;
- JAX/JAXLIB versions;
- backend/device;
- dtype;
- solver choices;
- physics gates;
- timing;
- memory when available.

---

## 16. Plotting requirements

Publication-ready plots must be generated from diagnostics, not manually assembled.

Required figure recipes:

- energy budget summary;
- `divB` history;
- current density map;
- flux function contours;
- reconnection rate vs time;
- Sweet-Parker scaling;
- tearing growth fit;
- island width vs time;
- plasmoid count with convergence warning;
- magnetic/kinetic spectra;
- Ohm’s-law term budget;
- Hall quadrupole plot;
- radiative cooling budget;
- performance roofline-style summary;
- gradient validation plot.

Plot API must allow:

- saving PNG and PDF;
- consistent labels and units;
- optional LaTeX-style labels;
- fixed style defaults;
- no hard-coded paths.

---

## 17. Testing and CI

### 17.1 Test categories

- Unit tests for operators.
- Unit tests for primitive/conservative conversion.
- Unit tests for boundary conditions.
- Unit tests for Riemann solvers.
- Unit tests for spectral derivatives.
- Unit tests for CT/GLM divergence behavior.
- Unit tests for physics gates.
- Regression tests for benchmark metrics.
- Gradient tests against finite differences.
- CLI smoke tests.
- Documentation build test.

### 17.2 CI tiers

Fast CI:

- tiny grids;
- CPU only;
- no long simulations;
- import and CLI smoke tests;
- a few gradient checks;
- docs build.

Slow CI / scheduled:

- medium grids;
- benchmark regression;
- comparison-data checks;
- performance trend snapshots.

GPU CI if available:

- JAX GPU install;
- one spectral run;
- one finite-volume run;
- one gradient run;
- one profiler smoke test.

---

## 18. Performance engineering plan

### 18.1 Do first

- Implement pure JAX kernels.
- Use `jax.jit`/`eqx.filter_jit` on whole evolution chunks.
- Avoid recompilation by keeping shapes/static configs stable.
- Donate buffers for long runs.
- Use `lax.scan` for timesteps.
- Use `vmap` for ensembles/scans.
- Benchmark X32 and X64 separately.
- Measure compile and execution separately.

### 18.2 Profile before optimizing

Use:

- `jax.profiler.trace`;
- XProf/TensorBoard;
- device memory profile;
- Python wall timers with `block_until_ready()`;
- optional Nsight for GPU deep dives.

### 18.3 Candidate optimizations after profiling

- Fuse stencil operations.
- Reduce primitive/conservative conversion count.
- Cache wave-speed computations.
- Use buffer donation.
- Use domain sharding for large 3D runs.
- Use Pallas only for proven bottlenecks.
- Implement custom kernels for reconstruction/flux divergence only after baseline is correct.

---

## 19. Phased implementation roadmap

### Phase 0: migration and skeleton

Deliverables:

- legacy migration;
- new package skeleton;
- config system;
- logging/run manifest;
- CI basics;
- docs skeleton;
- `plan.md` log discipline.

Acceptance:

- package imports;
- CLI prints version/help;
- docs build;
- old code preserved.

### Phase 1: core operators and spectral reduced MHD

Deliverables:

- Cartesian grid;
- FFT derivative operators;
- spectral Poisson/inverse Laplacian;
- reduced-MHD equations;
- RK4/SSPRK3;
- tearing initial condition;
- energy diagnostics;
- gradient check through short solve.

Acceptance:

- linear tearing benchmark runs;
- finite-difference gradient check passes on tiny grid;
- output and plot generated.

### Phase 2: finite-volume ideal/resistive MHD baseline

Deliverables:

- conservative/primitive conversion;
- Rusanov and HLL solvers;
- reconstruction options;
- SSPRK2/3;
- GLM divergence cleaning;
- Brio-Wu, rotor, Orszag-Tang;
- resistive induction terms.

Acceptance:

- hydro/MHD benchmarks match expected qualitative and quantitative tolerances;
- `divB` diagnostic works;
- CI smoke tests pass.

### Phase 3: benchmark registry and publication plots

Deliverables:

- benchmark registry schema;
- automated run/plot/report;
- tearing, Sweet-Parker, Orszag-Tang, KH, RT;
- publication plot recipes;
- docs pages.

Acceptance:

- `mhx benchmark run tearing` works;
- `mhx benchmark report` produces markdown/html and figures;
- at least five benchmark docs pages are complete.

### Phase 4: differentiable workflows

Deliverables:

- inverse tearing;
- resistivity inference;
- guide-field sensitivity;
- finite-difference gradient validator;
- checkpointed scan mode;
- Optax/Optimistix workflows.

Acceptance:

- three differentiable examples run end-to-end;
- gradient plots included;
- docs explain limitations.

### Phase 5: constrained transport and higher-order MHD

Deliverables:

- CT implementation;
- HLLD solver target;
- WENO3/WENO5;
- positivity-preserving improvements;
- comparison with Athena/OpenMHD/astronomix for core tests.

Acceptance:

- Orszag-Tang and blast compare favorably with references;
- divB control documented;
- performance baseline published.

### Phase 6: Hall/generalized-MHD modules

Deliverables:

- Hall induction term;
- Hall gate;
- electron inertia prototype;
- generalized Ohm diagnostics;
- GEM-like benchmark;
- Hall quadrupole plot.

Acceptance:

- Hall quadrupole appears in GEM-like run;
- Ohm-law term budget plots generated;
- comparison notes with Gkeyll/literature.

### Phase 7: radiative and thermal modules

Deliverables:

- optically thin cooling;
- bremsstrahlung cooling;
- anisotropic thermal conduction;
- implicit or subcycled cooling/conduction;
- radiative reconnection benchmark.

Acceptance:

- cooling/conduction tests pass;
- energy budget includes radiative losses;
- radiative reconnection example documented.

### Phase 8: 3D topology and turbulence

Deliverables:

- 3D finite-volume support;
- field-line tracing;
- `E_parallel` line integral;
- topology diagnostics prototype;
- turbulence spectra;
- 3D current-sheet/flux-rope demos.

Acceptance:

- at least one 3D reconnection demo with topology diagnostics;
- spectra diagnostics tested;
- docs warn about resolution and topology limitations.

### Phase 9: v1.0 release paper package

Deliverables:

- reproducible benchmark gallery;
- external comparison report;
- performance report;
- differentiability report;
- paper-ready figures;
- complete docs;
- release archive.

Acceptance:

- a reviewer can reproduce all main figures from CLI commands;
- external-code comparisons are documented;
- API is stable enough for users.

---

## 20. Immediate tasks for the coding agent

Start with these steps in order.

1. Update `plan.md` log entry: “Started rebuild.”
2. Move current code to `legacy/old_mhx/`.
3. Create root `README.md` for new MHX.
4. Create `pyproject.toml` with minimal dependencies.
5. Create `src/mhx/__init__.py` and `_version.py`.
6. Add CLI skeleton: `mhx --help`, `mhx version`.
7. Add config dataclasses and one example config.
8. Add grid module and simple periodic Cartesian grid.
9. Add spectral derivative operator and tests.
10. Add docs skeleton and `docs/index.md`.
11. Add CI smoke tests.
12. Update `plan.md` with files changed, tests run, failures, and next steps.

Do not start with Hall, radiative terms, or 3D. The first scientific target is a small spectral tearing benchmark plus gradient check.

---

## 21. Coding standards

- Python 3.10+; use `tomllib` on Python 3.11+ and `tomli` fallback on Python 3.10.
- Type annotations for public APIs.
- Shape annotations with jaxtyping for core kernels.
- Use `ruff` formatting/linting.
- Use `pytest`.
- No global mutable solver state.
- No hidden unit conventions.
- No hidden floors/limiters without diagnostics.
- Every public physics parameter must be documented.
- Every benchmark must have a reference.
- Every differentiable example must have a gradient check.

---

## 22. Final online pass addendum: code comparisons, discretization, docs, CI, figures, and manuscript-grade deliverables

This section resolves the remaining planning requirements from the final online pass. It supersedes any weaker or contradictory earlier wording. The coding agent must treat this section as binding unless a later log entry explicitly records a tested reason to change it.

### 22.1 Hard product requirements from the final pass

The new MHX must support all of the following from the first public release candidate:

1. **One-command CLI execution from TOML**: `mhx run examples/orszag_tang.toml` must run a complete simulation, write a manifest, write outputs, and produce at least one diagnostic plot when requested.
2. **Single Python driver script**: every benchmark must also have a short Python script demonstrating import, config loading, run, diagnostics, and plotting.
3. **TOML, not YAML, for user-facing input files**: Python 3.11+ can use `tomllib`; Python 3.10 must use `tomli` fallback. If writing TOML is needed, use `tomli-w` or `tomlkit`.
4. **Python 3.10 and newer**: all code, tests, docs, and wheels must support Python 3.10+.
5. **No exact dependency pinning in `pyproject.toml`**: use dependency names and broad extras, not exact version pins. Do not list `jaxlib` as a dependency. Users should install accelerator-specific JAX wheels using official JAX instructions; project dependencies should include `jax`.
6. **ReadTheDocs documentation**: root `.readthedocs.yaml` and a `docs/` folder must be committed.
7. **GitHub workflows**: CI, docs build, coverage, benchmark smoke tests, and PyPI publishing workflows must exist.
8. **Coverage above 95%**: CI must fail below 95% total coverage. Coverage must include unit, numerical, regression, physics-gate, CLI, and gradient tests. The target is not superficial coverage; test quality is part of acceptance.
9. **README at highly starred repo standard**: badges, one-line mission, installation, first CLI run, first Python run, plotting/movie demo, docs link, benchmark table, contribution guide, citation/reference section.
10. **Documentation as manuscript substrate**: equations, derivations, nondimensionalization, discretization, inputs, outputs, diagnostics, comparisons, performance, and figures must be sufficiently complete that a manuscript can be written from the docs.
11. **Plotting and movie API**: users must be able to create publication plots and movies with simple calls, not ad hoc scripts.
12. **External code comparison harness**: do not merely cite other codes; provide exact clone/build/run instructions and parsers where legal/public. When full reproducibility is blocked by registration or access restrictions, document that honestly and use published data or capability comparisons.

### 22.2 Exact packaging policy

Use a modern `pyproject.toml` with broad dependencies and extras. Do not pin exact versions unless a temporary known-bad compatibility issue is logged in `plan.md` and removed as soon as possible.

Recommended dependency structure:

```toml
[project]
name = "mhx"
description = "Differentiable JAX framework for MHD, magnetic reconnection, turbulence, and plasma instabilities"
requires-python = ">=3.10"
dependencies = [
  "jax",
  "numpy",
  "scipy",
  "equinox",
  "lineax",
  "diffrax",
  "optimistix",
  "optax",
  "jaxtyping",
  "matplotlib",
  "xarray",
  "h5py",
  "tomli; python_version < '3.11'",
  "rich",
  "typer",
]

[project.optional-dependencies]
docs = [
  "sphinx",
  "myst-parser",
  "sphinx-copybutton",
  "sphinxcontrib-bibtex",
  "pydata-sphinx-theme",
  "sphinx-design",
  "nbsphinx",
  "jupyter",
]
plotting = [
  "imageio",
  "imageio-ffmpeg",
  "pandas",
]
test = [
  "pytest",
  "pytest-cov",
  "pytest-benchmark",
  "hypothesis",
]
dev = [
  "ruff",
  "mypy",
  "pre-commit",
  "build",
  "twine",
]
checkpoint = [
  "orbax-checkpoint",
]
```

Notes:

- Do **not** list `jaxlib` in project dependencies. JAX installation is platform-specific; users can install CPU JAX via `pip install -U jax` or GPU JAX via official extras such as `jax[cuda13]` when appropriate.
- Do not pin exact library versions in `pyproject.toml`. CI may test against specific Python versions, but dependencies should remain broad unless a compatibility log entry justifies a temporary constraint.
- Build metadata should use `hatchling` or `setuptools`. Prefer `hatchling` for simple packaging unless the existing repository already has a strong reason to use another backend.
- CLI entrypoint:

```toml
[project.scripts]
mhx = "mhx.cli.main:app"
```

### 22.3 TOML input design

Every user-facing config should be TOML. Names must be readable and domain-specific. Avoid cryptic single-letter options except for standard physical symbols where documented.

Example complete finite-volume MHD input:

```toml
[run]
name = "orszag_tang_2d"
output_dir = "runs/orszag_tang_2d"
precision = "float64"
seed = 0
write_interval = 0.02
checkpoint_interval = 0.2
save_history = true

[mesh]
dimension = 2
nx = 256
ny = 256
nz = 1
xmin = 0.0
xmax = 1.0
ymin = 0.0
ymax = 1.0
zmin = 0.0
zmax = 1.0
boundary_x = "periodic"
boundary_y = "periodic"
boundary_z = "periodic"
magnetic_layout = "constrained_transport"  # constrained_transport or cell_centered

[physics]
model = "ideal_mhd"  # ideal_mhd, resistive_mhd, hall_mhd, reduced_mhd
normalization = "alfvenic"
equation_of_state = "ideal_gas"
gamma = 1.6666666666666667
resistivity = 0.0
viscosity = 0.0
hyper_resistivity = 0.0
hall_enabled = false
ion_inertial_length = 0.0
electron_inertial_length = 0.0
radiative_cooling_enabled = false
cooling_law = "none"
thermal_conduction_enabled = false

[numerics]
solver = "finite_volume"
reconstruction = "weno_z"          # constant, minmod, mc, ppm, weno3, weno5, weno_z
riemann_solver = "hlld"            # rusanov, hll, hlld
divergence_control = "constrained_transport"  # glm or constrained_transport
time_integrator = "ssprk3"         # euler, ssprk2, ssprk3, rk4
cfl = 0.3
fixed_dt = 0.0                      # 0 means use CFL controller
positivity_limiter = true
smooth_ad_mode = false

[diagnostics]
energy_budget = true
divergence_b = true
current_density = true
reconnection_rate = false
spectra = false
write_lineouts = true

[plots]
enabled = true
format = ["png", "pdf"]
variables = ["density", "pressure", "current_density_z"]
make_movie = true
movie_variable = "current_density_z"
movie_fps = 20

[benchmark]
case = "orszag_tang"
reference = "athena_orszag_tang"
resolution_suite = [64, 128, 256]
fast_ci = true
requires_x64 = true
```

Example spectral reduced-MHD tearing input:

```toml
[run]
name = "linear_tearing_spectral"
output_dir = "runs/linear_tearing_spectral"
precision = "float64"
write_interval = 1.0

[mesh]
dimension = 2
nx = 256
ny = 256
xmin = -3.141592653589793
xmax = 3.141592653589793
ymin = -3.141592653589793
ymax = 3.141592653589793
boundary_x = "periodic"
boundary_y = "periodic"

[physics]
model = "reduced_mhd"
normalization = "alfvenic"
resistivity = 0.0001
hyper_resistivity = 0.0
guide_field = 1.0

[numerics]
solver = "pseudo_spectral"
dealiasing = "two_thirds"
time_integrator = "rk4"
fixed_dt = 0.001
scan_chunk_size = 100
checkpoint_mode = "remat_chunks"

[problem]
name = "tearing_mode"
equilibrium = "harris_periodic"
wave_number = 1.0
perturbation_amplitude = 1.0e-6

[diagnostics]
tearing_growth_rate = true
island_width = true
energy_budget = true

[plots]
enabled = true
make_movie = true
movie_variable = "current_density_z"
```

### 22.4 CLI and Python driver requirements

CLI commands must be short and composable:

```bash
mhx --help
mhx version
mhx init orszag_tang --out examples/orszag_tang.toml
mhx run examples/orszag_tang.toml
mhx plot runs/orszag_tang_2d --variable density --time -1 --format png,pdf
mhx movie runs/orszag_tang_2d --variable current_density_z --fps 20 --out runs/orszag_tang_2d/figures/current.mp4
mhx benchmark run tearing_linear --resolution 64,128,256
mhx benchmark report runs/benchmarks/tearing_linear
mhx compare run openmhd sweet_parker --mhx-config examples/sweet_parker.toml --external-dir external_runs/openmhd
mhx physics check examples/hall_gem.toml
mhx docs figures --all
```

Every example must also provide a Python script:

```python
# examples/python/run_orszag_tang.py
import mhx

cfg = mhx.load_config("examples/orszag_tang.toml")
result = mhx.run(cfg)

mhx.plot.slice(result, variable="density", time=-1, save="runs/orszag_tang_2d/figures/density.png")
mhx.plot.slice(result, variable="current_density_z", time=-1, save="runs/orszag_tang_2d/figures/current_density.png")
mhx.movie.scalar_field(result, variable="current_density_z", save="runs/orszag_tang_2d/figures/current_density.mp4")
```

The `mhx.run(cfg)` function should return a lightweight result object with lazy access to output arrays. For long simulations, do not keep the entire time history in host memory unless requested.

### 22.5 README standard

Root `README.md` must be user-facing and polished. Minimum structure:

````markdown
# MHX

[badges: CI, docs, coverage, PyPI, Python versions, license]

**MHX is a differentiable JAX framework for magnetohydrodynamics, magnetic reconnection, turbulence, and plasma instabilities.**

## What MHX does

- finite-volume compressible MHD;
- pseudo-spectral reduced MHD;
- resistive, hyper-resistive, Hall, cooling, and conduction modules;
- benchmarked reconnection examples;
- differentiable inverse-design workflows;
- publication-ready plots and movies.

## Installation

```bash
pip install mhx
```

Development install:

```bash
git clone https://github.com/uwplasma/MHX.git
cd MHX
pip install -e ".[dev,test,docs,plotting]"
```

GPU JAX note:

```bash
pip install -U "jax[cuda13]"
pip install -e ".[dev,test,docs,plotting]"
```

## First run: CLI

```bash
mhx init orszag_tang --out examples/orszag_tang.toml
mhx run examples/orszag_tang.toml
mhx plot runs/orszag_tang_2d --variable current_density_z --time -1
mhx movie runs/orszag_tang_2d --variable current_density_z
```

## First run: Python

```python
import mhx
cfg = mhx.load_config("examples/orszag_tang.toml")
result = mhx.run(cfg)
mhx.plot.slice(result, "current_density_z", time=-1)
```

## Benchmarks

Table with benchmark name, equations, reference, status, command, docs page.

## Documentation

Link to ReadTheDocs.

## Citation

BibTeX once paper/preprint exists.
````

Badges should include:

- GitHub Actions CI status.
- ReadTheDocs status.
- Codecov coverage badge or equivalent.
- PyPI version.
- Supported Python versions.
- License.

### 22.6 GitHub workflows

Required workflows:

```text
.github/workflows/
  ci.yml
  docs.yml
  coverage.yml        # may be folded into ci.yml if badge still updates
  benchmark-smoke.yml
  publish.yml
```

#### 22.6.1 `ci.yml`

Requirements:

- Test Python 3.10, 3.11, 3.12, and newest stable 3.x available in GitHub Actions.
- Install with `pip install -e ".[test,plotting]"`.
- Run `ruff`, import checks, CLI help, unit tests, fast numerical tests, and fast gradient checks.
- Enforce `pytest --cov=mhx --cov-report=xml --cov-report=term-missing --cov-fail-under=95`.
- Upload coverage XML to Codecov or another coverage service.

Example skeleton:

```yaml
name: CI
on:
  pull_request:
  push:
    branches: [main]

jobs:
  tests:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12", "3.13"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - run: python -m pip install --upgrade pip
      - run: pip install -e ".[test,plotting]"
      - run: mhx --help
      - run: ruff check .
      - run: pytest tests --cov=mhx --cov-report=xml --cov-report=term-missing --cov-fail-under=95
      - uses: codecov/codecov-action@v4
        with:
          files: ./coverage.xml
```

#### 22.6.2 `docs.yml` and `.readthedocs.yaml`

Use Sphinx + MyST Markdown. ReadTheDocs requires a top-level `.readthedocs.yaml`.

Recommended root file:

```yaml
version: 2

build:
  os: ubuntu-24.04
  tools:
    python: "3.12"

sphinx:
  configuration: docs/conf.py

python:
  install:
    - method: pip
      path: .
      extra_requirements:
        - docs
        - plotting

formats:
  - pdf
  - epub
```

`docs.yml` should run the same Sphinx build locally in CI:

```bash
sphinx-build -b html docs docs/_build/html
sphinx-build -b linkcheck docs docs/_build/linkcheck
```

#### 22.6.3 `publish.yml`

Use PyPI trusted publishing with GitHub OIDC; do not store long-lived API tokens when trusted publishing is available.

```yaml
name: Publish
on:
  push:
    tags:
      - "v*"

jobs:
  publish:
    runs-on: ubuntu-latest
    permissions:
      id-token: write
      contents: read
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: python -m pip install --upgrade build
      - run: python -m build
      - uses: pypa/gh-action-pypi-publish@release/v1
```

#### 22.6.4 `benchmark-smoke.yml`

Use small grids only. This workflow protects physics gates without overloading CI.

Required fast benchmark tests:

- linear advection convergence at tiny grid;
- diffusion decay against analytic solution;
- Alfvén wave one-period error;
- Brio-Wu smoke with finite density/pressure and expected discontinuity structure;
- Orszag-Tang short run with finite conserved variables and bounded `divB`;
- reduced-MHD tearing growth-rate smoke at low resolution;
- finite-difference gradient check for a tiny differentiable objective;
- physics gates reject intentionally invalid Hall/cooling/hyper-resistive configs.

### 22.7 ReadTheDocs and manuscript-grade docs plan

Use Sphinx with MyST Markdown so docs can contain Markdown, math, references, API docs, and manuscript-style chapters.

Recommended docs tree:

```text
docs/
  conf.py
  index.md
  bibliography.bib
  _static/
  _templates/
  getting_started/
    installation.md
    first_cli_run.md
    first_python_run.md
    plotting_and_movies.md
    troubleshooting.md
  user_guide/
    configuration_toml.md
    outputs_and_manifests.md
    physics_gates.md
    precision_and_devices.md
  equations/
    nondimensionalization.md
    ideal_mhd.md
    resistive_mhd.md
    reduced_mhd.md
    hall_mhd.md
    generalized_ohm.md
    hyper_resistivity.md
    radiative_terms.md
    thermal_conduction.md
  numerics/
    finite_volume_overview.md
    conservative_variables.md
    reconstruction.md
    riemann_solvers.md
    constrained_transport.md
    glm_cleaning.md
    spectral_methods.md
    time_integration.md
    implicit_and_imex.md
    differentiability_and_adjoints.md
  diagnostics/
    energy_budgets.md
    reconnection.md
    plasmoids_and_islands.md
    turbulence_spectra.md
    topology_3d.md
    synthetic_spacecraft.md
    synthetic_emission.md
  benchmarks/
    index.md
    brio_wu.md
    rotor.md
    orszag_tang.md
    kelvin_helmholtz.md
    rayleigh_taylor.md
    linear_tearing.md
    rutherford.md
    sweet_parker.md
    plasmoid_instability.md
    ideal_tearing.md
    hyper_resistive_plasmoids.md
    gem_hall.md
    radiative_reconnection.md
    turbulent_reconnection.md
  comparisons/
    overview.md
    athena.md
    pluto.md
    mpi_amrvac.md
    flash.md
    openmhd.md
    gkeyll.md
    astronomix.md
    jax_fluids.md
    dedalus.md
    pic_reference_codes.md
  differentiable_workflows/
    overview.md
    inverse_tearing.md
    resistivity_inference.md
    cooling_law_calibration.md
    sensor_placement.md
    topology_aware_optimization.md
  api/
    index.rst
  manuscript/
    abstract.md
    methods.md
    verification.md
    validation.md
    differentiability.md
    performance.md
    limitations.md
    reproducibility.md
  contributing.md
  roadmap.md
```

Every benchmark page must include:

- equations solved;
- nondimensional parameters;
- initial and boundary conditions;
- numerical schemes used;
- exact TOML config;
- exact CLI command;
- exact Python script;
- output files produced;
- diagnostic definitions;
- expected plots and tolerances;
- comparison code instructions;
- references and links to code/discussion files;
- known failure modes and resolution caveats.

Every source-code discussion should link to actual files once implemented, for example:

```markdown
Implementation files:

- [`src/mhx/numerics/finite_volume/reconstruction.py`](../../src/mhx/numerics/finite_volume/reconstruction.py)
- [`src/mhx/numerics/riemann/hlld.py`](../../src/mhx/numerics/riemann/hlld.py)
- [`src/mhx/numerics/constrained_transport/uct.py`](../../src/mhx/numerics/constrained_transport/uct.py)
- [`src/mhx/physics/hall.py`](../../src/mhx/physics/hall.py)
- [`src/mhx/diagnostics/reconnection.py`](../../src/mhx/diagnostics/reconnection.py)
```

### 22.8 Numerical discretization decision tree

MHX must expose several methods, but defaults should be conservative, proven, and easy to validate.

#### 22.8.1 Default finite-volume MHD path

Default production solver:

```text
Conservative finite volume
+ primitive reconstruction
+ HLLD Riemann solver where stable
+ HLL/Rusanov fallback
+ SSPRK3 time integration
+ constrained transport for divB
+ positivity/floor diagnostics
+ optional GLM cell-centered mode
```

Implementation algorithm:

1. Store conservative cell-centered state `U = [rho, rho*vx, rho*vy, rho*vz, E, Bx, By, Bz]` for GLM mode.
2. For CT mode, store magnetic field face-centered in addition to cell-centered views/interpolants.
3. Convert `U -> W` primitives with density/pressure positivity handling.
4. Reconstruct primitive variables to faces using selected reconstruction:
   - `constant` for first-order debugging;
   - `minmod`/`mc` for robust second order;
   - `ppm` for classical high-resolution tests;
   - `weno3`, `weno5`, `weno_z` for high-order research runs.
5. Solve face Riemann problems:
   - `rusanov` first implementation and fallback;
   - `hll` robust intermediate;
   - `hlld` target default for ideal/resistive MHD.
6. Compute finite-volume flux divergence.
7. Update gas variables with SSPRK2/3.
8. Update magnetic field:
   - CT mode: compute edge-centered electric fields using upwind constrained transport and apply discrete curl update;
   - GLM mode: evolve divergence-cleaning scalar and damp divergence waves.
9. Apply source/split operators:
   - Ohmic diffusion;
   - viscosity;
   - hyper-resistivity;
   - Hall term;
   - cooling;
   - conduction.
10. Run physics gates before, during, and after evolution.
11. Emit diagnostics and manifest.

#### 22.8.2 Why CT and GLM both exist

- **CT mode** is the research-grade default for magnetic reconnection and Orszag-Tang-like problems because it maintains a discrete divergence-free magnetic field.
- **GLM mode** is useful for rapid prototyping, cell-centered implementation simplicity, comparison with codes using GLM, and early finite-volume milestones.
- Documentation must never hide which divergence method was used. Every figure and benchmark report must label CT or GLM.

#### 22.8.3 Differentiability-friendly finite-volume mode

Shock-capturing methods are often nonsmooth. MHX should offer a `smooth_ad_mode` for differentiable demos:

- fixed timestep schedule, no adaptive branching in the differentiated function;
- smooth limiter approximations where possible;
- smooth positivity transform or documented `stop_gradient` around floors;
- smooth objectives instead of event counts;
- `jax.checkpoint`/`jax.remat` around timestep blocks;
- gradient validation against finite differences at tiny resolution;
- warnings when users request gradients through HLLD/limiters/floors/shocks.

This mode is not a replacement for research-grade shock capturing. It is an honest differentiable surrogate mode for inverse problems.

#### 22.8.4 Pseudo-spectral reduced-MHD path

Default spectral solver:

```text
Fourier pseudo-spectral derivatives
+ 2/3 dealiasing
+ explicit RK4 or SSPRK3
+ optional IMEX/Lineax diffusion solve
+ checkpointed lax.scan
+ custom transpose roadmap
```

Implementation details:

- Use `jax.numpy.fft` for periodic domains.
- Build derivative operators as reusable PyTrees: `Dx`, `Dy`, `Laplacian`, `InverseLaplacian`, `DealiasMask`.
- Keep spectral arrays complex-valued internally and provide real physical views.
- Validate derivative accuracy using manufactured periodic functions.
- Validate Poisson/inverse Laplacian consistency.
- Use 2/3 dealiasing by default for nonlinear products in research mode.
- Use fixed-step scans for benchmark and gradient reproducibility.
- Implement operator-level adjoint tests early so custom adjoints can be added later.

#### 22.8.5 Resistive, hyper-resistive, Hall, conduction, and cooling terms

- **Resistive diffusion**: start explicit central-difference or spectral diffusion. Add implicit/semi-implicit Lineax path for stiff high-resolution runs.
- **Hyper-resistivity**: implement as a fourth-order operator with an explicit timestep gate `dt <= C dx^4 / eta_H` unless using implicit/IMEX. Always report `S_H` and effective dissipation scale.
- **Hall term**: implement only after baseline CT/GLM MHD is stable. Gate on `d_i/dx`, whistler timestep, and Ohm-law diagnostics. Provide HLLD/Hall compatibility notes.
- **Thermal conduction**: start with isotropic conduction tests, then anisotropic field-aligned conduction. Use Lineax/implicit or super-time-stepping if explicit timestep is prohibitive.
- **Radiative cooling**: start with optically thin analytic cooling-law tests. Use exact/subcycled/implicit source updates when cooling time is shorter than the MHD timestep. Always track energy loss and temperature floors.

#### 22.8.6 Method options exposed to users

Expose these options in TOML and Python:

| Option family | Choices | Default | Notes |
|---|---|---|---|
| solver | `finite_volume`, `pseudo_spectral` | problem-dependent | finite-volume for compressible MHD; spectral for reduced MHD |
| reconstruction | `constant`, `minmod`, `mc`, `ppm`, `weno3`, `weno5`, `weno_z` | `mc` early, `weno_z` research | HLLD+WENO-Z target after baseline |
| Riemann solver | `rusanov`, `hll`, `hlld` | `hll` early, `hlld` research | Rusanov fallback |
| divergence control | `glm`, `constrained_transport` | `constrained_transport` once implemented | GLM initial milestone acceptable |
| time integrator | `euler`, `ssprk2`, `ssprk3`, `rk4` | `ssprk3` FV, `rk4` spectral | fixed-step required for core AD tests |
| diffusion solve | `explicit`, `imex`, `implicit_lineax` | `explicit` early | implicit for stiff terms |
| AD mode | `off`, `direct`, `checkpointed`, `smooth_ad`, `custom_ad` | `checkpointed` for examples | custom later |
| precision | `float32`, `float64` | `float64` validation | performance demos can use float32 |

### 22.9 Final online pass: external code runbook

The comparison harness should live under:

```text
comparison_cases/
  README.md
  athena/
  pluto/
  mpi_amrvac/
  flash/
  openmhd/
  gkeyll/
  astronomix/
  jax_fluids/
  dedalus/
  parsers/
  reports/
```

Do not vendor third-party codes. Provide scripts that clone/build/run into `external_runs/`, plus parsers for outputs. Store only small reference data if licenses allow.

For every external run, capture:

```text
code name
repository URL
git commit or release
compiler/interpreter
build flags
input file
command
hardware
MPI ranks / threads / devices
dtype or precision when known
mesh
solver/reconstruction/Riemann/divB method
wall time
memory if available
output files
parser used
comparison metrics
```

#### 22.9.1 Athena / Athena++

Use cases:

- Brio-Wu shock tube;
- Orszag-Tang vortex;
- rotor/blast;
- Kelvin-Helmholtz;
- Rayleigh-Taylor;
- current sheet / reconnection setups where available;
- CT/Godunov reference behavior.

Public locations:

- Athena++ current repository: https://github.com/PrincetonUniversity/athena
- Athena++ project site: https://www.athena-astro.app/
- Athena classic tutorial/test pages remain useful for documented commands and initial conditions.

Documented run examples to mirror:

```bash
# Classic Athena-style Orszag-Tang run pattern from public docs
make clean
./configure --with-problem=orszag-tang --with-order=3
make all
cd bin
./athena -i ../tst/2D-mhd/athinput.orszag-tang
```

```bash
# Classic Athena-style 3D Rayleigh-Taylor pattern from public docs
cd bin
./athena -i ../tst/3D-mhd/athinput.rt \
  domain1/Nx1=32 domain1/Nx2=32 domain1/Nx3=64 time/tlim=3.0
```

MHX comparison tasks:

- Create `comparison_cases/athena/README.md` with exact clone/build notes for Athena++ and classic Athena references.
- Create Athena-like TOML cases in MHX for Orszag-Tang, KH, and RT.
- Build parsers for VTK/HDF5 where possible.
- Plot MHX vs Athena lineouts and 2D maps at matched times.
- For Orszag-Tang, use the published periodic unit-square ICs in the docs and compare density, pressure, current density, energy, and `divB`.

#### 22.9.2 PLUTO

Use cases:

- shock-capturing ideal/resistive MHD;
- Orszag-Tang;
- rotor/blast;
- Kelvin-Helmholtz / Rayleigh-Taylor where examples exist;
- non-ideal terms, anisotropic conduction, cooling, and AMR references later.

Public locations:

- PLUTO site: https://plutocode.ph.unito.it/
- User guide / essentials documentation: https://plutocode.ph.unito.it/user_guide.html

Documented build/run pattern:

```bash
# Typical PLUTO workflow, adapted from PLUTO documentation
export PLUTO_DIR=/path/to/PLUTO
cd $PLUTO_DIR/Test_Problems/MHD/Orszag_Tang
python $PLUTO_DIR/setup.py
make
./pluto

# MPI run, when configured with MPI
mpiexec -n 4 ./pluto
```

MHX comparison tasks:

- Build `comparison_cases/pluto/README.md` with install notes, problem directory, setup, make, and run commands.
- Compare Brio-Wu, Orszag-Tang, rotor/blast, and any available resistive current-sheet configuration.
- Record PLUTO module choices: reconstruction, Riemann solver, divergence control, resistivity, conduction/cooling.
- Use PLUTO comparisons especially for finite-volume shock-capturing and radiative/non-ideal source-term behavior.

#### 22.9.3 MPI-AMRVAC

Use cases:

- AMR MHD references;
- Orszag-Tang and standard MHD tests;
- solar/astrophysical current-sheet tests;
- radiative/cooling cases if available.

Public locations:

- Documentation: https://amrvac.org/
- Source repository: https://github.com/amrvac/amrvac

Typical documented build/run pattern to encode:

```bash
git clone https://github.com/amrvac/amrvac.git mpi-amrvac
cd mpi-amrvac
# follow docs: install Fortran compiler, MPI, perl, git
# in a chosen tests/mhd problem directory:
setup.pl -d=2
make -j 4
mpirun -np 4 ./amrvac -i amrvac.par
```

MHX comparison tasks:

- Add AMRVAC problem mapping for Orszag-Tang, blast/rotor, and any reconnection/current-sheet test.
- Record AMR settings carefully; start with uniform-grid AMRVAC when possible for fair comparison.
- Compare AMR benefit qualitatively only after uniform-grid validation is done.

#### 22.9.4 FLASH

Use cases:

- extended-MHD feature inventory;
- Hall, Biermann, Nernst, Seebeck, anisotropic resistivity/conduction references;
- radiation/conduction/HED-inspired benchmarks;
- not necessarily an immediate public code-to-code run if access/registration blocks reproducibility.

Public locations:

- FLASH site and docs: https://flash.rochester.edu/site/flashcode/

MHX comparison tasks:

- Document which FLASH modules correspond conceptually to MHX physics gates.
- Use FLASH docs for Hall timestep warnings, extended Ohm-law terms, anisotropic conduction, radiation diffusion, and HED examples.
- If code access is available, create one Brio-Wu/Orszag-Tang or Hall/current-sheet comparison.
- If not, keep FLASH as a documented capability comparator and cite public documentation.

#### 22.9.5 OpenMHD

Use cases:

- early direct resistive-MHD reconnection comparison;
- finite-volume HLLD/MUSCL/GLM comparison;
- Sweet-Parker/current-sheet style examples;
- lightweight public code-to-code comparisons.

Public locations:

- Site: https://sci.nao.ac.jp/MEMBER/zenitani/openmhd-e.html
- Code distribution linked from the site.

Documented usage pattern:

```bash
# General OpenMHD pattern from docs
# 1. Edit Makefile for compiler/MPI/OpenMP/CUDA options.
# 2. Select/edit main, model, and boundary-condition files.
make
./openmhd
# output is written under data/
python plot.py
# or in IPython:
%run -i plot.py
```

MHX comparison tasks:

- Make OpenMHD the first finite-volume reconnection comparison if build succeeds.
- Match HLLD/MUSCL/GLM settings where possible.
- Compare reconnection rate, current sheet width, energy, and `divB` diagnostics.

#### 22.9.6 Gkeyll

Use cases:

- GEM reconnection;
- two-fluid/five-moment/ten-moment comparisons;
- Hall, electron inertia, and pressure-tensor interpretation;
- boundaries between MHD, Hall MHD, and kinetic/fluid-kinetic models.

Public locations:

- Docs: https://gkeyll.readthedocs.io/
- Ten-moment overview: https://heliophysics.princeton.edu/gkeyll/overview
- GEM example pages from the Gkeyll/Hakim documentation.

Typical build/run approach:

```bash
# Follow Gkeyll docs for dependencies such as C compiler, MPI, CUDA where needed,
# OpenBLAS, and SuperLU.
# Then run a Lua input file, e.g. a GEM reconnection example:
gkeyll gem.lua
```

MHX comparison tasks:

- Use Gkeyll GEM/two-fluid outputs to show where resistive MHD fails and Hall/generalized-MHD improves.
- Do not claim kinetic equivalence.
- Compare Hall quadrupole, reconnection rate, generalized Ohm-law terms, and gross current-sheet structure.

#### 22.9.7 astronomix

Use cases:

- closest JAX-native MHD comparison;
- Orszag-Tang;
- Kelvin-Helmholtz;
- MHD blast;
- turbulence;
- differentiability and ease-of-use comparison;
- performance comparison on the same CPU/GPU.

Public location:

- https://github.com/leo1200/astronomix

Expected workflow:

```bash
git clone https://github.com/leo1200/astronomix.git
cd astronomix
pip install -e .
# Run documented examples/notebooks such as Kelvin-Helmholtz and Orszag-Tang.
```

MHX comparison tasks:

- Provide side-by-side examples: MHX and astronomix Orszag-Tang, KH, blast, and a reconnection case if astronomix supports it.
- Compare runtime, memory, compile time, output clarity, config length, plot/movie steps, and differentiability examples.
- Differentiate MHX by reconnection diagnostics, physics gates, benchmark registry, topology diagnostics, and inverse-problem workflows.

#### 22.9.8 JAX-Fluids

Use cases:

- JAX-native CFD performance and ergonomics reference;
- hydrodynamic KH/RT/shock comparisons;
- differentiability workflow comparison;
- not a direct MHD comparison unless MHD support appears.

Public locations:

- https://github.com/tumaer/JAXFLUIDS
- https://jax-fluids.readthedocs.io/

Expected workflow:

```bash
git clone https://github.com/tumaer/JAXFLUIDS.git
cd JAXFLUIDS
pip install .
# Run documented examples such as Sod or KH/RT when available.
```

MHX comparison tasks:

- Compare hydrodynamic examples and differentiability/performance methodology.
- Use JAX-Fluids as a standard for polished docs and benchmark reporting.

#### 22.9.9 Dedalus

Use cases:

- spectral PDE reference;
- reduced/incompressible MHD reference;
- eigenvalue/linear stability comparisons;
- adjoint methodology inspiration.

Public locations:

- https://dedalus-project.org/
- https://dedalus-project.readthedocs.io/

Expected workflow:

```bash
pip install dedalus
# or follow Dedalus docs for MPI/FFTW/HDF5 installation on HPC.
python examples/ivp_*.py
```

MHX comparison tasks:

- Compare spectral derivative accuracy, Poisson solves, tearing/eigenvalue problems where possible.
- Use the automated spectral-adjoint literature as inspiration for custom adjoint paths, not as a dependency.

#### 22.9.10 PIC and kinetic reference codes

Use VPIC, WarpX, Smilei, EPOCH, and related PIC/hybrid codes only as physics-reference comparators. They are not direct MHD runtime competitors.

Comparison targets:

- collisionless reconnection rate regimes;
- Hall/electron inertia/pressure tensor signatures;
- electron-only reconnection phenomena;
- particle acceleration diagnostics;
- radiative/relativistic signatures where relevant.

MHX should provide reduced MHD/Hall/generalized-MHD comparisons and clear statements of where kinetic physics is outside its model class.

### 22.10 Final benchmark and literature target list

This is the benchmark suite the docs and paper should be organized around.

#### 22.10.1 Canonical numerical tests

| Benchmark | Purpose | Required outputs | External comparisons |
|---|---|---|---|
| Linear advection | stencil and time integration convergence | L1/L2/Linf error vs resolution | analytic |
| Diffusion decay | resistive/thermal diffusion validation | amplitude decay, convergence | analytic |
| Alfvén wave | MHD wave propagation and divB control | error after one period, divB | Athena/PLUTO |
| Brio-Wu | 1D MHD shock capturing | lineouts of rho, p, v, B | Athena/PLUTO |
| Rotor | multidimensional MHD stress test | density/pressure/B maps | Athena/PLUTO |
| Magnetic blast | robust pressure/field dynamics | maps, positivity logs | Athena/PLUTO/astronomix |
| Orszag-Tang | MHD shocks/current sheets/turbulence onset | density, pressure, current, energy, divB, movie | Athena/PLUTO/OpenMHD/astronomix |

#### 22.10.2 Fluid and MHD instabilities

| Benchmark | Purpose | Required outputs | External comparisons |
|---|---|---|---|
| Kelvin-Helmholtz | instability, shear layers, MHD stabilization | density/vorticity/current maps, growth, movie | Athena, JAX-Fluids hydro, astronomix |
| Rayleigh-Taylor | buoyancy instability, 2D/3D visualization | density slices, growth, movie | Athena, JAX-Fluids hydro |
| Double shear layer | vortical/turbulent flow | vorticity, spectra | JAX-Fluids |
| MHD RT/KH with guide field | magnetic suppression/stretching | field lines, current density | Athena/PLUTO |

#### 22.10.3 Reconnection benchmarks

| Benchmark | Purpose | Required outputs |
|---|---|---|
| FKR linear tearing | linear growth and eigenfunction validation | growth rate vs theory, eigenfunction, convergence |
| Rutherford regime | nonlinear island growth | island width vs time, algebraic growth fit |
| Sweet-Parker | resistive reconnection scaling | rate vs `S`, sheet width, inflow/outflow |
| Loureiro plasmoid instability | high-Lundquist plasmoid scaling | growth scaling, plasmoid number, convergence |
| Ideal tearing | onset in thinning current sheets | growth vs aspect ratio, `a/L ~ S^-1/3` test |
| Coalescing islands | nonlinear reconnection and energy conversion | island merger, `J·E`, energy partition |
| Hyper-resistive plasmoids | fourth-order dissipation effects | `S_H` scaling, plasmoid flux distribution |
| Under-resolution caution | scientific honesty | resolved vs under-resolved plasmoid claims |
| Harris sheet | standard 2D reconnection setup | current sheet structure and rate |

#### 22.10.4 Generalized-MHD and Hall benchmarks

| Benchmark | Purpose | Required outputs |
|---|---|---|
| GEM-like reconnection | compare resistive and Hall regimes | reconnection rate, Hall quadrupole, Ohm-law terms |
| Hall whistler propagation | timestep and dispersion gate | dispersion curve, stability |
| Electron-inertia switch | generalized Ohm-law term test | term budgets, scale dependence |
| Anomalous resistivity closure | closure sensitivity | eta_eff maps, rate comparison |

#### 22.10.5 Radiative and thermal benchmarks

| Benchmark | Purpose | Required outputs |
|---|---|---|
| Cooling ODE | source-term verification | analytic/numerical cooling curve |
| Optically thin current-sheet cooling | radiative reconnection module | cooling maps, energy loss, rate change |
| Bremsstrahlung-like double current sheet | HED/astrophysical inspiration | density/temperature/current evolution |
| Radiative X-point collapse | cooling-enhanced collapse | collapse time, sheet width, emission proxy |
| Field-aligned conduction | anisotropic transport | heat flux along B, timestep/implicit gate |

#### 22.10.6 3D and turbulence benchmarks

| Benchmark | Purpose | Required outputs |
|---|---|---|
| 3D null collapse | topology diagnostics | null location, spine/fan, `int E_parallel dl` |
| Flux-rope coalescence | 3D reconnection and helicity | field-line movie, helicity proxy, energy |
| QSL/slipping reconnection | 3D connectivity changes | Q maps, field-line connectivity movie |
| Driven MHD turbulence | turbulence/reconnection statistics | spectra, current-sheet distribution, reconnection stats |
| Synthetic spacecraft crossing | space-plasma diagnostics | time series, inferred rate, optimized path demo |

### 22.11 Novel results MHX should be able to produce beyond reproducing literature

These are target research demonstrations for the manuscript and documentation.

1. **Differentiable reconnection-rate control**: optimize the initial perturbation or boundary driving to achieve a target reconnection-rate history in reduced MHD.
2. **Resistivity and hyper-resistivity inference**: recover `eta`, `eta_H`, or anomalous-resistivity closure parameters from synthetic reconnection data.
3. **Radiative cooling-law inference**: infer cooling-law parameters from density/temperature/emission proxies in radiative current-sheet simulations.
4. **Resolution-sensitivity maps for plasmoid onset**: quantify where plasmoid claims are resolution-driven by differentiable/surrogate and ensemble diagnostics.
5. **Topology-aware 3D optimization**: optimize boundary driving to create or suppress QSL/current-sheet formation using smooth proxies for topology changes.
6. **Synthetic spacecraft trajectory optimization**: place virtual probes or lineouts to best infer reconnection rates or Ohm-law term balance.
7. **MHD-to-extended-MHD closure calibration**: use Gkeyll/PIC outputs as target data to fit reduced Hall/anomalous-resistivity closures in MHX.
8. **Publication-ready differentiable benchmark gallery**: demonstrate not just that MHX solves MHD, but that gradients through physically validated reconnection workflows are usable and interpretable.

### 22.12 Figures, plots, and movies for docs and manuscript

Every figure must be reproducible from a documented command. Use `docs/figures/manifest.toml` to map figures to scripts, configs, source data, and expected output files.

#### 22.12.1 Required static figures

| ID | Figure | Source command | Notes |
|---|---|---|---|
| F01 | MHX architecture and physics gates | `mhx docs figures architecture` | package layout, equation hierarchy |
| F02 | Numerical method decision tree | `mhx docs figures methods` | spectral vs finite-volume vs Hall/radiative modules |
| F03 | Linear advection/diffusion convergence | `mhx benchmark run convergence` | log-log error slopes |
| F04 | Alfvén wave convergence and divB | `mhx benchmark run alfven_wave` | CT/GLM comparison |
| F05 | Brio-Wu shock tube | `mhx benchmark run brio_wu` | lineouts vs reference |
| F06 | Orszag-Tang maps | `mhx benchmark run orszag_tang` | density, pressure, current, divB |
| F07 | Kelvin-Helmholtz instability | `mhx benchmark run kelvin_helmholtz` | hydrodynamic and MHD variants |
| F08 | Rayleigh-Taylor instability | `mhx benchmark run rayleigh_taylor` | 2D and optional 3D |
| F09 | Linear tearing growth | `mhx benchmark run tearing_linear` | growth rate vs theory |
| F10 | Rutherford island width | `mhx benchmark run rutherford` | island width evolution |
| F11 | Sweet-Parker scaling | `mhx benchmark run sweet_parker --suite scaling` | rate and sheet width vs `S` |
| F12 | Plasmoid instability scaling | `mhx benchmark run plasmoid` | growth, number, convergence |
| F13 | Under-resolution caution | `mhx benchmark run plasmoid_resolution` | resolved vs under-resolved comparison |
| F14 | Hyper-resistive plasmoid scaling | `mhx benchmark run hyper_resistive_plasmoid` | `S_H` trends |
| F15 | GEM/Hall quadrupole | `mhx benchmark run gem_hall` | `B_z`, rate, Ohm terms |
| F16 | Generalized Ohm-law budget | `mhx benchmark run ohm_budget` | term-by-term maps/lineouts |
| F17 | Radiative current sheet | `mhx benchmark run radiative_sheet` | cooling budget, temperature/density |
| F18 | Radiative plasmoid collapse | `mhx benchmark run radiative_plasmoid` | collapse and emission proxy |
| F19 | 3D topology demo | `mhx benchmark run null_3d` | field lines, nulls, `E_parallel` |
| F20 | Turbulent reconnection statistics | `mhx benchmark run turbulent_reconnection` | spectra, current sheets, rate distribution |
| F21 | Differentiable inverse tearing | `mhx diff inverse-tearing` | optimization history, before/after fields |
| F22 | Resistivity inference | `mhx diff infer-resistivity` | posterior/loss/gradient checks |
| F23 | Sensor placement | `mhx diff sensor-placement` | probe trajectory and information metric |
| F24 | Performance scaling | `mhx benchmark performance` | compile/runtime/memory/cell updates/s |
| F25 | External comparison matrix | `mhx compare report` | MHX vs external codes summary |

#### 22.12.2 Required movies

| Movie | Variable(s) | Purpose |
|---|---|---|
| Orszag-Tang current-density movie | `current_density_z`, density | current-sheet formation and shock interaction |
| Kelvin-Helmholtz movie | density/vorticity/current | instability roll-up and magnetic stabilization |
| Rayleigh-Taylor movie | density | instability growth, 2D/3D visualization |
| Linear tearing movie | flux contours/current | island formation |
| Sweet-Parker movie | current density, velocity streamlines | inflow/outflow and sheet thinning |
| Plasmoid movie | current density, flux contours | plasmoid growth/coalescence |
| Hyper-resistive plasmoid movie | current density | fourth-order dissipation effects |
| Hall GEM movie | `B_z`, current, Ohm terms | Hall quadrupole and fast reconnection |
| Radiative collapse movie | density, temperature, cooling rate | cooling-driven dynamics |
| 3D field-line movie | field lines/QSL/current | topology evolution |
| Turbulent reconnection movie | current sheets and spectra | turbulence-driven reconnection |

Movie API requirements:

```python
mhx.movie.scalar_field(result, variable="current_density_z", save="current.mp4", fps=20)
mhx.movie.contours(result, variable="psi", overlay="current_density_z", save="tearing.mp4")
mhx.movie.field_lines_3d(result, seeds="qsl", save="fieldlines.mp4")
mhx.movie.comparison([mhx_result, athena_result], variable="density", save="compare.mp4")
```

Plot API requirements:

```python
mhx.plot.slice(result, variable="density", time=-1, save="density.pdf")
mhx.plot.lineout(result, variable="pressure", axis="x", index="center", save="pressure_lineout.pdf")
mhx.plot.energy_budget(result, save="energy_budget.pdf")
mhx.plot.reconnection_summary(result, save="reconnection_summary.pdf")
mhx.plot.spectra(result, fields=["magnetic", "kinetic"], save="spectra.pdf")
mhx.plot.ohm_budget(result, save="ohm_budget.pdf")
mhx.plot.performance(report, save="performance.pdf")
```

### 22.13 Physics and numerical tests required for 95%+ coverage

Coverage should be enforced, but test quality is more important than the number. Required tests:

#### 22.13.1 Unit/operator tests

- grid spacing and coordinates;
- periodic/reflecting/outflow boundary handling;
- primitive/conservative roundtrip;
- pressure/energy positivity checks;
- finite-difference derivative on manufactured fields;
- spectral derivative on periodic manufactured fields;
- divergence and curl identities;
- Poisson/inverse-Laplacian residual;
- reconstruction monotonicity for simple profiles;
- Riemann solver finite outputs and symmetry properties;
- CT update preserves discrete divergence for manufactured EMF;
- GLM damping reduces divergence in controlled tests.

#### 22.13.2 Physics-gate tests

- Hall config fails when `d_i` is unresolved;
- hyper-resistivity config fails/warns when explicit timestep violates `dx^4` constraint;
- cooling config fails/warns when cooling time is below timestep without subcycling/implicit solve;
- plasmoid benchmark warns when current sheet has too few cells across width;
- reconnection diagnostics warn when `divB` exceeds tolerance;
- differentiation through nondifferentiable diagnostics such as hard plasmoid count triggers warning or error.

#### 22.13.3 Literature-anchored numerical tests

- Alfvén wave convergence slope;
- Brio-Wu shock tube lineout tolerance at fixed time;
- Orszag-Tang density/current qualitative and quantitative regression;
- linear tearing growth rate within tolerance at small validated parameters;
- Rutherford island width trend;
- Sweet-Parker scaling on a small resolution suite;
- Loureiro plasmoid scaling on slow/nightly suite;
- hyper-resistive scaling on slow/nightly suite;
- Hall quadrupole emergence in GEM-like setup;
- cooling ODE exact or high-accuracy reference.

#### 22.13.4 Differentiability tests

- `jax.grad` through short spectral solve vs finite differences;
- checkpointed vs direct gradient agreement on tiny problem;
- `jax.jvp` for one-step finite-volume smooth mode vs finite differences;
- Lineax/custom-linear-solve transpose consistency;
- Optax inverse problem decreases loss in a tiny demonstration.

### 22.14 Performance and ease-of-use report requirements

The performance report must separate:

- installation time;
- first import time;
- JIT compile time;
- first-run wall time;
- steady-state runtime;
- memory peak;
- output/write time;
- gradient runtime;
- gradient memory;
- maximum stable resolution on the test hardware.

Use `block_until_ready()` around timed JAX computations. Include hardware/device details, precision, grid, solver choices, and number of devices. For JAX comparisons, report compilation and execution separately.

Ease-of-use report must include:

- commands needed from clone to first figure;
- number of config lines;
- number of Python lines for a first script;
- whether plots/movies are built in;
- whether benchmark docs include equations and tolerances;
- whether output loads as xarray/HDF5/Zarr easily;
- external code notes and reproducibility limitations.

### 22.15 Implementation order after final pass

The coding agent should proceed in this order:

1. Move old files to `legacy/old_mhx/`.
2. Create new `src/mhx` skeleton.
3. Create `pyproject.toml` with Python 3.10+, unpinned dependencies, no `jaxlib`, TOML fallback.
4. Create `README.md` with badges and first-run examples.
5. Create `.readthedocs.yaml`, `docs/conf.py`, and the docs tree.
6. Create `.github/workflows/ci.yml`, `docs.yml`, `benchmark-smoke.yml`, and `publish.yml`.
7. Implement config loading from TOML with `tomllib`/`tomli` fallback.
8. Implement CLI skeleton with `mhx version`, `mhx init`, `mhx run`, `mhx plot`, `mhx movie`, `mhx benchmark`, `mhx physics check`.
9. Implement grid, state, output manifest, and plotting stubs with real tests.
10. Implement spectral derivative/Poisson tests.
11. Implement reduced-MHD tearing smoke and gradient test.
12. Implement finite-volume first-order/Rusanov/GLM baseline.
13. Add Brio-Wu and Orszag-Tang tests and docs.
14. Add benchmark registry and figure-generation pipeline.
15. Add external comparison READMEs and first OpenMHD/astronomix comparison scripts.
16. Add CT, HLL/HLLD, WENO, and physics modules in staged milestones.


## 23. Decision records

Add architectural decisions here as they are made.

### ADR-0001: Rebuild, do not patch legacy MHX

Decision: Move old code into `legacy/old_mhx/` and build new MHX from a clean package skeleton.

Rationale: The old code is centered on pseudo-spectral reduced-MHD tearing/plasmoid workflows and does not have the architecture required for full MHD, generalized MHD, radiative terms, documentation, external comparisons, and robust differentiability.

Status: Planned.

### ADR-0002: Two-track numerical architecture

Decision: Implement both spectral reduced/incompressible MHD and finite-volume compressible MHD.

Rationale: Spectral methods are ideal for reduced-MHD tearing, turbulence, and differentiability; finite-volume MHD is mandatory for shocks, compressibility, radiative terms, and comparison with established MHD codes.

Status: Planned.

### ADR-0003: Fixed-step scan as baseline differentiable evolution path

Decision: Use fixed-step `lax.scan` evolution as the baseline for differentiable benchmarks.

Rationale: It is reproducible, benchmarkable, compatible with checkpointing, and avoids ambiguity from differentiating through adaptive CFL decisions.

Status: Planned.

### ADR-0004: Diffrax as optional/prototype solver path, not the only production path

Decision: Use Diffrax where useful but keep custom PDE timesteppers.

Rationale: Diffrax provides excellent adjoint machinery and adaptive solvers, but production PDE verification and external-code comparison require direct control over timestepping, output, CFL, and diagnostics.

Status: Planned.

---

## 24. Open questions

Coding agents should update this section when questions are resolved.

1. Resolved: use Sphinx + MyST Markdown on ReadTheDocs. Rationale: manuscript-grade equations, API autodoc, references, PDF/HTML builds, and ReadTheDocs integration.
2. Should storage default to HDF5 or Zarr? Initial recommendation: Zarr for chunked JAX/Python workflows, HDF5 optional.
3. Which finite-volume divergence strategy lands first: GLM or CT? Initial recommendation: GLM first for speed, CT as high-priority milestone.
4. Which external code comparison should be first? Initial recommendation: OpenMHD for reconnection and astronomix for JAX-native MHD.
5. Should finite-volume HLLD be implemented before WENO5? Initial recommendation: Rusanov/HLL first, then HLLD, then WENO5.
6. How much of old MHX inverse-design logic should be ported? Initial recommendation: use as reference only, reimplement cleanly.

---

## 25. Agent log

Every agent must append an entry here. Do not delete previous entries.

### Log entry template

```markdown
### YYYY-MM-DD HH:MM TZ — Agent: <name/model>

**Summary**

- ...

**Files changed**

- ...

**Tests run**

- Command: `...`
- Result: ...

**Benchmarks run**

- ...

**Decisions made**

- ...

**Problems / blockers**

- ...

**Next steps**

- ...
```

### 2026-05-04 — Agent: GPT-5.5 Pro

**Summary**

- Drafted the initial rebuild plan for the new MHX.
- Incorporated current JAX ecosystem guidance, differentiable PDE/adjoint strategy, external-code comparison targets, benchmark tiers, documentation requirements, and migration instructions.
- Established that this file must serve as both handoff plan and ongoing agent log.

**Files changed**

- `plan.md` drafted.

**Tests run**

- None. Planning-only step.

**Benchmarks run**

- None. Planning-only step.

**Decisions made**

- New MHX should be a clean rebuild with old files moved into `legacy/old_mhx/`.
- Use two numerical tracks: spectral reduced/incompressible MHD and finite-volume compressible MHD.
- Use fixed-step `lax.scan` as the baseline differentiable evolution path, with Diffrax as an optional/prototype path.
- Use Equinox for PyTree-based modules/configurable components, Lineax/custom linear solves for implicit pieces, Optax/Optimistix for inverse problems, and JAX checkpointing for memory control.

**Problems / blockers**

- No repository modifications have been made yet.
- External code comparison data still needs to be generated or collected.

**Next steps**

- A coding agent should perform the legacy migration, create the new package skeleton, add CLI/config/grid/operator basics, and implement the first spectral derivative tests.

### 2026-05-04 — Agent: GPT-5.5 Pro, final online planning pass

**Summary**

- Performed the requested final planning pass and expanded `plan.md` with hard product requirements, TOML input schema, CLI/Python-driver requirements, packaging policy, README standard, GitHub workflows, ReadTheDocs/Sphinx configuration, external-code runbooks, numerical discretization decisions, benchmark/literature targets, figure/movie plans, coverage requirements, and implementation ordering.
- Resolved contradictions from the initial plan: Python support is now 3.10+, user configs are TOML rather than YAML, Python 3.10 uses `tomli` fallback, documentation uses Sphinx + MyST on ReadTheDocs, and `pyproject.toml` must not depend directly on `jaxlib`.
- Added exact comparison planning for Athena/Athena++, PLUTO, MPI-AMRVAC, FLASH, OpenMHD, Gkeyll, astronomix, JAX-Fluids, Dedalus, and PIC/kinetic reference codes.
- Added manuscript-grade documentation and reproducible figure/movie requirements.

**Files changed**

- `plan.md` updated.

**Tests run**

- None. Planning-only step.

**Benchmarks run**

- None. Planning-only step.

**Decisions made**

- Use TOML as the user-facing config format.
- Support Python 3.10+.
- Use Sphinx + MyST on ReadTheDocs.
- Enforce 95%+ coverage with physics/numerical tests, not only smoke tests.
- Use finite-volume CT/HLLD/WENO-Z as the research-grade MHD target, GLM/HLL/Rusanov as early milestones/fallbacks, and pseudo-spectral reduced MHD for tearing, turbulence, and differentiable research workflows.
- Treat external codes as benchmark references with documented runbooks and parsers rather than vendored dependencies.

**Problems / blockers**

- External codes have not yet been cloned, built, or run.
- No numerical benchmark data have been generated yet.
- Some external-code workflows may require registration, compilers, MPI, CUDA, or site-specific setup.

**Next steps**

- Coding agent should execute the migration, create the clean package skeleton, add TOML config loading, add CLI help/version/init/run stubs, add docs/workflows, and implement first spectral derivative and reduced-MHD tearing tests.

### 2026-05-04 — Agent: Codex, rebuild implementation pass 1 started

**Summary**

- Started rebuild.
- Performed an online context pass for JAX differentiable numerics, Diffrax adjoints, Lineax solves, JAX-Fluids, and MHD/reconnection validation targets before editing.
- Moved the active pre-rebuild implementation, scripts, tests, docs, examples, and benchmark files into `legacy/old_mhx/`.
- Created the first clean `src/mhx/` package skeleton with a version module, Typer CLI, TOML config dataclasses, periodic Cartesian grid, spectral derivative operators, manifest IO, docs skeleton, and smoke tests.

**Files changed**

- Added new root `README.md`, `pyproject.toml`, `.readthedocs.yaml`, `legacy/README.md`, `src/mhx/`, `examples/linear_tearing.toml`, `examples/run_linear_tearing.py`, `docs/`, `tests/`, and updated `.github/workflows/ci.yml`.
- Preserved old README/docs/tests/scripts/package under `legacy/old_mhx/`.

**Tests run**

- Pending in this pass. Next command sequence is editable install, import check, CLI check, ruff, pytest with coverage, and Sphinx build.

**Benchmarks run**

- None yet. The first numerical check is the spectral derivative smoke workflow, not a physical benchmark.

**Decisions made**

- Use the new `src/` layout immediately so legacy `mhx/` cannot shadow the rebuilt package.
- Use TOML configs and JSON manifests for deterministic run records.
- Keep the first numerical kernel deliberately small: pure JAX periodic FFT derivatives with direct manufactured sinusoid tests.
- Keep `jaxlib` out of `pyproject.toml`; users install accelerator-specific JAX wheels separately.

**Problems / blockers**

- No scientific solver has been ported yet; this is the foundation pass.
- External code comparisons have been scoped from literature/code references but not cloned, built, or run.

**Next steps**

- Run validation locally, fix failures, then commit and push this first rebuild chunk.
- Next implementation chunk should add the first reduced-MHD spectral state/RHS, a linear tearing manufactured benchmark, and a gradient check.

### 2026-05-04 — Agent: Codex, rebuild implementation pass 1 validated

**Summary**

- Completed the first migration and package-skeleton chunk.
- Fixed a pathing issue from the first patch application by reapplying all active project files with absolute `/Users/rogerio/local/MHX/...` paths.
- Added enough tests to validate the active skeleton, IO schema, CLI, grid generation, and spectral operators.

**Files changed**

- Active root files: `.github/workflows/ci.yml`, `.gitignore`, `.readthedocs.yaml`, `README.md`, `pyproject.toml`, `plan.md`.
- Active source/docs/tests/examples: `src/mhx/`, `docs/`, `tests/`, `examples/`.
- Legacy archive: `legacy/README.md` and `legacy/old_mhx/`.

**Tests run**

- `python -m pip install -e ".[dev,docs]"` passed.
- `python -c "import mhx; print(mhx.__version__); print(mhx.__file__)"` passed and imports from `src/mhx/__init__.py`.
- `mhx version` and `mhx --help` passed.
- `python -m ruff check src tests examples` passed.
- `python -m pytest --cov=mhx --cov-report=term-missing --cov-fail-under=95` passed: 17 tests, 100% coverage on the active skeleton.
- `sphinx-build -b html docs docs/_build/html` passed without warnings after fixing MyST path references.
- `python examples/run_linear_tearing.py` passed.
- `mhx run examples/linear_tearing.toml --outdir outputs/smoke` passed and wrote `config_effective.json`, `diagnostics.json`, and `manifest.json`.

**Benchmarks run**

- No physical benchmark yet. The current smoke workflow validates the FFT derivative kernel against a manufactured sinusoid.

**Decisions made**

- Removed stale CI media/paper jobs from the active workflow until rebuilt equivalents exist; keeping dead CI jobs would make the repository fail immediately after the clean rebuild.
- Enforced coverage in CI from the beginning, but only against the new active package.
- Kept generated build/test/output artifacts out of git via `.gitignore`.

**Problems / blockers**

- The active solver is not yet a physical MHD time integrator.
- The old package is intentionally archived and no longer importable from the repository root.
- Full external-code comparisons and literature-derived benchmark data are still future work.

**Next steps**

- Commit and push this migration/skeleton chunk.
- Implement the first spectral reduced-MHD state, Poisson solve/operator utilities, linear tearing RHS, deterministic FAST integration, and gradient check.
- Add a real diagnostics API before porting inverse design, scans, figures, or neural ODE components.

### 2026-05-04 — Agent: Codex, reduced-MHD smoke benchmark

**Summary**

- Added the first active physical model: periodic pseudo-spectral reduced MHD in flux/vorticity form.
- Added inverse-Laplacian support with safe zero-mode handling so gradients do not produce NaNs.
- Added fixed-step differentiable RK4 evolution through `jax.lax.scan`.
- Added reduced-MHD diagnostics and a deterministic FAST tearing-like smoke benchmark.
- Updated the CLI so `mhx run examples/linear_tearing.toml --outdir ...` runs the reduced-MHD smoke model rather than only a derivative check.

**Files changed**

- Added `src/mhx/state/`, `src/mhx/equations/reduced_mhd.py`, `src/mhx/time_integrators/`, `src/mhx/diagnostics/reduced_mhd.py`, and `src/mhx/benchmarks/tearing.py`.
- Updated spectral operators, CLI run path, example config/driver, README, API docs, architecture docs, and added `docs/reduced_mhd.md`.
- Added `tests/test_reduced_mhd.py` and expanded spectral/CLI tests.

**Tests run**

- `python -m ruff check src tests examples` passed.
- `python -m pytest --cov=mhx --cov-report=term-missing --cov-fail-under=95` passed: 23 tests, 98.04% coverage.
- `sphinx-build -b html docs docs/_build/html` passed.
- `python examples/run_linear_tearing.py` passed and printed reduced-MHD energy diagnostics.
- `mhx run examples/linear_tearing.toml --outdir outputs/smoke` passed and wrote JSON outputs.
- Manual `jax.grad` check through a three-step RK4 solve passed with finite gradient.

**Benchmarks run**

- FAST reduced-MHD smoke run only. It validates the execution path, energy diagnostics, and differentiability; it is not yet an FKR growth-rate benchmark.

**Decisions made**

- Defined the initial equations as `ψ_t + [φ,ψ] = η∇²ψ`, `ω_t + [φ,ω] = [ψ,∇²ψ] + ν∇²ω`, with `∇²φ = ω` and zero-mean inverse Laplacian.
- Chose a smooth periodic initial condition `ψ = cos(y) + ε cos(x)cos(y)`, `ω = 0` for the FAST smoke test.
- Kept saved diagnostics scalar/JSON-safe for now; richer array outputs will wait for the formal output schema.

**Problems / blockers**

- The benchmark is not yet theory-calibrated against FKR/Coppi tearing rates.
- No plotting/movie output exists in the rebuilt active package yet.
- Diagnostics are physically meaningful but still minimal.

**Next steps**

- Add formal output schema for trajectories and diagnostic time series.
- Add growth-rate diagnostics and a small linear tearing validation target.
- Add plotting helpers for energy history and flux contours, then a deterministic FAST figure pipeline.

### 2026-05-04 — Agent: Codex, trajectory schema and figures

**Summary**

- Added the first versioned active output schema: `mhx.reduced_mhd.trajectory.v1`.
- Added NPZ write/read utilities for reduced-MHD trajectories with config and diagnostics JSON payloads.
- Updated `mhx run` to write `trajectory.npz` and include it in `manifest.json` hashes.
- Added `mhx figures <run_dir>` to regenerate deterministic `energy_history.png` and `flux_final.png` from saved outputs.

**Files changed**

- Added `src/mhx/io/trajectory.py`, `src/mhx/plotting/reduced_mhd.py`, `tests/test_io_schema.py`, and `docs/output_schema.md`.
- Updated CLI, IO exports, plotting exports, README, quickstart docs, docs index, and CLI/IO tests.

**Tests run**

- `python -m ruff check src tests examples` passed.
- `python -m pytest --cov=mhx --cov-report=term-missing --cov-fail-under=95` passed: 27 tests, 98.33% coverage.
- `sphinx-build -b html docs docs/_build/html` passed.
- `mhx run examples/linear_tearing.toml --outdir outputs/smoke` passed.
- `mhx figures outputs/smoke` passed and wrote `energy_history.png` and `flux_final.png`.

**Benchmarks run**

- FAST reduced-MHD smoke run only.

**Decisions made**

- Used compressed NPZ for the first active array schema because it is simple, inspectable, and sufficient for FAST workflows.
- Kept manifests JSON-only with SHA-256 hashes for reproducible diffs.
- Kept plotting regenerated from saved data rather than storing figures as authoritative data.

**Problems / blockers**

- No movie writer yet.
- No growth-rate extraction or validated tearing-rate benchmark yet.
- The current figure path uses index-space flux contours; physical coordinate axes should be added with the plotting API upgrade.

**Next steps**

- Add growth-rate/amplitude diagnostics for the perturbation mode.
- Add a first deterministic benchmark report command and docs page.
- Add basic movie/GIF support once the trajectory schema stores enough frames for useful animations.

### 2026-05-04 — Agent: Codex, mode amplitude and growth-rate diagnostics

**Summary**

- Added standardized reduced-MHD Fourier mode amplitude diagnostics.
- Added least-squares exponential growth-rate fitting for saved mode amplitudes.
- Updated the FAST benchmark to save every timestep, report `diagnostic_mode`, `initial_mode_amplitude`, `final_mode_amplitude`, and `gamma_fit`.
- Updated `mhx figures` to write `mode_amplitude.png`.

**Files changed**

- Updated `src/mhx/diagnostics/reduced_mhd.py`, plotting helpers, benchmark diagnostics, CLI figure generation, config defaults, example TOML, README, and docs.
- Expanded reduced-MHD and CLI tests for amplitude/growth metrics and regenerated figures.

**Tests run**

- `python -m ruff check src tests examples` passed.
- `python -m pytest --cov=mhx --cov-report=term-missing --cov-fail-under=95` passed: 30 tests, 98.43% coverage.
- `sphinx-build -b html docs docs/_build/html` passed before the final test-only fix.
- `mhx run examples/linear_tearing.toml --outdir outputs/smoke` passed.
- `mhx figures outputs/smoke` passed and wrote `energy_history.png`, `flux_final.png`, and `mode_amplitude.png`.
- Remote GitHub CI for commits `ce465e6`, `ccdfd31`, and `2205ff2` is green.

**Benchmarks run**

- FAST reduced-MHD smoke run only.

**Decisions made**

- The mode amplitude is the normalized FFT amplitude of `psi`.
- The FAST benchmark reports `gamma_fit` for plumbing and regression only; docs explicitly warn it is not yet an FKR tearing rate.

**Problems / blockers**

- No validated tearing eigenmode or fit-window policy yet.
- No benchmark report command yet.

**Next steps**

- Implement a deterministic benchmark report command that reads a run directory and writes a small JSON/Markdown report.
- Add a simple GIF/movie writer using saved `psi` frames.
- Start the theory-calibrated tearing benchmark by adding an equilibrium/eigenmode module and fit-window controls.

### 2026-05-04 — Agent: Codex, reports and GIF output

**Summary**

- Added `mhx report <run_dir>` to generate `report.json` and `report.md` from saved diagnostics and manifest hashes.
- Added optional `mhx figures <run_dir> --gif` support for `flux_movie.gif`.
- Added `imageio` to plotting/dev extras.

**Files changed**

- Added `src/mhx/benchmarks/report.py`.
- Updated plotting, CLI, tests, README, quickstart, output-schema docs, and `pyproject.toml`.

**Tests run**

- `python -m pip install -e ".[dev,docs]"` passed and installed `imageio`.
- `python -m ruff check src tests examples` passed.
- `python -m pytest --cov=mhx --cov-report=term-missing --cov-fail-under=95` passed: 31 tests, 98.41% coverage.
- `sphinx-build -b html docs docs/_build/html` passed.
- `mhx run examples/linear_tearing.toml --outdir outputs/smoke` passed.
- `mhx figures outputs/smoke --gif` passed and wrote PNGs plus `flux_movie.gif`.
- `mhx report outputs/smoke` passed and wrote `report.json` and `report.md`.

**Benchmarks run**

- FAST reduced-MHD smoke run only.

**Decisions made**

- Reports are regenerated artifacts, not authoritative data; source of truth remains config, diagnostics, trajectory NPZ, and manifest hashes.
- GIF writing is opt-in with `--gif` to keep default figure generation lightweight.

**Problems / blockers**

- Movie output is functional but basic; it does not yet use physical coordinates or contour overlays.
- Benchmark reports do not yet include external/theory comparison rows.

**Next steps**

- Add physical coordinate axes to plotting.
- Add benchmark comparison hooks and a first theory-calibrated tearing-growth fixture.
- Add a `mhx benchmark` command group once there is more than one benchmark.

### 2026-05-04 — Agent: Codex, benchmark command group and theory scaffolds

**Summary**

- Added first-class `mhx benchmark run` and `mhx benchmark validate` commands.
- Added physical-coordinate plotting for flux contours and GIF frames.
- Added validation output `validation.json` with lightweight energy/growth checks.
- Added analytic benchmark-planning scaffolds for FKR constant-psi tearing, Loureiro plasmoid scaling, and Pucci-Velli ideal-tearing aspect ratio.
- Added a benchmark docs page with equations, command sequence, expected artifacts, and linked references.

**Files changed**

- Added `src/mhx/benchmarks/theory.py`, `docs/benchmarks.md`, and `tests/test_benchmark_theory.py`.
- Updated `src/mhx/cli/main.py`, `src/mhx/benchmarks/report.py`, plotting helpers, README, quickstart, API docs, and CLI tests.

**Tests run**

- `python -m ruff check src tests examples` passed.
- `python -m pytest --cov=mhx --cov-report=term-missing --cov-fail-under=95` passed: 36 tests, 97.93% coverage.
- `sphinx-build -b html docs docs/_build/html` passed.
- `mhx benchmark run --config examples/linear_tearing.toml --outdir outputs/benchmarks/linear_tearing_fast --gif` passed.
- `mhx benchmark validate outputs/benchmarks/linear_tearing_fast` passed.

**Benchmarks run**

- FAST reduced-MHD benchmark pipeline only.

**Decisions made**

- Benchmark theory functions are scaling estimates only; docs explicitly avoid claiming calibrated validation.
- `mhx benchmark validate` writes machine-readable `validation.json` and exits nonzero on failed checks.
- Physical coordinates now appear in regenerated flux figures when mesh metadata is available.

**Problems / blockers**

- FKR/Coppi validation is still a future benchmark; the current theory code only provides planning estimates.
- There is still only one active benchmark, so `mhx benchmark` is intentionally small.

**Next steps**

- Add a fit-window policy and a calibrated linear tearing fixture.
- Add benchmark artifact upload to CI once the active benchmark pipeline is stable on GitHub Actions.
- Add the first physics plugin API after the benchmark API stabilizes.

### 2026-05-04 — Agent: Codex, fit-window policy for growth diagnostics

**Summary**

- Added explicit diagnostics config fields for Fourier mode and growth-fit time window.
- Updated FAST benchmark diagnostics to record `fit_time_window` and `fit_sample_count`.
- Added fit-window selection tests and config validation.
- Updated docs to make growth-rate fitting auditable and reproducible.

**Files changed**

- Updated `src/mhx/config/schema.py`, `src/mhx/diagnostics/reduced_mhd.py`, `src/mhx/benchmarks/tearing.py`, `examples/linear_tearing.toml`, tests, and docs.

**Tests run**

- `python -m ruff check src tests examples` passed.
- `python -m pytest --cov=mhx --cov-report=term-missing --cov-fail-under=95` passed: 36 tests, 98.00% coverage.
- `sphinx-build -b html docs docs/_build/html` passed.
- `mhx benchmark run --config examples/linear_tearing.toml --outdir outputs/benchmarks/linear_tearing_fast --gif` passed.
- `mhx benchmark validate outputs/benchmarks/linear_tearing_fast` passed.

**Benchmarks run**

- FAST reduced-MHD benchmark pipeline only.

**Decisions made**

- Growth-rate fit windows live under `[diagnostics]` in TOML, because they define how diagnostics are interpreted rather than how the solver evolves.
- If no fit window is supplied, MHX fits all saved samples.

**Problems / blockers**

- The fit policy is now explicit, but the fitted FAST mode remains a smoke diagnostic, not a calibrated FKR rate.

**Next steps**

- Add a synthetic eigenmode/growth fixture with known gamma to test the complete report path.
- Add CI artifact upload for benchmark outputs.
- Start the physics plugin interface after the benchmark/report/validation workflow is stable.

### 2026-05-04 — Agent: Codex, versioned physics plugin API

**Summary**

- Added a stable `mhx.physics.v1` plugin contract for RHS extension terms.
- Added built-in `hyper_resistivity` and `vorticity_drag` terms with registry metadata and CLI linting.
- Wired configured physics terms into the reduced-MHD RHS and benchmark diagnostics.
- Added TOML-driven model assembly via `[physics] rhs_terms` and `[physics.term_parameters.<term>]`.
- Added a plugin example, user-facing docs, API docs, README quickstart notes, and regression tests.

**Files changed**

- Added `src/mhx/physics/terms.py`, `docs/plugins.md`, `examples/linear_tearing_hyper.toml`, and `tests/test_physics_terms.py`.
- Updated `src/mhx/physics/__init__.py`, `src/mhx/config/schema.py`, `src/mhx/equations/reduced_mhd.py`, `src/mhx/benchmarks/tearing.py`, `src/mhx/cli/main.py`, README, docs index, and API docs.

**Tests run**

- `python -m ruff check src tests examples` passed.
- `python -m pytest --cov=mhx --cov-report=term-missing --cov-fail-under=95` passed: 42 tests, 97.86% coverage.
- `sphinx-build -b html docs docs/_build/html` passed.
- `mhx run examples/linear_tearing_hyper.toml --outdir outputs/plugin_smoke` passed.
- `mhx physics list` and `mhx physics lint hyper_resistivity` passed.

**Benchmarks run**

- FAST reduced-MHD benchmark with configured hyper-resistivity and vorticity drag only.

**Decisions made**

- The v1 extension point is an additive RHS contribution over `ReducedMHDState`, which is simple to test and keeps the core integrator unchanged.
- Built-in terms are registered by name and assembled from TOML so users can change physics without editing Python.
- Config validation rejects parameter tables for inactive terms, but it does not reject unknown term names so third-party registries can be introduced later.

**Problems / blockers**

- The current registry is process-local and built-in only; external package discovery is still future work.
- The plugin API covers RHS additions but not new evolved fields yet.

**Progress**

- Estimated plan completion: 31%.

**Next steps**

- Add external plugin loading and a `ModelConfig`-style registry object for equilibria, terms, and diagnostics.
- Add the first nontrivial extended-MHD toy term that approximates Hall/two-fluid effects within the current reduced-state API.
- Add benchmark artifact upload in CI and expand validation beyond smoke energy checks.
