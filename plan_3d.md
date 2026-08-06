# MHX 3D MHD Program Plan

Generated: 2026-08-05

This file is the implementation plan for extending MHX from two-dimensional
reduced MHD to full three-dimensional incompressible visco-resistive MHD,
and the persistent execution log for that work. It sits beside `plan.md`
(the original rebuild plan) and `plan_docs.md` (the documentation program).
Where this file and `plan.md` section 6 disagree, this file wins: it commits
to Track A3 below and defers finite-volume compressible MHD indefinitely.

---

## 0. Mission

Add a full 3D MHD solver to MHX that:

- solves the incompressible visco-resistive MHD equations in a periodic box
  with Fourier pseudo-spectral accuracy, no toy terms, no scaffolds;
- passes a validation ladder anchored to published numbers, from
  machine-precision wave dispersion to the Taylor--Green dynamo threshold
  and the reduced-MHD limit;
- shares the 2D API: the same `mhx.Simulation` call, the same TOML schema,
  the same output files, so a 2D script becomes a 3D script by changing
  `shape` and the equations name;
- differentiates end to end under the same gradient-validation contract as
  the 2D code;
- runs fast on one GPU, scales to the office pair by slab decomposition,
  and has a clean path to pencil decomposition on clusters;
- produces publication figures and movies, including 3D isosurface and
  field-line renders;
- adds the minimum code that meets the above. The target for the entire
  3D numerical core is under about 1,500 new source lines, because the
  existing spectral operators are already dimension-agnostic and a slab
  spectral MHD core is a few hundred lines in every reference
  implementation.

The flagship science deliverable is the 2D-versus-3D comparison: full 3D
MHD with a strong guide field converging to the existing validated 2D
reduced-MHD results, following the Oughton--Dmitruk--Matthaeus protocol.
That comparison turns the code extension itself into a publishable result.

---

## 1. Formulation decision

**Track A3: incompressible visco-resistive MHD, primitive variables, Fourier
pseudo-spectral, triply periodic.**

$$
\partial_t \mathbf{v} + (\mathbf{v}\cdot\nabla)\mathbf{v}
  = -\nabla p + \mathbf{j}\times\mathbf{B} + \nu\nabla^2\mathbf{v},
\qquad
\partial_t \mathbf{B} = \nabla\times(\mathbf{v}\times\mathbf{B})
  + \eta\nabla^2\mathbf{B},
$$

with $\nabla\cdot\mathbf{v} = \nabla\cdot\mathbf{B} = 0$, in Alfvén units,
optionally with a uniform guide field $\mathbf{B}_0$.

Why this and not finite-volume compressible MHD:

1. **Literature coherence.** The canonical periodic-box 3D MHD results
   (Orszag--Tang 3D, Taylor--Green dynamo, decaying-turbulence decay laws,
   ABC dynamos, ideal tearing) are all incompressible spectral
   computations. Every gate in section 7 checks against numbers produced
   by codes of exactly this class (GHOST, SNOOPY, spectralDNS, the
   Boldyrev-group codes).
2. **Minimal code.** spectralDNS demonstrates a slab-parallel 3D spectral
   solver in about one hundred lines over an FFT library. MHX's spectral
   operators (`numerics/spectral/operators.py`) already take
   `lengths: tuple[float, ...]` and dimension-generic shapes, so the 3D
   core reuses them unchanged. Finite-volume MHD would need
   reconstruction, Riemann solvers, constrained transport, and limiters:
   thousands of lines before the first validated result, against
   entrenched competition (Athena++, PLUTO, AMRVAC).
3. **Exact constraints.** The spectral Leray projector keeps
   $\nabla\cdot\mathbf{B}=0$ and $\nabla\cdot\mathbf{v}=0$ at round-off.
   No divergence cleaning, no CT, no pressure solve.
4. **The 2D bridge.** Reduced MHD is the strong-guide-field limit of this
   system (Strauss 1976; Zank--Matthaeus 1992). The same spectral
   machinery on both sides makes the comparison clean and reviewable.
5. **Differentiability.** Fixed-step spectral updates are smooth; no
   limiter or Riemann-solver nonsmoothness enters the gradient path.

Variables: primitive $(\mathbf{v}, \mathbf{B})$ with the projector
$P_{ij}(\mathbf{k}) = \delta_{ij} - k_i k_j/k^2$, matching the production
choice of GHOST, SNOOPY, TARANG, and spectralDNS. An Elsässer wrapper
$\mathbf{z}^\pm = \mathbf{v} \pm \mathbf{b}$ is provided for diagnostics,
cross-helicity-controlled forcing, and the exact $\mathbf{B}_0$ phase
rotation. The vector potential is rejected: in Fourier space it buys
nothing that the projector does not already guarantee.

---

## 2. Numerics decisions

| Choice | Decision | Anchor |
| --- | --- | --- |
| Spectrum | `rfftn` half-spectrum, `complex64[3, nx, ny, nz//2+1]` per field | real fields; halves memory; XLA lowers to cuFFT R2C |
| Dealiasing | 2/3 truncation mask | Orszag 1971; community default; 3/2 padding costs 3.4x memory in 3D |
| Nonlinearity | rotational form $\mathbf{v}\times\boldsymbol{\omega} + \mathbf{j}\times\mathbf{B}$, curl form for induction | conserves energy under truncation; 18 FFTs per RHS |
| Time stepper | Williamson 2N low-storage RK3 with exact integrating factors $e^{-\nu k^2\Delta t}$, $e^{-\eta k^2\Delta t}$ | SNOOPY and DNS-standard; removes diffusive step limit; two registers |
| Guide field | optional exact Elsässer phase rotation $e^{\pm i k_\parallel v_A \Delta t}$ | removes the $B_0$ Alfvén CFL for the strong-guide-field campaign |
| CFL | $\Delta t \le C/[k_{\max}(u_{\max} + v_{A,\max})]$, $C=0.3$ to $0.5$ | with exact $B_0$ rotation only fluctuation $v_A$ counts |
| Implicit option | backward Euler / trapezoid through `solvax.newton_krylov`, Fourier-diagonal preconditioner | steady states and stiff extensions; mirrors the 2D path |
| Hyperdissipation | allowed for exploratory movies only; every gate and published number is Laplacian DNS | Beresnyak's bottleneck criticism; Haugen--Brandenburg artifacts |
| Precision | complex64 fields with float64 reductions for invariants; gates run under x64 in CI | the float32 Newton-tolerance trap is already documented in 2D |

The classical RK4 path (`evolve_rk4`) stays available; the 2N-RK3 with
integrating factors is the production stepper because reverse-mode
recomputation cost is set by the per-step live set, which the 2N form
minimizes.

---

## 3. Architecture

### 3.1 State and modules

```text
src/mhx/
  state/mhd3d.py          MHD3DState(v_hat, b_hat), trajectory container
  equations/mhd3d.py      RHS, projector, curl, Elsasser transforms
  numerics/spectral/      operators.py unchanged (already n-dimensional)
    pfft.py               pfft3/pifft3: rfftn signature, sharded slab path
  time_integrators/
    low_storage.py        2N-RK3 with integrating factors
  physics/equilibria3d.py OT3D, TaylorGreen, ABC, ForceFree, CPAlfven,
                          HarrisSheet3D (guide field)
  physics/forcing.py      OU band forcing, ABC/TG deterministic,
                          Elsasser-controlled
  diagnostics/mhd3d.py    shell spectra, budgets, helicities, Shebalin
                          angles, Politano-Pouquet law, structure functions
  plotting/mhd3d.py       slice mosaics; pyvista isosurfaces behind viz3d
```

The state lives permanently in spectral space; real space exists only
inside the RHS. Estimated new-core line counts, informed by the reference
codes: transforms ~100, RHS ~60, stepper ~60, state ~80, equilibria ~150,
forcing ~120, diagnostics ~300, plotting ~150. Core under ~1,000 lines
before benchmarks and tests.

### 3.2 API symmetry, 2D to 3D

```python
# 2D today
result = mhx.Simulation(shape=(256, 256)).run()

# 3D after this program
result = mhx.Simulation(
    shape=(256, 256, 256),
    equations="mhd3d",
    equilibrium=mhx.OrszagTang3DEquilibrium(beta=0.8),
    viscosity=1.0e-3,
    resistivity=1.0e-3,
).run()
```

- `Simulation` dispatches on `equations` (default stays `"reduced_mhd"`;
  a three-entry `shape` with the default equations is an error, not a
  guess).
- `resistivity`/`viscosity`, `dt`, `t_end`, `save_every`, `device_count`,
  `integrator`, and `dealiasing` keep their exact 2D meanings.
- TOML: the same schema with `[mesh] nz` and
  `model = "incompressible_mhd_3d"`. One config diff turns a recorded 2D
  campaign into a 3D campaign.
- Output: the same NPZ-plus-manifest contract with a `dimension` field and
  the field list `(vx, vy, vz, bx, by, bz)` replacing `(psi, omega)`.
  `mhx figures` grows slice-mosaic recipes; `result.plot` gives the
  four-panel 3D summary (midplane |j| slice, energy history, spectra,
  divergence check).

### 3.3 Distributed transforms

One module, `numerics/spectral/pfft.py`, with the `jnp.fft.rfftn`
signature:

- mesh size 1: plain `jnp.fft.rfftn`/`irfftn`, zero divergence from the
  single-device path;
- sharded: `shard_map` slab path over MHX's existing one-dimensional
  device mesh: local `rfft2` over the unsharded axes, one
  `jax.lax.all_to_all` transpose, local `fft` along the remaining axis;
- wrapped in `jax.custom_vjp` whose backward pass is the hand-written
  inverse transform, because `custom_partitioning` has no differentiation
  rule (jax-ml/jax issue 29954) and naive sharded `jnp.fft` all-gathers
  the field;
- a compiled-HLO regression test asserts no field-sized `all-gather`
  appears in either the forward or backward program;
- `jaxDecomp` (pencil decomposition, cuDecomp or pure-JAX backend) slots
  behind the same interface for multi-node work later. It is c2c-only
  today, so it stays strictly optional until its r2c lands.

### 3.4 Memory plan

Spectral state at 512-cubed, six components, complex64: about 3.2 GB;
with two 2N registers and FFT scratch, roughly 10 to 15 GB per GPU on the
office pair under slab sharding. 256-cubed fits one GPU with wide margin.
Techniques, all already proven in the referenced JAX codes: buffer
donation at the jit boundary, `lax.scan` carry reuse, diagnostics streamed
to host through `io_callback` instead of stored trajectories, and saved
fields restricted to a configured subset.

---

## 4. Differentiability plan

The 2D contract carries over verbatim: every differentiable claim is
validated against central finite differences under x64, at small
resolution, with step-size and dtype checks.

- **Short trajectories** (hundreds of steps): direct reverse-mode through
  the scan.
- **Long trajectories**: chunked scans with
  `jax.checkpoint(policy=nothing_saveable)` per chunk, chunk length near
  the square root of the step count; nested rematerialization
  (treeverse-style, as in Diffrax's recursive adjoint) documented for
  extreme horizons; host offload of named residuals as the escape hatch.
- **Steady or statistically steady objectives**: implicit differentiation
  through `solvax` root solves instead of unrolled backprop.
- **Nonsmooth diagnostics** (plasmoid counts, critical-point counts in
  slices) keep the 2D policy: report them, never optimize through them.

Gradient gates: d(final total energy)/d(nu) and d/d(eta) on a small ABC
decay run, checked against finite differences; one sharded-vs-single-device
gradient equality check to pin the distributed-FFT adjoint.

---

## 5. Validation ladder

Anchored to published, paper-extracted numbers. G1 through G6 run in CI
under x64 at 64-cubed or smaller, or kinematically. G7 through G12 are
office-GPU campaigns with committed evidence bundles, following the
existing claim-level and promotion machinery.

| Gate | Content | Reference anchor | Tolerance |
| --- | --- | --- | --- |
| G1 | single-mode viscous/resistive decay, divergence at machine epsilon, projector/curl spectral convergence | exact | round-off; stepper order verified |
| G2 | damped oblique Alfvén and pseudo-Alfvén dispersion, $\nu\ne\eta$, including one overdamped case | exact complex $\omega = \pm\sqrt{k_\parallel^2 v_A^2 - \frac{1}{4}(\nu-\eta)^2 k^4} - \frac{i}{2}(\nu+\eta)k^2$ | $10^{-8}$ relative, spectral part |
| G3 | large-amplitude circularly polarized Alfvén wave, oblique, ten crossing times | exact Walén solution; no parametric decay in incompressible MHD | phase/amplitude error at stepper order |
| G4 | ideal truncated run conserving $E$, $H_C$, $H_M$; drift toward absolute equilibria | Frisch, Pouquet, Léorat, Mazure 1975 | relative drift below $10^{-6}$ per dynamical time |
| G5 | ABC kinematic dynamo windows and rates | Galloway--Frisch 1986; Bouya--Dormy 2013: windows near $R_m \approx 8.9$--$17.5$ and reopening at $\approx 27$; frequency jump at $R_{m1} \in [24.05, 24.10]$ | window edges 2 percent; $\sigma(R_m{=}100)$ 5 percent |
| G6 | resistive tearing: FKR $S^{-3/5}$ branch and the ideal-tearing point | Pucci--Velli 2014; Landi et al. 2015: $\gamma\tau_A = 0.63$ at $a/L = S^{-1/3}$ | exponent and point to 5 percent |
| G7 | 3D Orszag--Tang, exact PPS95 initial condition with $\beta = 0.8$ | Mininni, Pouquet, Montgomery 2006: $\max|j|$ exponential to $t^3$ with transition near $t \approx 0.6$; $\varepsilon_{\mathrm{peak}} \approx 0.3$, Re-independent | 10 percent on peak dissipation at 256--512 cubed |
| G8 | Taylor--Green dynamo threshold | Ponty et al. 2005: $R_M^c = 28.8$ at $P_m \approx 1$ on 64-cubed | 10 percent; Kazantsev $k^{3/2}$ kinematic spectrum |
| G9 | decaying Taylor--Green MHD non-universality, insulating/alternative/conducting classes | Lee et al. 2010: $k^{-2}$ / $k^{-5/3}$ / $k^{-3/2}$ from one velocity field | class-resolved compensated spectra at 512-cubed |
| G10 | decay laws and helicity conservation | Biskamp--Müller 1999/2000 with the Christensson bracket: nonhelical exponent in $[-1.1, -0.9]$, helical in $[-0.7, -0.5]$; $dH_M/dt = -2\eta\langle\mathbf{j}\cdot\mathbf{b}\rangle$ exactly | bracket gates, exact helicity budget |
| G11 | forced steady turbulence: exact third-order law | Politano--Pouquet 1998 4/3-law plateau equal to measured $\varepsilon^\pm$; $\zeta_3 = 1$; slope in $[-1.7, -1.5]$ | 10 percent on the plateau; do not claim $-5/3$ versus $-3/2$ below 1024-cubed |
| G12 | reduced-MHD limit, the flagship | Oughton--Dmitruk--Matthaeus 2004; Dmitruk--Matthaeus--Oughton 2005; Oughton et al. 2017: full 3D at $B_0/b_{\mathrm{rms}} = 1, 5, 10$ converging to MHX-2D spectra, Shebalin angles, budgets | protocol match, trend convergence |

Two literature disputes are deliberately bracketed, never pinned: the
helical decay exponent and the $-5/3$ versus $-3/2$ inertial-range slope.
Gates assert the brackets and the docs state the controversy with both
citations.

---

## 6. Diagnostics (first-class, tested)

Shell-averaged kinetic and magnetic spectra; axisymmetric
$E(k_\perp, k_\parallel)$ and reduced perpendicular/parallel spectra under
a guide field; residual-energy and normalized cross-helicity spectra;
energy, cross-helicity, and magnetic-helicity budgets with closure
tolerances; dissipation history and $C_\varepsilon(Re)$; Taylor scale and
$R_\lambda$; Shebalin angles; scale-dependent alignment; Politano--Pouquet
third-order structure functions; Elsässer structure-function exponents
against the Müller--Biskamp sheet model. Every one lands as a tested
function in `diagnostics/mhd3d.py` before any campaign uses it.

---

## 7. SOLVAX pull requests

The explicit 3D path needs nothing from SOLVAX on day one. The implicit
path, eigenvalue gates at scale, and multi-device Krylov need, in
dependency order:

1. **Sharded-operand pass**: `gmres`/`gcrot`/`newton_krylov`/`pcg` on
   `NamedSharding` complex pytrees; no host syncs in orthogonalization;
   an HLO test forbidding field-sized all-gathers.
2. **Axis-aware inner products**: optional collective axis so solvers work
   inside `shard_map` regions.
3. **Fourier-diagonal preconditioner helper** in `solvax.precond`,
   composing with the existing Fourier--Helmholtz machinery.
4. **Multi-leaf pytree GCROT operands** (already on the SOLVAX roadmap;
   the MHD state is a two-leaf pytree).
5. **Device-resident Arnoldi/Krylov--Schur** for large complex sharded
   operators; the current eigensolvers orchestrate on the host.
6. **Fixed-work solver variants** with static iteration counts for
   scan-embedded implicit steps.
7. **Complex mixed-precision Krylov wrappers**: complex64 orthogonalization
   with float64 scalar reductions.

Items 1, 2, and 6 gate the implicit 3D path; items 3 to 5 gate eigenvalue
campaigns at scale; item 7 is an optimization. These slot into the SOLVAX
plan's own sharding milestone.

---

## 8. Execution phases

Each phase is a reviewable unit that keeps every existing 2D gate green,
updates docs beside code (the `plan_docs.md` structure has reserved slots),
and appends a log entry here.

- **P3D-0. Decisions and scaffolding of record.** Commit this plan. File
  SOLVAX issues for section 7 items 1, 2, 6. Add the `equations` dispatch
  seam to `Simulation` with `"reduced_mhd"` as the only value, so the 2D
  API change lands first and alone.
  *Acceptance: 2D suite green; dispatch seam covered by tests.*
- **P3D-1. Transforms and state.** `pfft.py` with the sharded slab path
  and custom adjoint; `MHD3DState`; G1 operator gates; the
  no-all-gather HLO test; sharded-versus-single-device forward and
  gradient equality tests.
  *Acceptance: G1 at round-off on CPU and both office GPUs.*
- **P3D-2. Equations and steppers.** RHS with projection, 2N-RK3 with
  integrating factors, optional exact $B_0$ rotation, `Simulation`
  three-dimensional path, TOML and NPZ schema extensions. Gates G2, G3,
  G4. Gradient gate d(energy)/d(nu) against finite differences.
  *Acceptance: G2 to G4 in CI under x64; a 2D config diffing to a 3D
  config in one hunk.*
- **P3D-3. Linear physics.** Kinematic induction mode (frozen velocity),
  ABC dynamo gate G5, tearing gate G6 reusing the eigen tools where
  possible. First 3D docs pages: model, numerics, validation.
  *Acceptance: G5, G6 in CI; docs pages pass the prose and link gates.*
- **P3D-4. First nonlinear campaigns.** 3D Orszag--Tang G7 on the office
  GPUs with committed evidence bundles; slice-mosaic plotting; the
  `viz3d` pyvista extra with isosurface and field-line movie recipes; a
  3D hero for the gallery through the existing motion-gated media
  pipeline.
  *Acceptance: G7 bundle passes and is committed; movies pass the motion
  gate; the landing gains the 3D hero.*
- **P3D-5. Forcing and turbulence.** OU, deterministic, and
  Elsässer-controlled forcing; decay-law gate G10; forced steady
  turbulence with the exact-law gate G11; Taylor--Green dynamo G8; the
  non-universality campaign G9 as evidence bundles.
  *Acceptance: G8, G10, G11 pass; G9 bundles committed with claim levels.*
- **P3D-6. The flagship comparison.** G12: full 3D at
  $B_0/b_{\mathrm{rms}} = 1, 5, 10$ against the validated 2D results,
  same diagnostics, one campaign runner. Paper-pipeline integration and
  the 2D-versus-3D docs chapter.
  *Acceptance: G12 protocol bundle complete; the comparison chapter
  passes review gates; a manuscript-substrate section exists.*
- **P3D-7. Scale and implicit.** jaxDecomp behind the `pfft` interface
  for pencil/multi-node; SOLVAX items 1 to 6 consumed as they merge;
  implicit 3D steps and steady-state solves; strong and weak scaling
  report in the performance docs.
  *Acceptance: scaling table with recorded settings; implicit path
  converges with the same convergence-flag contract as 2D.*

Phases P3D-1 through P3D-3 are laptop-plus-CI work. P3D-4 onward needs the
office GPUs at 256-cubed and 512-cubed. Nothing in P3D-0 through P3D-3
waits on SOLVAX.

---

## 9. Publication strategy

Two natural papers fall out of the ladder:

1. **Methods and code**: differentiable spectral 3D MHD in JAX with the
   G1--G11 ladder, the distributed-FFT adjoint, and the gradient
   contract. Venue class: Computer Physics Communications or JOSS plus a
   methods preprint.
2. **Physics**: the 2D-versus-3D reduced-MHD-limit study (G12), with the
   differentiable-sensitivity angle (gradients of spectra and budgets
   with respect to $B_0$, $\nu$, $\eta$) as the novel axis beyond the
   classical protocol.

No claim ships without its gate, and the two bracketed disputes stay
bracketed in both papers.

---

## 10. Risks

1. **Distributed-FFT adjoint.** `custom_partitioning` cannot
   differentiate; the shard_map-plus-all_to_all path with a hand-written
   adjoint is the mitigation, pinned by the HLO regression test and the
   sharded-gradient equality gate.
2. **Reverse-mode cost at 512-cubed over long horizons.** Chunked remat
   is quadratic-free but doubles compute; some objectives must move to
   implicit adjoints. Budgeted in P3D-7, not on the critical path.
3. **jaxDecomp coupling.** c2c-only and version-pinned; kept strictly
   optional behind the in-house slab path until its r2c support lands.
4. **Precision.** float64 on workstation GPUs runs at a large throughput
   penalty; the policy (complex64 fields, float64 reductions, x64 gates
   in CI) must be enforced by tests, since this stack has hit silent
   precision traps twice already.
5. **SOLVAX distributed Krylov is greenfield.** The implicit 3D path
   depends on section 7 items 1, 2, 6; the explicit path does not, so
   the program cannot be blocked by solver work.

---

## 11. Out of scope

Compressible MHD, shocks, finite-volume methods, AMR, non-periodic
boundaries, Hall and kinetic extensions, and relativistic MHD. The plugin
seams stay open for all of them; none of them may dilute this program.

---

## 12. Readiness

Ready to implement now. The preconditions are met:

- the spectral operators are already dimension-agnostic (verified against
  the source);
- the office GPUs run gate-passing 256-squared campaigns today, and the
  memory budget puts 256-cubed on one A4000 and 512-cubed on the sharded
  pair;
- the benchmark ladder has paper-extracted numbers with stated
  tolerances, so every phase has an objective finish line;
- the explicit path needs zero new SOLVAX features, and the SOLVAX PR
  list is scoped for the implicit path in parallel;
- the API seam (P3D-0) is a small, safe change to land first.

The first code phase, P3D-0 plus P3D-1, can start immediately.

---

## 13. Log

### 2026-08-05 — Plan created

- Formulation, numerics, architecture, ladder, SOLVAX list, and phases
  fixed from a two-track deep dive: physics literature with
  paper-extracted gate numbers, and implementation survey of GHOST,
  spectralDNS, SNOOPY, Dedalus, jax-cfd, Exponax, astronomix, JAX-Fluids,
  and jaxDecomp, plus the JAX distributed-FFT and checkpointing state of
  the art.
- Next: P3D-0.
