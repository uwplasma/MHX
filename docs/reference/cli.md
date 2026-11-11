# Benchmark commands

Run `mhx --help` for the full command tree. This page documents the
benchmark and validation workflows, which are the largest command group.

MHX benchmark workflows are intentionally command-line reproducible. The active
catalog now includes a FAST reduced-MHD smoke run, exact linear physics gates,
Harris tearing eigenvalue checks, nonlinear energy-budget gates, and
Orszag--Tang and turbulence media.

```bash
mhx benchmark run \
  --config examples/linear_tearing.toml \
  --outdir outputs/benchmarks/linear_tearing_fast \
  --gif
mhx benchmark validate outputs/benchmarks/linear_tearing_fast
```

Expected files include:

- `manifest.json`
- `diagnostics.json`
- `trajectory.npz`
- `figures/energy_history.png`
- `figures/flux_final.png`
- `figures/mode_amplitude.png`
- `figures/flux_movie.gif`
- `report.json`
- `report.md`
- `validation.json`

## Detailed command index

Use this section as the detailed command catalog that the README intentionally
does not carry. Most benchmark commands write `claim_level = "validation"`;


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
mhx benchmark current-sheet-timedomain --outdir outputs/benchmarks/periodic_current_sheet_timedomain
mhx benchmark current-sheet-nonlinear-bridge --outdir outputs/benchmarks/periodic_current_sheet_nonlinear_bridge
mhx benchmark double-harris-growth --outdir outputs/benchmarks/periodic_double_harris_nonlinear_growth
mhx benchmark double-harris-long-run --outdir outputs/benchmarks/periodic_double_harris_seeded_long_run --movies
mhx benchmark double-harris-convergence --outdir outputs/benchmarks/periodic_double_harris_convergence
mhx benchmark nonlinear-energy-budget --outdir outputs/benchmarks/nonlinear_energy_budget
mhx benchmark orszag-tang --outdir outputs/benchmarks/orszag_tang_vortex --movies
mhx benchmark decaying-turbulence --outdir outputs/benchmarks/decaying_mhd_turbulence --movies
mhx benchmark forced-turbulent-reconnection --outdir outputs/benchmarks/forced_turbulent_reconnection --movies
mhx benchmark forced-turbulent-reconnection-readiness-check outputs/benchmarks/forced_turbulent_reconnection
mhx benchmark diffusion-eigenvalue --outdir outputs/benchmarks/diffusion_eigenvalue
mhx benchmark power-iteration --outdir outputs/benchmarks/power_iteration
mhx benchmark arnoldi --outdir outputs/benchmarks/arnoldi
python examples/run_orszag_tang.py --outdir outputs/examples/orszag_tang --nx 64 --ny 64 --t-end 6
```

The equation-heavy physics gates and still figures are on
[validation.md](../validation/index.md). 
## Orszag--Tang nonlinear vortex example

```bash
mhx benchmark orszag-tang \
  --outdir outputs/benchmarks/orszag_tang_vortex \
  --nx 64 --ny 64 --t-end 6 --save-every 40 --movies
```

Expected files include:

- `manifest.json`
- `diagnostics.json`
- `validation.json`
- `orszag_tang_vortex.npz`
- `figures/orszag_tang_summary.png`
- `figures/orszag_tang_flux.gif`
- `figures/orszag_tang_current.gif`
- `figures/orszag_tang_vorticity.gif`

This is an incompressible reduced-MHD adaptation of the
[Orszag--Tang 1979](https://doi.org/10.1017/S002211207900210X) vortex. It is
intended as a nonlinear morphology, energy-decay, divergence, and high-$k$
cascade gate, not as a compressible full-MHD shock benchmark.

## Decaying turbulence and forced turbulent reconnection

```bash
mhx benchmark decaying-turbulence \
  --outdir outputs/benchmarks/decaying_mhd_turbulence \
  --nx 64 --ny 64 --t-end 8 --save-every 20 --movies

mhx benchmark forced-turbulent-reconnection \
  --outdir outputs/benchmarks/forced_turbulent_reconnection \
  --nx 64 --ny 64 --t-end 80 --save-every 100 --movies
```

The decaying case starts from a deterministic band-limited reduced-MHD state
and gates finite fields, dissipative total energy, current-density growth, and
high-wavenumber transfer. The forced case adds a periodic current sheet,
broadband perturbations, weak large-scale vorticity forcing, an X/O-aware
reconnection proxy when critical points are found, and a conservative fallback
proxy when they are not.

Expected files include:

- `diagnostics.json`
- `validation.json`
- `decaying_mhd_turbulence.npz` or `forced_turbulent_reconnection.npz`
- `figures/decaying_mhd_turbulence_summary.png` or
  `figures/forced_turbulent_reconnection_summary.png`
- optional `figures/*_flux.gif` and `figures/*_current.gif` with `--movies`

These examples are anchored to 2-D MHD turbulence/current-sheet studies and
turbulent reconnection ideas, but they remain validation media. They do not
claim 3-D Lazarian--Vishniac fast reconnection or converged turbulence
statistics.

## Exact-decay validation benchmark

The first physics-gated numerical validation is a single Fourier mode in the
resistive induction limit:

```bash
mhx benchmark decay --outdir outputs/benchmarks/resistive_decay
```

This checks the literature-standard diffusion law
$\psi_k(t)=\psi_k(0)\exp(-\eta |k|^2t)$ and
$E_B(t)=E_B(0)\exp(-2\eta |k|^2t)$. It writes:

- `diagnostics.json`
- `validation.json`
- `decay_history.npz`
- `figures/decay_amplitude.png`
- `figures/decay_energy.png`
- `figures/decay_relative_error.png`

The same benchmark is documented with figures on the
[validation page](../validation/index.md).

## Reconnection and linear-operator gates

The benchmark roadmap includes analytic power-law gates for expected FKR,
Sweet-Parker plasmoid, and ideal-tearing exponents plus linear operator and
direct eigenvalue gates:

```bash
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
mhx benchmark current-sheet-timedomain --outdir outputs/benchmarks/periodic_current_sheet_timedomain
mhx benchmark current-sheet-nonlinear-bridge --outdir outputs/benchmarks/periodic_current_sheet_nonlinear_bridge
mhx benchmark double-harris-growth --outdir outputs/benchmarks/periodic_double_harris_nonlinear_growth
mhx benchmark double-harris-long-run --outdir outputs/benchmarks/periodic_double_harris_seeded_long_run
mhx benchmark double-harris-convergence --outdir outputs/benchmarks/periodic_double_harris_convergence
mhx benchmark nonlinear-energy-budget --outdir outputs/benchmarks/nonlinear_energy_budget
mhx benchmark diffusion-eigenvalue --outdir outputs/benchmarks/diffusion_eigenvalue
mhx benchmark power-iteration --outdir outputs/benchmarks/power_iteration
mhx benchmark arnoldi --outdir outputs/benchmarks/arnoldi
```

This writes:

- `diagnostics.json`
- `validation.json`
- `scaling_history.npz`
- `figures/fkr_scaling.png`
- `figures/plasmoid_scaling.png`
- `figures/ideal_tearing_scaling.png`

The FKR-window command writes:

- `diagnostics.json`
- `validation.json`
- `fkr_window.npz`
- `figures/fkr_constant_psi_window.png`

The FKR growth-rate command writes:

- `diagnostics.json`
- `validation.json`
- `fkr_growth_rate.npz`
- `figures/fkr_growth_rate.png`

The Harris Delta-prime command writes:

- `diagnostics.json`
- `validation.json`
- `harris_delta_prime.npz`
- `figures/harris_delta_prime.png`

The direct Harris-sheet tearing eigenvalue command writes:

- `diagnostics.json`
- `validation.json`
- `linear_tearing_eigenvalue.npz`
- `figures/linear_tearing_eigenvalue.png`

The finite-domain tearing dispersion command writes:

- `diagnostics.json`
- `validation.json`
- `linear_tearing_dispersion.npz`
- `figures/linear_tearing_dispersion.png`

The Harris eigenfunction-layer command writes:

- `diagnostics.json`
- `validation.json`
- `linear_tearing_layer.npz`
- `figures/linear_tearing_layer.png`

The time-domain Harris eigenmode replay command writes:

- `diagnostics.json`
- `validation.json`
- `linear_tearing_timedomain.npz`
- `figures/linear_tearing_timedomain.png`

The linearized-RHS command writes:

- `diagnostics.json`
- `validation.json`
- `linearized_rhs.npz`
- `figures/linearized_rhs_errors.png`

The reduced-MHD eigenmode command writes:

- `diagnostics.json`
- `validation.json`
- `reduced_mhd_linear_eigenmode.npz`
- `figures/reduced_mhd_linear_eigenmode_errors.png`

The cosine-equilibrium linearization command writes:

- `diagnostics.json`
- `validation.json`
- `cosine_equilibrium_linearization.npz`
- `figures/cosine_equilibrium_linearization_errors.png`

The periodic current-sheet eigenvalue command writes:

- `diagnostics.json`
- `validation.json`
- `periodic_current_sheet_eigenvalue.npz`
- `figures/periodic_current_sheet_spectrum.png`

The periodic current-sheet time-domain replay command writes:

- `diagnostics.json`
- `validation.json`
- `periodic_current_sheet_timedomain.npz`
- `figures/periodic_current_sheet_timedomain.png`

The nonlinear current-sheet differentiability bridge command writes:

- `diagnostics.json`
- `validation.json`
- `periodic_current_sheet_nonlinear_bridge.npz`
- `figures/periodic_current_sheet_nonlinear_bridge.png`

The periodic double-Harris nonlinear-growth command writes:

- `diagnostics.json`
- `validation.json`
- `periodic_double_harris_nonlinear_growth.npz`
- `figures/periodic_double_harris_nonlinear_growth.png`

The periodic double-Harris seeded long-run command writes:

- `diagnostics.json`
- `validation.json`
- `periodic_double_harris_seeded_long_run.npz`, including
  `reconnected_flux`, `seed_mode_reconnected_flux`,
  `rutherford_island_width`, dominant low-mode indices, X/O counts, energies,
  and saved base/seeded fields
- `figures/periodic_double_harris_seeded_long_run.png`
- optional `figures/periodic_double_harris_flux.gif` and
  `figures/periodic_double_harris_current.gif` with `--movies`

The periodic double-Harris convergence command writes:

- `diagnostics.json`
- `validation.json`
- `periodic_double_harris_convergence.npz`
- `figures/periodic_double_harris_convergence.png`

The periodic double-Harris parameter-sweep command writes:

- `diagnostics.json`
- `validation.json`
- `periodic_double_harris_parameter_sweep.npz`
- `figures/periodic_double_harris_parameter_sweep.png`

The periodic double-Harris promotion checker writes:

- `promotion_readiness.json`
- `validation.json`
- `figures/promotion_matrix.png`
- `manifest.json` and `artifact_manifest.json`

The nonlinear energy-budget command writes:

- `diagnostics.json`
- `validation.json`
- `nonlinear_energy_budget.npz`
- `figures/nonlinear_energy_budget.png`

The diffusion-eigenvalue command writes:

- `diagnostics.json`
- `validation.json`
- `diffusion_eigenvalue.npz`
- `figures/diffusion_eigenvalue_errors.png`

The power-iteration command writes:

- `diagnostics.json`
- `validation.json`
- `power_iteration_history.npz`
- `figures/power_iteration_history.png`

The Arnoldi command writes:

- `diagnostics.json`
- `validation.json`
- `arnoldi_spectrum.npz`
- `figures/arnoldi_ritz_values.png`

## Theory scaffolds

MHX includes analytic scaling estimates for benchmark planning and reports. They
are not replacements for calibrated eigenvalue calculations.

For the constant-$\psi$ Furth-Killeen-Rosenbluth tearing regime, MHX uses the
Harris-sheet outer-region proxy

$$
\Delta' a = 2\left[(ka)^{-1} - ka\right],
$$

and the order-unity-coefficient-free scaling

$$
\gamma \tau_a \sim S_a^{-3/5}(ka)^{2/5}(\Delta'a)^{4/5}.
$$

For Sweet-Parker plasmoid estimates, MHX includes the Loureiro scaling

$$
\gamma_{\max}\tau_A \sim S^{1/4}, \qquad k_{\max}L \sim S^{3/8}.
$$

For ideal tearing planning, MHX includes the Pucci-Velli aspect-ratio scaling

$$
a/L \sim S^{-1/3}.
$$

The separate FKR-window gate checks the sampled constant-$\psi$ regime by
plotting $\gamma\tau_a$ and $\Delta'\delta$ versus $ka$ at fixed $S_a$. This
prevents future numerical tearing comparisons from silently mixing FKR and
large-$\Delta'$ Coppi regimes.

The Harris Delta-prime gate numerically integrates the ideal outer tearing
equation for a Harris sheet and recovers
$\Delta'a=2[(ka)^{-1}-ka]$. It validates the outer-region target but still does
not solve the resistive FKR inner-layer eigenvalue problem.

The direct Harris-sheet tearing eigenvalue gate solves a finite-difference
linear reduced-MHD eigenproblem for $S=1000$, $ka=0.5$, and $d/a=10$. It checks
the unstable tearing eigenvalue against a published growth rate
$\gamma\simeq0.0131$, verifies grid extrapolation in $\Delta x^2$, checks the
dense eigenpair residual, confirms tearing parity, and verifies a stable
$ka=1.2$ control has no positive-growth eigenvalue. This is a single reference
eigenproblem, not yet a full FKR/Coppi dispersion scan.

The finite-domain tearing dispersion gate repeats the same eigenproblem over a
small $ka$ scan, verifies positive growth for sampled $0<ka<1$, verifies stable
controls above $ka=1$, checks dense eigenpair residuals, and retains the
$ka=0.5$ literature anchor. This is the first dispersion-shaped tearing gate,
but it is still FAST and finite-domain rather than a production asymptotic
FKR/Coppi study.

The Harris eigenfunction-layer gate repeats the direct eigenproblem over a
small $S$ scan and measures half-maximum widths of the flux, flow, and current
perturbation profiles. It gates monotonic flow-layer narrowing, outer-flux
width stability, broad fitted-slope ranges, and eigen residuals. This is a
shape/localization validation, not an asymptotic exponent claim.

The time-domain Harris eigenmode replay gate integrates the same discrete
operator as $\dot q=Lq$ from the selected tearing eigenvector. It fits
$\log\|q(t)\|_2$ and gates the recovered growth rate against the dense
eigenvalue, while also checking RK4 amplitude error and final mode-shape
alignment. This closes the diagnostic loop between eigenvalue selection and
growth fitting, but remains a linear finite-domain validation rather than a
nonlinear island-growth claim.

The linearized-RHS gate compares JAX's matrix-free Jacobian-vector product
against a centered finite difference of the reduced-MHD RHS. This is the
operator-level scaffold for later calibrated tearing eigenmode benchmarks.

The reduced-MHD eigenmode gate applies the flattened JVP operator at the
zero-state linear limit and checks the analytic resistive and viscous Fourier
diffusion eigenvalues for the $\psi$ and $\omega$ blocks.

The cosine-equilibrium linearization gate moves the same JVP machinery to a
nonzero current-sheet equilibrium $\psi_0=A\cos y$. It checks two exact
Poisson-bracket couplings: flow advection of equilibrium flux and magnetic
tension from perturbed flux. This is a physics gate for the current-sheet terms
that calibrated FKR eigenmode benchmarks will use next.

The periodic current-sheet eigenvalue gate then assembles the tiny dense
spectrum of that same nonzero-equilibrium operator. It verifies mean/gauge
modes, checks the selected dense eigenpair residual, and fails if the
non-gauge spectrum has spurious positive growth. This remains a periodic
operator-stability gate, not a calibrated FKR/Coppi growth-rate benchmark.

The periodic current-sheet time-domain replay gate selects a real decaying
eigenmode of the same dense JVP, advances $dq/dt=Lq$ with RK4, and compares
the whole state vector to $q(t)=\exp(\lambda t)q(0)$. This catches errors
between operator assembly, time stepping, and decay/growth fitting before
moving the workflow to nonlinear perturbation studies.

The nonlinear current-sheet bridge differentiates the complete RK4 saved
trajectory map with JAX, then compares it to centered finite differences of
full nonlinear trajectories. The expected $O(\epsilon^2)$ convergence is an
evidence gate for differentiable programming claims; it is the step
between linear operator replay and future adjoint/inverse-design workflows.

The nonlinear energy-budget gate advances a multi-mode nonlinear reduced-MHD
state and checks
$dE/dt=-\eta\langle j^2\rangle-\nu\langle\omega^2\rangle$ by comparing the
measured total energy to integrated resistive/viscous dissipation. This is now
the primary FAST nonlinear PDE consistency gate, but it is still not a
Rutherford, Sweet--Parker, or plasmoid validation.

The diffusion-eigenvalue gate validates the Rayleigh-quotient/residual path on a
known Fourier eigenpair before using the same matrix-free machinery for tearing
operators.

The power-iteration gate validates a minimal dominant-eigenpair iteration on a
known diagonal matrix-free operator. This keeps the eigensolver control path
tested independently before later coupling to reduced-MHD tearing operators.

The Arnoldi gate validates a small Krylov Ritz-spectrum path on a known
non-normal upper-triangular operator. It is the direct scaffold for future
matrix-free tearing eigenmode calculations where the linearized reduced-MHD JVP
is too large to assemble explicitly.

References used for the benchmark roadmap include
[Furth, Killeen, and Rosenbluth 1963](https://cir.nii.ac.jp/crid/1363107370207531008),
[MacTaggart 2019](https://eprints.gla.ac.uk/191898/),
[MacTaggart and Stewart 2017](https://www.maths.gla.ac.uk/~dmactaggart/papers/dmac17c.pdf),
[Loureiro, Schekochihin, and Cowley 2007](https://arxiv.org/abs/astro-ph/0703631),
and [Pucci and Velli ideal tearing context](https://arxiv.org/abs/1704.08793).
