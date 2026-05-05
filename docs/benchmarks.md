# Benchmarks

MHX benchmark workflows are intentionally command-line reproducible. The current
active benchmark is a FAST reduced-MHD spectral smoke run. It verifies
configuration loading, spectral operators, RK4 time integration, diagnostics,
output schema, figures, GIF generation, reports, and validation checks.

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
[validation page](validation.md).

## Analytic reconnection scaling gates

The benchmark roadmap also includes analytic power-law gates for the expected
FKR, Sweet-Parker plasmoid, and ideal-tearing exponents:

```bash
mhx benchmark scaling --outdir outputs/benchmarks/reconnection_scaling
mhx benchmark fkr-window --outdir outputs/benchmarks/fkr_window
mhx benchmark linearized-rhs --outdir outputs/benchmarks/linearized_rhs
mhx benchmark reduced-mhd-eigenmode --outdir outputs/benchmarks/reduced_mhd_eigenmode
mhx benchmark cosine-equilibrium-linearization --outdir outputs/benchmarks/cosine_equilibrium_linearization
mhx benchmark current-sheet-eigenvalue --outdir outputs/benchmarks/periodic_current_sheet_eigenvalue
mhx benchmark diffusion-eigenvalue --outdir outputs/benchmarks/diffusion_eigenvalue
mhx benchmark power-iteration --outdir outputs/benchmarks/power_iteration
mhx benchmark arnoldi --outdir outputs/benchmarks/arnoldi
mhx benchmark catalog --outdir outputs/benchmarks/catalog
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

The catalog command writes:

- `benchmark_catalog.json`
- `benchmark_catalog.md`
- `manifest.json`

## CI artifacts

Every push runs a `benchmark-artifacts` CI job. It executes deterministic FAST
pipelines:

```bash
mhx benchmark run --config examples/linear_tearing.toml --outdir outputs/ci/linear_tearing_fast --gif
mhx benchmark validate outputs/ci/linear_tearing_fast
mhx benchmark decay --outdir outputs/ci/resistive_decay
mhx benchmark scaling --outdir outputs/ci/reconnection_scaling
mhx benchmark fkr-window --outdir outputs/ci/fkr_window
mhx benchmark linearized-rhs --outdir outputs/ci/linearized_rhs
mhx benchmark reduced-mhd-eigenmode --outdir outputs/ci/reduced_mhd_eigenmode
mhx benchmark cosine-equilibrium-linearization --outdir outputs/ci/cosine_equilibrium_linearization
mhx benchmark current-sheet-eigenvalue --outdir outputs/ci/periodic_current_sheet_eigenvalue
mhx benchmark diffusion-eigenvalue --outdir outputs/ci/diffusion_eigenvalue
mhx benchmark power-iteration --outdir outputs/ci/power_iteration
mhx benchmark arnoldi --outdir outputs/ci/arnoldi
mhx benchmark timing --outdir outputs/ci/timing --repeats 1 --warmups 0
mhx benchmark catalog --outdir outputs/ci/catalog
mhx validate all --outdir outputs/ci/validation_suite
mhx run examples/linear_tearing_twofluid_toy.toml --outdir outputs/ci/twofluid_toy
mhx figures outputs/ci/twofluid_toy --gif
mhx report outputs/ci/twofluid_toy
mhx run examples/linear_tearing_plugin_demo.toml --outdir outputs/ci/plugin_demo
mhx figures outputs/ci/plugin_demo --gif
mhx report outputs/ci/plugin_demo
mhx artifact-manifest outputs/ci
```

The job uploads `outputs/ci` as the `mhx-fast-artifacts` GitHub Actions
artifact. Reviewers can download it to inspect manifests, reports, PNG figures,
GIF movies, and `artifact_manifest.json` checksums generated from the exact
commit under test.

The catalog file `outputs/ci/catalog/benchmark_catalog.md` gives reviewers a
single table of active FAST gates, schemas, commands, and expected artifacts.

For a single reviewer-facing pass/fail bundle, run:

```bash
mhx validate all --outdir outputs/validation_suite
```

This command executes the active deterministic FAST validation gates and writes:

- `outputs/validation_suite/validation_suite.json`
- `outputs/validation_suite/validation_suite.md`
- `outputs/validation_suite/artifact_manifest.json`
- one subdirectory per validation case, each with its own `validation.json`
  and `manifest.json`.

## FAST timing benchmark

MHX records performance as an artifact, not as a brittle absolute CI gate:

```bash
mhx benchmark timing --outdir outputs/benchmarks/timing --repeats 3 --warmups 1
```

This measures wall-clock time and Python allocation peaks for:

- `linear_tearing_fast`: the active reduced-MHD smoke run.
- `resistive_decay_fast`: the exact single-mode resistive validation case.
- `reconnection_scaling`: analytic FKR/plasmoid/ideal-tearing scaling gates.

The command writes:

- `timing.json`
- `timing.md`
- `figures/timing_summary.png`
- `manifest.json`

The CI job uses `--repeats 1 --warmups 0` to keep runtime low and uploads the
result in `outputs/ci/timing/`. Timing comparisons should be made between
artifacts from the same runner class; MHX does not currently fail CI on
absolute speed.

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
[Loureiro, Schekochihin, and Cowley 2007](https://arxiv.org/abs/astro-ph/0703631),
and [Pucci and Velli ideal tearing context](https://arxiv.org/abs/1704.08793).
