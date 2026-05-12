# Paper plan and claim boundaries

This page is the working blueprint for a complete MHX methods/validation paper.
It is intentionally conservative: it separates claims already supported by CI
artifacts from nonlinear reconnection claims that still require long production
runs.

## Proposed paper title

**MHX: a validation-first differentiable reduced-MHD framework for magnetic
reconnection, physics plugins, and neural surrogate experiments**

## Core contributions

1. A pure-JAX reduced-MHD solver stack with explicit spectral operators,
   deterministic fixed-step integration, x64 validation gates, and artifact
   schemas.
2. A public diagnostics API with shared definitions for energy, Fourier-mode
   growth, divergence error, reconnecting-flux amplitude, and Rutherford-style
   island-width proxies.
3. Literature-anchored tearing/reconnection benchmarks: exact resistive decay,
   FKR regime formulas, Harris $\Delta'$ matching, direct Harris eigenvalues,
   finite-domain dispersion, eigenfunction-layer localization, and time-domain
   eigenmode replay.
4. A physics-plugin interface for Hall, hyper-resistive, anisotropic-pressure,
   two-fluid, and local user-defined terms without changing solver internals.
5. Differentiability gates for nonlinear trajectory maps, establishing the
   local tangent checks required before inverse-design or neural-ODE claims.
6. A reproducible paper pipeline with figures, GIFs, manifests, checksums, and
   CI artifact validation.

## Current claim status

| Claim | Current status | Required before paper claim |
| --- | --- | --- |
| Spectral derivative signs and diffusion rates | Supported by exact Fourier tests and exact-decay benchmark. | Maintain x64 regression gates. |
| Linear tearing eigenvalue at one Harris reference case | Supported against the $S=1000$, $ka=0.5$ benchmark value $\gamma\simeq0.0131$. | Add medium-grid reproducibility run and artifact hash bundle. |
| Finite-domain tearing dispersion branch | Supported as a small sampled branch/residual gate. | Extend to a documented FKR/Coppi parameter sweep. |
| Nonlinear reduced-MHD budget | Supported for a periodic multi-mode state. | Add longer current-sheet runs with same budget diagnostics. |
| Rutherford island growth | Not yet supported. | Run long enough for many linear e-folds, then track $W(t)$ and compare algebraic growth. |
| Sweet-Parker/plasmoid chains | Not yet supported. | Resolve long thin sheets and secondary islands at adequate Lundquist number. |
| Neural ODE surrogate for reconnection metrics | Deterministic FAST dataset/split/baseline/calibration artifacts implemented; no trainable surrogate claim yet. | Train a latent/neural ODE against the frozen `mhx.neural_ode.dataset.v1` contract and beat the documented baselines. |

## New large-push lanes

Two validation lanes have been added with explicit claim boundaries:

| Lane | Current status | Source-code boundary |
| --- | --- | --- |
| `seed_robust_qi.md` | Implemented as FAST seed-ensemble validation for smooth perturbation sensitivity. It is not production UQ. | QI implementation lives in [seed_robust_qi.py](https://github.com/uwplasma/MHX/blob/main/src/mhx/benchmarks/seed_robust_qi.py); deterministic initial conditions remain in [equilibria.py](https://github.com/uwplasma/MHX/blob/main/src/mhx/physics/equilibria.py#L35). |
| FAST Rutherford campaign runner | Implemented for validation-grade histories and schema checks. It is not a long production campaign. | Template writing lives in [campaigns.py](https://github.com/uwplasma/MHX/blob/main/src/mhx/benchmarks/campaigns.py#L42); FAST runner lives in [campaign_runner.py](https://github.com/uwplasma/MHX/blob/main/src/mhx/benchmarks/campaign_runner.py). |
| `neural_ode_reproducibility.md` | Implemented as deterministic FAST dataset, train/validation/test splits, no-training baselines, and residual calibration. It is not a trained neural surrogate. | Dataset and baseline implementation lives in [reproducibility.py](https://github.com/uwplasma/MHX/blob/main/src/mhx/neural_ode/reproducibility.py). |

Both lanes are exposed through the CLI in
[main.py](https://github.com/uwplasma/MHX/blob/main/src/mhx/cli/main.py).

The production operations guide now exists as [campaign_runner.md](campaign_runner.md).
It documents the FAST runner, production plan, duration gate, checkpoint/resume
metadata, required production artifacts, and review questions. The actual
expensive long-run executor is still planned, so paper text should say "FAST
campaign validation" or "production campaign scaffold" rather than "production
campaign result" until a completed production artifact bundle exists.

## Nonlinear duration requirement

For a linear eigenmode with growth rate $\gamma$, observing $N_e$ e-folds
requires

$$
t_\mathrm{end} \ge \frac{N_e}{\gamma}.
$$

The direct Harris benchmark used by MHX has $\gamma\simeq0.0131$ at
$S=1000$, $ka=0.5$. Ten e-folds therefore require

$$
t_\mathrm{end} \simeq \frac{10}{0.0131} \approx 763.4.
$$

The current nonlinear energy-budget gate runs to $t=0.8$, which is only about
$1.0\times 10^{-3}$ of this ten-e-fold window. It is a nonlinear code-validity
gate, not an island-growth or plasmoid result.

```bash
mhx benchmark nonlinear-duration-audit \
  --outdir outputs/benchmarks/nonlinear_duration_audit
mhx benchmark duration-policy \
  --outdir outputs/benchmarks/duration_policy
mhx campaign rutherford-template \
  --outdir outputs/campaigns/rutherford_template
```

![Nonlinear duration audit](_static/validation/nonlinear_duration_audit/nonlinear_duration_audit.png)

## Proposed figure set

| Figure | Contents | Status |
| --- | --- | --- |
| 1 | Architecture: configs → model registry → solver → diagnostics → artifacts. | Draft in docs/architecture. |
| 2 | Exact decay and spectral-operator identities. | CI artifact exists. |
| 3 | FKR, plasmoid, and ideal-tearing analytic scaling targets. | CI artifact exists; analytic only. |
| 4 | Harris $\Delta'$ and direct eigenvalue benchmark. | CI artifact exists. |
| 5 | Finite-domain dispersion and eigenfunction layer localization. | CI artifact exists. |
| 6 | Time-domain eigenmode replay and growth-fit recovery. | CI artifact exists. |
| 7 | Nonlinear differentiability and energy-budget gates. | CI artifact exists. |
| 8 | Nonlinear duration audit and production-run requirements. | CI artifact exists. |
| 9 | Rutherford island growth campaign. | Duration-guarded template exists; production run still planned. |
| 10 | Sweet-Parker/plasmoid nonlinear campaign. | Planned. |
| 11 | Neural-ODE dataset/baselines/calibration/failure cases. | Implemented as a deterministic no-training reproducibility lane in [neural_ode_reproducibility.md](neural_ode_reproducibility.md). |

The reviewer-facing figure readiness table is maintained in
[publication_checklist.md](publication_checklist.md). That page is the preferred
place to decide whether a plot can be promoted from validation/demo status to
paper evidence.

## Production nonlinear campaign checklist

Every nonlinear island/plasmoid result should archive:

- full config, code commit, API version, dependency lock file, and manifest;
- a passing duration-policy check for the declared growth rate and e-fold count;
- x64 setting and JIT setting;
- grid, timestep, CFL or fixed-step stability rationale, and tolerances;
- reconnected flux $\psi_1(t)$ and island width
  $W=4\sqrt{|\psi_1|/|B_y'(0)|}$;
- reconnection proxy $E_\mathrm{rec}$, current-sheet length/thickness, and aspect ratio;
- magnetic, kinetic, total energy, resistive dissipation, viscous dissipation, and budget residual;
- at least one resolution/time-step comparison;
- visual flux/current movies with fixed color ranges;
- explicit claim boundary: smoke, validation, or production physics result.
- generated `claim_level` metadata in every manifest and artifact manifest.

## Literature anchors

- [Furth, Killeen & Rosenbluth (1963)](https://doi.org/10.1063/1.1706761) for constant-$\psi$ tearing theory.
- [Rutherford (1973)](https://doi.org/10.1063/1.1694232) for nonlinear tearing/island growth.
- [Loureiro, Schekochihin & Cowley (2007)](https://arxiv.org/abs/astro-ph/0703631) for Sweet-Parker plasmoid instability scalings.
- [Pucci & Velli (2014)](https://doi.org/10.1088/2041-8205/780/2/L19) for ideal tearing.
- [MacTaggart Harris benchmark PDF](https://eprints.gla.ac.uk/191898/1/191898.pdf) for the direct Harris-sheet eigenvalue anchor used in MHX.
