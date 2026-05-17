# Skeptical validation audit

This page records the current scientific maturity of the rebuilt MHX package.
It is intentionally conservative: a passing CI badge means the documented gates
passed, not that MHX is already a publication-grade nonlinear reconnection
solver.

## Audit result

| Area | Current status | Research-grade? |
| --- | --- | --- |
| Spectral derivatives and Laplacian signs | Tested by exact Fourier identities, exact resistive decay, and matrix-free eigenvalue gates. | Yes for smooth periodic FAST fixtures. |
| RK4 smoke integration | Runs deterministically and preserves expected dissipative trends in tiny examples. | Engineering-grade only. |
| Exact resistive decay | Matches $\psi_k(t)=\psi_k(0)e^{-\eta |k|^2t}$ and $E_B(t)=E_B(0)e^{-2\eta |k|^2t}$ with x64 validation. | Yes for this linear limit. |
| Analytic FKR/plasmoid/ideal-tearing scaling plots | Reproduce expected literature exponents from formulas. | Yes as analytic scaffolds; no PDE-solver claim. |
| Harris-sheet $\Delta'$ outer solve | Numerically integrates the ideal outer tearing ODE and recovers $\Delta'a=2[(ka)^{-1}-ka]$. | Yes for outer-region matching; not a growth-rate eigenvalue. |
| FKR growth-rate assembly | Propagates numerical Harris $\Delta'$ into $\gamma\tau_a\sim S_a^{-3/5}(ka)^{2/5}(\Delta'a)^{4/5}$ and gates both exponents. | Yes as an asymptotic growth-rate target; not a direct eigenvalue solve. |
| Direct Harris-sheet tearing eigenvalue | Solves the 1D reduced-MHD Harris eigenproblem at $S=1000$, $ka=0.5$, $d/a=10$ and gates against $\gamma\simeq0.0131$. | Yes for this single reference eigenproblem; not yet a full FKR/Coppi scan. |
| Finite-domain tearing dispersion | Scans the same eigenproblem over sampled $ka$, checks the unstable band below $ka=1$, stable controls above $ka=1$, residuals, and the $ka=0.5$ anchor. | Useful FAST dispersion gate; not yet an asymptotic Lundquist-number scan. |
| Harris eigenfunction layer shape | Scans the direct eigenproblem over $S$, checks flow-layer narrowing, outer-flux width stability, broad fitted-slope ranges, and residuals. | Useful eigenfunction-localization gate; not an asymptotic exponent validation. |
| Time-domain Harris eigenmode replay | Integrates the same direct eigenvector under $\dot q=Lq$ and refits $\gamma$ from $\log\|q(t)\|_2$. | Yes for linear growth fitting; not a nonlinear island-growth validation. |
| Matrix-free JVP and eigen scaffolds | JAX JVP, zero-state eigenmodes, Arnoldi, and power iteration are tested on controlled fixtures. | Good scaffolding for larger matrix-free tearing spectra. |
| Nonzero current-sheet linearization | Exact bracket couplings around $\psi_0=A\cos y$ are tested. | Good operator gate; still not an FKR benchmark. |
| Periodic current-sheet dense spectrum | Tiny dense spectrum around $\psi_0=A\cos y$ checks gauge modes, eigenpair residuals, and absence of spurious positive growth. | Useful stability/operator gate; not an FKR/Coppi tearing-growth validation. |
| Periodic current-sheet time-domain replay | A real decaying eigenmode of the same JVP is advanced with RK4 and compared to $q(t)=e^{\lambda t}q(0)$. | Good operator/time-step bridge; not a nonlinear reconnection validation. |
| Nonlinear current-sheet differentiability bridge | JAX JVP of the nonlinear RK4 trajectory map is compared with centered finite differences and gates $O(\epsilon^2)$ convergence. | Good differentiability gate for inverse design and neural ODE data; not a nonlinear reconnection result. |
| Nonlinear reduced-MHD energy budget | A multi-mode nonlinear periodic state is advanced and checked against $dE/dt=-\eta\langle j^2\rangle-\nu\langle\omega^2\rangle$. | Yes for this nonlinear conservation/dissipation identity; not an island-growth or plasmoid result. |
| Nonlinear duration audit | Current nonlinear FAST durations are compared with Harris linear e-fold and Rutherford/plasmoid target windows. | Yes as a claim-boundary gate; it proves the current nonlinear runs are too short for nonlinear-island/plasmoid claims. |
| Duration policy | Current short runs and future production templates are checked against explicit e-fold requirements. | Yes as an enforceable guard; production physics still needs long actual runs and convergence. |
| Orszag--Tang and turbulence media | Reduced-MHD Orszag--Tang, decaying turbulence, and forced turbulent-reconnection examples run with finite-field, energy, high-$k$, and reconnection-proxy gates. | Yes as validation media; not as compressible MHD, 3-D turbulence, or production reconnection statistics. |
| FAST reduced-MHD run | Produces stable outputs, diagnostics, figures, and GIFs. Kinetic energy remains tiny and mode amplitudes change weakly. | Smoke test only. |
| Two-fluid and plugin examples | Exercise extension paths and output schemas. | API examples only; not validated extended-MHD physics. |
| Nonlinear tearing/plasmoid dynamics | Not demonstrated by the current FAST runs. | No. |
| Neural ODE / inverse design | Deterministic dataset, baselines, calibration, and fitted latent-ODE FAST workflow exist. | Validation protocol only; not production surrogate or inverse-design evidence. |

## Plot audit notes

The current validation plots are internally consistent. The exact-decay plot is
the cleanest linear PDE result: numerical and analytic amplitude/energy curves
overlap, and relative errors remain near roundoff in x64. The Harris $\Delta'$
and FKR growth-rate plots are useful tearing-theory gates: they validate
outer-region matching and asymptotic exponent assembly. The direct
Harris-sheet eigenvalue plot is the strongest tearing-specific point result so
far: the finite-grid growth rates converge linearly in $\Delta x^2$ toward the
published $\gamma\simeq0.0131$ reference, the selected dense eigenpair residual
is near roundoff, and the eigenfunction parity is correct. The finite-domain
dispersion plot adds a useful branch-level check: sampled $0<ka<1$ points grow,
sampled $ka>1$ controls are stable/oscillatory, and residuals remain near
roundoff. The eigenfunction-layer plot checks a separate failure mode:
the selected branch has a localized flow/current response near the resonant
surface, and the flow layer narrows as $S$ rises while the outer flux width
stays nearly fixed. The time-domain replay plot is a diagnostic-loop check: the RK4
amplitude follows $\exp(\gamma t)$ and the fitted growth rate matches the
eigenvalue, so future nonlinear growth fits have a calibrated linear baseline.
The periodic current-sheet time-domain plot is a separate solver-operator
bridge: it verifies that the periodic JVP spectrum, RK4 replay, and exponential
decay diagnostics agree on the same selected eigenmode.
The nonlinear bridge plot is a differentiability audit: centered finite
differences of complete nonlinear trajectories converge to the JAX trajectory
JVP at second order, so the code has a defensible local tangent before adjoint
or neural-ODE claims are made.
The nonlinear energy-budget plot is the strongest current nonlinear PDE check:
the full nonlinear trajectory dissipates total energy at the rate predicted by
the reduced-MHD theorem, with the integrated residual below the documented gate.
It tests bracket cancellation and dissipative signs, but it does not prove
tearing-island growth, Rutherford saturation, or plasmoid onset.
The Orszag--Tang, decaying-turbulence, and forced-turbulent-reconnection media
are useful nonlinear reduced-MHD exercises: they show current filamentation,
high-wavenumber transfer, bounded energy behavior, and a reconnection proxy
where appropriate. They are not calibrated compressible-MHD, 3-D turbulence,
or production reconnection-rate results.
The nonlinear duration-audit plot is deliberately skeptical. It shows that the
default FAST nonlinear budget gate reaches only $t=0.8$, whereas ten e-folds of
the direct Harris-sheet benchmark with $\gamma\simeq0.0131$ require
$t\simeq763.4$. Longer validation runs exercise the executor and diagnostics,
but this boundary prevents using short CI gates as evidence for nonlinear
island growth or plasmoid chains.
The broader scaling plots remain analytic reference plots; they
should not be cited as solver recovery of FKR, plasmoid, or ideal-tearing
regimes.

The reduced-MHD FAST simulation plots are intentionally unimpressive: energy is
nearly flat, kinetic energy is near zero, and the mode amplitude changes only at
the fourth significant figure over $t\le 0.1$. That is acceptable for a smoke
workflow, but it is not a nonlinear reconnection result. Any paper-quality claim
about tearing growth, island evolution, plasmoid formation, or inverse-design
success still requires new calibrated simulations.

## Precision policy

Validation gates use JAX x64 precision. The CLI validation commands call the
central runtime helper before generating validation artifacts, and
`mhx validate all` records `jax_enable_x64` in `validation_suite.json`.

Exploratory runs may use x32, but x32 results should not be compared to tight
validation tolerances. If a benchmark is intended to support a scientific claim,
run it in x64 and archive the complete output directory plus
`artifact_manifest.json`.

## Reviewer-ready today

The repository is currently defensible as a validation-first rebuild with:

- installable package and CLI;
- deterministic output schemas;
- exact linear physics gates;
- numerical Harris-sheet outer matching gate;
- FKR asymptotic growth-rate target assembled from numerical $\Delta'$;
- direct reduced-MHD Harris-sheet tearing eigenvalue gate for one published
  reference case;
- finite-domain Harris-sheet tearing dispersion sign/residual gate;
- Harris eigenfunction localization gate over a small $S$ scan;
- time-domain Harris eigenmode growth-fit replay;
- matrix-free linearization scaffolds;
- tiny nonzero-equilibrium dense-spectrum gate;
- periodic current-sheet linear time-domain replay;
- nonlinear current-sheet trajectory-map differentiability bridge;
- Orszag--Tang, decaying turbulence, and forced turbulent-reconnection
  validation media with explicit claim boundaries;
- nonlinear duration audit that flags short CI nonlinear runs against
  literature-scale target windows;
- FAST neural-ODE dataset/baseline/calibration and latent-ODE training
  protocol;
- plugin and diagnostics extension examples;
- CI-checked docs, tests, figures, GIFs, reports, and artifact manifests.

## Not reviewer-ready yet

The following items should not be overclaimed:

- full calibrated FKR/Coppi tearing dispersion scans;
- nonlinear magnetic-island growth;
- Sweet-Parker plasmoid instability from the PDE solver;
- production performance on large grids;
- neural-ODE production surrogate accuracy/generalization;
- inverse-design superiority or Pareto-front claims.

The next scientific milestone should extend the single direct tearing
eigenvalue gate into a parameter scan with documented asymptotic windows,
resolution studies, and tolerances against accepted FKR/Coppi theory or a
reference code.
