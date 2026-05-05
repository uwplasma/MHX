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
| Matrix-free JVP and eigen scaffolds | JAX JVP, zero-state eigenmodes, Arnoldi, and power iteration are tested on controlled fixtures. | Good scaffolding; not yet a tearing spectrum. |
| Nonzero current-sheet linearization | Exact bracket couplings around $\psi_0=A\cos y$ are tested. | Good operator gate; still not an FKR benchmark. |
| Periodic current-sheet dense spectrum | Tiny dense spectrum around $\psi_0=A\cos y$ checks gauge modes, eigenpair residuals, and absence of spurious positive growth. | Useful stability/operator gate; not an FKR/Coppi tearing-growth validation. |
| FAST reduced-MHD run | Produces stable outputs, diagnostics, figures, and GIFs. Kinetic energy remains tiny and mode amplitudes change weakly. | Smoke test only. |
| Two-fluid and plugin examples | Exercise extension paths and output schemas. | API examples only; not validated extended-MHD physics. |
| Nonlinear tearing/plasmoid dynamics | Not demonstrated by the current FAST runs. | No. |
| Neural ODE / inverse design | Roadmap item after validation base stabilizes. | No. |

## Plot audit notes

The current validation plots are internally consistent. The exact-decay plot is
the strongest numerical result: numerical and analytic amplitude/energy curves
overlap, and relative errors remain near roundoff in x64. The scaling plots are
useful, but they are analytic reference plots; they should not be cited as
solver recovery of FKR, plasmoid, or ideal-tearing regimes.

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
- matrix-free linearization scaffolds;
- tiny nonzero-equilibrium dense-spectrum gate;
- plugin and diagnostics extension examples;
- CI-checked docs, tests, figures, GIFs, reports, and artifact manifests.

## Not reviewer-ready yet

The following items should not be overclaimed:

- calibrated FKR/Coppi tearing eigenvalues;
- nonlinear magnetic-island growth;
- Sweet-Parker plasmoid instability from the PDE solver;
- production performance on large grids;
- neural-ODE surrogate accuracy;
- inverse-design superiority or Pareto-front claims.

The next scientific milestone should be a calibrated tearing eigenvalue
benchmark with a documented equilibrium, resolution study, fit window, and
tolerance against an accepted asymptotic or reference-code result.
