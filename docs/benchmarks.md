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

## CI artifacts

Every push runs a `benchmark-artifacts` CI job. It executes two deterministic
FAST pipelines:

```bash
mhx benchmark run --config examples/linear_tearing.toml --outdir outputs/ci/linear_tearing_fast --gif
mhx benchmark validate outputs/ci/linear_tearing_fast
mhx run examples/linear_tearing_twofluid_toy.toml --outdir outputs/ci/twofluid_toy
mhx figures outputs/ci/twofluid_toy --gif
mhx report outputs/ci/twofluid_toy
```

The job uploads `outputs/ci` as the `mhx-fast-artifacts` GitHub Actions
artifact. Reviewers can download it to inspect manifests, reports, PNG figures,
and GIF movies generated from the exact commit under test.

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

References used for the benchmark roadmap include
[Furth, Killeen, and Rosenbluth 1963](https://cir.nii.ac.jp/crid/1363107370207531008),
[Loureiro, Schekochihin, and Cowley 2007](https://arxiv.org/abs/astro-ph/0703631),
and [Pucci and Velli ideal tearing context](https://arxiv.org/abs/1704.08793).
