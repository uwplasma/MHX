# Physics validation

MHX validation runs are physics gates, not smoke tests. Every gate states the
quantity it checks, the analytic or literature reference it checks against,
a numeric tolerance, and the command that reproduces it. A gate that cannot
name its reference and tolerance does not ship.

## How to read a gate

Each gate page section gives:

1. **The claim.** The physical statement under test, for example, a single
   Fourier mode must decay at the exact rate $\eta |\mathbf{k}|^2$.
2. **The reference.** An exact solution, an eigenvalue computation, or a
   published scaling such as the tearing growth rate of {cite}`furth1963`.
3. **The tolerance.** The largest accepted deviation, stated with the grid,
   time step, and precision it was measured at.
4. **The command.** One `mhx benchmark` invocation that regenerates the
   figures and the `validation.json` verdict.

## Claim levels

Every artifact carries a claim level in its manifest, and the level bounds
what the artifact can support:

| Level | Meaning |
| --- | --- |
| `smoke` | the code path runs and writes valid artifacts |
| `validation` | a physics gate passed at documented settings |
| `production_template` | the workflow for a production run, not its result |
| `production` | a converged, seed-checked result meeting the promotion gate |

Most committed evidence sits at `validation`. The
[reviewer evidence map](../project/reviewer_evidence.md) tracks what each
level currently covers, and the promotion machinery for `production` claims
lives in the [campaign runner](../project/campaign_runner.md).

## The gate families

| Page | Gates | Anchor |
| --- | --- | --- |
| [Exact limits and scaffolds](exact_limits.md) | resistive decay, diffusion eigenvalues, linearization consistency | exact solutions |
| [Linear tearing](linear_tearing.md) | FKR window, growth rates, $\Delta'$, eigenvalues, layer structure | {cite}`furth1963` |
| [Nonlinear gates](nonlinear.md) | Orszag--Tang, turbulence, energy budget, duration audits | {cite}`orszag1979,biskamp2003` |
| [Reconnection campaigns](reconnection_campaigns.md) | double-Harris long runs, convergence, seed sweeps | {cite}`harris1962,rutherford1973` |
| [Scaling theory](scaling_theory.md) | FKR, ideal-tearing, and plasmoid scaling gates | {cite}`pucci2014,loureiro2007` |

## The first gate: exact resistive decay

The entry point of the suite is exact. With the nonlinearity removed by a
single-mode initial condition, the flux equation reduces to
$\partial_t\hat\psi_{\mathbf{k}} = -\eta|\mathbf{k}|^2\hat\psi_{\mathbf{k}}$,
so the amplitude must follow $A_0\exp(-\eta|\mathbf{k}|^2 t)$ and the
magnetic energy $E_B(0)\exp(-2\eta|\mathbf{k}|^2 t)$.

![Exact resistive-decay relative errors](../_static/validation/exact_decay/decay_relative_error.png)

The unit test fails if amplitude, energy, fitted rate, monotonicity, or
final-field L2 deviations exceed the documented tolerances. Run it yourself:

```bash
mhx benchmark decay --outdir outputs/validation/resistive_decay
```

- [Implementation](https://github.com/uwplasma/MHX/blob/main/src/mhx/benchmarks/decay.py)
- [Tests](https://github.com/uwplasma/MHX/blob/main/tests/test_resistive_decay_validation.py)
- [Plotting helpers](https://github.com/uwplasma/MHX/blob/main/src/mhx/plotting/reduced_mhd.py)

## Run the whole suite

```bash
mhx validate all --outdir outputs/validation/suite
mhx benchmark catalog --outdir outputs/validation/catalog
```

The catalog command writes a machine-readable inventory of every gate, its
tolerances, and its artifacts. Read each artifact manifest before you use a
run as research evidence: the manifest, not the figure, bounds the claim.
