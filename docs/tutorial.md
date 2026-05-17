# Tutorial: from first run to validation media

This tutorial keeps the first-pass path short while pointing to the evidence
pages that carry the full reviewer context.

## 1. Create and run a config

```bash
mhx init outputs/tutorial/linear_tearing.toml
mhx run outputs/tutorial/linear_tearing.toml --outdir outputs/tutorial/smoke
mhx figures outputs/tutorial/smoke --gif
mhx report outputs/tutorial/smoke
mhx artifact-manifest outputs/tutorial/smoke
```

Expected outputs:

- `outputs/tutorial/smoke/config_effective.json`
- `outputs/tutorial/smoke/diagnostics.json`
- `outputs/tutorial/smoke/trajectory.npz`
- `outputs/tutorial/smoke/figures/flux_movie.gif`
- `outputs/tutorial/smoke/report.md`
- `outputs/tutorial/smoke/artifact_manifest.json`

## 2. Run a physics gate

```bash
mhx benchmark decay --outdir outputs/tutorial/resistive_decay
```

This checks the exact reduced-MHD induction limit
$\psi_k(t)=\psi_k(0)\exp(-\eta k^2t)$ and writes the amplitude, energy, and
relative-error plots documented on [validation.md](validation.md).

## 3. Generate nonlinear validation media

```bash
mhx benchmark orszag-tang --outdir outputs/tutorial/orszag_tang --nx 64 --ny 64 --t-end 6 --movies
mhx benchmark decaying-turbulence --outdir outputs/tutorial/decaying_turbulence --nx 64 --ny 64 --t-end 8 --movies
mhx benchmark forced-turbulent-reconnection --outdir outputs/tutorial/forced_reconnection --nx 64 --ny 64 --t-end 80 --save-every 100 --movies
```

These are `claim_level = "validation"` artifacts. They are good for learning
the API, diagnostics, and plotting workflow; they are not production plasmoid
or turbulence-statistics results.

## 4. Inspect extension points

```bash
mhx physics list
mhx diagnostics list
mhx run examples/linear_tearing_plugin_demo.toml --outdir outputs/tutorial/plugin_demo
mhx report outputs/tutorial/plugin_demo
```

The plugin demo shows how to register a local RHS term and diagnostic without
editing MHX source. See [plugins.md](plugins.md) for the external package
template and testing checklist.
