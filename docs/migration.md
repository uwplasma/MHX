# Migration from legacy scripts

The original exploratory scripts are preserved under `legacy/old_mhx/`. They are
not imported by the rebuilt package and are not part of the public API.

Use the active CLI instead:

| Legacy workflow | Active replacement |
| --- | --- |
| `run_MHD.py` or `run_MHD_box.py` | `mhx run examples/linear_tearing.toml --outdir outputs/smoke` |
| `mhd_tearing_solve.py` | `mhx benchmark run --config examples/linear_tearing.toml --outdir outputs/benchmarks/linear_tearing_fast` |
| `mhd_tearing_postprocess.py` | `mhx figures <run_dir> --gif` and `mhx report <run_dir>` |
| `mhd_linear_benchmarks.py` | `mhx benchmark decay`, `mhx benchmark linearized-rhs`, `mhx benchmark reduced-mhd-eigenmode` |
| `mhd_tearing_scan.py` | Roadmap: TOML-driven scan command after the v1 validation core is complete. |
| `mhd_tearing_inverse_design.py` | Roadmap: differentiable inverse-design command after calibrated tearing eigenvalue validation. |
| `mhd_tearing_ml.py` and `mhd_tearing_ml_v2.py` | `mhx neural-ode dataset --outdir outputs/neural_ode/seed_qi_fast` and `mhx neural-ode train --outdir outputs/neural_ode/latent_ode_fast` |

## Why the old scripts are archived

The old scripts were valuable exploratory tooling, but they mixed solver code,
plotting, hard-coded parameters, objective functions, and output paths. The new
package keeps these concerns separate:

- configs live in TOML and are saved as `config_effective.json`;
- diagnostics are registry entries with stable output keys;
- physics terms are versioned plugins;
- artifacts are schema-versioned and checksumed;
- validation commands have explicit pass/fail gates.

## Enforcement

Run the same check used in CI:

```bash
python tools/check_legacy_imports.py
```

This fails if active Python files import `legacy.old_mhx` or any archived
top-level script module such as `mhd_tearing_solve`.
