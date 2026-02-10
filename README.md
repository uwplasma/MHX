# MHX

Differentiable pseudo-spectral reduced MHD tearing/plasmoid solver and analysis tools (JAX-based).

## Install

```bash
pip install -e .
```

Inverse design / ML extras:

```bash
pip install -e ".[ml]"
```

Docs extras:

```bash
pip install -e ".[docs]"
```

## Quickstart (FAST)

Run a tiny simulation (seconds) and generate figures:

```bash
mhx simulate --fast --equilibrium original --eta 1e-3 --nu 1e-3
mhx figures --run outputs/runs/<timestamp>_simulate
```

## Outputs

Runs are written under:

```
outputs/runs/<timestamp>_<tag>/
  config.yaml
  history.npz
  solution_initial.npz
  solution_mid.npz
  solution_final.npz
  figures/
```

Grid scans and figure outputs from the inverse-design figure driver:

```
outputs/scans/reachable_region_scan_<eq_mode>.npz
outputs/figures/*.png
```

## Objective consistency (important)

The inverse-design objective is persisted into `history.npz` (`target_f_kin`,
`target_complexity`, `lambda_complexity`). The figure generator will load these
values by default to avoid apples-to-oranges comparisons.

## Citation

If you use MHX, please cite it. See `CITATION.cff`.

## Notes

- Many scripts assume 64-bit JAX. Enable with:

```bash
export JAX_ENABLE_X64=1
```
