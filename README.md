# MHX

Collection of reduced MHD / tearing-mode / reconnection model scripts (JAX-based).

This repository was split out of the earlier `uwplasma/LX` umbrella repo to keep
vacuum-field solvers (now `uwplasma/BIMFx`) separate from MHD tooling.

## What’s here

The codebase currently consists of research scripts (not yet packaged as a
library), including:

- Tearing-mode solvers and scans
- Postprocessing/visualization scripts
- Inverse-design / ML-assisted variants

## Install

```bash
pip install -r requirements.txt
```

or for editable development:

```bash
pip install -e .
```

For inverse-design / ML scripts:

```bash
pip install -e ".[ml]"
```

## Quickstart (FAST)

Run a tiny tearing simulation and generate a couple of basic figures:

```bash
mhx simulate --fast --equilibrium original --eta 1e-3 --nu 1e-3
mhx figures --run outputs/runs/<timestamp>_simulate
```

The `--fast` mode is intended for smoke tests and CI (seconds, not minutes).

## Inverse design objective consistency

The inverse-design training script now persists the objective used for training
(`target_f_kin`, `target_complexity`, `lambda_complexity`) into the saved
`inverse_design_history_<eq_mode>.npz`. The figure generator prefers loading the
objective from that history file to avoid apples-to-oranges comparisons.

## Notes

- Many scripts assume JAX 64-bit mode. Consider setting:

  ```bash
  export JAX_ENABLE_X64=1
  ```

## License

MIT. See `LICENSE`.
