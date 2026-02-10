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

## Notes

- Many scripts assume JAX 64-bit mode. Consider setting:

  ```bash
  export JAX_ENABLE_X64=1
  ```

## License

MIT. See `LICENSE`.

