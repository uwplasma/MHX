# MHX plugin template

This is a minimal installable third-party plugin layout for MHX.

It demonstrates:

- a physics term registered through the `mhx.physics` entry-point group;
- a diagnostic registered through the `mhx.diagnostics` entry-point group;
- a small test that can run without modifying MHX source.

Install from this directory in editable mode:

```bash
python -m pip install -e .
mhx physics list-with-plugins --entry-point-group mhx.physics
mhx diagnostics list-with-plugins --entry-point-group mhx.diagnostics
```

Use the registered names in a normal MHX TOML config:

```toml
[physics]
plugin_entry_point_groups = ["mhx.physics"]
rhs_terms = ["template_flux_sink"]

[physics.term_parameters.template_flux_sink]
rate = 1e-3

[diagnostics]
plugin_entry_point_groups = ["mhx.diagnostics"]
quantities = ["energy", "mode_growth", "template_final_psi_mean"]
```
