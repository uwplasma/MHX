Reproducibility
===============

This section provides exact command sequences and the expected output files.
All commands assume you are at the repository root.

Setup
-----

```bash
export JAX_ENABLE_X64=1
pip install -e ".[ml,docs]"
```

1) Tiny simulation
------------------

```bash
mhx simulate --fast --equilibrium original --eta 1e-3 --nu 1e-3
```

Expected outputs (under the printed run directory):

- `config.yaml`
- `solution_final.npz`
- `figures/energy.png`
- `figures/az_midplane.png`

2) Tiny scan
------------

```bash
mhx scan --equilibrium forcefree --grid 4x4
```

Expected outputs:

- `outputs/scans/reachable_region_scan_forcefree.npz`

3) Tiny inverse design
----------------------

```bash
mhx inverse-design --equilibrium forcefree --steps 2 --fast
```

Expected outputs (under the printed run directory):

- `config.yaml`
- `history.npz`
- `solution_initial.npz`
- `solution_mid.npz`
- `solution_final.npz`

4) Reachable region & comparison figures
-----------------------------------------

```bash
python mhd_tearing_inverse_design_figures.py
```

Expected outputs:

- `outputs/figures/fig_reachable_heatmaps_forcefree.png`
- `outputs/figures/fig_reachable_region_fkin_Cplasmoid.png`
- `outputs/figures/fig_gamma_vs_fkin.png`
- `outputs/figures/fig_inverse_vs_grid_forcefree.png`

5) Docs media (optional)
------------------------

```bash
python examples/make_inverse_design_media.py
```

Expected outputs:

- `docs/_static/fig_reachable_heatmap.png`
- `docs/_static/fig_reachable_region.png`
- `docs/_static/fig_cost_history.png`
