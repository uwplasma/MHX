Reproducibility
===============

This section provides exact command sequences and the expected output files.
All commands assume you are at the repository root.

See also :doc:`output_schema` for detailed NPZ keys and file layout.

Setup
-----

.. code-block:: bash

   export JAX_ENABLE_X64=1
   export MHX_API_VERSION=1
   pip install -e ".[ml,docs]"

1) Tiny simulation
------------------

.. code-block:: bash

   mhx simulate --fast --equilibrium original --eta 1e-3 --nu 1e-3

Expected outputs (under the printed run directory):

- `config.yaml`
- `solution_final.npz`
- `figures/energy.png`
- `figures/az_midplane.png`

2) Tiny scan
------------

.. code-block:: bash

   mhx scan --equilibrium forcefree --grid 4x4

Expected outputs:

- `outputs/scans/reachable_region_scan_forcefree.npz`

3) Tiny inverse design
----------------------

.. code-block:: bash

   mhx inverse-design --equilibrium forcefree --steps 2 --fast

Expected outputs (under the printed run directory):

- `config.yaml`
- `history.npz`
- `solution_initial.npz`
- `solution_mid.npz`
- `solution_final.npz`

4) Reachable region & comparison figures
-----------------------------------------

.. code-block:: bash

   # FAST mode:
   MHX_FIGURES_FAST=1 python mhd_tearing_inverse_design_figures.py

Expected outputs:

- `outputs/figures/fig_reachable_heatmaps_forcefree.png`
- `outputs/figures/fig_reachable_region_fkin_Cplasmoid.png`
- `outputs/figures/fig_gamma_vs_fkin.png`
- `outputs/figures/fig_inverse_vs_grid_forcefree.png`

5) Docs media (optional)
------------------------

.. code-block:: bash

   python examples/make_inverse_design_media.py

Expected outputs:

- `docs/_static/fig_reachable_heatmap.png`
- `docs/_static/fig_reachable_region.png`
- `docs/_static/fig_cost_history.png`

5b) Extended-MHD media (optional)
---------------------------------

.. code-block:: bash

   python examples/make_extended_mhd_media.py

Expected outputs:

- `docs/_static/energy_hall.png`
- `docs/_static/az_midplane_hall.gif`
- `docs/_static/energy_anisotropic.png`
- `docs/_static/az_midplane_anisotropic.gif`
- `docs/_static/energy_electron_pressure.png`
- `docs/_static/az_midplane_electron_pressure.gif`
- `docs/_static/energy_two_fluid.png`
- `docs/_static/az_midplane_two_fluid.gif`

6) Latent ODE dataset
---------------------

.. code-block:: bash

   python examples/latent_ode_dataset.py

Expected outputs:

- `outputs/datasets/latent_ode_dataset.npz`

7) Reproduce all figures (FAST)
--------------------------------

.. code-block:: bash

   python examples/reproduce_figures.py

Expected outputs include:

- `docs/_static/energy.png`
- `docs/_static/az_midplane.gif`
- `docs/_static/fig_reachable_heatmap.png`
- `docs/_static/fig_reachable_region.png`
- `docs/_static/fig_cost_history.png`
- `docs/_static/latent_ode_fit.png`
- `docs/_static/latent_ode_ablation.rst`
- `docs/_static/timing_table.rst`
- `docs/_static/latent_ode_experiment.png`
- `docs/_static/latent_ode_experiment.rst`
- `docs/_static/latent_ode_calibration.png`
- `docs/_static/latent_ode_failure.png`
- `docs/_static/benchmark_matrix.rst`
- `outputs/figures/fig_reachable_heatmaps_forcefree.png`
- `outputs/figures/fig_inverse_vs_grid_forcefree.png`
- `outputs/benchmarks/timing_table.json`
- `outputs/benchmarks/latent_ode_ablation.json`
- `outputs/benchmarks/latent_ode_experiment.json`
- `outputs/benchmarks/benchmark_matrix.json`
- `outputs/datasets/latent_ode_dataset.npz`
- `outputs/manifest.json`
- `outputs/run_configs/*.yaml`

8) Timing table (FAST + small)
------------------------------

.. code-block:: bash

   python examples/benchmark_timings.py

Expected outputs:

- `outputs/benchmarks/timing_table.json`
- `docs/_static/timing_table.rst`

9) Benchmark matrix
-------------------

.. code-block:: bash

   MHX_BENCH_FAST=1 python bench/benchmark_matrix.py

Expected outputs:

- `outputs/benchmarks/benchmark_matrix.json`
- `docs/_static/benchmark_matrix.rst`
