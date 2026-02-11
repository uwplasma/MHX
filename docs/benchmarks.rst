Benchmarks
==========

This page collects quick validation plots for FAST settings. For publication
benchmarks, replace FAST with production settings and regenerate figures.

Energy evolution (FAST):

.. image:: _static/energy.png
   :width: 600
   :alt: Energy time series

Reachable region (FAST):

.. image:: _static/fig_reachable_region.png
   :width: 600
   :alt: Reachable region

Inverse-design cost history (FAST):

.. image:: _static/fig_cost_history.png
   :width: 600
   :alt: Cost history

Timing table
------------

Generate timing data on your machine:

.. code-block:: bash

   python examples/benchmark_timings.py
   # optionally include a production-sized run (slow):
   python examples/benchmark_timings.py --production
   # optionally enable JIT for timing:
   python examples/benchmark_timings.py --jit

The generated table (FAST + small by default) is stored in:

- `outputs/benchmarks/timing_table.json`
- `docs/_static/timing_table.rst`

.. include:: _static/timing_table.rst
