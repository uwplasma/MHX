Performance Guide
=================

This page summarizes the main performance knobs and how to benchmark MHX.

Key knobs
---------

- ``Nx, Ny, Nz``: resolution (dominant cost)
- ``n_frames``: number of saved frames (memory + IO)
- ``t1`` and ``dt0``: integration length and initial step size
- ``--jit``: JIT on/off for time integration
- ``JAX_ENABLE_X64``: precision control (accuracy vs speed)

Benchmarking
------------

Use the built-in timing script:

.. code-block:: bash

   python examples/benchmark_timings.py
   python examples/benchmark_timings.py --jit
   python examples/benchmark_timings.py --production
   python bench/benchmark_matrix.py
   MHX_BENCH_FAST=1 python bench/benchmark_matrix.py

Outputs:

- ``outputs/benchmarks/timing_table.json``
- ``docs/_static/timing_table.rst``
- ``outputs/benchmarks/benchmark_matrix.json``
- ``docs/_static/benchmark_matrix.rst``

If ``psutil`` is installed, the JSON also includes ``rss_mb``. The script
always records a Python `tracemalloc` peak as ``tracemalloc_peak_mb``.

Benchmark matrix
----------------

.. include:: _static/benchmark_matrix.rst

Tips
----

- Use ``--fast`` for smoke tests and CI.
- Enable JIT for production runs, but keep ``JAX_DISABLE_JIT=1`` for CI.
- Tune ``n_frames`` to reduce IO overhead if you only need final diagnostics.
