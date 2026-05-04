Migration Guide (Legacy Scripts)
================================

The legacy scripts were moved under ``scripts/legacy/`` and replaced with
thin shims at the repo root. Prefer the CLI equivalents below.

Old → New
---------

- ``mhd_tearing_solve.py`` → ``mhx simulate``
- ``mhd_tearing_scan.py`` → ``mhx scan``
- ``mhd_tearing_inverse_design.py`` → ``mhx inverse-design``
- ``mhd_tearing_inverse_design_figures.py`` → ``mhx figures`` / paper pipeline
- ``mhd_tearing_postprocess.py`` → ``mhx figures``
- ``mhd_tearing_postprocess_ml.py`` → ``mhx figures`` + ``mhx inverse-design``
- ``mhd_tearing_postprocess_ml_v2.py`` → ``mhx figures`` + ``mhx inverse-design``
- ``mhd_tearing_ml.py`` → ``mhx inverse-design`` + ``mhx ml`` (future)
- ``mhd_tearing_ml_v2.py`` → ``mhx inverse-design`` + ``mhx ml`` (future)
- ``mhd_linear_benchmarks.py`` → ``bench/benchmark_matrix.py``
- ``mhd_reconnection_rate.py`` → diagnostics in ``mhx/solver/diagnostics.py``
- ``run_MHD.py`` / ``run_MHD_box.py`` → ``mhx simulate`` + ``mhx figures``

Deprecation window
------------------

Legacy shims remain for two minor releases, after which they will be removed.
Use the CLI to avoid breakage.
