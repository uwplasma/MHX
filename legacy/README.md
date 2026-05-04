# Legacy MHX code

The previous MHX implementation has been moved to `legacy/old_mhx/` as part of
the May 2026 rebuild.

The legacy tree is preserved for reference, reproducibility, and migration work.
It is no longer the active package imported by `import mhx`; the active package
now lives under `src/mhx/`.

To inspect or run legacy workflows, work inside `legacy/old_mhx/` and install
its saved metadata/dependencies separately. Legacy scripts may still write
outputs in their working directory, so run them in a scratch directory.

