Benchmarks and Validation
=========================

This page summarizes the automated validation checks and their tolerances.

Linear tearing benchmark (FKR)
------------------------------

We compare the fitted growth rate ``gamma_fit`` against the FKR estimate
``gamma_FKR`` for a FAST configuration.

Acceptance criterion (tests):

- ``0.01 < gamma_fit / gamma_FKR < 100``

This is intentionally loose for FAST runs where resolution and runtime are
minimal; tighter bounds are expected for production settings.

Manufactured-solution test
--------------------------

We validate spectral derivatives on a known analytic field:

``f(x,y) = sin(kx x) cos(ky y)``

The test checks that the max error in ``∂f/∂x`` and ``∂f/∂y`` is ``< 1e-6``.

Convergence and invariants
--------------------------

Automated tests also cover:

- time-step sensitivity for ``f_kin`` (coarse vs smaller ``dt0``)
- energy budget sanity (finite, non-negative energies)
- Sweet–Parker scaling consistency (``E_SP`` vs ``eta`` trend)

See ``tests/`` for the precise thresholds used in CI.
