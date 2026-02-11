Physics Model
=============

We solve a 2D/2.5D incompressible reduced MHD tearing-mode setup in a periodic
box using a pseudo-spectral method and diffusive dissipation. Classic tearing
mode theory references include the FKR and Coppi regimes, and plasmoid
instability work builds on Sweet–Parker sheet scalings and subsequent linear
instability analyses.

Key references are listed in :doc:`references`.

Source: https://github.com/uwplasma/MHX/blob/main/mhx/solver/tearing.py

Extended-MHD options
--------------------

Reconnection studies often extend resistive MHD with additional physics such as
Hall terms or anisotropic pressure closures. MHX supports additive RHS terms
via the plugin interface; see :doc:`plugins` and :doc:`adding_physics` for
examples, and :doc:`references` for Hall-MHD reconnection literature.

Hall-MHD reconnection is a standard extension for collisionless regimes and
features prominently in the GEM reconnection challenge and related studies
(see the Hall-MHD references in :doc:`references`).
