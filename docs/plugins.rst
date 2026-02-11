Physics Plugins
===============

The solver supports additive physics terms via a plugin interface. Each term
adds contributions to the spectral RHS for ``v_hat`` and ``B_hat``.

Physics API version
-------------------

Current API version: ``1`` (see ``PHYSICS_API`` in ``mhx/solver/plugins.py``).

Compatibility matrix:

- ``API_VERSION = 1``: ``rhs_additions(t, v_hat, B_hat, kx, ky, kz, k2, mask_dealias)``

Interface
---------

Implement ``PhysicsTerm`` with a ``rhs_additions`` method:

.. code-block:: python

   from mhx.solver.plugins import PhysicsTerm, API_VERSION

   class MyTerm(PhysicsTerm):
       name = "my_term"
       api_version = API_VERSION
       def rhs_additions(self, *, t, v_hat, B_hat, kx, ky, kz, k2, mask_dealias):
           dv = 0.0 * v_hat
           dB = 0.0 * B_hat
           return dv, dB

Registry
--------

.. code-block:: python

   from mhx.solver.plugins import register_term, list_terms

   register_term(MyTerm())
   print(list_terms())

Template
--------

.. code-block:: python

   from mhx.solver.plugins import PhysicsTerm, PHYSICS_API

   class MyTerm(PhysicsTerm):
       name = "my_term"
       api_version = PHYSICS_API.version
       def rhs_additions(self, *, t, v_hat, B_hat, kx, ky, kz, k2, mask_dealias):
           dv = 0.0 * v_hat
           dB = 0.0 * B_hat
           return dv, dB

Minimal example
---------------

See:

- https://github.com/uwplasma/MHX/blob/main/examples/physics_plugin_minimal.py
