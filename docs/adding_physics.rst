Adding New Physics Terms
========================

Checklist
---------

1. Implement a ``PhysicsTerm`` with a stable ``api_version``.
2. Register a factory for config-driven use.
3. Add a minimal example and a test.
4. Update docs if the term affects diagnostics.
5. Run ``mhx plugin lint`` to validate term metadata/signature.

How to test
-----------

.. code-block:: bash

   pytest -q tests/test_plugins.py
   python examples/physics_plugin_minimal.py
   python examples/physics_plugin_extended_mhd.py
   mhx plugin lint

How to add a diagnostic
-----------------------

1. Add a new key or group in ``mhx/solver/diagnostics.py``.
2. Reference it in ``ModelConfig.diagnostics`` via YAML.
3. Verify in a FAST run:

.. code-block:: bash

   mhx simulate --fast --model-config model.yaml

Template
--------

.. code-block:: python

   from mhx.solver.plugins import PhysicsTerm, API_VERSION

   class MyTerm(PhysicsTerm):
       name = "my_term"
       api_version = API_VERSION
       def rhs_additions(self, *, t, v_hat, B_hat, kx, ky, kz, k2, mask_dealias):
           dv = 0.0 * v_hat
           dB = 0.0 * B_hat
           return dv, dB

Extended-MHD toy example
------------------------

See:

- https://github.com/uwplasma/MHX/blob/main/examples/physics_plugin_extended_mhd.py

Plugin template
---------------

Copy the template under ``templates/physics_plugin_template/`` to bootstrap
your own plugin with tests and docs.

Built-in terms
--------------

MHX ships a few vetted toy terms for quick experimentation:

- ``hall`` (Hall-like induction term)
- ``hyper_resistivity`` (``-eta4 * k^4 * B``)
- ``anisotropic_pressure`` (parallel velocity damping)
