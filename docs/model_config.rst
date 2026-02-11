Model Configuration
===================

You can assemble physics from a YAML/JSON config without code changes.

Example
-------

.. code-block:: yaml

   model:
     equilibrium_mode: original
     rhs_terms: [linear_drag, hyper_resistivity, hall_toy]
     term_params:
       linear_drag:
         mu: 0.05
       hyper_resistivity:
         eta4: 1e-3
       hall_toy:
         d_h: 1e-2

Usage
-----

.. code-block:: bash

   mhx simulate --fast --model-config model.yaml
   mhx scan --fast --model-config model.yaml
   mhx inverse-design --fast --model-config model.yaml

If ``equilibrium_mode`` is omitted in the model file, the CLI ``--equilibrium``
value is used.

Diagnostics selection
---------------------

You can request specific diagnostics (or groups) with:

.. code-block:: yaml

   model:
     diagnostics: [energies, tearing, plasmoid]

See :doc:`diagnostics` for available groups.
