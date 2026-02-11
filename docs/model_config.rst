Model Configuration
===================

You can assemble physics from a YAML/JSON config without code changes.

Example
-------

.. code-block:: yaml

   model:
     equilibrium_mode: original
     rhs_terms: [linear_drag, hyper_resistivity]
     term_params:
       linear_drag:
         mu: 0.05
       hyper_resistivity:
         eta4: 1e-3

Usage
-----

.. code-block:: bash

   mhx simulate --fast --model-config model.yaml
   mhx scan --fast --model-config model.yaml
   mhx inverse-design --fast --model-config model.yaml

If ``equilibrium_mode`` is omitted in the model file, the CLI ``--equilibrium``
value is used.
