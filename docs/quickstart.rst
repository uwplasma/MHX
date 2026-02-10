Quickstart
==========

Run a tiny simulation (seconds):

.. code-block:: bash

   mhx simulate --fast --equilibrium original --eta 1e-3 --nu 1e-3

Generate figures from the run directory:

.. code-block:: bash

   mhx figures --run outputs/runs/<timestamp>_simulate

Example outputs
---------------

.. image:: _static/energy.png
   :width: 600
   :alt: Energy time series

.. image:: _static/az_midplane.gif
   :width: 600
   :alt: A_z midplane evolution

Media generation script:

- https://github.com/uwplasma/MHX/blob/main/examples/make_fast_media.py
