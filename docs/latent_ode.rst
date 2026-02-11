Latent ODE
==========

We provide a minimal latent ODE module for learning reduced dynamics from
solver trajectories. The core model follows

.. math::

   \dot{z} = f_\theta(z), \quad y = g_\phi(z)

where ``z`` is a latent state and ``y`` is an observed diagnostic.

FAST tutorial
-------------

.. code-block:: bash

   python examples/latent_ode_fast.py

This produces:

.. image:: _static/latent_ode_fit.png
   :width: 600
   :alt: Latent ODE fit

Source:

- https://github.com/uwplasma/MHX/blob/main/mhx/ml/latent_ode.py
