Diagnostics
===========

Core diagnostics live in `mhx/solver/diagnostics.py`.

Definitions
-----------

Late-time kinetic-energy fraction:

.. math::

   f_{\mathrm{kin}} = \frac{\langle E_{\mathrm{kin}} \rangle_{t\in\mathrm{tail}}}
   {\langle E_{\mathrm{kin}} + E_{\mathrm{mag}} \rangle_{t\in\mathrm{tail}}}

Plasmoid complexity (smooth curvature proxy on a 1D cut):

.. math::

   C_{\mathrm{plasmoid}} = \left\langle \left(\frac{\partial^2 A_z}{\partial s^2}\right)^2 \right\rangle_s

Tearing growth rate (linear regression on log amplitude):

.. math::

   \gamma_{\mathrm{fit}} = \frac{d}{dt} \ln |B_x(k_x=0, k_y=1, k_z=0)|

Implementation details
----------------------

- Growth fitting uses a masked, JAX-friendly regression window.
- Complexity metric is differentiable (no discrete island counting).

Source link:

- https://github.com/uwplasma/MHX/blob/main/mhx/solver/diagnostics.py
