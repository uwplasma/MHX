Installation
============

Basic install:

.. code-block:: bash

   pip install -e .

For inverse-design / ML:

.. code-block:: bash

   pip install -e ".[ml]"

Notes:

- Enable 64-bit JAX for consistency:

.. code-block:: bash

   export JAX_ENABLE_X64=1
