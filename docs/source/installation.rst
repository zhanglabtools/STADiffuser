Installation
============

Install from source code
------------------------

To install STADiffuser from source code, clone the repository and install the dependencies using the following commands:

.. code-block:: bash

    git clone https://github.com/messcode/STADiffuser.git

It's recommended to create a new virtual environment before installing the dependencies:

.. code-block:: bash

    conda create -n STADiffuser python=3.9

Activate the virtual environment:

.. code-block:: bash

    conda activate STADiffuser

Now you can install the code from the source directory:

.. code-block:: bash

    cd STADiffuser
    pip install -e .

Install from PyPI
-----------------

To install STADiffuser from PyPI, use the following command:

.. code-block:: bash

    pip install stadiffuser

Trouble shooting
----------------

If you fail to install, you may need to install the following dependencies manually:

- `SCANPY <https://scanpy.readthedocs.io/en/stable/installation.html>`_: recommended version 1.9.3
- `PyTorch <https://pytorch.org/get-started/locally/>`_: recommended version 1.13.1
- `PyG (PyTorch Geometric) <https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html>`_: recommended version 2.3.1
