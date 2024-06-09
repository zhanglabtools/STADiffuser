Installation
============

It's recommended to create a new virtual environment before installing the dependencies:

.. code-block:: bash

    conda create -n STADiffuser python=3.9


Activate the virtual environment:

.. code-block:: bash

    conda activate STADiffuser


Prerequisites
-------------

Make sure you have installed a version of `PyTorch <https://pytorch.org/get-started/locally/>`_ (version >= 1.13.1) that is compatible with your GPU (if applicable) first.

.. code-block:: bash

   pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

Replace ``cu117`` with the appropriate CUDA version for your system if it differs from CUDA 11.8. If you are using a CPU-only version, you can simply install PyTorch without specifying a CUDA version.

.. code-block:: bash

   pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.1+cu117.html

.. _PyTorch: https://pytorch.org/



Install from source code
------------------------

To install STADiffuser from source code, clone the repository and install the dependencies using the following commands:

.. code-block:: bash

    git clone https://github.com/messcode/STADiffuser.git


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
