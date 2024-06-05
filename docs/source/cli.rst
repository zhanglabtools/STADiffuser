.. _CLI:

STADiffuser CLI documentation
=============================

The command-line interface is provided by the `stadiffuser.cli` module. The main function is `main`, which is the entry point of the command-line interface.
We provide the following command-line templates for single slice, multiple slices, and 3D slices data. Please refer to :ref:`usage` for more details.


Run STADiffuser CLI for single slice
------------------------------------


.. code-block:: bash
   :linenos:

    stadiffuser-cli --input_file path-to-your-processed-h5ad-file \
    --output_dir path-to-your-output-directory \
    --autoencoder-path path-to-your-autoencoder-model.pth \
    --new-spatial-division spatial-division-value \
    --label label-column-name \
    --device your-cuda-device \
    --autoencoder-batch-size 128 \
    --denoiser-batch-size 128 \
    --input_dim input-dimension-value


Run STADiffuser CLI for multiple slices
---------------------------------------

.. code-block:: bash
   :linenos:

    stadiffuser-cli python scripts/main.py --input_file \
    path-to-your-processed-multi-slice-h5ad-file \
    --output_dir path-to-your-output-directory \
    --new-spatial-division spatial-division-value \
    --input_dim input-dimension-value \
    --multi-slice \
    --use-batch used-batch-name-in-adata.obs \
    --pretrain-epochs 200 \
    --autoencoder-max-epochs 500 \
    --update-interval 50 \
    --autoencoder-batch-size 128 \
    --denoiser-max-epochs 1000 \
    --denoiser-batch-size 256 \
    --device cuda:0


Run STADiffuser CLI for 3D slices data
--------------------------------------

.. code-block:: bash
   :linenos:

        stadiffuser-cli --input_file \
        path-to-your-processed-h5ad-file \
        --output_dir path-to-your-output-directory \
        --new-spatial-division spatial-division-value  \
        --new-spatial-z-division z-axis-division-value  \
        --input_dim number-of-genes \
        --multi-slice \
        --use-batch column-name-for-batch \
        --pretrain-epochs 200 \
        --autoencoder-max-epochs 500 \
        --update-interval 50 \
        --autoencoder-batch-size 256 \
        --denoiser-3d \
        --denoiser-max-epochs integer-value \
        --denoiser-batch-size integer-value \
        --label label-column-name \
        --device cuda:0

.. _usage:
Usage
-----

.. automodule:: stadiffuser.cli
    :members: main

