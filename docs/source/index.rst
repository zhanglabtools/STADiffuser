.. _home:

Welcome to STADiffuser's documentation!
=======================================


STADiffuser is a cutting-edge deep generative model designed to simulate high-fidelity spatial transcriptomic (ST) data. By providing a versatile simulation framework that can generate accurate and detailed ST data, this tool enables various downstream tasks, including: imputation, super-resolution, in silico experiments, and cell type-specific gene identification.


Please refer to the :ref:`Tutorials` for a step-by-step guide on how to use STADiffuser. We also provide a command-line interface (:ref:`CLI`) for users who prefer to run STADiffuser from the terminal. For more advanced users, we offer an :ref:`API` that allows for more flexibility and customization.

.. toctree::
   :maxdepth: 1

   self
   installation
   tutorials
   cli
   api


Architecture
------------

.. image::  _static/STADiffuser-backbone.png
    :align: center
    :width: 75%
    :alt: STADiffuser framework

STADiffuser's architecture is composed of a two-stage framework designed for high-fidelity simulation:

- **Autoencoder with Graph Attention Mechanism**： The autoencoder learns embeddings for spatial spots using a graph attention mechanism, which captures the intricate spatial relationships and gene expression patterns in the data.

- **Latent Diffusion Model with Spatial Denoising Network**： The latent diffusion model generates realistic ST data by diffusing the learned embeddings through a spatial denoising network, which refines the spatial patterns and gene expression profiles.


Functionality and Applications
------------------------------
STADiffuser offers a range of functionalities and applications that make it a powerful tool for simulating and analyzing spatial transcriptomic data.
See :doc:`tutorials` for more details.


.. image::  _static/STADiffuser-app.png
    :align: center
    :width: 75%
    :alt: STADiffuser empowers various downstream analyses


