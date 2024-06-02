# STADiffuser: versatile deep generative model for high fidelity simulation of spatial transcriptomics

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Functionality and Applications](#functionality-and-applications)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)



## Overview
STADiffuser is a cutting-edge deep generative model designed to simulate high-fidelity spatial transcriptomic (ST) data. This tool addresses the limitations of current spatial transcriptomics technologies, such as high costs, data sparsity, and limited resolution, by providing a versatile simulation framework that can generate accurate and detailed ST data.

## Architecture
STADiffuser's architecture is composed of a two-stage framework designed for high-fidelity simulation: 
- **Autoencoder with Graph Attention Mechanism**： The autoencoder learns embeddings for spatial spots using a graph attention mechanism, which captures the intricate spatial relationships and gene expression patterns in the data.
- **Latent Diffusion Model with Spatial Denoising Network**： The latent diffusion model generates realistic ST data by diffusing the learned embeddings through a spatial denoising network, which refines the spatial patterns and gene expression profiles.
![STADiffuser](./docs/_static/STADiffuser-backbone.png)

## Functionality and Applications
STADiffuser offers a range of functionalities and applications that make it a powerful tool for simulating and analyzing spatial transcriptomic data.
![STADiffuser](./docs/_static/STADiffuser-app.png)

Functionality and applicaitons includes:
- **Multi-Sample and 3D Coordinate Modeling**: Handle multiple samples simultaneously and model data in 3D space for comprehensive and realistic simulations.
- **User-Defined Conditions**: Specify various conditions and parameters to customize the simulation process.
- **Accurate Imputation**: Perform accurate imputation of missing data, enhancing dataset completeness and usability.
- **Super-Resolution**: Enhance the resolution of spatial transcriptomic data, enabling detailed study of gene expression patterns.

- **In Silico Experiments**: Enhance statistical power and reduce experimental costs by conducting in silico experiments with simulated data.
- **Cell Type-Specific Gene Identification**: Identify genes specific to particular cell types while controlling for confounding factors.
- **3D Slice Imputation**: Impute missing slices in 3D spatial transcriptomic data, providing a more continuous and complete spatial map of gene expression.

## Installation


## Usage


## Contributing


## License



