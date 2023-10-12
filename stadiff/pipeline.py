"""
Pipeline for STADiffuser model
"""
import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import scipy.sparse as sp
from anndata import AnnData
from diffusers import SchedulerMixin
from torch_geometric.data import Data
from tqdm.auto import tqdm


def remove_edge(G, is_masked):
    """
    Remove edges in the graph.
    Parameters
    ----------
    G:
        The graph in scipy.sparse.coo_matrix format.
    is_masked:
        The mask of edges to be removed.
    Returns
    -------
    G:
        The new graph in scipy.sparse.coo_matrix format.
    """
    is_masked = is_masked.astype(int)
    mask = np.tile(is_masked, (G.shape[0], 1))
    mask = 1 - np.minimum(mask + mask.T, 1)
    new_G = G.multiply(mask)
    return new_G


def prepare_dataset(adata: AnnData,
                    use_rep=None,
                    use_spatial='spatial',
                    use_net='spatial_net',
                    use_label = None,
                    device='cpu'):
    """
    Transfer adata to pytorch_geometric dataset
    Parameters
    ----------
    adata:
        AnnData object, adata.obsm["spatial"] must exist.
    factor:
        The factor to divide the spatial coordinates.
    used_mask:
        The key of the mask in adata.obs.
    use_rep:
        The key of the expression matrix in adata.obsm. If None, use adata.X.
    use_spatial:
        The key of the spatial coordinates in adata.obsm.
    use_label:
        The key of the label in adata.obs.
    device:
        "cpu" or "cuda"
    """
    G_df = adata.uns[use_net].copy()
    cells = np.array(adata.obs_names)
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))
    G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
    G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)
    G = sp.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])), shape=(adata.n_obs, adata.n_obs))
    G = G + sp.eye(G.shape[0])
    if use_rep is not None:
        x = adata.obsm[use_rep]
    else:
        x = adata.X
    if sp.issparse(x):
        x = x.todense()
    spatial = adata.obsm[use_spatial]
    edge_list = np.nonzero(G)
    normalized_spatial = torch.LongTensor(spatial)
    if use_label is not None:
        label = adata.obs[use_label]
        label = torch.LongTensor(label)
    else:
        label = None
    data = Data(edge_index=torch.LongTensor(np.array([edge_list[0], edge_list[1]])),
                x=torch.FloatTensor(x),
                spatial=normalized_spatial,
                label=label)
    return data.to(device)


def train_autoencoder(train_loader, model, n_epochs=1000, gradient_clip=5,
                      save_dir=None, model_name="model.pth", lr=1e-4, weight_decay=1e-6):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    loss_list = []
    pbar = tqdm(range(n_epochs))
    model.train()
    for epoch in range(1, n_epochs + 1):
        for batch in train_loader:
            optimizer.zero_grad()
            z, out = model(batch.x, batch.edge_index)
            loss = F.mse_loss(out, batch.x)
            loss_list.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            pbar.set_description(f"Epoch: {epoch}, Loss: {loss.item():.4f}")
            loss_list.append(loss.item())
        scheduler.step()
        pbar.update(1)
    # check if save_dir is not None and exists
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(model.state_dict(), os.path.join(save_dir, model_name))
    return model, loss_list


def simulate(model: torch.nn.Module=None,
             noise_scheduler: SchedulerMixin=None,
             spatial_coord: torch.Tensor=None,
             labels: Optional[torch.Tensor] = None,
             n_features: int = None,
             init_x: torch.Tensor = None,
             seed=None,
             progress=False,
             n_samples: int = 1,
             return_tuple=True,
             **kwargs):
    r"""
    Simulate the diffusion process of the model.
    Parameters
    -----------
    model:
        the denoising model
    spatial_coord:
        the spatial coordinates of the spots (n_spots, 2) or (n_spots, 3)
    labels:
        optional, the labels of the spots when the model is conditional
    noise_scheduler:
        the noise scheduler
    seed:
        random seed to control torch and numpy
    progress:
        whether to show the progress bar
    n_samples:
        `int`, number of samples to simulate
    return_tuple:
        `bool`, whether to return a tuple of tensors
    device:
        `str`, "cpu" or "cuda"
    kwargs:
        other arguments to pass to the model

    Returns
    --------
    denoised:
        denoised expression matrix, shape (n_spots, n_samples * p) if `return_tuple` is False,
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    n_spots = spatial_coord.shape[0]
    assert n_features is not None or init_x is not None, "Either n_features or init_x must be provided."
    if init_x is None:
        denoised = torch.randn((n_spots * n_samples, n_features), device=model.device)
        denoised = denoised.unsqueeze(1)
    else:
        denoised = init_x.unsqueeze(1)
    # check if spatial_coord in kwargs
    with torch.no_grad():
        if progress:
            iterator = tqdm(noise_scheduler.timesteps)
        else:
            iterator = noise_scheduler.timesteps
        for t in iterator:
            model_output = model(denoised, t, spatial_coord, class_labels=labels).sample
            denoised = noise_scheduler.step(model_output, t, denoised).prev_sample
    denoised = torch.squeeze(denoised, 1)
    if n_samples > 1:
        denoised = torch.chunk(denoised, n_samples, dim=0)
        if not return_tuple:
            denoised = torch.cat(denoised, dim=1)
    return denoised
