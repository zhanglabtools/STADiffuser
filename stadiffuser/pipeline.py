"""
Pipeline for STADiffuser model
"""
import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import scipy.sparse as sp
import scanpy as sc
from anndata import AnnData
from diffusers import SchedulerMixin, DDPMScheduler
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
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
                    use_label=None,
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
    Notes: side effect: add edge_list to adata.uns
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
    edge_list = np.array([edge_list[0], edge_list[1]])
    adata.uns['edge_list'] = edge_list
    data = Data(edge_index=torch.LongTensor(edge_list),
                x=torch.FloatTensor(x),
                spatial=normalized_spatial,
                label=label)
    return data.to(device)



def get_recon(adata, autoencoder, device="cuda:0", use_net="spatial_net", apply_normalize=True, use_rep="latent",
              use_spatial="spatial", batch_mode=False, batch_size=256, num_neighbors=[5, 3],
              inplace=True, show_progress=True):
    adata_recon = adata.copy()
    if apply_normalize:
        sc.pp.normalize_total(adata_recon, target_sum=1e4)
        sc.pp.log1p(adata_recon)
    data = prepare_dataset(adata_recon, use_net=use_net, use_spatial=use_spatial)
    autoencoder = autoencoder.to(device)
    autoencoder.eval()
    if not batch_mode:
        data = data.to(device)
        with torch.no_grad():
            latent, recon = autoencoder(data.x, data.edge_index)
        # realase memory
        del data
    else:
        train_loader = NeighborLoader(data, num_neighbors=num_neighbors, batch_size=batch_size, shuffle=False)
        latent = []
        recon = []
        n_batches = len(train_loader)
        if show_progress:
            pbar = tqdm(total=n_batches)
        for batch in train_loader:
            if show_progress:
                pbar.update(1)
                pbar.set_description("Batch: {} / {}".format(pbar.n, n_batches))
                pbar.refresh()
            batch = batch.to(device)
            with torch.no_grad():
                latent_batch, recon_batch = autoencoder(batch.x, batch.edge_index)
            input_num = batch.input_id.shape[0]
            latent_batch = latent_batch[:input_num, :]
            recon_batch = recon_batch[:input_num, :]
            latent.append(latent_batch.cpu())
            recon.append(recon_batch.cpu())
        latent = torch.cat(latent, dim=0)
        recon = torch.cat(recon, dim=0)
    if inplace:
        adata_recon.X = recon.cpu().numpy()
        adata_recon.obsm[use_rep] = latent.cpu().numpy()
        return adata_recon
    else:
        return latent.cpu().numpy(), recon.cpu().numpy()


def train_autoencoder(train_loader, model,
                      n_epochs=1000, gradient_clip=5, lr=1e-4, weight_decay=1e-6,
                      save_dir=None, model_name="autoencoder",
                      device="cpu",
                      check_points=None):
    # check if save_dir is not None and exists
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    loss_list = []
    pbar = tqdm(range(n_epochs))
    model.train()
    for epoch in range(1, n_epochs + 1):
        for batch in train_loader:
            batch = batch.to(device)
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
        if check_points is not None and epoch in check_points:
            torch.save(model, os.path.join(save_dir, "{}_{}.pth".format(model_name, epoch)))
    if save_dir is not None:
        torch.save(model, os.path.join(save_dir, "{}.pth".format(model_name)))
    return model, loss_list


def get_latent_embedding(adata, autoencoder, data,
                         add_rep="latent",
                         add_recon=None,
                         return_recon=False,
                         device="cpu"):
    # get device of the model
    # get latent representation
    autoencoder.eval()
    with torch.no_grad():
        data = data.to(device)
        latent, recon = autoencoder(data.x, data.edge_index)
        latent = latent.cpu().numpy()
        recon = recon.cpu().numpy()
    # add latent representation to adata
    adata.obsm[add_rep] = latent
    if return_recon:
        if add_recon is not None:
            adata.obsm[add_recon] = recon
        return adata, recon
    else:
        return adata


def train_denoiser(train_loader,
                   model,
                   noise_scheduler: SchedulerMixin,
                   n_epochs: int = 1000,
                   lr: float = 1e-4,
                   lr_scheduler: str = "cosine_annealing",
                   weight_decay: float = 1e-6,
                   gradient_clip: float = 5,
                   num_class_embeds: Optional[int] = None,
                   save_dir: Optional[str] = None,
                   evaluate_interval: Optional[int] = None,
                   model_name: str = "denoiser",
                   device: str = "cpu",
                   check_points=None,
                   eval_fn=None,
                   eval_kwargs=None,
                   ):
    r"""
    Train the denoising model with the noise scheduler.
    Parameters
    ----------
    train_loader: torch_geometric.loader or torch_geometric.data.DataLoader
        The data loader for training.
    model: torch.nn.Module
        The denoising model.
    noise_scheduler: SchedulerMixin
        The noise scheduler.
    n_epochs: int
        The number of epochs to train.
    lr: float
        The learning rate.
    lr_scheduler: str
        The learning rate scheduler. Only "cosine_annealing" is supported.
    weight_decay: float
        The weight decay.
    gradient_clip: float
        The gradient clip.
    num_class_embeds: int
        The number of class embeddings if None the model is unconditional.
    """
    pbar = tqdm(range(n_epochs))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if lr_scheduler == "cosine_annealing":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    else:
        raise NotImplementedError
    loss_list = []
    for epoch in range(1, n_epochs + 1):
        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(device)
            clean_data = batch.x
            clean_data = clean_data.unsqueeze(1)  # (batch_size, 1, num_channels)
            optimizer.zero_grad()
            noise = torch.randn_like(clean_data)
            timesteps = torch.randint(0, 1000, (clean_data.shape[0],), device=batch.x.device, dtype=torch.long, )
            noisy_data = noise_scheduler.add_noise(clean_data, noise, timesteps)
            if num_class_embeds is None:
                noise_pred = model(noisy_data, timesteps, batch.spatial).sample
            else:
                noise_pred = model(noisy_data, timesteps, batch.spatial, batch.label).sample
            loss = F.mse_loss(noise_pred, noise)
            loss.backward()
            loss_list.append(loss.item())
            # clip gradient
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            pbar.set_description(f"Epoch: {epoch}, Loss: {loss.item():.4f}, batch_id: {batch_idx}")
        lr_scheduler.step()
        pbar.update(1)
        if eval_fn is not None:
            if epoch % evaluate_interval == 1:
                eval_fn(model, noise_scheduler, epoch, **eval_kwargs)
        if check_points is not None and epoch in check_points:
            torch.save(model, os.path.join(save_dir, "{}_{}.pth".format(model_name, epoch)))

    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(model, os.path.join(save_dir, "{}.pth".format(model_name)))
        print(" Save model to {}.pth".format(os.path.join(save_dir, "{}.pth".format(model_name))))
    print("-------------------Training Finished-------------------")
    return model, loss_list


def simulate(denoiser: torch.nn.Module = None,
             autoencoder: torch.nn.Module = None,
             ref_data=None,
             spatial_coord: np.ndarray = None,
             labels: Optional[np.ndarray] = None,
             noise_scheduler: SchedulerMixin = None,
             init_x: torch.Tensor = None,
             seed=None,
             progress=True,
             n_samples: int = 1,
             return_tuple=True,
             normarlizer=None,
             add_rep="latent",
             use_spatial="spatial",
             device="cpu",
             **kwargs):
    r"""
    Simulate the diffusion process of the model.
    Parameters
    -----------
    denoiser:
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
    if ref_data is not None:
        assert autoencoder is not None, "autoencoder must be provided when ref_data is not None"
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    denoiser = denoiser.to(device)
    spatial_coord = torch.from_numpy(spatial_coord).to(device)
    n_spots = spatial_coord.shape[0]
    n_features = denoiser.sample_size
    if init_x is None:
        sim_embed = torch.randn((n_spots * n_samples, n_features), device=denoiser.device)
        sim_embed = sim_embed.unsqueeze(1)
    else:
        sim_embed = init_x.unsqueeze(1)
    #  noise scheduler
    if noise_scheduler is None:
        noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    # check if spatial_coord in kwargs
    if labels is not None:
        print("Simulate with labels")
        labels = torch.from_numpy(labels).to(device).long()
    with torch.no_grad():
        if progress:
            iterator = tqdm(noise_scheduler.timesteps)
        else:
            iterator = noise_scheduler.timesteps
        for t in iterator:
            model_output = denoiser(sim_embed, t, spatial_coord, class_labels=labels).sample
            sim_embed = noise_scheduler.step(model_output, t, sim_embed).prev_sample
    sim_embed = torch.squeeze(sim_embed, 1)

    if n_samples > 1:
        sim_embed = torch.chunk(sim_embed, n_samples, dim=0)
        if not return_tuple:
            sim_embed = torch.cat(sim_embed, dim=1)

    if normarlizer is not None:
        sim_embed = sim_embed.cpu().numpy()
        sim_embed = normarlizer.denormalize(sim_embed)
        sim_embed = torch.from_numpy(sim_embed).float().to(device)

    if n_samples > 1 and ref_data is not None:
        # return simulated x directly if n_samples > 1
        return sim_embed

    if ref_data is not None:
        sim_data = ref_data.copy()
        sim_data.obsm[use_spatial] = spatial_coord.cpu().numpy()
        autoencoder = autoencoder.to(device)
        # check ref_data.uns["edge_list"] exist
        if "edge_list" not in ref_data.uns:
            raise ValueError("ref_data.uns['edge_list'] must exist. Call prepare_dataset first.")
        else:
            edge_list = ref_data.uns["edge_list"]
            edge_list = torch.LongTensor(edge_list).to(device)
        with torch.no_grad():
            sim_count = autoencoder.decode(sim_embed, edge_list)
        sim_data.obsm[add_rep] = sim_embed.cpu().numpy()
        sim_data.X = sim_count.cpu().numpy()
        return sim_data
    return sim_embed


