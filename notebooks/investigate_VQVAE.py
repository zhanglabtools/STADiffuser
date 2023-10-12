import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch
import warnings
import time
import stadiff.utils as sutils
from tqdm.auto import tqdm
from stadiff.dataset import SpatialExpressionDataset
from stadiff.utils import spatial2index, exp_tensor2mat
from stadiff.pipeline import prepare_dataset
from stadiff.vqvae import VQVAE
from collections import defaultdict

warnings.filterwarnings("ignore")


def load_DLPFC(section_id=151507):
    data_path = "E:/Datasets/DLPFC/{}".format(section_id)
    adata = sc.read_visium(data_path, count_file="{}_filtered_feature_bc_matrix.h5".format(section_id),
                           library_id=str(section_id))
    adata.var_names_make_unique()
    annotation = pd.read_csv(os.path.join(data_path, "{}_truth.txt".format(section_id)), sep='\t', header=None,
                             index_col=0)
    adata.obs["Manual annotation"] = annotation.loc[adata.obs.index, 1]
    # preprocessing data
    sc.pp.filter_cells(adata, min_genes=20)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata = adata[:, adata.var['highly_variable']]
    adata = spatial2index(adata, tol=50, spot_width=137, spot_height=120)
    return adata


def train_loop(model, train_loader, config, device='cuda'):
    model = model.to(device)
    model.train()
    train_loss = 0
    n_epochs = config["optimizer"]["n_epochs"]
    optimizer = torch.optim.Adam(model.parameters(), lr=config["optimizer"]["lr"],
                                 weight_decay=config["optimizer"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config["scheduler"]["step_size"],
                                                gamma=config["scheduler"]["gamma"])
    beta = config["optimizer"]["beta"]
    pbar = tqdm(range(1, n_epochs + 1))
    tracker = defaultdict(list)
    for epoch in range(n_epochs):
        for batch_idx, (data, slice_info) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = out["recon_loss"] + beta * out["emb_loss"]
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            tracker["recon_loss"].append(out["recon_loss"].item())
            tracker["emb_loss"].append(out["emb_loss"].item())
            tracker["loss"].append(loss.item())
        pbar.set_description("Epoch {}, Loss {:.4f}".format(epoch, loss.item()))
        scheduler.step()
        pbar.update(1)
    pbar.close()
    return model, tracker


def visualize_genes(adata, adata_, gene_name, save_path=None):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    sc.pl.spatial(adata, color=gene_name, ax=ax[0], show=False, title="Raw".format(gene_name))
    sc.pl.spatial(adata_, color=gene_name, ax=ax[1], show=False, title="Reconstructed")
    if save_path is not None:
        plt.savefig(save_path)
    plt.close()


def evaluation_1slice(model, train_loader, tracker, adata, device='cuda',
                      save_dir=None, monitor_genes=None):
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    model = model.to(device)
    model.eval()
    adata_ = adata.copy()
    with torch.no_grad():
        for batch_idx, (data, slice_info) in enumerate(train_loader):
            data = data.to(device)
            out = model(data)
            recon = out["x_recon"].squeeze(0).cpu().numpy()
            recon = exp_tensor2mat(recon, adata, adata.uns['padding_info'])
            adata_.X = recon
            latent = out["z"].squeeze(0).to('cpu').detach().numpy()
            adata_.obsm["embedding"] = exp_tensor2mat(latent, adata, adata.uns['padding_info'])
            sc.pp.neighbors(adata_, use_rep="embedding")
            sc.tl.umap(adata_, n_components=2, random_state=42)
            # plot the UMAP
            fig, ax = plt.subplots(figsize=(5, 5))
            sc.pl.umap(adata_, color=["Manual annotation"], ax=ax, show=False)
            if save_dir is not None:
                plt.savefig(os.path.join(save_dir, "slice_{}_umap.png".format(batch_idx)))
            if monitor_genes is not None:
                for gene_name in monitor_genes:
                    visualize_genes(adata, adata_, gene_name,
                                    save_path=os.path.join(save_dir, "slice_{}_{}.png".format(batch_idx, gene_name)))
    # visulization loss curve
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].plot(tracker["recon_loss"], label="recon_loss")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Reconstruction loss")
    axs[1].plot(tracker["emb_loss"], label="emb_loss")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Embedding loss")
    axs[2].plot(tracker["loss"], label="loss")
    axs[2].set_xlabel("Epoch")
    axs[2].set_ylabel("Total loss")
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, "loss_curve.png"))
    # destroy the fig
    plt.close()


def save_folder_formatter(note):
    time_str = time.strftime("%Y%m%d-%H%M%S")
    return "{}_{}".format(time_str, note)


def train_model_eval(adata, config, train_loader, save_folder, model_class=VQVAE):
    model = model_class(**config["model"])
    model, tracker = train_loop(model, train_loader, config)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    sutils.save_config(os.path.join(save_folder, "config.yaml"), config)
    evaluation_1slice(model, train_loader, tracker, adata,
                      save_dir=save_folder,
                      monitor_genes=["SLC17A7", "CD47"])


CONFIG = {"model": {"in_channels": 3000, "out_channels": 3000,
                    "latent_channels": 64, "num_vq_embeddings": 128,
                    "norm_num_groups": 16},
          "optimizer": {"lr": 1e-3, "weight_decay": 1e-5, "step_size": 200, "n_epochs": 1000, "beta": 0.1},
          "scheduler": {"step_size": 200, "gamma": 0.5}}

CONFIG_CNNVQVAE = {"model": {"in_channels": 3000,
                             "latent_channels": [512, 128, 30],
                             "num_vq_embeddings": 128,
                             "vq_embed_dim": 30},
                   "optimizer": {"lr": 1e-3, "weight_decay": 1e-5, "step_size": 200, "n_epochs": 1000, "beta": 0.1},
                   "scheduler": {"step_size": 200, "gamma": 0.5}}

if "__main__" == __name__:
    import copy
    from stadiff.vqvae import CNNVQVAE

    EXP_FOLDER = "./output"
    adata = load_DLPFC(section_id=151507)
    exp_tensor, adata = prepare_dataset(adata, padding=96)
    exp_tensor = torch.from_numpy(exp_tensor).float()
    dataset = SpatialExpressionDataset(exp_tensor, adata.uns['padding_info'])
    # Varying \beta weight parameter of the embedding loss
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    # vary the beta parameter
    beta_list = [0.1, 0.5, 1, 2, 5, 10]
    for beta in beta_list:
        config = copy.deepcopy(CONFIG_CNNVQVAE)
        config["optimizer"]["beta"] = beta
        save_folder = os.path.join(EXP_FOLDER, save_folder_formatter("CNNVQVAE_beta_{}".format(beta)))
        train_model_eval(adata, config, train_loader, save_folder, model_class=CNNVQVAE)
    # vary the number of embedding
    num_vq_embeddings_list = [16, 32, 64, 128, 256]
    for num_vq_embeddings in num_vq_embeddings_list:
        config = copy.deepcopy(CONFIG_CNNVQVAE)
        config["model"]["num_vq_embeddings"] = num_vq_embeddings
        save_folder = os.path.join(EXP_FOLDER, save_folder_formatter("CNNVQVAE_num_vq_embeddings_{}".format(num_vq_embeddings)))
        train_model_eval(adata, config, train_loader, save_folder, model_class=CNNVQVAE)
    # vary the vq_embed_dim
    vq_embed_dim_list = [16, 32, 64, 128, 256]
    for vq_embed_dim in vq_embed_dim_list:
        config = copy.deepcopy(CONFIG_CNNVQVAE)
        config["model"]["vq_embed_dim"] = vq_embed_dim
        save_folder = os.path.join(EXP_FOLDER, save_folder_formatter("CNNVQVAE_vq_embed_dim_{}".format(vq_embed_dim)))
        train_model_eval(adata, config, train_loader, save_folder, model_class=CNNVQVAE)