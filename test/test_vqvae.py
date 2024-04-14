import torch
from tqdm import tqdm
from stadiffuser.models import SpaUNet1DModel
from stadiffuser.pipeline import prepare_dataset
from diffusers import DDPMScheduler
from collections import defaultdict
from torch_geometric.loader import NeighborLoader
import scanpy as sc
import os
import matplotlib.pyplot as plt
import stadiffuser.pipeline as pipeline
import warnings
warnings.filterwarnings("ignore")


# if i import scanpy first, debugger will raise error!

def eval_fn(denoise_net, niose_scheduler, epoch, adata=None, spatial=None, labels=None,
            n_features=32, normalizer=None, save_dir=None, progress=True, device="cpu"):
    simulated = pipeline.simulate(denoise_net, niose_scheduler, spatial.to(device), labels=labels.to(device),
                                  n_features=n_features, progress=progress)
    simulated = simulated.cpu().numpy()
    if normalizer is not None:
        simulated = normalizer.denormalize(simulated)
    adata_sim = sc.AnnData(simulated)
    sc.pp.neighbors(adata_sim, n_neighbors=30)
    sc.tl.umap(adata_sim, min_dist=0.2)
    adata_sim.obs["labels"] = labels.cpu().numpy().astype("str")
    adata_sim.obsm["spatial"] = adata.obsm["spatial"].copy()
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    sc.pl.umap(adata_sim, color=["labels"], show=False, ax=ax, frameon=False)
    plt.tight_layout()
    if save_dir is not None:
        fig.savefig(os.path.join(save_dir, "epoch{}-embedding.png".format(epoch)), dpi=300)
    return adata_sim

# free gpu memory
torch.cuda.empty_cache()
save_dir = "E:\Projects\diffusion\output\Hippocampus_condition"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
adata = sc.read_h5ad('E:\Projects\diffusion\output\Hippocampus_condition\J20_hippocampus_rep3.h5ad')
data_latent = prepare_dataset(adata, use_rep="normalized_latent", use_spatial="spatial_new", use_net="Spatial_Net",
                              use_label="cell_type")
loader = NeighborLoader(data_latent, num_neighbors=[5, 5], batch_size=64)
eval_kwargs = {"adata": adata, "spatial": data_latent.spatial,
               "labels": data_latent.label, "n_features": 32,
               "save_dir": os.path.join(save_dir, "_figures"),
               "device": device}

num_class_embeds = len(adata.obs["first_type"].unique())
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
denoise_net = SpaUNet1DModel(in_channels=17, out_channels=1, num_class_embeds=num_class_embeds).to(device)
denoise_net, denoise_loss = pipeline.train_denoiser(loader, denoise_net, noise_scheduler,
                                                    lr=1e-4, weight_decay=1e-6, n_epochs=50,
                                                    num_class_embeds=num_class_embeds,
                                                    save_dir=save_dir,
                                                    device=device, eval_fn=eval_fn,
                                                    evaluate_interval=100,
                                                    eval_kwargs=eval_kwargs)
# adata = sc.read_h5ad("E:\Projects\diffusion\dataset\E16-18h_a_count_normal_stereoseq.h5ad")
# batch_list = adata.obs["slice_ID"].unique().tolist()
# used_batches = batch_list[:3]
# adata = adata[adata.obs["slice_ID"].isin(used_batches)]
# print(adata)
# iter_comb = [(used_batches[i], used_batches[i+1]) for i in range(len(used_batches)-1)]
# adata = utils.cal_spatial_net3D(adata, iter_comb=iter_comb, batch_id="slice_ID", rad_cutoff=1.5, z_k_cutoff=3,
#                                 add_key="spatial_net")
