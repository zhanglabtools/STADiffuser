import os
# os.chdir("..")
import scanpy as sc
import matplotlib.pyplot as plt
import torch
import seaborn as sns
import warnings
from tqdm.auto import tqdm
warnings.filterwarnings("ignore")
sns.set_theme("paper", style="ticks", font_scale=1.25)




if __name__ == "__main__":
    from stadiffuser.models import SpaUNet1DModel
    from stadiffuser.pipeline import prepare_dataset
    from diffusers import DDPMScheduler
    from collections import defaultdict
    from torch_geometric.loader import NeighborLoader
    # os.environ["R_HOME"] = r"D:\Program Files\R\R-4.0.3"
    adata = sc.read_h5ad('E:\Projects\diffusion\output\Hippocampus_condition\J20_hippocampus_rep3.h5ad')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data_latent = prepare_dataset(adata, use_rep="normalized_latent", use_spatial="spatial_new", use_net="Spatial_Net",
                                  use_label="cell_type")
    loader = NeighborLoader(data_latent, num_neighbors=[5, 5], batch_size=64)
    num_class_embeds = len(adata.obs["first_type"].unique())
    denoise_net = SpaUNet1DModel(in_channels=18, out_channels=1, num_class_embeds=num_class_embeds).to(device)
    optimizer = torch.optim.AdamW(denoise_net.parameters(), lr=1e-4, weight_decay=1e-6)
    niose_scheduler = DDPMScheduler(num_train_timesteps=1000)
    n_epochs = 1
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    pbar = tqdm(range(n_epochs))
    evaluate_interval = 100
    logging = defaultdict(list)
    batch = next(iter(loader))
    batch = batch.to(device)
    clean_data = batch.x
    clean_data = clean_data.unsqueeze(1)  # (batch_size, 1, num_channels)
    optimizer.zero_grad()
    noise = torch.randn_like(clean_data)
    timesteps = torch.randint(0, 1000, (clean_data.shape[0],), device=batch.x.device, dtype=torch.long, )
    noisy_data = niose_scheduler.add_noise(clean_data, noise, timesteps)
    noise_pred = denoise_net(noisy_data, timesteps, batch.spatial, batch.label).sample




