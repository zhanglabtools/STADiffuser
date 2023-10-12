
import torch
from tqdm import tqdm
from stadiff.models import SpaUNet1DModel
from stadiff.pipeline import prepare_dataset
from diffusers import DDPMScheduler
from collections import defaultdict
from torch_geometric.loader import NeighborLoader
import scanpy as sc
# if i import scanpy first, debugger will raise error!

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = "cpu"
adata = sc.read_h5ad('E:\Projects\diffusion\output\Hippocampus_condition\J20_hippocampus_rep3.h5ad')
data_latent = prepare_dataset(adata, use_rep="normalized_latent", use_spatial="spatial_new", use_net="Spatial_Net",
                              use_label="cell_type")
loader = NeighborLoader(data_latent, num_neighbors=[5, 5], batch_size=64)
num_class_embeds = len(adata.obs["first_type"].unique())
denoise_net = SpaUNet1DModel(in_channels=17, out_channels=1, num_class_embeds=num_class_embeds).to(device)
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

# adata = sc.read_h5ad("E:\Projects\diffusion\dataset\E16-18h_a_count_normal_stereoseq.h5ad")
# batch_list = adata.obs["slice_ID"].unique().tolist()
# used_batches = batch_list[:3]
# adata = adata[adata.obs["slice_ID"].isin(used_batches)]
# print(adata)
# iter_comb = [(used_batches[i], used_batches[i+1]) for i in range(len(used_batches)-1)]
# adata = utils.cal_spatial_net3D(adata, iter_comb=iter_comb, batch_id="slice_ID", rad_cutoff=1.5, z_k_cutoff=3,
#                                 add_key="spatial_net")

