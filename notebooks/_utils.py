import scanpy as sc
import pandas as pd
import torch
import os
import rdata
import numpy as np
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
from stadiffuser import pipeline
from stadiffuser import metrics



# Set environment variables for R
os.environ["R_HOME"] = r"D:\Program Files\R\R-4.0.3"

method_palette = {
    'splatter':  '#1f78b4',  # Blue
    'kersplatter': '#a6cee3',  # Lighter blue
    'zinb_spatial': "#ff7f0e",  # coral
    'SRT_domain':  '#33a02c',  # Green
    'scDesign_label': '#6a3d9a',  # Purple
    'stadiff': '#e31a1c',  #red
    "stadiff_label": "#fb9a99", #lighter red
}

def prep_count(X):
    adata = sc.AnnData(X)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata.X


def get_rep(autoencoder, x, edge_index):
    """
    Get the latent representation of x and the reconstructed x.
    """
    with torch.no_grad():
        latent, recon_X = autoencoder(x, edge_index)
    return latent.detach().cpu().numpy(), recon_X.detach().cpu().numpy()


class SimRes:
    def __init__(self, name, **kwargs):
        self._data = dict()
        self.name = name
        for key, value in kwargs.items():
            self._data[key] = value

    def __repr__(self):
        """
        Return a string representation of the object.
        `Sim{name}(key=value.shape, ...)` If the value is None show None.
        """
        repr_str = "{}(".format(self.name)
        for key, value in self._data.items():
            if value is None:
                repr_str += f"{key}=None, "
            else:
                repr_str += f"{key}={value.shape}, "
        repr_str = repr_str[:-2] + ")"
        return repr_str

    def __getitem__(self, key):
        return self._data[key]


# define a function that
def prepare_eval(name, autoencoder, sim_latent=None, sim_count=None, edge_index=None, device="cpu"):
    """
    Prepare the data for evaluation.Return a dictionary with keys:
    - sim_latent: latent representation of the data
    - sim_recon: reconstructed data from simulated data
    - sim_count: simulated count data
    """
    if sim_count is not None:
        normalized_count = prep_count(sim_count)
        sim_latent, sim_recon = get_rep(autoencoder.to(device), torch.from_numpy(normalized_count).float().to(device),
                                        edge_index.to(device))
    if sim_latent is not None:
        with torch.no_grad():
            sim_recon = autoencoder.decode(torch.from_numpy(sim_latent).to(device).float(),
                                           edge_index.to(device)).detach().cpu().numpy()
    return SimRes(name, latent=sim_latent, recon=sim_recon, count=sim_count)


def read_simRDS(path, ref_adata=None, keep_all=False, compress=True):
    res =  rdata.parser.parse_file(path)
    res = rdata.conversion.convert(res)
    # convert res["sim_count"] to np.array with dtype float32
    res["sim_count"] = np.array(res["sim_count"], dtype=np.float32)
    if compress:
        # save use scipy.sparse.csr_matrix
        res["sim_count"] = scipy.sparse.csr_matrix(res["sim_count"])
    if ref_adata is not None:
        if keep_all:
            sim_adata = ref_adata.copy()
            sim_adata.obs["labels"] = res["sim_labels"]
            sim_adata.obsm["spatial"] = np.array(res["sim_spatial"])
            sim_adata.X = res["sim_count"].T
            return sim_adata
        else:
            sim_adata = sc.AnnData(X=res["sim_count"].T, obs={"labels": res["sim_labels"]})
        return sim_adata
    else:
        return res


def plot_umap(adata, latent, title=None, ax=None, frameon=False, seed=None):
    # if x is not torch tensor, convert it to torch tensor
    # if x is sparse matrix, convert it to numpy
    # set seed
    if seed is not None:
        np.random.seed(seed)
    adata.obsm["latent"] = latent
    sc.pp.neighbors(adata, use_rep="latent", n_neighbors=30)
    sc.tl.umap(adata, min_dist=0.5)
    sc.pl.umap(adata, color=["ground_truth"], ncols=1, frameon=frameon, wspace=0.4,
               use_raw=False, show=False, title=title, ax=ax, legend_loc='none')


def plot_performance(perf_df, method_palette, plot_methods=None):
    if plot_methods is None:
        plot_methods = ["splatter", "kersplatter", "zinb_spatial",  "scDesign", "SRT_domain", "stadiff_label"]
    plot_palette = {method: method_palette[method] for method in plot_methods}
    plot_palette["stadiff_label"] = method_palette["stadiff"]
    perf_df_drop = perf_df.loc[plot_methods]
    fig, ax = plt.subplots(1, 1, figsize=(3.25, 3.5))
    # set color palette
    perf_df_drop["method"] = perf_df_drop.index
    sns.scatterplot(data=perf_df_drop, x="enhanced_gene_diversity", y="enhanced_gene_corr",
                    hue="method",
                    ax=ax,
                    style="method",
                    palette=plot_palette,
                    markers="o",
                    alpha=0.7,
                    legend=False, s=100)
    # show legend
    # ax.set_ylim(0.2, 1)
    ax.set_xlabel("")
    ax.set_ylabel("")
    return fig

def go_plot_palette(go_df, pattern_dict, unknown_color="grey"):
    plot_palette = dict()
    # interate on the row of go_df
    for i in range(go_df.shape[0]):
        row = go_df.iloc[i]
        # if row matches the pattern_dict, then set the color to the pattern_dict
        matched = False
        for pattern, color in pattern_dict.items():
            if pattern in row["name"].lower():
                plot_palette[row["name"]] = color
                matched = True
        if not matched:
            plot_palette[row["name"]] = unknown_color
    return plot_palette


def compute_perf_dict(comp_sim_dict, comp_methods, autoencoder, adata_raw, adata_real_recon, device="cuda:0"):
    comp_perf_dict = {}
    for name in comp_methods:
        print("==== {} ====".format(name))
        adata_list = comp_sim_dict[name]
        if name == "stadiff_label" or name == "stadiff":
            comp_perf_dict[name] = compute_sim_perf(comp_sim_dict[name], autoencoder, adata_raw, adata_real_recon,
                                                    enhanced=True, n_rep=5, device=device)
        else:
            adata_temp_list = []
            for adata in adata_list:
                adata = adata.copy()
                adata.X = adata.X.toarray()
                adata.uns["spatial_net"] = adata_raw.uns["spatial_net"]
                adata.obsm["spatial"] = adata_raw.obsm["spatial"]
                # set obs_name
                adata.obs_names = adata_raw.obs_names
                adata_temp_list.append(adata)
            adata_list = adata_temp_list
            comp_perf_dict[name] = compute_sim_perf(adata_list, autoencoder, adata_raw, adata_real_recon,
                                                   enhanced=False, n_rep=5, device=device)
    return comp_perf_dict


def compute_sim_perf(sim_adata_list, autoencoder, adata_raw, adata_real_recon,
                     enhanced = False,
                     n_rep=5,
                     device="cuda:0"):
    sim_perf = dict()
    metric_names = ["cell_corr", "enhanced_cell_corr", "pairwise_cell_corr", "pairwise_enhanced_cell_corr"]
    metric_names += ["gene_corr", "enhanced_gene_corr", "pairwise_gene_corr", "pairwise_enhanced_gene_corr"]
    for name in metric_names:
        sim_perf[name] = []
    if not enhanced:
        adata_sim_recon_list = []
    else:
        adata_sim_recon_list = sim_adata_list
    for ind in range(n_rep):
        adata_sim = sim_adata_list[ind]
        if not enhanced:
            adata_sim_recon = pipeline.get_recon(adata_sim, autoencoder, device=device, use_net="spatial_net")
            adata_sim_recon_list.append(adata_sim_recon)
        else:
            adata_sim_recon = adata_sim
        for dim in ["cell", "gene"]:
            sim_perf[dim + "_corr"].append(np.nanmean(metrics.compute_corr(adata_sim, adata_raw, dim=dim)))
            sim_perf["enhanced_" + dim + "_corr"].append(np.nanmean(metrics.compute_corr(adata_sim_recon, adata_real_recon, dim=dim)))

    for (i, j) in itertools.combinations(range(n_rep), 2):
        for dim in ["cell", "gene"]:
            sim_perf["pairwise_" + dim + "_corr"].append(np.nanmean(metrics.compute_corr(sim_adata_list[i], sim_adata_list[j], dim=dim)))
            sim_perf["pairwise_enhanced_" + dim + "_corr"].append(np.nanmean(metrics.compute_corr(adata_sim_recon_list[i], adata_sim_recon_list[j], dim=dim)))
    return sim_perf


def compute_mLISI(comp_sim_dict, comp_methods, adata_real_recon, autoencoder, device):
    comp_perf_dict = {}
    for method_name in comp_methods:
        print("Method: ", method_name)
        comp_perf_dict[method_name] = {"mLISI": []}
        adata_sim = comp_sim_dict[method_name][0]
        if method_name == "stadiff_label" or method_name == "stadiff":
            adata_sim_recon = adata_sim
        else:
            adata_sim = adata_sim.copy()
            adata_sim.X = adata_sim.X.toarray()
            adata_sim.uns["spatial_net"] = adata_real_recon.uns["spatial_net"]
            adata_sim.obsm["spatial"] = adata_real_recon.obsm["spatial"]
            adata_sim.obs_names = adata_real_recon.obs_names
            adata_sim_recon = pipeline.get_recon(adata_sim, autoencoder, device=device, use_net="spatial_net")
        comp_perf_dict[method_name]["mLISI"].append(metrics.compute_paired_lisi(adata_sim_recon, adata_real_recon))
    return comp_perf_dict


def extract_perf_df(comp_methods, comp_perf_dict):
    metric_names = ["cell_corr", "enhanced_cell_corr", "pairwise_cell_corr", "pairwise_enhanced_cell_corr"]
    metric_names += ["gene_corr", "enhanced_gene_corr", "pairwise_gene_corr", "pairwise_enhanced_gene_corr"]
    perf_df = []
    for sim_name in comp_methods:
        perf_df.append([np.nanmean(comp_perf_dict[sim_name][name]) for name in metric_names])
    perf_df = pd.DataFrame(perf_df, index=comp_methods, columns=metric_names)
    for group in ["cell", "gene"]:
        perf_df["enhanced_" + group + "_diversity"] = 1 - perf_df["pairwise_enhanced_" + group + "_corr"]
        perf_df[group + "_diversity"] = 1 - perf_df["pairwise_" + group + "_corr"]
    return perf_df
