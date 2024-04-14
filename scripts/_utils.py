import warnings
import matplotlib as mpl
import matplotlib.pyplot as plt
import itertools
import seaborn as sns
import rdata
import numpy as np
import torch
import scanpy as sc
from stadiffuser import pipeline, metrics

sns.set_theme(context="paper", style="ticks", font_scale=1.5)


method_palette = {
    'splatter':  '#8A98B8',  # Blue
    'kersplatter': '#B5BDD2',  # Lighter blue
    'zinb_spatial': "#EBAB71",  # coral
    'SRT_domain':  '#8CCF87',  # Green
    'scDesign_label': '#C0AFE2',  # Purple
    'stadiff': '#EA6666',  #coral
    "stadiff_label": "#CF4144", #lighter red
}


def read_simRDS(path, ref_adata=None):
    res =  rdata.parser.parse_file(path)
    res = rdata.conversion.convert(res)
    if ref_adata is not None:
        sim_adata = ref_adata.copy()
        sim_adata.obs["labels"] = res["sim_labels"]
        sim_adata.obsm["spatial"] = np.array(res["sim_spatial"])
        sim_adata.X = res["sim_count"].T
        return sim_adata
    else:
        return res
    

def read_adata_list(path, name, n_rep, ref_data=None):
    adata_list = []
    for i in range(1, 1 + n_rep):
        adata_sim = read_simRDS(path + name.format(i), ref_adata=ref_data)
        adata_list.append(adata_sim)
    return adata_list


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
            adata_sim_recon = get_recon(adata_sim, autoencoder, device=device, use_net="spatial_net")
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