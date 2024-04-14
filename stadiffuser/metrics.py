"""
This module contains functions for calculating metrics on the simulated data.
The expected input is AnnData object
"""
import numpy as np
import pandas as pd
import scanpy as sc
import scipy
from scanpy import AnnData
from scipy.stats import spearmanr
from typing import Iterable
from sklearn.neighbors import NearestNeighbors


def compute_corr(sim_ad,
                 ref_ad,
                 corr_type="pearson",
                 dim="gene",
                 mask=None,
                 retrun_df=False):
    """
    Calculate the gene-wise correlation between the simulated data and the real data.
    :param sim_ad: AnnData object of the simulated data
    :param ref_ad: AnnData object of the real data
    :param corr_type: "pearson" or "spearman"
    :param mask: mask of the region of interest
    """
    sim_data = sim_ad.X
    ref_data = ref_ad.X
    n_spots, n_genes = sim_data.shape
    if mask is not None:
        print("Compute correlation on masked region")
        sim_data = sim_data[mask, :]
        ref_data = ref_data[mask, :]
    if corr_type == "pearson":
        corr_fn = lambda x, y: np.corrcoef(x, y)[0, 1]
    elif corr_type == "spearman":
        corr_fn = lambda x, y: spearmanr(x, y)[0]
    else:
        raise ValueError("corr_type must be one of 'pearson' or 'spearman'")
    if dim == "gene":
        corr = np.zeros(sim_data.shape[1])
        for i in range(sim_data.shape[1]):
            corr[i] = corr_fn(sim_data[:, i], ref_data[:, i])
        if retrun_df:
            corr = pd.DataFrame(corr, index=sim_ad.var_names, columns=["corr"])
        else:
            corr = np.array(corr)
    elif dim == "cell":
        corr = np.zeros(sim_data.shape[0])
        for i in range(sim_data.shape[0]):
            corr[i] = corr_fn(sim_data[i, :], ref_data[i, :])
        corr = pd.DataFrame(corr, index=sim_ad.obs_names, columns=["corr"])
    else:
        raise ValueError("dim must be one of 'gene' or 'cell'")
    return corr


def compute_lisi(
        X: np.array,
        metadata: pd.DataFrame,
        label_colnames: Iterable[str],
        perplexity: float = 30
):
    """Compute the Local Inverse Simpson Index (LISI) for each column in metadata.

    LISI is a statistic computed for each item (row) in the data matrix X.

    The following example may help to interpret the LISI values.

    Suppose one of the columns in metadata is a categorical variable with 3 categories.

        - If LISI is approximately equal to 3 for an item in the data matrix,
          that means that the item is surrounded by neighbors from all 3
          categories.

        - If LISI is approximately equal to 1, then the item is surrounded by
          neighbors from 1 category.

    The LISI statistic is useful to evaluate whether multiple datasets are
    well-integrated by algorithms such as Harmony [1].

    [1]: Korsunsky et al. 2019 doi: 10.1038/s41592-019-0619-0
    Borrowed from harmonypy package
    """
    n_cells = metadata.shape[0]
    n_labels = len(label_colnames)
    # We need at least 3 * n_neigbhors to compute the perplexity
    knn = NearestNeighbors(n_neighbors=perplexity * 3, algorithm='kd_tree').fit(X)
    distances, indices = knn.kneighbors(X)
    # Don't count yourself
    indices = indices[:, 1:]
    distances = distances[:, 1:]
    # Save the result
    lisi_df = np.zeros((n_cells, n_labels))
    for i, label in enumerate(label_colnames):
        labels = pd.Categorical(metadata[label])
        n_categories = len(labels.categories)
        simpson = compute_simpson(distances.T, indices.T, labels, n_categories, perplexity)
        lisi_df[:, i] = 1 / simpson
    return lisi_df


def compute_simpson(
        distances: np.ndarray,
        indices: np.ndarray,
        labels: pd.Categorical,
        n_categories: int,
        perplexity: float,
        tol: float = 1e-5
):
    n = distances.shape[1]
    P = np.zeros(distances.shape[0])
    simpson = np.zeros(n)
    logU = np.log(perplexity)
    # Loop through each cell.
    for i in range(n):
        beta = 1
        betamin = -np.inf
        betamax = np.inf
        # Compute Hdiff
        P = np.exp(-distances[:, i] * beta)
        P_sum = np.sum(P)
        if P_sum == 0:
            H = 0
            P = np.zeros(distances.shape[0])
        else:
            H = np.log(P_sum) + beta * np.sum(distances[:, i] * P) / P_sum
            P = P / P_sum
        Hdiff = H - logU
        n_tries = 50
        for t in range(n_tries):
            # Stop when we reach the tolerance
            if abs(Hdiff) < tol:
                break
            # Update beta
            if Hdiff > 0:
                betamin = beta
                if not np.isfinite(betamax):
                    beta *= 2
                else:
                    beta = (beta + betamax) / 2
            else:
                betamax = beta
                if not np.isfinite(betamin):
                    beta /= 2
                else:
                    beta = (beta + betamin) / 2
            # Compute Hdiff
            P = np.exp(-distances[:, i] * beta)
            P_sum = np.sum(P)
            if P_sum == 0:
                H = 0
                P = np.zeros(distances.shape[0])
            else:
                H = np.log(P_sum) + beta * np.sum(distances[:, i] * P) / P_sum
                P = P / P_sum
            Hdiff = H - logU
        # distancesefault value
        if H == 0:
            simpson[i] = -1
        # Simpson's index
        for label_category in labels.categories:
            ix = indices[:, i]
            q = labels[ix] == label_category
            if np.any(q):
                P_sum = np.sum(P[q])
                simpson[i] += P_sum * P_sum
    return simpson


def compute_paired_lisi(sim_data: AnnData,
                        ref_data: AnnData,
                        use_rep="latent",
                        zscore=True):
    """
    Compute the local inverse Simpson index for simulated data and reference data.
    """
    # combine the simulated data and the reference data
    if zscore:
        sim_data_ = sim_data.copy()
        ref_data_ = ref_data.copy()
        sim_data_.obsm[use_rep] = scipy.stats.zscore(sim_data_.obsm[use_rep], axis=0)
        ref_data_.obsm[use_rep] = scipy.stats.zscore(ref_data_.obsm[use_rep], axis=0)
    combined_data = sim_data_.concatenate(ref_data_)
    combined_data.obs["batch"] = ["sim"] * sim_data.shape[0] + ["ref"] * ref_data.shape[0]
    # perform umap on the combined data
    sc.pp.neighbors(combined_data, use_rep=use_rep, n_neighbors=30)
    sc.tl.umap(combined_data)
    # compute the lisi
    lisi = compute_lisi(combined_data.obsm["X_umap"], combined_data.obs, ["batch"])
    return lisi





