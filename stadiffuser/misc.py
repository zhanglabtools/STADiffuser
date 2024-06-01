import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

class KNN3dImputation:
    def __init__(self, adata_ref, adj_ids=None, use_label=None, use_batch="slice_ID"):
        self.ref = adata_ref.copy()
        # construct the KDTree for each label
        self.use_label = use_label
        self.use_batch = use_batch
        self.tree_adj = dict()
        self.adj_ids = adj_ids
        if use_label is not None:
            unique_labels = adata_ref.obs[use_label].unique()
            self.unique_labels = unique_labels
        else:
            self.unique_labels = ["ref"]
        for ind, id in enumerate(adj_ids):
            self.tree_adj[id] = dict()
            # compute the kdtree between the spatial coordinates of adata_out and adata_adj for each label
            adata_adj = adata_ref[adata_ref.obs[use_batch] == id].copy()
            for label in unique_labels:
                adata_adj_spatial = adata_adj[adata_adj.obs[use_label] == label].obsm["spatial"].copy()
                # computet the KDTree between adata_out and adata_adj
                self.tree_adj[id][label] = cKDTree(adata_adj_spatial)
                self.tree_adj[id][label + "_names"] = adata_adj[adata_adj.obs[use_label] == label].obs_names.to_numpy()

    def impute(self, adata_target, k_neighbors=5):
        adata_out_imp = adata_target.copy()
        adata_adj_list =[]
        for id in self.adj_ids:
            adata_adj = self.ref[self.ref.obs[self.use_batch] == id].copy()
            adata_adj_list.append(adata_adj)
        for label in self.unique_labels:
            X_imp = 0
            for ind, id in enumerate(self.adj_ids):
                adata_adj = adata_adj_list[ind]
                adata_out_spatial = adata_target[adata_target.obs[self.use_label] == label].obsm["spatial"].copy()
                tree = self.tree_adj[id][label]
                dist, indices = tree.query(adata_out_spatial, k=k_neighbors)
                names = self.tree_adj[id][label + "_names"][indices]
                names = names.flatten()
                X_neighbor = adata_adj[names].X
                X_imp += pd.DataFrame(X_neighbor).groupby(np.arange(len(X_neighbor))//k_neighbors).mean()
            X_imp /= len(self.adj_ids)
            adata_out_imp[adata_target.obs[self.use_label] == label].X = X_imp.to_numpy()
        return adata_out_imp