"""
This file contains the functionals and classes for dataset construction and processing.
"""
import numpy as np
import scanpy as sc
import torch
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
# import cKDTree
from scipy.spatial import cKDTree


def get_slice_loader(adata, data, batch_name, use_batch="slice_ID", batch_size=32, **kwargs):
    input_nodes = np.where(adata.obs[use_batch] == batch_name)[0]
    train_loader = NeighborLoader(data, num_neighbors=[5, 3], batch_size=batch_size,
                                  input_nodes=input_nodes, **kwargs)
    return train_loader


class MaskNode:
    r"""
    Mask the node features in the data and remove the edges connected to the masked nodes. Record the masked nodes
    and the removed edges.
    Parameters
    ----------
    data:
        The pytorch_geometric.data.Data object, containing the node features `data.x` and the edges `data.edge_index`.

    mask:
        The mask of the nodes to be masked. 1 for masked and 0 for unmasked.
    """

    def __init__(self,
                 data: Data,
                 mask: np.ndarray,
                 lazy: bool = True
                 ):
        assert data.x.shape[0] == mask.shape[0], "The number of nodes in data and mask must be the same."
        self.mask = mask
        self.masked_nodes = np.where(mask == 1)[0]
        self.remained_nodes = np.where(mask == 0)[0]
        self.node_mapping = dict(zip(self.remained_nodes, range(
            self.remained_nodes.shape[0])))  # map the original node index to the new node index

    def get_data(self, data):
        r"""
        Mask the node features in the data and remove the edges connected to the masked nodes. Record the masked nodes
        and the removed edges.
        Parameters
        ----------
        data:
            The pytorch_geometric.data.Data object, containing the node features `data.x` and the edges `data.edge_index`.
        """
        print("Mask {} nodes and there are {} spots remaining".format(len(self.masked_nodes), len(self.remained_nodes)))
        x = data.x[~self.mask.astype(bool), :]
        remained_nodes = torch.from_numpy(self.remained_nodes).to(data.x.device)
        remained_edges = torch.isin(data.edge_index[0, :], remained_nodes) & \
                         torch.isin(data.edge_index[1:], remained_nodes).squeeze(0)
        edge_index = data.edge_index[:, remained_edges]
        # chagne the node index in edge_index
        edge_index[0] = torch.tensor([self.node_mapping[i.item()] for i in edge_index[0]])
        edge_index[1] = torch.tensor([self.node_mapping[i.item()] for i in edge_index[1]])
        return Data(x=x, edge_index=edge_index)


def _tuples_to_dict(mutual_pairs):
    """
    Convert the tuple of (target_node_index, reference_node_index) to a dictionary of {target_node_index: reference_node_index}.
    """
    mutual_dict = {}
    for target_index, reference_index in mutual_pairs:
        # Check if the target_index is already in the dictionary
        if target_index in mutual_dict:
            # If it is, append the reference_index to the list
            mutual_dict[target_index].append(reference_index)
        else:
            # If it's not, create a new entry with a list containing the reference_index
            mutual_dict[target_index] = [reference_index]
    return mutual_dict


class TripletSampler:
    """
    Construct the triplet data for training.
    Parameters
    ----------
    adata:
        The AnnData object containing the original data of multiple batches.
    data:
        The pytorch_geometric.data.Data object, containing the node features `data.x` and the edges `data.edge_index`.
    target:
        The target batch id.
    reference:
        The reference batch id.
    use_batch:
        The batch id to be used for training.
    use_rep:
        The representation to be used for searching the nearest neighbors.
    num_neighbors:
        The number of nearest neighbors to be used for triplet construction.
    """

    def __init__(self,
                 adata: sc.AnnData,
                 data,
                 target: str,
                 reference: str,
                 use_batch: str = None,
                 use_rep: str = None,
                 num_neighbors: int = 30,
                 random_seed: int = 0
                 ):
        # check the input shape
        assert data.x.shape[0] == adata.shape[0], "The number of nodes in data and adata must be the same."
        rep = adata.obsm[use_rep] if use_rep is not None else adata.X
        target_rep = rep[adata.obs[use_batch] == target, :]
        reference_rep = rep[adata.obs[use_batch] == reference, :]
        # get the indice of all the nodes
        all_indices = np.arange(data.x.shape[0])
        self.target_indices = all_indices[adata.obs[use_batch] == target]
        self.reference_indices = all_indices[adata.obs[use_batch] == reference]
        # construct the tree for query
        self.num_neighbors = num_neighbors
        self.rng = np.random.RandomState(random_seed)
        # find the k nearest neighbors for all the nodes in the target batch
        target_tree = cKDTree(target_rep)
        reference_tree = cKDTree(reference_rep)
        _, target_neighbor_indices = reference_tree.query(target_rep, k=num_neighbors)
        _, reference_neighbor_indices = target_tree.query(reference_rep, k=num_neighbors)
        # construct the tuple of (target_node_index, reference_node_index)
        pairs1 = [(self.target_indices[i], self.reference_indices[neighbor_index]) for i, neighbors in enumerate(target_neighbor_indices)
                  for neighbor_index in neighbors]
        pairs2 = [(self.target_indices[neighbor_index], self.reference_indices[i]) for i, neighbors in enumerate(reference_neighbor_indices)
                  for neighbor_index in neighbors]
        # find the mutual nearest neighbors of the target batch
        mutual_pairs = set(pairs1).intersection(set(pairs2))
        self.mutual_dict = _tuples_to_dict(mutual_pairs)

    def query(self, anchor_indices):
        """
        Query the positive and negative indices for the anchor indices.
        """
        # query the mutual nearest neighbors
        anchor_indices_list = []
        positive_indices_list = []
        negative_indices_list = []
        for anchor_index in anchor_indices:
            try:
                positive_indices = np.array(self.mutual_dict[anchor_index])
                negative_indices = self.rng.choice(self.target_indices, size=len(positive_indices), replace=False)
                anchor_indices_list.append(anchor_index * np.ones(len(positive_indices)))
                positive_indices_list.append(positive_indices)
                negative_indices_list.append(negative_indices)
            except KeyError:
                continue
        anchor_indices = np.concatenate(anchor_indices_list)
        positive_indices = np.concatenate(positive_indices_list)
        negative_indices = np.concatenate(negative_indices_list)
        return anchor_indices, positive_indices, negative_indices
