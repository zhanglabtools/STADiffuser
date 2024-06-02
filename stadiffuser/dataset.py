"""
This file contains the functionals and classes for dataset_hub construction and processing.
"""
import numpy as np
import scanpy as sc
import torch
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from scipy.spatial import cKDTree


def get_slice_loader(adata, data, batch_name, use_batch="slice_ID", batch_size=32, **kwargs):
    """
    Get the NeighborLoader for the slice batch.

    Parameters
    ----------
    adata: `anndata.AnnData`
        The AnnData object containing the original data of multiple batches.

    data: `torch_geometric.data.Data`
        The pytorch_geometric.data.Data object, containing the node features `data.x` and the edges `data.edge_index`.
        Constructed by the `prepare_data` function in the `stadiffuser.pipeline` module.

    batch_name: str
        The name of the slice batch in the `use_batch` column of the `adata.obs`.

    use_batch: str
        The column name in the `adata.obs` to indicate the batch information.

    batch_size: int
        The batch size for the NeighborLoader.

    **kwargs:
        The additional arguments for the NeighborLoader.

    Returns
    -------
    train_loader: `torch_geometric.loader.NeighborLoader`
        The NeighborLoader for the slice batch.
    """
    input_nodes = np.where(adata.obs[use_batch] == batch_name)[0]
    train_loader = NeighborLoader(data, num_neighbors=[5, 3], batch_size=batch_size,
                                  input_nodes=input_nodes, **kwargs)
    return train_loader


class MaskNode:
    r"""
       A class to mask node features in a graph and remove edges connected to masked nodes.

       This class provides a way to mask specific nodes in a graph and remove the edges that are connected
       to these nodes. It keeps track of the masked nodes and the edges that have been removed, allowing
       for the creation of a modified graph data object with the masked nodes and remaining edges.

       Parameters
       ----------
       data : torch_geometric.data.Data
           The input data containing the node features `data.x` and the edges `data.edge_index`.
       mask : numpy.ndarray
           A boolean array indicating which nodes to mask. True for masked nodes and False for unmasked nodes.

       Attributes
       ----------
       mask : numpy.ndarray
           The boolean mask indicating which nodes are masked.
       masked_nodes : numpy.ndarray
           The indices of the masked nodes.
       remained_nodes : numpy.ndarray
           The indices of the nodes that remain after masking.
       node_mapping : dict
           A dictionary mapping the indices of the remained nodes to a new consecutive index.
    """
    def __init__(self,
                 data: Data,
                 mask: np.ndarray,
                 ):
        r"""
               Initialize the MaskNode object by setting up the mask and computing the masked and remained nodes.

               Parameters
               ----------
               data : torch_geometric.data.Data
                   The input data containing the node features `data.x` and the edges `data.edge_index`.
               mask : numpy.ndarray
                   A boolean array indicating which nodes to mask. True for masked nodes and False for unmasked nodes.
        """
        assert data.x.shape[0] == mask.shape[0], "The number of nodes in data and mask must be the same."
        self.mask = mask
        self.masked_nodes = np.where(mask == 1)[0]
        self.remained_nodes = np.where(mask == 0)[0]
        self.node_mapping = dict(zip(self.remained_nodes, range(
            self.remained_nodes.shape[0])))  # map the original node index to the new node index

    def get_data(self, data):
        r"""
        Create a new data object with masked node features and removed edges.

        This method applies the mask to the node features and removes edges connected to masked nodes.
        It returns a new `Data` object with the modified node features and edge indices.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data containing the node features `data.x` and the edges `data.edge_index`.

        Returns
        -------
        torch_geometric.data.Data
            The modified data with masked node features and removed edges.
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
       Construct triplet data for training from batch-disjoint AnnData and PyG Data objects.

       This class creates triplets consisting of an anchor node, a positive node (nearest neighbor
       from a specified reference batch), and a negative node (randomly chosen from the target batch).
       It uses a specified representation from the AnnData object to find nearest neighbors and construct
       the triplets.

       Parameters
       ----------
       adata : anndata.AnnData
           The AnnData object containing the original data of multiple batches.
       target : str, optional
           The target batch id for anchor and negative nodes. Defaults to None.
       reference : str, optional
           The reference batch id for positive nodes. Defaults to None.
       use_batch : str, optional
           The batch id to be used for training. Defaults to None.
       use_rep : str, optional
           The key for the representation in `adata.obsm` to be used for nearest neighbor search.
           Defaults to None, in which case `adata.X` is used.
       num_neighbors : int, optional
           The number of nearest neighbors to be used for triplet construction. Defaults to 30.
       random_seed : int, optional
           The random seed for numpy's random number generator. Defaults to 0.

       Attributes
       ----------
       target_indices : numpy.ndarray
           The indices of nodes in the target batch.
       reference_indices : numpy.ndarray
           The indices of nodes in the reference batch.
       num_neighbors : int
           The number of nearest neighbors for triplet construction.
       rng : numpy.random.RandomState
           The random number generator for sampling negative nodes.
       mutual_dict : dict
           A dictionary mapping anchor node indices to lists of positive node indices.

       Examples
       --------
       >>> sampler = TripletSampler(adata, target='batch1', reference='batch2', use_batch='batch', use_rep='X_pca')
       >>> anchor_indices, positive_indices, negative_indices = sampler.query(anchor_indices)
       """

    def __init__(self,
                 adata: sc.AnnData,
                 target: str = None,
                 reference: str = None,
                 use_batch: str = None,
                 use_rep: str = None,
                 num_neighbors: int = 30,
                 random_seed: int = 0
                 ):
        # check the input shape
        rep = adata.obsm[use_rep] if use_rep is not None else adata.X
        target_rep = rep[adata.obs[use_batch] == target, :]
        reference_rep = rep[adata.obs[use_batch] == reference, :]
        # get the indice of all the nodes
        all_indices = np.arange(adata.shape[0])
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
        r"""
        Query the positive and negative indices for the given anchor indices.

        For each anchor index, this method finds the corresponding positive indices (nearest neighbors
        from the reference batch) and samples negative indices (randomly chosen from the target batch).

        Parameters
        ----------
        anchor_indices : numpy
            The indices of the anchor nodes.
        """
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
