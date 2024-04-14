import torch
import yaml
import scanpy as sc
from scanpy import AnnData
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from typing import Union, List


def create_mask(region_type, spatial, **kwargs):
    assert region_type in ["circle", "rectangle"], "region_type must be one of 'circle' or 'rectangle"
    if region_type == "circle":
        assert "center" in kwargs.keys() and "radius" in kwargs.keys(), \
            "Center and radius must be provided for circle masking"
        center = kwargs["center"]
        radius = kwargs["radius"]
        assert len(center) == 2, "Center must be a 2-dim vector"
        assert radius > 0, "Radius must be positive"
        mask = np.linalg.norm(spatial - center, axis=1) <= radius
    elif region_type == "rectangle":
        assert "left_bottom" in kwargs.keys() and "right_top" in kwargs.keys(), \
            "Left bottom and right top must be provided for rectangle masking"
        left_bottom = kwargs["left_bottom"]
        right_top = kwargs["right_top"]
        assert len(left_bottom) == 2 and len(right_top) == 2, \
            "Left bottom and right top must be 2-dim vectors"
        assert left_bottom[0] < right_top[0] and left_bottom[1] < right_top[1], \
            "Left bottom must be smaller than right top"
        mask = np.logical_and(spatial[:, 0] >= left_bottom[0], spatial[:, 0] <= right_top[0])
        mask = np.logical_and(mask, spatial[:, 1] >= left_bottom[1])
        mask = np.logical_and(mask, spatial[:, 1] <= right_top[1])
    else:
        raise NotImplementedError
    return mask


def get_mask(adata: AnnData,
             use_spatial: str = "spatial",
             mask_args_list=None):
    """
    Mask a region of the AnnData object. Notes: Side effect: adata.obs[add_key] is added to adata.
    Parameters
    ----------
    adata:
        AnnData object and adata.obsm["spatial"] must exist.
    add_mask:
        The key of mask to add to adata.obs
    use_spatial:
        The key of spatial coordinates in adata.obsm
    use_net:
        The key of spatial network in adata.uns
    mask_args_list:
        list arguments for masking (region_type, other_args)
        Other arguments for masking. For circle, "center" and "radius" must be provided.
        For rectangle, "left_bottom" and "right_top" must be provided.

    Returns
    -------
    adata: AnnData with masked region in adata.obs[add_key], 1 for unmasked and 0 for masked.
    """
    # check adata.obsm["spatial"]
    spatial = adata.obsm[use_spatial]
    mask_list = []
    for (region_type, mask_arg) in mask_args_list:
        mask = create_mask(region_type, spatial, **mask_arg)
        mask_list.append(mask)
    for mask in mask_list:
        print(f"Mask contains {np.sum(mask)} spots")
    if len(mask_list) > 1:
        mask = np.logical_or(*mask_list)
    else:
        mask = mask_list[0]
    return mask


def mask_region(adata: AnnData,
                use_mask: str,
                use_net: str = "spatial_net"
                ):
    """
    Mask a region of the AnnData object using adata.obs[use_mask].
    """
    mask = adata.obs[use_mask]
    adata_mask = adata[mask, :]
    net = adata.uns[use_net]
    masked_nodes = adata.obs_names[np.logical_not(mask)]
    net_ = net[np.logical_not(np.logical_or(net['Cell1'].isin(masked_nodes), net['Cell2'].isin(masked_nodes)))]
    adata_mask.uns[use_net] = net_
    print("-" * 20)
    print("Remove {} nodes and {} edges from the original data".format(len(masked_nodes), len(net) - len(net_)))
    return adata_mask


def upsample_spatial(adata: AnnData, ratio:float, use_net: str = "spatial_net", use_spatial: str = "spatial",
                     seed=2024):
    """
    Upsample the spatial coordinates by insert new points between the existing points. ratio > 1.
    """
    # check valid ratio
    assert ratio > 1, "ratio must be greater than 1"
    spatial = adata.obsm[use_spatial]
    # net has four columns "Cell1", "Cell2", "Distance", "EdgeType"
    net = adata.uns[use_net]
    n_spots = spatial.shape[0]
    n_new_spots = int(n_spots * (ratio - 1))
    # random choose n row indices from net
    np.random.seed(seed)
    indices = np.random.choice(net.shape[0], n_new_spots)
    start_points = spatial[net["Cell1"].iloc[indices]].values
    end_points = spatial[net["Cell2"].iloc[indices]].values
    new_points = (start_points + end_points) / 2
    new_spatial = np.vstack((spatial, new_points))
    print("Upsample ratio = {}, add {} new spots".format(ratio, n_new_spots))
    return new_spatial



class MinMaxNormalize:
    """
    Minmax normalization for a torch tensor to (-1, 1)

    Parameters
    ----------
    tensor: torch.Tensor
        The tensor to be normalized
    dim: int
        The dimension to be normalized
    """

    def __init__(self, x, dim=0, dtype="float64"):
        self.dim = dim
        self.dtype = dtype
        x = x.astype(self.dtype)
        self.min = np.min(x, axis=dim, keepdims=True)
        self.max = np.max(x, axis=dim, keepdims=True)

    def normalize(self, x):
        x = x.astype(self.dtype)
        return 2 * (x - self.min) / (self.max - self.min) - 1

    def __call__(self, x):
        return self.normalize(x)

    def denormalize(self, x):
        x = x.astype(self.dtype)
        return (x + 1) / 2 * (self.max - self.min) + self.min


def _cal_spatial_net(spatial_df, rad_cutoff=None, k_cutoff=None, verbose=True):
    """
    Construct the spatial neighbor networks from the spatial coordinates dataframe.

    Parameters
    ----------
    spatial_df: pd.DataFrame
        The spatial coordinates dataframe with index cell names and columns for coordinates.
    rad_cutoff: float
        The radius cutoff when model="Radius"
    k_cutoff: int
        The number of nearest neighbors when model="KNN"
    verbose: bool
        Whether to print the information of the spatial network
    """
    assert rad_cutoff is not None or k_cutoff is not None, "Either rad_cutoff or k_cutoff must be provided"
    assert rad_cutoff is None or k_cutoff is None, "Only one of rad_cutoff and k_cutoff must be provided"
    if verbose:
        print('------Calculating spatial graph...')

    coor = spatial_df.values
    tree = cKDTree(coor)
    if rad_cutoff is not None:
        indices = tree.query_ball_point(coor, r=rad_cutoff)
    else:
        distances, indices = tree.query(coor, k=k_cutoff + 1)

    # construct the spatial df
    spatial_net_data = []
    for i, neighbors in enumerate(indices):
        cell1 = spatial_df.index[i]
        for j in neighbors:
            if i != j:  # Avoid self-loop
                cell2 = spatial_df.index[j]
                if rad_cutoff is not None:
                    distance = np.linalg.norm(coor[i] - coor[j])
                else:
                    distance = distances[i, j]
                spatial_net_data.append((cell1, cell2, distance))

    spatial_net = pd.DataFrame(spatial_net_data, columns=['Cell1', 'Cell2', 'Distance'])
    # add the edge type information "within"
    spatial_net['EdgeType'] = 'within'
    if verbose:
        print('------Spatial graph calculated.')
        print('The graph contains %d edges, %d cells, %.4f neighbors per cell on average.' % (
            spatial_net.shape[0], spatial_df.shape[0], spatial_net.shape[0] / spatial_df.shape[0]))
    return spatial_net


def _cal_spatial_bipartite(spatial_df1, spatial_df2, rad_cutoff=None, k_cutoff=None, verbose=True):
    """
    Construct the spatial neighbor across two spatial coordinates dataframe.

    Parameters
    ----------
    spatial_df1: pd.DataFrame
        The spatial coordinates dataframe with index cell names and columns for coordinates.
    spatial_df2: pd.DataFrame
        The spatial coordinates dataframe with index cell names and columns for coordinates.
    model: str
        "Radius" or "KNN"
    rad_cutoff: float
        The radius cutoff when model="Radius"
    k_cutoff: int
        The number of nearest neighbors when model="KNN"
    verbose: bool
        Whether to print the information of the spatial network
    """
    # only of rad_cutoff and k_cutoff must be provided
    assert rad_cutoff is not None or k_cutoff is not None, "Either rad_cutoff or k_cutoff must be provided"
    assert rad_cutoff is None or k_cutoff is None, "Only one of rad_cutoff and k_cutoff must be provided"
    if verbose:
        print('------Calculating spatial bipartite graph...')

    coor1 = spatial_df1.values
    coor2 = spatial_df2.values

    tree1 = cKDTree(coor1)
    tree2 = cKDTree(coor2)

    if rad_cutoff is not None:
        indices = tree1.query_ball_tree(tree2, r=rad_cutoff)  # indices is a list of lists in spatial_df2
    else:
        distances, indices = tree2.query(coor1, k=k_cutoff + 1)

    # construct the spatial bipartite df
    spatial_bipartite_data = []
    for i, neighbors in enumerate(indices):
        cell1 = spatial_df1.index[i]
        for j in neighbors:
            cell2 = spatial_df2.index[j]
            distance = np.linalg.norm(coor1[i] - coor2[j])
            spatial_bipartite_data.append((cell1, cell2, distance))

    spatial_bipartite = pd.DataFrame(spatial_bipartite_data, columns=['Cell1', 'Cell2', 'Distance'])
    # add the edge type iinformation "across"
    spatial_bipartite['EdgeType'] = 'across'
    if verbose:
        print('------Spatial bipartite graph calculated.')
        print('The graph contains %d edges, %d cells, %.4f neighbors per cell on average.' \
              % (spatial_bipartite.shape[0], spatial_df1.shape[0] + spatial_df2.shape[0],
                 spatial_bipartite.shape[0] / (spatial_df1.shape[0])))
    return spatial_bipartite


def cal_spatial_net2D(adata, rad_cutoff=None, k_cutoff=None, use_obsm="spatial",
                      add_key="spatial_net", verbose=True):
    spatial_df = pd.DataFrame(adata.obsm[use_obsm])
    spatial_df.index = adata.obs.index
    spatial_net = _cal_spatial_net(spatial_df, rad_cutoff=rad_cutoff, k_cutoff=k_cutoff,
                                   verbose=verbose)
    adata.uns[add_key] = spatial_net.copy()
    return adata


def cal_spatial_net3D(adata, batch_id=None, iter_comb=None, rad_cutoff=None, k_cutoff=None,
                      z_rad_cutoff=None, z_k_cutoff=None, use_obsm="spatial", add_key="Spatial_Net",
                      verbose=True):
    """
    Calculate the spatial network for 3D data.
    First, calculate the spatial network for each layer.
    Then, calculate the spatial bipartite network between layers.
    Finally, combine the two networks.
    """
    assert batch_id is not None, "batch_id must be provided"
    # assert iter_comb is not None, "iter_comb must be provided"
    batch_list = adata.obs[batch_id].unique()
    # iter_comb must be a list of tuples with length 2 and each tuple contains two batch ids belonging to the batch_list
    if iter_comb is not None:
        assert all([len(comb) == 2 and comb[0] in batch_list and comb[1] in batch_list for comb in iter_comb]), \
            "`iter_comb` must be a list of tuples with length 2 and each tuple contains two batch ids belonging to the " \
            "batch_list"
    if verbose:
        print("------Calculating spatial network for each batch...")
    spatial_net_list = []
    for batch in batch_list:
        if verbose:
            print(f"Calculating spatial network for batch {batch}...")
        adata_batch = adata[adata.obs[batch_id] == batch]
        spatial_df = pd.DataFrame(adata_batch.obsm[use_obsm])
        spatial_df.index = adata_batch.obs.index
        spatial_net = _cal_spatial_net(spatial_df, rad_cutoff=rad_cutoff, k_cutoff=k_cutoff, verbose=verbose)
        spatial_net_list.append(spatial_net)
    # construct the spatial bipartite network
    if verbose:
        print("------Calculating spatial bipartite network...")
    spatial_bipartite_list = []
    if iter_comb is not None:
        for comb in iter_comb:
            if verbose:
                print(f"Calculating spatial bipartite network for {comb}...")
            adata1 = adata[adata.obs[batch_id] == comb[0]]
            adata2 = adata[adata.obs[batch_id] == comb[1]]
            spatial_df1 = pd.DataFrame(adata1.obsm[use_obsm])
            spatial_df1.index = adata1.obs.index
            spatial_df2 = pd.DataFrame(adata2.obsm[use_obsm])
            spatial_df2.index = adata2.obs.index
            spatial_bipartite = _cal_spatial_bipartite(spatial_df1, spatial_df2, rad_cutoff=z_rad_cutoff,
                                                       k_cutoff=z_k_cutoff, verbose=verbose)
            spatial_bipartite_list.append(spatial_bipartite)
        # combine the spatial network and spatial bipartite network
        spatial_net = pd.concat(spatial_net_list + spatial_bipartite_list, axis=0)
    else:
        spatial_net = pd.concat(spatial_net_list, axis=0)
    if verbose:
        print("------Spatial network calculated.")
    adata.uns[add_key] = pd.DataFrame(spatial_net)
    return adata


def binary_search_alpha(x, tol, left, right):
    best_alpha = None
    while left <= right:
        mid = (left + right) / 2
        deviations = np.abs((x * mid - np.round(x * mid)) / mid)
        max_deviation = np.mean(deviations)

        if max_deviation <= tol:
            best_alpha = mid
            right = mid - 1  # Continue searching for smaller alpha
        else:
            left = mid + 1
    return best_alpha


def quantize_coordination(coordinates: Union[np.ndarray, pd.DataFrame],
                          methods: List[tuple] = None,
                          verbose: bool = True):
    """
    Quantize the spatial coordinates to integers.
    Parameters
    ----------
    coordinates: np.ndarray or pd.DataFrame of size (n, k)
    methods: List of tuples (method_name, paramters)  to quantize the spatial coordinates
    """
    assert len(methods) == coordinates.shape[1], "The length of methods must be equal to the number of columns of " \
                                                 "coordinates"
    new_coordinates = np.zeros_like(coordinates)
    if verbose:
        print("Quantizing spatial coordinates...")
    for ind, (method_name, param) in enumerate(methods):
        coord = coordinates[:, ind]
        coord = coord - np.min(coord)
        if method_name == "binary_search":
            alpha = binary_search_alpha(coord, param, 0.1, 100)
        elif method_name == "division":
            alpha = 1 / param
        else:
            raise NotImplementedError(f"Method {method_name} is not implemented")
        new_coord = np.round(coord * alpha)
        new_coordinates[:, ind] = new_coord
        if verbose:
            # compute pearson correlation
            from scipy.stats import pearsonr
            corr = pearsonr(coord, new_coord)[0]
            diff = np.diff(np.sort(np.unique(new_coord)))
            # get meadian difference
            median_diff = np.median(diff)
            mean_deviation = np.mean(np.abs(coord * alpha - new_coord) / median_diff)
            print(f"Quantize {ind}th dimension of spatial coordinates to {alpha}, mean deviation: {mean_deviation}, "
                  f"pearson correlation: {corr}")
    return new_coordinates


def match_points_hungarian(set1, set2):
    """
    Match two sets of points using the Hungarian algorithm.

    Parameters:
    - set1: numpy array of shape (n, 2) representing the first set of points (x, y).
    - set2: numpy array of shape (m, 2) representing the second set of points (x, y).

    Returns:
    - matched_pairs: List of tuples containing the indices of matched pairs [(i, j)].
    """
    from scipy.optimize import linear_sum_assignment
    # Ensure the sets have the same number of dimensions
    if set1.shape[1] != set2.shape[1] or set1.shape[1] != 2:
        raise ValueError("Both sets of points must have the same number of dimensions (2).")

    # Calculate the Euclidean distance matrix between set1 and set2
    distance_matrix = np.linalg.norm(set1[:, np.newaxis, :] - set2, axis=2)

    # Use the Hungarian algorithm to find the optimal assignment
    row_ind, col_ind = linear_sum_assignment(distance_matrix)

    # Create a list of matched pairs
    matched_pairs = [(i, j) for i, j in zip(row_ind, col_ind)]

    return np.array(matched_pairs)


def find_affine_transformation(source_points, target_points):
    """
    Find the affine transformation matrix between source and target points using least squares.

    Parameters:
    - source_points: numpy array of shape (n, 2) representing the source points.
    - target_points: numpy array of shape (n, 2) representing the target points.

    Returns:
    - affine_matrix: 3x3 numpy array representing the affine transformation matrix.
    """

    # Augment source points with ones for the translation component
    augmented_source = np.hstack((source_points, np.ones((len(source_points), 1))))

    # Solve the least squares problem to find the transformation matrix parameters
    params, _, _, _ = np.linalg.lstsq(augmented_source, target_points, rcond=None)

    # Build the affine transformation matrix
    affine_matrix = np.vstack((params.T, [0, 0, 1]))

    return affine_matrix

