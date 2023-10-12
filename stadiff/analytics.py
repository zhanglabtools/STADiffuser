import numpy as np


def mclust_R(adata, num_cluster, modelNames='EEE', use_rep='STAGATE', random_seed=2023,
             add_key='mclust'):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    Parameters
    ----------
    adata：
        Anndata object
    num_cluster:
        `int`, Number of clusters
    modelNames:
        `str`, The model names to be used in mclust. Default: 'EEE'
    use_rep:
        `str`, The obsm key to be used for clustering. Default: 'STAGATE'
    random_seed:
        `int`, Random seed for mclust. Default: 2023
    add_key:
        `str`, The key to add to adata.obs. Default: 'mclust'

    Returns
    -------
    adata: AnnData with adata.obs[add_key] added.
    """
    np.random.seed(random_seed)

    try:
        import rpy2.robjects as robjects
    except ImportError:
        raise ImportError("Please install rpy2 first.")
    try:
        robjects.r.library("mclust")
    except:
        raise ImportError("Please install mclust package in R first.")
    import rpy2.robjects.numpy2ri

    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']
    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[use_rep]), num_cluster, modelNames)
    mclust_res = np.array(res[-2])
    adata.obs[add_key] = mclust_res
    adata.obs[add_key] = adata.obs[add_key].astype('int')
    adata.obs[add_key] = adata.obs[add_key].astype('category')
    return adata