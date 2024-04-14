"""
This script is used to convert the Rdata file to an AnnData object.
"""

import rdata
import glob
import os
import scipy
import scanpy as sc
import numpy as np
import tqdm

# App3

config_template = {
    "source_dir": "./output/App2-HPC/rep1/full_sim/",
    "target_dir": "./output/App2-HPC/rep1/full_sim/",
    "ref_adata": "./output/App2-HPC/rep1/adata_raw.h5ad",
}


def read_simRDS(path, ref_adata=None, keep_all=False, compress=True):
    res = rdata.parser.parse_file(path)
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


def convert_rdata_files(config, ref_adata):
    rdata_files = glob.glob("{}/*.rds".format(config['source_dir']))
    print("Found {} Rdata files".format(len(rdata_files)))
    for rf in tqdm.tqdm(rdata_files):
        adata_sim = read_simRDS(rf, ref_adata=ref_adata)
        adata_sim = adata_sim.copy()
        adata_sim.write("{}/{}.h5ad".format(config['target_dir'], rf.split("\\")[-1].split(".")[0]))
        # remove the file in source
        os.remove(rf)

# missing_rate = [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9]
#
# for r in missing_rate:
#     mask_name = "random_{}".format(r)
#     print("Processing mask: {}".format(mask_name))
#     # update the config
#     config["source_dir"] = "./output/App1-DLPFC/151676/{}".format(mask_name)
#     config["target_dir"] = "./output/App1-DLPFC/151676/{}".format(mask_name)
# # get ths list of Rdata files end with "rds"

if __name__ == "__main__":
    config = config_template
    ref_adata = sc.read_h5ad(config_template["ref_adata"])
    convert_rdata_files(config, ref_adata)