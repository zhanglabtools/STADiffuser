# cli.py

"""
Script to run the main program of STADiffuser.

Usage: python scripts/cli.py --args

Arguments:
    --input_file: str, path to the processed h5ad file. The AnnData object should contain the gene expression matrix,
    `spatial_net` and `spatial` in the `.obsm` and the metadata in the `.obs`.
    --output_dir: str, path to the output directory
    --input_dim: int, the input dimension of the autoencoder model.
    --gat_dim: list, the hidden dimensions of the GAT layers in the autoencoder model.
    --block_out_dims: list, the output dimensions of the blocks in the autoencoder model.
    --label: str, the column name of the label in the metadata to be used for diffusion model training. The key should
    be in the `.obs` of the AnnData object.
    --mask: str, the mask file to specify the region of interest. The key should be in the `.obs` of the AnnData object.
    If not specified, the whole slide will be used.
    --autoencoder-max-epochs: int, the maximum number of epochs for training the autoencoder model. Default is 500.
    --denoiser-max-epochs: int, the maximum number of epochs for training the diffusion model. Default is 1000.
    --device: str, "cuda" or "cpu", specify the device to run the model.
    --autoencoder-path: str, the path to the pre-trained autoencoder model.
    --autoencoder-check-points: str, the check points of the autoencoder model to be used for training the diffusion model.
    --denoiser-check-points: str, the check points of the diffusion model to be used for training the diffusion model.
    --autoencoder-name: str, the name of the autoencoder model.
    --autoencoder-batch-size: int, the batch size for training the autoencoder model.
    --autoencoder-batch-mode: bool, whether to use the batch mode for training the autoencoder model.
    --multi-slice: bool, whether to train the autoencoder model with multiple slices.
    --use-batch: str, the column name of the batch in the metadata to be used for training the autoencoder model.
    --pretrain-epochs: int, the number of epochs for pretraining the autoencoder model with multiple slices.
    --exclude-batches: str, the list of batches to be excluded for training the autoencoder model.
    --align-lr: float, the learning rate for aligning the autoencoder model with triplet loss.
    --update-interval: int, the update interval for aligning the autoencoder model with triplet loss.
    --margin: float, the margin for aligning the autoencoder model with triplet loss.
    --new-spatial-division: float, the new spatial is divided by this number.
    --new-spatial-z-division: float, the new spatial z is divided by this number.
    --remove-na-label: bool, whether to remove the NA labels in the metadata.
    --denoiser-batch-size: int, the batch size for training the diffusion model.

Example:
    python scripts/cli.py --input_file data/processed.h5ad --output_dir output --input_dim 3000 --gat_dim 512 32
    --block_out_dims 32 32 --label cell_type --mask mask --autoencoder-max-epochs 500 --denoiser-max-epochs 1000
    --device cuda:0 --autoencoder-path autoencoder.pth --autoencoder-check-points 100 200 300
    --denoiser-check-points 100 200 300 --autoencoder-name autoencoder_attn2 --autoencoder-batch-size 64
    --autoencoder-batch-mode --multi-slice --use-batch slice_id --pretrain-epochs 200 --exclude-batches slice1
    --align-lr 1e-4 --update-interval 50 --margin 1 --new-spatial-division 125 --new-spatial-z-division 1
    --remove-na-label --denoiser-in-channels 17 --denoiser-batch-size 64
"""

import os
import sys
import numpy as np
import torch
import argparse
from vae import SpaAE
from .models import SpaUNet1DModel
import pipeline
from torch_geometric.loader import NeighborLoader
import scanpy as sc
from diffusers import DDPMScheduler
from utils import mask_region
from dataset import get_slice_loader
import warnings
import logging
warnings.filterwarnings("ignore")


def build_autoencoder(input_dim=3000,
                      gat_dim=[512, 32],
                      block_out_dims=[32, 32]):
    """
    Build the autoencoder model.
    """
    autoencoder = SpaAE(input_dim=input_dim,
                        block_list=["AttnBlock"],
                        gat_dim=gat_dim,
                        block_out_dims=block_out_dims)
    return autoencoder


def _check_args_validity(args):
    def _check_list_int(x):
        try:
            [int(i) for i in x.split(",")]
        except ValueError:
            return False
        return True
    if args.autoencoder_path is not None:
        assert os.path.exists(args.autoencoder_path), f"Autoencoder model path {args.autoencoder_path} does not exist"
    if args.autoencoder_check_points is not None:
        # check the autoencoder check points is valid: a list of integers separated by comma
        if not _check_list_int(args.autoencoder_check_points):
            raise ValueError(f"Autoencoder check points {args.autoencoder_check_points} is not valid")
    if args.denoiser_check_points is not None:
        if not _check_list_int(args.denoiser_check_points):
            raise ValueError(f"Denoiser check points {args.denoiser_check_points} is not valid")


def _check_adata_validity(adata, args):
    if args.mask is not None:
        assert args.mask in adata.obs, f"Mask key {args.mask} is not in the .obs of the AnnData object"
    if args.label is not None:
        assert args.label in adata.obs, f"Label key {args.label} is not in the .obs of the AnnData object"
    if args.label is not None:
        assert "label_" not in adata.obs, "label_ key is already in the .obs of the AnnData object. Please rename it."
    # check that
    # check output_dir is valid: must be a string
    assert "spatial" in adata.obsm, "spatial key is not in the .obsm of the AnnData object"
    assert "spatial_net" in adata.uns, "spatial_net key is not in the .obsm of the AnnData object"
    # assert “label_" is not in the .obs of the AnnData object


# --- Set up the argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, help="Path to the processed input h5ad file")
parser.add_argument("--output_dir", type=str, help="Path to the output directory")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--input_dim", type=int, help="The input dimension of the autoencoder model", default=3000)
parser.add_argument("--gat_dim", type=int, nargs="+",
                    help="The hidden dimensions of the GAT layers in the autoencoder model", default=[512, 32])
parser.add_argument("--block_out_dims", type=int, nargs="+",
                    help="The output dimensions of the blocks in the autoencoder model", default=[32, 32])
parser.add_argument("--scale-exp", action="store_true", default=False)
parser.add_argument("--label", type=str, default=None)
parser.add_argument("--mask", type=str, default=None)
parser.add_argument("--autoencoder-name", type=str, default="autoencoder_attn2")
parser.add_argument("--autoencoder-max-epochs", type=int, default=500)
parser.add_argument("--autoencoder-batch-size", type=int, default=64,
                    help="The batch size for training the autoencoder")
parser.add_argument("--autoencoder-batch-mode", action="store_true", default=False)
parser.add_argument("--denoiser-max-epochs", type=int, default=1000)
parser.add_argument("--autoencoder-path", type=str, default=None)
parser.add_argument("--autoencoder-check-points", type=str, default=None)
#---------- parse arguments for multiple slices autoencoder training
parser.add_argument("--multi-slice", action="store_true", default=False)
parser.add_argument("--use-batch", type=str, default=None)
parser.add_argument("--pretrain-epochs", type=int, default=200)
parser.add_argument("--exclude-batches", type=str, default=None)
parser.add_argument("--align-lr", type=float, default=1e-4)
parser.add_argument("--update-interval", type=int, default=50)
parser.add_argument("--margin", type=float, default=1)
#----------- parse arguments for denoiser model
parser.add_argument("--new-spatial-division", type=float, default=125,
                    help="The new spatial is divided by this number.")
parser.add_argument("--new-spatial-z-division", type=float, default=1)
parser.add_argument("--remove-na-label", action="store_true", default=False)
parser.add_argument("--denoiser-in-channels", type=int, default=17)
parser.add_argument("--denoiser-batch-size", type=int, default=64)
parser.add_argument("--denoiser-3d", action="store_true", default=False)
parser.add_argument("--spatial-3d-concat", action="store_true", default=False)
parser.add_argument("--denoiser-check-points", type=str, default=None)
parser.add_argument("--run-autoencoder-only", action="store_true", default=False)

def main():
    # set up logger and format. Time should be YYYY-MM-DD HH:MM:SS
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s (%(asctime)s) >> %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger("STADiffuser")
    logger.info("Start running the STADiffuser pipeline")
    args = parser.parse_args()
    # ars.output_dir must be specificed
    if args.output_dir is None:
        raise ValueError("Output directory must be specified")
    if not os.path.exists(args.output_dir):
        logger.debug(f"Create output directory {args.output_dir}")
        os.makedirs(args.output_dir)
    output_dir = args.output_dir
    device = torch.device(args.device)
    # Load the data
    logger.info(f"Load the data from {args.input_file}")
    _check_args_validity(args)
    adata = sc.read_h5ad(args.input_file)
    if args.scale_exp:
        logger.info("Scale the gene expression matrix")
        sc.pp.scale(adata)
    _check_adata_validity(adata, args)

    if args.mask is not None:
        logger.info(f"Mask the region of interest using {args.mask}")
        adata = mask_region(adata, args.mask)

    if args.autoencoder_check_points is not None:
        autoencoder_ckpt = [int(x) for x in args.autoencoder_check_points.split(",")]

    if args.denoiser_check_points is not None:
        denoiser_ckpt = [int(x) for x in args.denoiser_check_points.split(",")]
    if args.multi_slice:
        logger.info("Train the autoencoder model with multiple slices...")
        use_batch = args.use_batch
        batch_list = adata.obs[use_batch].unique()
        if args.exclude_batches is not None:
            exclude_batches = args.exclude_batches.split(",")
            # remove extra space in the exclude_batches
            exclude_batches = [x.strip() for x in exclude_batches]
            batch_list = [x for x in batch_list if x not in exclude_batches]
            adata = adata[adata.obs[use_batch].isin(batch_list)]
            spatial_net = adata.uns["spatial_net"].copy()
            obs_names = adata.obs_names
            spatial_net = spatial_net[spatial_net["Cell1"].isin(obs_names) & spatial_net["Cell2"].isin(obs_names)]
            adata.uns["spatial_net"] = spatial_net
        logger.info("Use batches: {}".format(batch_list))
        logger.info("The dataset_hub contains {} slices, {} spots and {} genes".format(len(batch_list),
                                                                                    adata.shape[0], adata.shape[1]))
    data = pipeline.prepare_dataset(adata, use_rep=None)
    if args.autoencoder_path is not None:
        logger.info(f"Load the autoencoder model from {args.autoencoder_path}")
        autoencoder = torch.load(args.autoencoder_path).to(device)
    else:
        logger.info("Build the autoencoder model ...")
        autoencoder = build_autoencoder(input_dim=args.input_dim,
                                        gat_dim=args.gat_dim,
                                        block_out_dims=args.block_out_dims).to(device)
        # convert autoenoder-ckpt to list
        if args.autoencoder_check_points is not None:
            autoencoder_ckpt = [int(x) for x in args.autoencoder_check_points.split(",")]
        else:
            autoencoder_ckpt = None
        logger.info("Train the autoencoder model ...")
        if not args.multi_slice:
            train_loader = NeighborLoader(data, num_neighbors=[5, 3], batch_size=args.autoencoder_batch_size)
            autoencoder, autoencoder_loss = pipeline.train_autoencoder(train_loader, autoencoder,
                                                                       n_epochs=args.autoencoder_max_epochs,
                                                                       save_dir=output_dir, device=device,
                                                                       model_name=args.autoencoder_name,
                                                                       check_points=autoencoder_ckpt)
        else:
            # pretrain the autoencoder model with multiple slices
            train_loaders = [get_slice_loader(adata, data, batch, use_batch=use_batch,
                                              batch_size=args.autoencoder_batch_size) for batch in batch_list]
            logger.info("Pretrain the autoencoder model with multiple slices...")
            autoencoder, autoencoder_loss = pipeline.pretrain_autoencoder_multi(train_loaders,
                                                                                autoencoder,
                                                                                pretrain_epochs=args.pretrain_epochs,
                                                                                save_dir=output_dir,
                                                                                check_points=autoencoder_ckpt,
                                                                                device=device)
            # align the autoencoder model
            logger.info("Align the autoencoder model with triplet loss...")
            tri_epochs = args.autoencoder_max_epochs - args.pretrain_epochs
            autoencoder, autoencoder_loss = pipeline.train_autoencoder_multi(adata, autoencoder, use_batch=use_batch,
                                                                             batch_list=batch_list,
                                                                             n_epochs=tri_epochs,
                                                                             margin=args.margin,
                                                                             lr=args.align_lr,
                                                                             update_interval=args.update_interval,
                                                                             check_points=autoencoder_ckpt,
                                                                             save_dir=output_dir, device=device)
    if args.run_autoencoder_only:
        logger.info("Finish running the STADiffuser pipeline")
        sys.exit(0)
    # ------------------------    Train the diffusion model
    if args.denoiser_check_points is not None:
        denoiser_ckpt = [int(x) for x in args.denoiser_check_points.split(",")]
    else:
        denoiser_ckpt = None
    if args.autoencoder_batch_mode:
        adata = pipeline.get_recon(adata, autoencoder, device=device,
                                   apply_normalize=False, show_progress=True, batch_mode=True)
    else:
        adata = pipeline.get_recon(adata, autoencoder, device=device,
                                   apply_normalize=False, show_progress=False, batch_mode=False)
    normalizer = utils.MinMaxNormalize(adata.obsm["latent"], dim=0)
    adata.obsm["normalized_latent"] = normalizer.normalize(adata.obsm["latent"])
    # convert the spatial coordination to the new spatial coordination
    spatial_new = adata.obsm["spatial"].copy()
    dvision = args.new_spatial_division
    logger.debug(f"Quantize the spatial coordination by division {dvision}")
    if not args.denoiser_3d:
        spatial_new = utils.quantize_coordination(spatial_new, methods=[("division", dvision), ("division", dvision)])
    else:
        logger.info("Run 3D denoiser model...")
        spatial_new = utils.quantize_coordination(spatial_new, methods=[("division", dvision), ("division", dvision),
                                                                        ("division", args.new_spatial_z_division)])
    adata.obsm["spatial_new"] = spatial_new
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    in_channels = args.denoiser_in_channels
    spatial_3d_concat = args.spatial_3d_concat
    if spatial_3d_concat:
        in_channels += 1
        logger.info("Concatenate the 3D spatial encoding to the input channels")
    else:
        logger.info("Add the 3D spatial encoding to the input spaital positional channels")
    if args.label is None:
        logger.info("Train diffusion model...")
        data_latent = pipeline.prepare_dataset(adata, use_rep="normalized_latent", use_spatial="spatial_new",
                                               use_net="spatial_net")
        train_loader = NeighborLoader(data_latent, num_neighbors=[5, 3], batch_size=args.denoiser_batch_size)
        if not args.denoiser_3d:
            denoiser = SpaUNet1DModel(in_channels=in_channels, out_channels=1).to(device)
        else:
            denoiser = SpaUNet1DModel(in_channels=in_channels, out_channels=1, spatial_encoding="sinusoidal3d",
                                      spatial3d_concat=spatial_3d_concat).to(device)
        denoiser, denoise_loss = pipeline.train_denoiser(train_loader, denoiser, noise_scheduler,
                                                         lr=1e-4, weight_decay=1e-6,
                                                         n_epochs=args.denoiser_max_epochs,
                                                         save_dir=output_dir, device=device,
                                                         check_points=denoiser_ckpt)
    else:
        label_name = args.label
        if args.remove_na_label:
            na_num = adata.obs[label_name].isna().sum()
            logger.debug("Remove {} NA labels".format(na_num))
            adata = adata[~adata.obs[label_name].isna()]
            #
        num_class_embeds = len(np.unique(adata.obs[label_name]))
        class_dict = dict(zip(np.unique(adata.obs[label_name]), range(num_class_embeds)))
        # write class_dict to logger
        logger.info(f"Class dictionary:\n {class_dict}")
        adata.obs["label_"] = adata.obs[label_name].map(class_dict)
        data_latent = pipeline.prepare_dataset(adata, use_rep="normalized_latent", use_spatial="spatial_new",
                                               use_net="spatial_net", use_label="label_")
        train_loader = NeighborLoader(data_latent, num_neighbors=[5, 3], batch_size=args.denoiser_batch_size)
        if not args.denoiser_3d:
            denoiser = SpaUNet1DModel(in_channels=in_channels, out_channels=1, num_class_embeds=num_class_embeds).to(device)
        else:
            # construct the 3D denoiser model
            denoiser = SpaUNet1DModel(in_channels=in_channels, out_channels=1, num_class_embeds=num_class_embeds,
                                      spatial_encoding="sinusoidal3d",
                                      spatial3d_concat=spatial_3d_concat).to(device)
        logger.info("Train diffusion model with label {}".format(args.label))
        denoiser, denoise_loss = pipeline.train_denoiser(train_loader, denoiser, noise_scheduler,
                                                         lr=1e-4, weight_decay=1e-6,
                                                         n_epochs=args.denoiser_max_epochs,
                                                         num_class_embeds=num_class_embeds,
                                                         model_name="denoiser_{}".format(label_name),
                                                         device=device, save_dir=output_dir, check_points=denoiser_ckpt)
    logger.info("Finish running the STADiffuser pipeline")


if __name__ == "__main__":
    main()
