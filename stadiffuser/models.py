"""
Denoising network for STADiffuser
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from dataclasses import dataclass
from diffusers.utils import BaseOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.embeddings import GaussianFourierProjection, TimestepEmbedding, Timesteps
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.unet_1d_blocks import get_down_block, get_mid_block, get_out_block, get_up_block
from .modules import SpatialEncoding, SpatialEncoding3D


@dataclass
class StaDiffuserOutput(BaseOutput):
    sample: torch.Tensor
    spatial_coord: torch.Tensor


class SpaUNet1DModel(ModelMixin, ConfigMixin):
    """
    The denoising network for STADiffuser, extending the `UNet1DModel` from `diffusers` with spatial processing.

    This model is designed to process one-dimensional data, such as time series or signals, with an emphasis on spatial information. It includes features for spatial encoding and can be configured to use three-dimensional spatial concatenation for enhanced performance.

    Parameters
    ----------
    sample_size : int, optional
        The default length of the input sample. Defaults to 32.
    spatial_encoding : str, optional
        The type of spatial encoding to use. Defaults to "sinusoidal".
    sample_rate : int, optional
        The sample rate of the input data, if applicable.
    in_channels : int, optional
        The number of channels in the input sample. Defaults to 2.
    out_channels : int, optional
        The number of channels in the output sample. Defaults to 2.
    extra_in_channels : int, optional
        Additional channels to append to the input. Defaults to 0.
    time_embedding_type : str, optional
        The type of time embedding to use. Defaults to "fourier".
    flip_sin_to_cos : bool, optional
        Whether to convert sine to cosine for Fourier time embedding. Defaults to True.
    use_timestep_embedding : bool, optional
        Whether to use timestep embedding. Defaults to False.
    freq_shift : float, optional
        Frequency shift for Fourier time embedding. Defaults to 0.0.
    down_block_types : Tuple[str], optional
        Tuple of downsample block types.
    up_block_types : Tuple[str], optional
        Tuple of upsample block types.
    mid_block_type : str, optional
        The block type for the middle of the UNet. Defaults to "UNetMidBlock1D".
    out_block_type : str, optional
        The optional output processing block type for the UNet.
    block_out_channels : Tuple[int], optional
        Tuple of block output channels.
    act_fn : str, optional
        The optional activation function to use in UNet blocks.
    norm_num_groups : int, optional
        The number of groups for group normalization. Defaults to 8.
    layers_per_block : int, optional
        The number of layers per block in the UNet. Defaults to 1.
    downsample_each_block : bool, optional
        Whether to downsample in each block. Defaults to False.
    spatial3d_concat : bool, optional
        Whether to use three-dimensional spatial concatenation.
    class_embed_type : str, optional
        The type of class embedding to use.
    num_class_embeds : int, optional
        The number of class embeddings.

    Attributes
    ----------
    time_embedder : object
        The time embedding module.
    spatial_embedder : object
        The spatial embedding module.
    model : object
        The core UNet model.
    """
    @register_to_config
    def __init__(
            self,
            sample_size: int = 32,
            spatial_encoding: str = "sinusoidal",
            sample_rate: Optional[int] = None,
            in_channels: int = 2,
            out_channels: int = 1,
            extra_in_channels: int = 0,
            time_embedding_type: str = "fourier",
            flip_sin_to_cos: bool = True,
            use_timestep_embedding: bool = False,
            freq_shift: float = 0.0,
            down_block_types: Tuple[str] = ("DownBlock1DNoSkip", "DownBlock1D", "AttnDownBlock1D"),
            up_block_types: Tuple[str] = ("AttnUpBlock1D", "UpBlock1D", "UpBlock1DNoSkip"),
            mid_block_type: Tuple[str] = "UNetMidBlock1D",
            out_block_type: str = None,
            block_out_channels: Tuple[int] = (32, 32, 64),
            act_fn: str = None,
            norm_num_groups: int = 8,
            layers_per_block: int = 1,
            downsample_each_block: bool = False,
            spatial3d_concat: Optional[bool] = False,
            class_embed_type: Optional[str] = "embedding",
            num_class_embeds: Optional[int] = None,
    ):
        super().__init__()
        self.sample_size = sample_size
        self.spatial3d_concat = spatial3d_concat
        if spatial_encoding == "sinusoidal":
            self.spa_encoder = SpatialEncoding(sample_size)
        elif spatial_encoding == "sinusoidal3d":
            self.spa_encoder = SpatialEncoding3D(concat=spatial3d_concat, channels=sample_size)
        else:
            raise NotImplementedError(f"Unknown spatial encoding {spatial_encoding}")
        # time enconding
        if time_embedding_type == "fourier":
            self.time_proj = GaussianFourierProjection(
                embedding_size=8, set_W_to_weight=False, log=False, flip_sin_to_cos=flip_sin_to_cos
            )
            timestep_input_dim = 2 * block_out_channels[0]
        elif time_embedding_type == "positional":
            self.time_proj = Timesteps(
                block_out_channels[0], flip_sin_to_cos=flip_sin_to_cos, downscale_freq_shift=freq_shift
            )
            timestep_input_dim = block_out_channels[0]

        if use_timestep_embedding:
            time_embed_dim = block_out_channels[0] * 4
            self.time_mlp = TimestepEmbedding(
                in_channels=timestep_input_dim,
                time_embed_dim=time_embed_dim,
                act_fn=act_fn,
                out_dim=block_out_channels[0],
            )
        # class embedding
        if num_class_embeds is not None:
            if class_embed_type == "embedding":
                self.class_embedding = nn.Embedding(num_class_embeds, sample_size)
            elif class_embed_type == "timestep":
                self.class_embedding = TimestepEmbedding(timestep_input_dim, sample_size)
            else:
                raise NotImplementedError(f"Unknown class embedding type {class_embed_type}")
        else:
            self.class_embedding = None

        self.down_blocks = nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])
        self.out_block = None
        # down
        output_channel = in_channels
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]

            if i == 0:
                input_channel += extra_in_channels

            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=block_out_channels[0],
                add_downsample=not is_final_block or downsample_each_block,
            )
            self.down_blocks.append(down_block)

        # mid
        self.mid_block = get_mid_block(
            mid_block_type,
            in_channels=block_out_channels[-1],
            mid_channels=block_out_channels[-1],
            out_channels=block_out_channels[-1],
            embed_dim=block_out_channels[0],
            num_layers=layers_per_block,
            add_downsample=downsample_each_block,
        )

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        if out_block_type is None:
            final_upsample_channels = out_channels
        else:
            final_upsample_channels = block_out_channels[0]

        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = (
                reversed_block_out_channels[i + 1] if i < len(up_block_types) - 1 else final_upsample_channels
            )

            is_final_block = i == len(block_out_channels) - 1

            up_block = get_up_block(
                up_block_type,
                num_layers=layers_per_block,
                in_channels=prev_output_channel,
                out_channels=output_channel,
                temb_channels=block_out_channels[0],
                add_upsample=not is_final_block,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        num_groups_out = norm_num_groups if norm_num_groups is not None else min(block_out_channels[0] // 4, 32)
        self.out_block = get_out_block(
            out_block_type=out_block_type,
            num_groups_out=num_groups_out,
            embed_dim=block_out_channels[0],
            out_channels=out_channels,
            act_fn=act_fn,
            fc_dim=block_out_channels[-1] // 4,
        )

    def forward(
            self,
            sample: torch.FloatTensor,
            timestep: Union[torch.Tensor, float, int],
            spatial_coord: torch.FloatTensor = None,
            class_labels: Optional[torch.Tensor] = None,
            return_dict: bool = True,
    ) -> Union[StaDiffuserOutput, Tuple]:
        r"""
        Forward pass of the STADiffuser model.

        Args:
            sample (torch.FloatTensor):
                Input tensor of shape `(batch_size, num_channels, sample_size)` containing noisy inputs.
            timestep (Union[torch.Tensor, float, int]):
                Batched timestep values.
            spatial_coord (torch.FloatTensor, optional):
                Spatial coordinates tensor of shape `(batch_size, 2)`. Defaults to None.
            class_labels (torch.LongTensor, optional):
                Class labels tensor of shape `(batch_size)`. Defaults to None.
            return_dict (bool, optional):
                Whether to return a `StaDiffuserOutput` instead of a plain tuple. Defaults to True.

        Returns:
            StaDiffuserOutput or tuple: If `return_dict` is True, returns a StaDiffuserOutput instance, otherwise returns a tuple.
        """
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        timestep_embed = self.time_proj(timesteps)
        if self.config.use_timestep_embedding:
            timestep_embed = self.time_mlp(timestep_embed)
        else:
            timestep_embed = timestep_embed[..., None]
            timestep_embed = timestep_embed.repeat([1, 1, sample.shape[2]]).to(sample.dtype)
            timestep_embed = timestep_embed.broadcast_to((sample.shape[:1] + timestep_embed.shape[1:]))
        # 2. spatial
        if spatial_coord is None:
            spatial_embed = 0
        else:
            spatial_embed = self.spa_encoder(spatial_coord)
        if self.spatial3d_concat:
            emb = spatial_embed[:, [0], :] + timestep_embed
            z_emb = spatial_embed[:, [1], :]
            # concat
            emb = torch.cat([emb, z_emb], dim=1)
        else:
            # spatial_embed (batch_size, 1, sample_size)
            #  timestep_embed (batch_size, 16, sample_size)
            # broadcast spatial_embed to (batch_size, 16, sample_size)
            emb = spatial_embed + timestep_embed

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels must be provided if class_embedding is not None")
            if self.config.class_embed_type == "timestep":
                class_emb = self.time_proj(class_labels)
            else:
                class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
                class_emb = torch.unsqueeze(class_emb, dim=1)
            emb = emb + class_emb
        # 2. down
        down_block_res_samples = ()
        for downsample_block in self.down_blocks:
            sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
            down_block_res_samples += res_samples

        # 3. mid
        if self.mid_block:
            sample = self.mid_block(sample, emb)

        # 4. up
        for i, upsample_block in enumerate(self.up_blocks):
            res_samples = down_block_res_samples[-1:]
            down_block_res_samples = down_block_res_samples[:-1]
            sample = upsample_block(sample, res_hidden_states_tuple=res_samples, temb=emb)

        # 5. post-process
        if self.out_block:
            sample = self.out_block(sample, emb)

        if not return_dict:
            return (sample,)

        return StaDiffuserOutput(sample=sample, spatial_coord=spatial_coord)
