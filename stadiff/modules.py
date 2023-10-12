"""
Modified from STAGATE
"""
from typing import Union, Tuple, Optional
import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import List


class SpatialEncoding(nn.Module):
    """
    Encode spatial information of spots. Similar to the position encoding in transformers.
    :param channels: the dimension of the tensor to apply spatial encoding to.
    :param max_len: the maximum length of the input tensor.
    """

    def __init__(self,
                 channels: int,
                 max_len: int = 1000):
        super(SpatialEncoding, self).__init__()
        self.channels = channels
        self.max_len = max_len
        pos_enc = torch.zeros(1, max_len, channels)
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / \
            torch.pow(10000, torch.arange(0, channels, 2, dtype=torch.float32) / channels)
        pos_enc[:, :, 0::2] = torch.sin(X)
        pos_enc[:, :, 1::2] = torch.cos(X)
        self.register_buffer('pos_enc', pos_enc)

    def forward(self,
                spatial_coord: torch.Tensor):
        """
        :param spatial_coord: spatial coordinates of spots, shape (batch_size,  2)
        :return pos_enc: positional encoding of the input spatial coordinates, shape (batch_size, 1, channels)
        """
        num_spots, _ = spatial_coord.shape
        row_pos_enc_dim = self.channels // 2
        row_pos_enc = self.pos_enc[:, spatial_coord[:, 0].long(), :row_pos_enc_dim]
        col_pos_enc = self.pos_enc[:, spatial_coord[:, 1].long(), row_pos_enc_dim:]
        pos_enc = torch.cat([row_pos_enc, col_pos_enc], dim=-1).permute(1, 0, 2)
        return pos_enc


class SpatialEncoding3D(SpatialEncoding):
    def __init__(self,
                 concat: bool = True,
                 channels: int = 128,
                 max_len: int = 1000):
        super(SpatialEncoding3D, self).__init__(channels, max_len)
        self.concat = concat

    def forward(self,
                spatial_coord: torch.Tensor):
        """
        Parameters
        ----------
            spatial_coord: spatial coordinates of spots, shape (batch_size,  3)
        """
        num_spots, _ = spatial_coord.shape
        if self.concat:
            row_pos_enc_dim = self.channels // 2
            col_pos_enc_dim = self.channels - row_pos_enc_dim
            z_pos_enc_dim = self.channels
        else:
            row_pos_enc_dim = self.channels // 3
            col_pos_enc_dim = self.channels // 3
            z_pos_enc_dim = self.channels - row_pos_enc_dim * 2
        row_pos_enc = self.pos_enc[:, spatial_coord[:, 0].long(), :row_pos_enc_dim]
        col_pos_enc = self.pos_enc[:, spatial_coord[:, 1].long(), :col_pos_enc_dim]
        z_pos_enc = self.pos_enc[:, spatial_coord[:, 2].long(), :z_pos_enc_dim]
        if self.concat:
            pos_enc = torch.cat([row_pos_enc, col_pos_enc], dim=-1).permute(1, 0, 2)
            z_pos_enc = z_pos_enc.permute(1, 0, 2)
            # concatenate the spatial encoding of x-y plane and z axis (batch_size, 2, channels)
            pos_enc = torch.cat([pos_enc, z_pos_enc], dim=1)
        else:
            pos_enc = torch.cat([row_pos_enc, col_pos_enc, z_pos_enc], dim=-1).permute(1, 0, 2)
        return pos_enc


class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()
        self.hidden_size = hidden_size

        self.query_projection = nn.Linear(hidden_size, hidden_size)
        self.key_projection = nn.Linear(hidden_size, hidden_size)
        self.value_projection = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        query = self.query_projection(x)
        key = self.key_projection(x)
        value = self.value_projection(x)

        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.hidden_size).float())
        attention_weights = F.softmax(attention_scores, dim=-1)
        attended_values = torch.matmul(attention_weights, value)

        return attended_values


class EncoderAttnBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim=None, out_dim=None):
        super(EncoderAttnBlock, self).__init__()
        self.attention = SelfAttention(out_dim)
        self.norm1 = nn.LayerNorm(out_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        if out_dim is None:
            out_dim = in_dim
        if hidden_dim is None:
            hidden_dim = in_dim
        self.lin_proj = nn.Linear(in_dim, out_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(out_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        x = self.lin_proj(x)
        attended_values = self.attention(x)
        residual1 = x + attended_values
        norm1_output = self.norm1(residual1)

        feed_forward_output = self.feed_forward(norm1_output)
        residual2 = norm1_output + feed_forward_output
        norm2_output = self.norm2(residual2)

        return norm2_output


class DecoderAttnBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim=None, out_dim=None):
        super(DecoderAttnBlock, self).__init__()
        self.attention = SelfAttention(in_dim)
        self.norm1 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(in_dim)
        if out_dim is None:
            out_dim = in_dim
        if hidden_dim is None:
            hidden_dim = out_dim
        self.lin_proj = nn.Linear(in_dim, out_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_dim)
        )

    def forward(self, x):
        attended_values = self.attention(x)
        residual1 = x + attended_values
        norm1_output = self.norm1(residual1)

        feed_forward_output = self.feed_forward(norm1_output)
        residual2 = norm1_output + feed_forward_output
        norm2_output = self.norm2(residual2)
        out = self.lin_proj(norm2_output)
        return out


class EncoderResBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim=None, out_dim=None):
        super(EncoderResBlock, self).__init__()
        self.norm1 = nn.LayerNorm(out_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        if out_dim is None:
            out_dim = in_dim
        if hidden_dim is None:
            hidden_dim = in_dim
        self.lin_proj = nn.Linear(in_dim, out_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(out_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        x = self.lin_proj(x)
        norm1_output = self.norm1(x)
        feed_forward_output = self.feed_forward(norm1_output)
        residual2 = norm1_output + feed_forward_output
        norm2_output = self.norm2(residual2)

        return norm2_output


class DecoderResBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim=None, out_dim=None):
        super(DecoderResBlock, self).__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(in_dim)
        if out_dim is None:
            out_dim = in_dim
        if hidden_dim is None:
            hidden_dim = out_dim
        self.lin_proj = nn.Linear(in_dim, out_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_dim)
        )

    def forward(self, x):
        norm1_output = self.norm1(x)
        feed_forward_output = self.feed_forward(norm1_output)
        residual2 = norm1_output + feed_forward_output
        norm2_output = self.norm2(residual2)
        out = self.lin_proj(norm2_output)
        return out


def get_blocks(block_list: List[str],
               out_dims: List[int],
               hidden_dims: Union[int, List[int]] = 128) -> nn.ModuleList:
    """
     Get the list of blocks to be used in the model.
     Parameters
     ----------
        block_list: list of str
     """
    blocks = []
    assert len(block_list) == len(out_dims) - 1, "length of block_list should be one less than length of out_dims"
    if isinstance(hidden_dims, int):
        hidden_dims = [hidden_dims] * len(out_dims)
    else:
        assert len(hidden_dims) == len(out_dims), "length of hidden_dims should be the same as length of out_dims"
    for ind, block in enumerate(block_list):
        if block == "EncoderAttnBlock":
            blocks.append(EncoderAttnBlock(out_dims[ind], out_dim=out_dims[ind + 1], hidden_dim=hidden_dims[ind]))
        elif block == "EncoderResBlock":
            blocks.append(EncoderResBlock(out_dims[ind], out_dim=out_dims[ind + 1], hidden_dim=hidden_dims[ind]))
        elif block == "DecoderAttnBlock":
            blocks.append(DecoderAttnBlock(out_dims[ind], out_dim=out_dims[ind + 1], hidden_dim=hidden_dims[ind]))
        elif block == "DecoderResBlock":
            blocks.append(DecoderResBlock(out_dims[ind], out_dim=out_dims[ind + 1], hidden_dim=hidden_dims[ind]))
        else:
            raise ValueError(f"block {block} not recognized")
    return nn.ModuleList(blocks)
