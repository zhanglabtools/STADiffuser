import torch
import torch.nn as nn
import torch.nn.functional as F
import contextlib
from torch_geometric.nn.conv import GATConv
from .modules import get_blocks, SelfAttention
from typing import Union, List

EPS = 1e-7


class SpaAE(nn.Module):
    """
    Spatial Autoencoder
    """
    def __init__(self,
                 input_dim: int = 3000,
                 gat_dim: List[int] = [512],
                 block_out_dims: List[int] = [512, 32],
                 block_list: List[str] = ["AttnBlock"],
                 dropout: float = 0.1,
                 act_fn: nn.Module = nn.ELU()
                 ):
        super(SpaAE, self).__init__()
        self.input_dim = input_dim
        self.gat_dim = gat_dim
        self.dropout = dropout
        self.act_fn = act_fn
        dropouts = len(gat_dim) * [0]
        dropouts[0] = dropout
        gat_dim = [input_dim] + gat_dim
        self._construct_gat_layer("in_gat", gat_dim, dropouts)
        self._construct_gat_layer("out_gat", gat_dim[::-1], dropouts[::-1])
        self.encoder = get_blocks(["Encoder" + block for block in block_list],
                                  block_out_dims)
        self.decoder = get_blocks(["Decoder" + block for block in block_list],
                                  block_out_dims[::-1])

    def _construct_gat_layer(self, name, gat_dim, dropouts=None):
        assert len(dropouts) == len(gat_dim) - 1, "The length of dropouts should be equal to the length of gat_dim - 1"
        setattr(self, name, nn.ModuleList())
        for i in range(len(gat_dim) - 1):
            getattr(self, name).append(GATConv(gat_dim[i], gat_dim[i+1], heads=1, concat=False, dropout=dropouts[i],
                                               add_self_loops=False, bias=False))

    def _call_forward(self, layer, features, edge_index):
        if isinstance(layer, GATConv):
            features = self.act_fn(layer(features, edge_index))
        else:
            features = layer(features)
        return features

    def decode(self, z, edge_index, no_grad=True):
        if no_grad:
            self.eval()
        with torch.no_grad() if no_grad else contextlib.suppress():
            for layer in self.decoder:
                z = self._call_forward(layer, z, edge_index)

            for layer in self.out_gat:
                z = layer(z, edge_index)
        return z

    def forward(self, features, edge_index):
        for layer in self.in_gat:
            features = layer(features, edge_index)
        for layer in self.encoder:
            features = self._call_forward(layer, features, edge_index)
        hidden_state = features
        for layer in self.decoder:
            features = self._call_forward(layer, features, edge_index)
        for layer in self.out_gat:
            features = layer(features, edge_index)
        return hidden_state, features


class SpaVAE(SpaAE):
    def __init__(self,
                 input_dim=3000,
                 gat_dim=512,
                 hidden_dim=32,
                 block_out_dims=[512, 32],
                 block_list=["AttnBlock"],
                 dropout=0.1,
                 act_fn=nn.ELU()
                 ):
        super(SpaVAE, self).__init__(input_dim, gat_dim, block_out_dims, block_list, dropout, act_fn)
        self.mu = nn.Linear(block_out_dims[-1], hidden_dim)
        self.log_var = nn.Linear(block_out_dims[-1], hidden_dim)

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var / 2) + EPS
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, features, edge_index):
        features = self.in_gat(features, edge_index)
        for layer in self.encoder:
            features = self._call_forward(layer, features, edge_index)
        hidden_state = features
        mu = self.mu(features)
        log_var = self.log_var(features) + EPS
        z = self.reparameterize(mu, log_var)
        for layer in self.decoder:
            features = self._call_forward(layer, features, edge_index)
        features = self.out_gat(features, edge_index)
        return hidden_state, features, mu, log_var, z

    @staticmethod
    def loss(features, recon_features, mu, log_var, kld_weight=1):
        recon_loss = F.mse_loss(recon_features, features)
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return recon_loss + kld_loss * kld_weight, recon_loss, kld_loss
