import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv

# Time2Vec (inchangé)
class Time2Vec(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.lin = nn.Linear(1, 1)
        self.weights = nn.Parameter(torch.randn(1, kernel_size))
        self.bias = nn.Parameter(torch.zeros(1, kernel_size))
    def forward(self, t: torch.Tensor):
        t = t.unsqueeze(-1)
        v_lin = self.lin(t)
        v_periodic = torch.sin(t * self.weights + self.bias)
        return torch.cat([v_lin, v_periodic], dim=-1)

# GNN temporel
class MultiLayerTimeAwareGNN(nn.Module):
    def __init__(self, in_channels, memory_dim, hidden_channels, out_channels,
                 msg_dim, time_kernel, num_layers=3, heads=2, dropout=0.1):
        super().__init__()
        self.time_enc = Time2Vec(time_kernel)
        edge_dim = msg_dim + (1 + time_kernel)
        self.lin = nn.Linear(memory_dim, in_channels) if memory_dim!=in_channels else nn.Identity()
        self.norms, self.convs, self.dropouts = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for i in range(num_layers):
            in_c = in_channels if i==0 else hidden_channels*heads
            out_c= out_channels//heads if i==num_layers-1 else hidden_channels
            self.convs.append(TransformerConv(in_c, out_c, heads=heads, dropout=dropout, edge_dim=edge_dim))
            self.norms.append(nn.LayerNorm(in_c if i==0 else hidden_channels*heads))
            self.dropouts.append(nn.Dropout(dropout))
    def forward(self, x, last_update, edge_index, t, msg):
        rel_t = last_update[edge_index[0]] - t
        rel_enc = self.time_enc(rel_t.to(x.dtype))
        edge_attr = torch.cat([rel_enc, msg], dim=-1)
        x = self.lin(x)
        for conv, norm, drop in zip(self.convs, self.norms, self.dropouts):
            x_new = conv(x, edge_index, edge_attr)
            x = norm(x + drop(x_new))
            x = F.silu(x)
        return x

# Petit décodeur pour forecast
class ForecastDecoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
    def forward(self, z):
        return self.net(z)