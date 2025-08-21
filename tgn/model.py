import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv
import torch.nn.functional as F
class MultiLayerTimeAwareGNN(nn.Module):
    def __init__(self, in_channels,memory_dim, hidden_channels, out_channels,
                 msg_dim, time_enc, num_layers=3, heads=2, dropout=0.1):
        super().__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        self.in_channels = in_channels
        self.memory = memory_dim
        self.lin = nn.Linear(memory_dim,in_channels)
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            norm_dim = in_channels if i == 0 else hidden_channels * heads
            self.norms.append(nn.LayerNorm(norm_dim))

        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_c = in_channels if i == 0 else hidden_channels * heads
            out_c = out_channels // heads if i == num_layers-1 else hidden_channels
            self.convs.append(
                TransformerConv(
                    in_c, out_c, heads=heads,
                    dropout=dropout, edge_dim=edge_dim
                )
            )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, last_update, edge_index, t, msg):
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        if self.memory != self.in_channels:
            x = self.lin(x)
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x_new = conv(x, edge_index, edge_attr)
            x = norm(x + self.dropout(x_new))
            if i < len(self.convs) - 1:  # Activation seulement sur les couches avant la dernière
                x = F.silu(x)


        return x



class RelationAttentionMultiHead(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.out = nn.Linear(embed_dim, embed_dim)
    def forward(self, z_src, z_dst):
        # z_src,z_dst: (B,D)
        pair = torch.stack([z_src, z_dst], dim=1)  # (B,2,D)
        attn_out, _ = self.mha(pair, pair, pair)
        rel = attn_out[:,0,:]
        return self.out(rel)

# === 4) WinPredictor updated ===
class WinPredictor(nn.Module):
    def __init__(self, embed_dim, hidden_dim, context_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.rel_att = RelationAttentionMultiHead(embed_dim, num_heads)
        total_dim = embed_dim + context_dim
        self.ctx_proj = nn.Sequential(
            nn.Linear(context_dim, embed_dim),
            nn.SiLU()
        )
        self.shared = nn.Sequential(
            nn.Linear(embed_dim*3, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        self.head_win = nn.Linear(hidden_dim,1)
        self.head_margin = nn.Linear(hidden_dim,1)
        self.head_elo = nn.Linear(hidden_dim,1)
    def forward(self, z_src, z_dst, match_feats):
        r_ij = self.rel_att(z_src, z_dst)
        r_ji = self.rel_att(z_dst, z_src)
        rel = torch.cat([r_ij, r_ji], dim=-1)
        ctx = self.ctx_proj(match_feats)
        x = torch.cat([rel, ctx], dim=-1)
        h = self.shared(x)
        return self.head_win(h), self.head_margin(h), self.head_elo(h)

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class SmallWinPredictor(nn.Module):
    """
    Tête de prédiction :
      - Backbone embeddings-only : (2*embed_dim) -> 512 -> 64
      - Optionnel : contexte match_feats via concat ou FiLM
      - Sortie : logit (B,1)
    """
    def __init__(self, embed_dim, hidden,match_dim=None, drop=0.4, mode="concat"):
        super().__init__()
        self.mode = mode
        self.drop = nn.Dropout(drop)
        self.act  = nn.ReLU()

        # Backbone (embeddings -> 64)
        self.first_head  = nn.Linear(2 * embed_dim, hidden[0])
        self.norm1       = nn.LayerNorm(hidden[0])
        self.second_head = nn.Linear(hidden[0], hidden[1])
        self.norm2       = nn.LayerNorm(hidden[1])

        # Branche match_feats (optionnelle)
        if mode == "concat":
            if match_dim is None:
                self.match_proj = nn.Sequential(
                    nn.LazyLinear(128), nn.ReLU(), nn.Dropout(drop),
                    nn.Linear(128, 64)
                )
            else:
                self.match_proj = nn.Sequential(
                    nn.Linear(match_dim, 2*hidden[1]), nn.ReLU(), nn.Dropout(drop),
                    nn.Linear(2*hidden[1], hidden[1])
                )
            self.fuse = nn.Sequential(
                nn.Linear(2*hidden[1], hidden[1]), nn.ReLU(), nn.Dropout(drop)
            )
        elif mode == "film":
            out_dim = 2 * 64  # gamma(64) + beta(64)
            if match_dim is None:
                self.film = nn.Sequential(
                    nn.LazyLinear(128), nn.ReLU(), nn.Linear(128, out_dim)
                )
            else:
                self.film = nn.Sequential(
                    nn.Linear(match_dim, 128), nn.ReLU(), nn.Linear(128, out_dim)
                )
        elif mode is None:
            pass
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Tête finale
        self.head_win = nn.Linear(hidden[1], 1)

    def forward(self, z_src, z_dst, match_feats=None):
        # Embeddings-only backbone -> h in R^64
        h = torch.cat([z_src, z_dst], dim=-1)            # [B, 2*embed_dim]
        h = self.drop(self.norm1(self.act(self.first_head(h))))  # [B,512]
        h = self.drop(self.norm2(self.act(self.second_head(h)))) # [B, 64]

        # Contexte (optionnel)
        if match_feats is not None and self.mode is not None:
            # harmoniser device/dtype
            if match_feats.device != h.device:
                match_feats = match_feats.to(h.device)
            if match_feats.dtype != h.dtype:
                match_feats = match_feats.to(h.dtype)

            if self.mode == "concat":
                m = self.match_proj(match_feats)         # [B, 64]
                h = torch.cat([h, m], dim=-1)            # [B, 128]
                h = self.fuse(h)                         # [B, 64]
            elif self.mode == "film":
                gb = self.film(match_feats)              # [B, 128]
                gamma, beta = gb.chunk(2, dim=-1)        # [B,64], [B,64]
                h = h * (1.0 + torch.tanh(gamma)) + beta

        return self.head_win(h)                          # [B,1] (logits)
                         # logit [B,1]


    
class WinPredictorMLP(nn.Module):
    def __init__(self, in_channels: int,
                 hidden_channels_list: list[int] = [256, 128],
                 dropout: float = 0.3,
                 residual: bool = False):
        super().__init__()
        layers = []
        dims = [5 * in_channels] + hidden_channels_list + [1]
        self.residual = residual

        for i in range(len(dims) - 2):
            in_dim, out_dim = dims[i], dims[i+1]
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.LayerNorm(out_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*layers)
        self.head_win = nn.Linear(dims[-2], dims[-1])
        self.head_margin = nn.Linear(dims[-2], dims[-1])
        self.head_elo = nn.Linear(dims[-2], dims[-1])

    def forward(self, z_src, z_dst):
        h_diff = z_src - z_dst
        h_abs = torch.abs(z_src - z_dst)
        h_mult = z_src * z_dst

        h = torch.cat([z_src, z_dst, h_diff, h_abs, h_mult], dim=-1)
        h = self.net(h)

        return self.head_win(h), self.head_margin(h), self.head_elo(h)




class MessageMLP(nn.Module):
    def __init__(self, raw_msg_dim: int, memory_dim: int, time_dim: int, hidden_dim: int):
        super().__init__()
        in_channels = raw_msg_dim + 2 * memory_dim + time_dim

        self.fc1 = nn.Linear(in_channels, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.2)
        self.act = nn.SiLU()
        self.out_channels = hidden_dim

    def forward(self, z_src, z_dst, raw_msg, t_enc):
        x = torch.cat([z_src, z_dst, raw_msg, t_enc], dim=-1)

        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout1(x)
        x = self.norm1(x)

        x = self.fc2(x)
        x = self.act(x)
        x = self.dropout2(x)
        x = self.norm2(x)

        return x
