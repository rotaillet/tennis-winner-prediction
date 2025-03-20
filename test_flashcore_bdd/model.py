import torch.nn as nn
import torch
import math
from torch_geometric.nn import GATConv
import torch.nn.functional as F

class PositionalEncodingBatch(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncodingBatch, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TennisModelGCN(nn.Module):
    def __init__(self, player_feature_dim, hidden_dim, output_dim, dropout=0.3):
        super(TennisModelGCN, self).__init__()
        self.gcn1 = GCNConv(player_feature_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, edge_index, edge_weight=None):
        # Appliquer la première couche GCN
        x = self.gcn1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.dropout(x)
        # Appliquer la seconde couche GCN
        x = self.gcn2(x, edge_index, edge_weight)
        return x

    
class TennisModelGAT(nn.Module):
    def __init__(self, player_feature_dim, hidden_dim, output_dim, num_heads=4, dropout=0.3):
        super(TennisModelGAT, self).__init__()
        self.gat1 = GATConv(player_feature_dim, hidden_dim, heads=num_heads, dropout=dropout, edge_dim=1)
        self.gat2 = GATConv(hidden_dim * num_heads, output_dim, heads=1, concat=False, dropout=dropout, edge_dim=1)
    
    def forward(self, x, edge_index, edge_weight):
        edge_attr = edge_weight.unsqueeze(-1)  # (num_edges, 1)
        # Supposons que x est l'entrée initiale
        # Supposons que la sortie de self.gat1 a une dimension 1024
        residual = x  
        x = self.gat1(x, edge_index, edge_weight)
        x = F.elu(x)

        
        x = self.gat2(x, edge_index, edge_weight)
        x = F.elu(x)
        # Adapter residual pour qu'il ait la dimension 1024
        return x
    
class TemporalTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dropout=0.3):
        super(TemporalTransformer, self).__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncodingBatch(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True,dim_feedforward=1024)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.batchnorm = nn.BatchNorm1d(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, return_sequence=False):
        # x: (batch, window_size, input_dim)
        x = self.input_proj(x)  # (batch, window_size, d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)  # (batch, window_size, d_model)
        if return_sequence:
            return x  # renvoie la squence complte
        # Sinon, on effectue l'agrgation par moyenne
        x = x.mean(dim=1)  # (batch, d_model)
        x = self.batchnorm(x)
        x = self.dropout(x)
        return x
    
class CrossAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.3):
        super(CrossAttention, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=False)
    
    def forward(self, query, key_value):
        """
        query: (batch, d_model) --> on la traite comme une squence de longueur 1
        key_value: (batch, seq_len, d_model) --> squence temporelle
        """
        # Reshape pour nn.MultiheadAttention qui attend des tenseurs de forme (seq_len, batch, d_model)
        query = query.unsqueeze(0)           # (1, batch, d_model)
        key = key_value.transpose(0, 1)        # (seq_len, batch, d_model)
        value = key.clone()                    # (seq_len, batch, d_model)
        attn_output, _ = self.mha(query, key, value)  # attn_output: (1, batch, d_model)
        return attn_output.squeeze(0)          # (batch, d_model)

# 3. Modifier le mod�le hybride pour int�grer la cross attention
class HybridTennisModel(nn.Module):
    def __init__(self, player_feature_dim, gat_hidden_dim, gat_output_dim,
                 hist_feature_dim, static_feature_dim, d_model,
                 num_players, num_tournois, num_heads=4, dropout=0.3):
        super(HybridTennisModel, self).__init__()
        self.gat = TennisModelGAT(player_feature_dim, gat_hidden_dim, gat_output_dim, num_heads, dropout)
        self.static_proj = nn.Linear(static_feature_dim, d_model)
        # Le temporal transformer reste inchang�
        self.temporal_transformer = TemporalTransformer(input_dim=hist_feature_dim, d_model=d_model, nhead=8, num_layers=2, dropout=dropout)
        # Module de cross attention pour fusionner static et temporel
        self.cross_attention = CrossAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)

        self.tournoi_embedding = nn.Embedding(num_tournois, d_model)
        # Nouveau calcul de la dimension d'entrée pour le classifieur :
        # emb_p1 (gat_output_dim) + emb_p2 (gat_output_dim) + static_repr (d_model)
        # + p1_cross (d_model) + p2_cross (d_model) + tournoi_embedding (d_model)
        classifier_input_dim = 2 * gat_output_dim + 4 * d_model
        # Nouvelle dimension d'entr�e pour le classifieur : 2 * gat_output_dim + static_repr + 2 * (output cross-attention)
        self.classifier = nn.Sequential(
        nn.Linear(classifier_input_dim, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(256, 64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(64, 2),
        )

    
    def forward(self, p1_history, p2_history, static_feat, player1_idx, player2_idx, tournoi_idx,
                node_features, edge_index, edge_weight):
        # Obtenir les embeddings GAT
        player_embeddings = self.gat(node_features, edge_index, edge_weight)
        emb_p1 = player_embeddings[player1_idx]
        emb_p2 = player_embeddings[player2_idx]
        # Projection des features statiques
        static_repr = self.static_proj(static_feat)  # (batch, d_model)
        
        # Obtenir la s�quence temporelle pour chaque joueur
        p1_temp_seq = self.temporal_transformer(p1_history, return_sequence=True)  # (batch, window_size, d_model)
        p2_temp_seq = self.temporal_transformer(p2_history, return_sequence=True)  # (batch, window_size, d_model)
        # Appliquer la cross attention : la repr�sentation statique guide l'attention sur la s�quence temporelle
        p1_cross = self.cross_attention(static_repr, p1_temp_seq)  # (batch, d_model)
        p2_cross = self.cross_attention(static_repr, p2_temp_seq)  # (batch, d_model)
        
        tournoi_repr = self.tournoi_embedding(tournoi_idx)
        # Fusionner toutes les repr�sentations
        combined = torch.cat([emb_p1, emb_p2, static_repr, p1_cross, p2_cross,tournoi_repr], dim=1)
        out = self.classifier(combined)
        return out