import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class TennisMatchPredictor(nn.Module):
    def __init__(self, static_dim, num_players, num_tournois, d_model, gnn_hidden,perf_dim, num_gnn_layers=2, dropout=0.3,seq_length=5):
        super(TennisMatchPredictor, self).__init__()
        # --- GNN pour les joueurs ---
        # On suppose que les features initiales des joueurs sont de dimension 1 (ex. rank normalisé)
        self.gnn_convs = nn.ModuleList()
        self.gnn_convs.append(GCNConv(2, gnn_hidden))
        for _ in range(num_gnn_layers - 1):
            self.gnn_convs.append(GCNConv(gnn_hidden, d_model))
        
        # --- Embedding pour les tournois ---
        self.tournoi_embedding = nn.Embedding(num_tournois, d_model)
        self.perf_gru = nn.GRU(input_size=perf_dim, hidden_size=d_model, num_layers=1, batch_first=True)
        
        total_input_dim = static_dim + 5 * d_model
        # On concatène : static features + embedding joueur 1 + embedding joueur 2 + embedding tournoi
        self.classifier = nn.Sequential(
            nn.Linear(total_input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)  # sortie à 2 dimensions pour la classification (ex. logits)
        )
    
    def forward(self, static_feat, player1_idx, player2_idx, tournoi_idx, graph_data,player1_seq, player2_seq):
        # --- Propagation dans le GNN ---
        x = graph_data.x  # features initiales des joueurs, de dimension (num_players, 1)
        edge_index = graph_data.edge_index
        edge_attr = graph_data.edge_attr  # si vous voulez utiliser les edge attributes
            
        for conv in self.gnn_convs:
            x = conv(x, edge_index, edge_weight=edge_attr)
            x = F.relu(x)
        # x est de taille (num_players, d_model)
        
        # Récupération des embeddings pour les deux joueurs
        embed_player1 = x[player1_idx]  # shape: (batch, d_model)
        embed_player2 = x[player2_idx]  # shape: (batch, d_model)
        
        # Récupération de l'embedding du tournoi
        embed_tournoi = self.tournoi_embedding(tournoi_idx)  # shape: (batch, d_model)
        _,hn1 = self.perf_gru(player1_seq)
        _,hn2 = self.perf_gru(player2_seq)
        form_player1 = hn1[-1]  # (batch, d_model)
        form_player2 = hn2[-1]  # (batch, d_model)      
        # Concaténation des features statiques et des embeddings
        combined = torch.cat([static_feat, embed_player1, embed_player2, embed_tournoi,form_player1,form_player2], dim=1)
        
        # Passage par le classifieur
        out = self.classifier(combined)
        return out