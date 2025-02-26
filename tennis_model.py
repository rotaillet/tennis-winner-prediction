import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import SAGEConv,GATConv
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

class TennisModelGATTransformer(nn.Module):
    def __init__(self, feature_dim, hidden_dim, transformer_dim, combined_dim, match_feat_dim,
                 nhead=8, num_transformer_layers=4, dropout_p=0.3):
        """
        feature_dim     : dimension des features statiques d'un joueur (ex: 1)
        hidden_dim      : dimension cachée pour le GAT (ex: 64)
        transformer_dim : dimension d'embedding utilisée dans le Transformer (ex: 32 ou 64)
        combined_dim    : dimension finale après fusion (ex: 64)
        match_feat_dim  : dimension des features de match (ex: 36)
        nhead           : nombre de têtes d'attention pour le Transformer
        num_transformer_layers : nombre de couches Transformer
        dropout_p       : taux de dropout
        """
        super(TennisModelGATTransformer, self).__init__()
        # --- Partie GNN avec GAT ---
        self.gat1 = GATConv(feature_dim, hidden_dim, heads=1, dropout=dropout_p)
        self.gat2 = GATConv(hidden_dim, hidden_dim, heads=1, dropout=dropout_p)
        
        # --- Partie Transformer pour l'historique dynamique ---
        # On projette d'abord les features (par exemple, le ranking) vers une dimension adaptée au Transformer
        self.history_proj = nn.Linear(feature_dim, transformer_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=transformer_dim, nhead=nhead, dropout=dropout_p)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        # Pooling sur la séquence (ici, moyenne sur la dimension temporelle)
        self.history_pool = nn.AdaptiveAvgPool1d(1)
        # Projection finale pour harmoniser avec le GAT
        self.history_fc = nn.Linear(transformer_dim, hidden_dim)
        
        # --- Mécanisme d'attention pour fusionner l'embedding global et l'embedding dynamique ---
        self.att_layer = nn.Linear(2 * hidden_dim, 2)
        
        # --- Fusion finale ---
        self.fc_player = nn.Linear(hidden_dim, combined_dim)
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc_match = nn.Sequential(
            nn.Linear(2 * combined_dim + match_feat_dim, combined_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(combined_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, edge_index, dynamic_histories, match_data, match_features):
        """
        x                : features statiques de tous les joueurs [num_players, feature_dim]
        edge_index       : indices des arêtes du graphe
        dynamic_histories: tuple de deux listes (p1_hist_list, p2_hist_list) où chaque élément est un tenseur [seq_len, feature_dim]
        match_data       : tenseur [batch_size, 2] contenant les identifiants des joueurs du match (p1, p2)
        match_features   : tenseur [batch_size, match_feat_dim] contenant les features de match
        """
        # --- Partie GNN avec GAT ---
        x_gnn = self.gat1(x, edge_index)
        x_gnn = F.elu(x_gnn)
        x_gnn = self.gat2(x_gnn, edge_index)
        x_gnn = F.elu(x_gnn)
        
        batch_size = match_data.size(0)
        p1_transformer_outputs = []
        p2_transformer_outputs = []
        p1_hist_list, p2_hist_list = dynamic_histories
        
        for i in range(batch_size):
            # Pour chaque match, on récupère l'historique dynamique de chaque joueur
            p1_hist = p1_hist_list[i].to(x.device)  # [seq_len, feature_dim]
            p2_hist = p2_hist_list[i].to(x.device)  # [seq_len, feature_dim]
            
            # Projection vers l'espace du Transformer
            p1_proj = self.history_proj(p1_hist)  # [seq_len, transformer_dim]
            p2_proj = self.history_proj(p2_hist)  # [seq_len, transformer_dim]
            
            # Le Transformer attend un tenseur de forme (seq_len, batch, d_model). Ici, batch=1.
            p1_proj = p1_proj.unsqueeze(1)  # [seq_len, 1, transformer_dim]
            p2_proj = p2_proj.unsqueeze(1)  # [seq_len, 1, transformer_dim]
            
            # Passage dans le Transformer Encoder
            p1_encoded = self.transformer_encoder(p1_proj)  # [seq_len, 1, transformer_dim]
            p2_encoded = self.transformer_encoder(p2_proj)
            
            # On retire la dimension batch et on transpose pour appliquer un pooling sur la séquence
            p1_encoded = p1_encoded.squeeze(1).transpose(0, 1)  # [transformer_dim, seq_len]
            p2_encoded = p2_encoded.squeeze(1).transpose(0, 1)
            
            # Pooling (moyenne) sur la séquence temporelle
            p1_pooled = self.history_pool(p1_encoded).squeeze(1)  # [transformer_dim]
            p2_pooled = self.history_pool(p2_encoded).squeeze(1)
            
            # Projection finale pour obtenir un vecteur de dimension hidden_dim
            p1_transformer_outputs.append(self.history_fc(p1_pooled).unsqueeze(0))  # [1, hidden_dim]
            p2_transformer_outputs.append(self.history_fc(p2_pooled).unsqueeze(0))
        
        # Concaténation sur le batch
        p1_transformer_outputs = torch.cat(p1_transformer_outputs, dim=0)  # [batch_size, hidden_dim]
        p2_transformer_outputs = torch.cat(p2_transformer_outputs, dim=0)  # [batch_size, hidden_dim]
        
        # --- Fusion des embeddings globaux et dynamiques ---
        p1_global = x_gnn[match_data[:, 0]]  # [batch_size, hidden_dim]
        p2_global = x_gnn[match_data[:, 1]]
        
        # Concaténation pour chaque joueur
        fusion_p1 = torch.cat([p1_global, p1_transformer_outputs], dim=1)  # [batch_size, 2*hidden_dim]
        fusion_p2 = torch.cat([p2_global, p2_transformer_outputs], dim=1)
        
        # Calcul des poids d'attention
        att_weights_p1 = F.softmax(self.att_layer(fusion_p1), dim=1)  # [batch_size, 2]
        att_weights_p2 = F.softmax(self.att_layer(fusion_p2), dim=1)
        
        # Fusion pondérée des embeddings
        fused_p1 = att_weights_p1[:, 0].unsqueeze(1) * p1_global + att_weights_p1[:, 1].unsqueeze(1) * p1_transformer_outputs
        fused_p2 = att_weights_p2[:, 0].unsqueeze(1) * p2_global + att_weights_p2[:, 1].unsqueeze(1) * p2_transformer_outputs
        
        # Projection finale pour obtenir les embeddings des joueurs
        player_emb_p1 = self.fc_player(fused_p1)
        player_emb_p2 = self.fc_player(fused_p2)
        player_emb_p1 = self.dropout(player_emb_p1)
        player_emb_p2 = self.dropout(player_emb_p2)
        
        # Fusion des embeddings des joueurs avec les features de match
        match_input = torch.cat([player_emb_p1, player_emb_p2, match_features], dim=1)
        out = self.fc_match(match_input)  # [batch_size, 1]
        return out