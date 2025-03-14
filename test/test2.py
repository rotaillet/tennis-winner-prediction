import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from tqdm import tqdm
import math

# Import PyTorch Geometric modules
from torch_geometric.nn import GATConv

###############################################
# Paramètres globaux
###############################################
WINDOW_SIZE = 10            # Taille de la fenêtre glissante pour l'historique
HIST_FEATURE_DIM = 10       # Nombre de features historiques utilisées
# 46 features statiques de base, 6 pour la forme, 4 pour le timing, 5 pour la surface, 2 pour head-to-head
STATIC_FEATURE_DIM = 68    
D_MODEL = 256               # Dimension pour la fusion et le Transformer temporel
GAT_HIDDEN_DIM = 64         # Dimension cachée pour le GAT
GAT_OUTPUT_DIM = 256        # Dimension de sortie du GAT

###############################################
# 0. Mappings pour joueurs et tournois
###############################################
def build_mappings(df):
    players = pd.concat([df["Joueur 1"], df["Joueur 2"]]).unique()
    player_to_idx = {player: idx for idx, player in enumerate(players)}
    tournois = df["Tournoi"].unique()
    tournoi_to_idx = {tournoi: idx for idx, tournoi in enumerate(tournois)}
    return player_to_idx, tournoi_to_idx

###############################################
# 0bis. Découpage par le dernier match de chaque joueur
###############################################
def get_last_match_indices(df):
    last_match_idx = set()
    players = pd.concat([df["Joueur 1"], df["Joueur 2"]]).unique()
    for player in players:
        df_player = df[(df["Joueur 1"] == player) | (df["Joueur 2"] == player)]
        last_date = df_player["Date"].max()
        idxs = df_player[df_player["Date"] == last_date].index.tolist()
        for idx in idxs:
            last_match_idx.add(idx)
    return list(last_match_idx)

def split_last_match(df):
    test_indices = get_last_match_indices(df)
    train_df = df.drop(index=test_indices).reset_index(drop=True)
    test_df = df.loc[test_indices].reset_index(drop=True)
    return train_df, test_df

###############################################
# 1. Prétraitement et préparation des données
###############################################
def extract_history_features(row, player):
    if row["Joueur 1"] == player:
        return np.array([
            row["prev_ACES_p1"],
            row["prev_DOUBLE_FAULTS_p1"],
            row["prev_1st_SERVE_%_p1_pct"],
            row["prev_1st_SERVE_POINTS_WON_p1_pct"],
            row["prev_2nd_SERVE_POINTS_WON_p1_pct"],
            row["prev_BREAK_POINTS_WON_p1_pct"],
            row["prev_TOTAL_RETURN_POINTS_WON_p1_pct"],
            row["prev_TOTAL_POINTS_WON_p1_pct"],
            row["prev_total_games_p1"],
            row["prev_set_win_p1"]
        ], dtype=np.float32)
    else:
        return np.array([
            row["prev_ACES_p2"],
            row["prev_DOUBLE_FAULTS_p2"],
            row["prev_1st_SERVE_%_p2_pct"],
            row["prev_1st_SERVE_POINTS_WON_p2_pct"],
            row["prev_2nd_SERVE_POINTS_WON_p2_pct"],
            row["prev_BREAK_POINTS_WON_p2_pct"],
            row["prev_TOTAL_RETURN_POINTS_WON_p2_pct"],
            row["prev_TOTAL_POINTS_WON_p2_pct"],
            row["prev_total_games_p2"],
            row["prev_set_win_p2"]
        ], dtype=np.float32)

def build_player_history(df):
    history = {}
    df_sorted = df.sort_values("Date")
    for idx, row in df_sorted.iterrows():
        for player in [row["Joueur 1"], row["Joueur 2"]]:
            if player not in history:
                history[player] = []
        feat1 = extract_history_features(row, row["Joueur 1"])
        feat2 = extract_history_features(row, row["Joueur 2"])
        win1 = 1 if row["winner"] == row["Joueur 1"] else 0
        win2 = 1 if row["winner"] == row["Joueur 2"] else 0
        history[row["Joueur 1"]].append((row["Date"], feat1, win1))
        history[row["Joueur 2"]].append((row["Date"], feat2, win2))
    return history

def get_player_history(history, player, current_date, window_size=WINDOW_SIZE):
    matches = history.get(player, [])
    past_feats = [feat for (date, feat, win) in matches if date < current_date]
    past_feats = past_feats[-window_size:]
    if len(past_feats) < window_size:
        pad = [np.zeros(HIST_FEATURE_DIM, dtype=np.float32) for _ in range(window_size - len(past_feats))]
        past_feats = pad + past_feats
    return np.stack(past_feats)  # (window_size, HIST_FEATURE_DIM)

def compute_static_features_max(row):
    cols = [
        "Rank_Joueur_1", "Rank_Joueur_2",
        "Age_Joueur_1", "Age_Joueur_2",
        "Points_Joueur_1", "Points_Joueur_2",
        "prev_DOUBLE_FAULTS_p1", "prev_DOUBLE_FAULTS_p2",
        "prev_ACES_p1", "prev_ACES_p2",
        "prev_1st_SERVE_%_p1_num", "prev_1st_SERVE_%_p1_den", "prev_1st_SERVE_%_p1_pct",
        "prev_1st_SERVE_%_p2_num", "prev_1st_SERVE_%_p2_den", "prev_1st_SERVE_%_p2_pct",
        "prev_1st_SERVE_POINTS_WON_p1_num", "prev_1st_SERVE_POINTS_WON_p1_den", "prev_1st_SERVE_POINTS_WON_p1_pct",
        "prev_1st_SERVE_POINTS_WON_p2_num", "prev_1st_SERVE_POINTS_WON_p2_den", "prev_1st_SERVE_POINTS_WON_p2_pct",
        "prev_2nd_SERVE_POINTS_WON_p1_num", "prev_2nd_SERVE_POINTS_WON_p1_den", "prev_2nd_SERVE_POINTS_WON_p1_pct",
        "prev_2nd_SERVE_POINTS_WON_p2_num", "prev_2nd_SERVE_POINTS_WON_p2_den", "prev_2nd_SERVE_POINTS_WON_p2_pct",
        "prev_BREAK_POINTS_WON_p1_num", "prev_BREAK_POINTS_WON_p1_den", "prev_BREAK_POINTS_WON_p1_pct",
        "prev_BREAK_POINTS_WON_p2_num", "prev_BREAK_POINTS_WON_p2_den", "prev_BREAK_POINTS_WON_p2_pct",
        "prev_TOTAL_RETURN_POINTS_WON_p1_num", "prev_TOTAL_RETURN_POINTS_WON_p1_den", "prev_TOTAL_RETURN_POINTS_WON_p1_pct",
        "prev_TOTAL_RETURN_POINTS_WON_p2_num", "prev_TOTAL_RETURN_POINTS_WON_p2_den", "prev_TOTAL_RETURN_POINTS_WON_p2_pct",
        "prev_TOTAL_POINTS_WON_p1_num", "prev_TOTAL_POINTS_WON_p1_den", "prev_TOTAL_POINTS_WON_p1_pct",
        "prev_TOTAL_POINTS_WON_p2_num", "prev_TOTAL_POINTS_WON_p2_den", "prev_TOTAL_POINTS_WON_p2_pct",
        "prev_total_games_p1", "prev_total_games_p2", "prev_set_win_p1", "prev_set_win_p2"
    ]
    features = np.array([row[col] for col in cols], dtype=np.float32)
    return features

def compute_player_form(history, player, current_date, window_size):
    matches = history.get(player, [])
    past_matches = [match for match in matches if match[0] < current_date]
    outcomes = [match[2] for match in past_matches[-window_size:]]
    if len(outcomes) == 0:
        return 0.5
    return np.mean(outcomes)

###############################################
# 2bis. Dataset PyTorch
###############################################
class TennisMatchDataset(Dataset):
    def __init__(self, df, history, player_to_idx, tournoi_to_idx):
        self.df = df.sort_values("Date").reset_index(drop=True)
        self.history = history
        self.player_to_idx = player_to_idx
        self.tournoi_to_idx = tournoi_to_idx
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        current_date = row["Date"]
        
        p1_history = get_player_history(self.history, row["Joueur 1"], current_date)
        p2_history = get_player_history(self.history, row["Joueur 2"], current_date)
        static_feat = compute_static_features_max(row)
        
        p1_form_10 = compute_player_form(self.history, row["Joueur 1"], current_date, 10)
        p1_form_3  = compute_player_form(self.history, row["Joueur 1"], current_date, 3)
        p1_form_last = compute_player_form(self.history, row["Joueur 1"], current_date, 1)
        p2_form_10 = compute_player_form(self.history, row["Joueur 2"], current_date, 10)
        p2_form_3  = compute_player_form(self.history, row["Joueur 2"], current_date, 3)
        p2_form_last = compute_player_form(self.history, row["Joueur 2"], current_date, 1)
        form_features = np.array([p1_form_10, p1_form_3, p1_form_last,
                                  p2_form_10, p2_form_3, p2_form_last], dtype=np.float32)
        
        def get_last_match_info(player):
            matches = self.history.get(player, [])
            past_dates = [m[0] for m in matches if m[0] < current_date]
            if past_dates:
                last_date = max(past_dates)
                days_since = (current_date - last_date).days
            else:
                days_since = 0
            three_months_ago = current_date - pd.Timedelta(days=90)
            count_3m = sum(1 for m in matches if three_months_ago <= m[0] < current_date)
            return days_since, count_3m
        
        p1_days_since, p1_matches_3m = get_last_match_info(row["Joueur 1"])
        p2_days_since, p2_matches_3m = get_last_match_info(row["Joueur 2"])
        timing_features = np.array([p1_days_since, p1_matches_3m, p2_days_since, p2_matches_3m], dtype=np.float32)
        
        def one_hot_surface(surface):
            surfaces = ["Clay", "Hard", "Grass", "Carpet", "I. hard"]
            vec = np.zeros(len(surfaces) + 1, dtype=np.float32)
            if surface in surfaces:
                idx = surfaces.index(surface)
            else:
                idx = len(surfaces)
            vec[idx] = 1.0
            return vec
        surface_vec = one_hot_surface(row["Surface"])
        
        def compute_head_to_head(p1, p2):
            df_h2h = self.df[((self.df["Joueur 1"] == p1) & (self.df["Joueur 2"] == p2)) | 
                              ((self.df["Joueur 1"] == p2) & (self.df["Joueur 2"] == p1))]
            df_h2h = df_h2h[df_h2h["Date"] < current_date]
            total = len(df_h2h)
            if total == 0:
                win_ratio = 0.5
            else:
                wins = (df_h2h["winner"] == p1).sum()
                win_ratio = wins / total
            return np.array([total, win_ratio], dtype=np.float32)
        
        head2head_features = compute_head_to_head(row["Joueur 1"], row["Joueur 2"])
        combined_static = np.concatenate([static_feat, form_features, timing_features, surface_vec, head2head_features])
        
        if row["winner"] == row["Joueur 1"]:
            target = 0
        elif row["winner"] == row["Joueur 2"]:
            target = 1
        else:
            raise ValueError(f"Nom de gagnant inattendu : {row['winner']}")
        
        player1_idx = self.player_to_idx[row["Joueur 1"]]
        player2_idx = self.player_to_idx[row["Joueur 2"]]
        tournoi_idx = self.tournoi_to_idx[row["Tournoi"]]
        
        return {
            "p1_history": torch.tensor(p1_history, dtype=torch.float),  # (WINDOW_SIZE, HIST_FEATURE_DIM)
            "p2_history": torch.tensor(p2_history, dtype=torch.float),
            "static_feat": torch.tensor(combined_static, dtype=torch.float),  # (STATIC_FEATURE_DIM,)
            "target": torch.tensor(target, dtype=torch.long),
            "player1_idx": torch.tensor(player1_idx, dtype=torch.long),
            "player2_idx": torch.tensor(player2_idx, dtype=torch.long),
            "tournoi_idx": torch.tensor(tournoi_idx, dtype=torch.long)
        }

###############################################
# Positional Encoding pour batch_first
###############################################
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

###############################################
# Transformer temporel pour l'historique
###############################################
class TemporalTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dropout=0.3):
        super(TemporalTransformer, self).__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncodingBatch(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.batchnorm = nn.BatchNorm1d(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: (batch, window_size, input_dim)
        x = self.input_proj(x)  # (batch, window_size, d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)  # (batch, window_size, d_model)
        # Moyenne sur la dimension temporelle
        x = x.mean(dim=1)  # (batch, d_model)
        x = self.batchnorm(x)
        x = self.dropout(x)
        return x

###############################################
# 3. Modèle GAT et modèle hybride amélioré
###############################################
class TennisModelGAT(nn.Module):
    def __init__(self, player_feature_dim, hidden_dim, output_dim, num_heads=4, dropout=0.3):
        super(TennisModelGAT, self).__init__()
        self.gat1 = GATConv(player_feature_dim, hidden_dim, heads=num_heads, dropout=dropout, edge_dim=1)
        self.gat2 = GATConv(hidden_dim * num_heads, output_dim, heads=1, concat=False, dropout=dropout, edge_dim=1)
    
    def forward(self, x, edge_index, edge_weight):
        edge_attr = edge_weight.unsqueeze(-1)  # (num_edges, 1)
        x = self.gat1(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)
        x = self.gat2(x, edge_index, edge_attr=edge_attr)
        return x

# 1. Modifier le TemporalTransformer pour optionnellement renvoyer la séquence complète
class TemporalTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dropout=0.3):
        super(TemporalTransformer, self).__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncodingBatch(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.batchnorm = nn.BatchNorm1d(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, return_sequence=False):
        # x: (batch, window_size, input_dim)
        x = self.input_proj(x)  # (batch, window_size, d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)  # (batch, window_size, d_model)
        if return_sequence:
            return x  # renvoie la séquence complète
        # Sinon, on effectue l'agrégation par moyenne
        x = x.mean(dim=1)  # (batch, d_model)
        x = self.batchnorm(x)
        x = self.dropout(x)
        return x

# 2. Créer un module de CrossAttention
class CrossAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.3):
        super(CrossAttention, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=False)
    
    def forward(self, query, key_value):
        """
        query: (batch, d_model) --> on la traite comme une séquence de longueur 1
        key_value: (batch, seq_len, d_model) --> séquence temporelle
        """
        # Reshape pour nn.MultiheadAttention qui attend des tenseurs de forme (seq_len, batch, d_model)
        query = query.unsqueeze(0)           # (1, batch, d_model)
        key = key_value.transpose(0, 1)        # (seq_len, batch, d_model)
        value = key.clone()                    # (seq_len, batch, d_model)
        attn_output, _ = self.mha(query, key, value)  # attn_output: (1, batch, d_model)
        return attn_output.squeeze(0)          # (batch, d_model)

# 3. Modifier le modèle hybride pour intégrer la cross attention
class HybridTennisModel(nn.Module):
    def __init__(self, player_feature_dim, gat_hidden_dim, gat_output_dim,
                 hist_feature_dim, static_feature_dim, d_model,
                 num_players, num_tournois, num_heads=4, dropout=0.3):
        super(HybridTennisModel, self).__init__()
        self.gat = TennisModelGAT(player_feature_dim, gat_hidden_dim, gat_output_dim, num_heads, dropout)
        self.static_proj = nn.Linear(static_feature_dim, d_model)
        # Le temporal transformer reste inchangé
        self.temporal_transformer = TemporalTransformer(input_dim=hist_feature_dim, d_model=d_model, nhead=4, num_layers=2, dropout=dropout)
        # Module de cross attention pour fusionner static et temporel
        self.cross_attention = CrossAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        # Nouvelle dimension d'entrée pour le classifieur : 2 * gat_output_dim + static_repr + 2 * (output cross-attention)
        self.classifier = nn.Sequential(
            nn.Linear(2 * gat_output_dim + d_model + 2 * d_model, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)
        )
    
    def forward(self, p1_history, p2_history, static_feat, player1_idx, player2_idx, tournoi_idx,
                node_features, edge_index, edge_weight):
        # Obtenir les embeddings GAT
        player_embeddings = self.gat(node_features, edge_index, edge_weight)
        emb_p1 = player_embeddings[player1_idx]
        emb_p2 = player_embeddings[player2_idx]
        # Projection des features statiques
        static_repr = self.static_proj(static_feat)  # (batch, d_model)
        
        # Obtenir la séquence temporelle pour chaque joueur
        p1_temp_seq = self.temporal_transformer(p1_history, return_sequence=True)  # (batch, window_size, d_model)
        p2_temp_seq = self.temporal_transformer(p2_history, return_sequence=True)  # (batch, window_size, d_model)
        # Appliquer la cross attention : la représentation statique guide l'attention sur la séquence temporelle
        p1_cross = self.cross_attention(static_repr, p1_temp_seq)  # (batch, d_model)
        p2_cross = self.cross_attention(static_repr, p2_temp_seq)  # (batch, d_model)
        
        # Fusionner toutes les représentations
        combined = torch.cat([emb_p1, emb_p2, static_repr, p1_cross, p2_cross], dim=1)
        out = self.classifier(combined)
        return out


###############################################
# 4. Boucle d'entraînement et de test
###############################################
def test_model(model, dataloader, criterion, node_features, edge_index, edge_weight, device="cpu"):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Test", leave=False):
            p1_history = batch["p1_history"].to(device)
            p2_history = batch["p2_history"].to(device)
            static_feat = batch["static_feat"].to(device)
            targets = batch["target"].to(device)
            player1_idx = batch["player1_idx"].to(device)
            player2_idx = batch["player2_idx"].to(device)
            tournoi_idx = batch["tournoi_idx"].to(device)
            outputs = model(p1_history, p2_history, static_feat, player1_idx, player2_idx, tournoi_idx,
                            node_features.to(device), edge_index.to(device), edge_weight.to(device))
            loss = criterion(outputs, targets)
            running_loss += loss.item() * p1_history.size(0)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    avg_loss = running_loss / len(dataloader.dataset)
    accuracy = correct / total
    return avg_loss, accuracy

def train_model(model, train_dataloader, test_dataloader, criterion, optimizer,
                node_features, edge_index, edge_weight, num_epochs=30, device="cpu"):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} - Train", leave=False)
        for batch in train_progress_bar:
            p1_history = batch["p1_history"].to(device)
            p2_history = batch["p2_history"].to(device)
            static_feat = batch["static_feat"].to(device)
            targets = batch["target"].to(device)
            player1_idx = batch["player1_idx"].to(device)
            player2_idx = batch["player2_idx"].to(device)
            tournoi_idx = batch["tournoi_idx"].to(device)
            
            optimizer.zero_grad()
            outputs = model(p1_history, p2_history, static_feat, player1_idx, player2_idx, tournoi_idx,
                            node_features.to(device), edge_index.to(device), edge_weight.to(device))
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * p1_history.size(0)
            train_progress_bar.set_postfix(loss=loss.item())
            
        train_loss = running_loss / len(train_dataloader.dataset)
        test_loss, test_accuracy = test_model(model, test_dataloader, criterion, node_features, edge_index, edge_weight, device)
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_accuracy:.4f}")

###############################################
# 5. Fonctions pour construire le graphe pondéré
###############################################
def build_player_graph_with_weights(df, player_to_idx, lambda_=0.001):
    reference_date = df["Date"].max()
    edge_dict = {}
    for idx, row in df.iterrows():
        p1 = player_to_idx[row["Joueur 1"]]
        p2 = player_to_idx[row["Joueur 2"]]
        match_date = row["Date"]
        days_diff = (reference_date - match_date).days
        weight = np.exp(-lambda_ * days_diff)
        for (src, dst) in [(p1, p2), (p2, p1)]:
            key = (src, dst)
            edge_dict[key] = edge_dict.get(key, 0) + weight
    edges = []
    weights = []
    for (src, dst), w in edge_dict.items():
        edges.append([src, dst])
        weights.append(w)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(weights, dtype=torch.float)
    return edge_index, edge_weight

def build_node_features(df, player_to_idx):
    num_players = len(player_to_idx)
    player_ranks = {player: [] for player in player_to_idx.keys()}
    for idx, row in df.iterrows():
        player_ranks[row["Joueur 1"]].append(row["Rank_Joueur_1"])
        player_ranks[row["Joueur 2"]].append(row["Rank_Joueur_2"])
    features = np.zeros((num_players, 1))
    for player, idx in player_to_idx.items():
        if player_ranks[player]:
            features[idx] = np.mean(player_ranks[player])
        else:
            features[idx] = 0
    features = (features - np.mean(features)) / np.std(features)
    return torch.tensor(features, dtype=torch.float)

###############################################
# 6. Pipeline principale
###############################################
def main():
    df = pd.read_csv("test2.csv", parse_dates=["Date"])
    player_to_idx, tournoi_to_idx = build_mappings(df)
    history = build_player_history(df)
    train_df, test_df = split_last_match(df)
    
    train_dataset = TennisMatchDataset(train_df, history, player_to_idx, tournoi_to_idx)
    test_dataset = TennisMatchDataset(test_df, history, player_to_idx, tournoi_to_idx)
    
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    num_players = len(player_to_idx)
    num_tournois = len(tournoi_to_idx)
    
    edge_index, edge_weight = build_player_graph_with_weights(df, player_to_idx, lambda_=0.001)
    node_features = build_node_features(df, player_to_idx)  # (num_players, player_feature_dim)
    player_feature_dim = node_features.shape[1]
    
    model = HybridTennisModel(
        player_feature_dim=player_feature_dim,
        gat_hidden_dim=GAT_HIDDEN_DIM,
        gat_output_dim=GAT_OUTPUT_DIM,
        hist_feature_dim=HIST_FEATURE_DIM,
        static_feature_dim=STATIC_FEATURE_DIM,
        d_model=D_MODEL,
        num_players=num_players,
        num_tournois=num_tournois,
        num_heads=16,
        dropout=0.3
    )
    
    criterion = nn.CrossEntropyLoss()
    # Ajout de weight_decay dans l'optimiseur pour la régularisation
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_model(model, train_dataloader, test_dataloader, criterion, optimizer,
                node_features, edge_index, edge_weight, num_epochs=30, device=device)

if __name__ == "__main__":
    main()
