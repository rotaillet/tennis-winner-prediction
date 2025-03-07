import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import SAGEConv
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from tqdm import tqdm

###############################################
# 1. Chargement et prétraitement des données
###############################################

# Lecture du fichier CSV
df = pd.read_csv("atp_tennis.csv")  # Assurez-vous que le fichier est dans le même dossier

# Conversion de la colonne Date en datetime et tri par ordre chronologique
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# Encodage des joueurs en identifiants numériques (colonnes en français)
players = list(set(df["Joueur 1"].unique()) | set(df["Joueur 2"].unique()))
player_encoder = {name: i for i, name in enumerate(players)}
df["Joueur1_ID"] = df["Joueur 1"].map(player_encoder)
df["Joueur2_ID"] = df["Joueur 2"].map(player_encoder)
df["Winner_ID"]  = df["winner"].map(player_encoder)

# Construction du graphe : création d'arêtes bidirectionnelles
edges = []
for _, row in df.iterrows():
    edges.append([row["Joueur1_ID"], row["Joueur2_ID"]])
    edges.append([row["Joueur2_ID"], row["Joueur1_ID"]])
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # [2, num_edges]

# Création des features statiques pour chaque joueur (ici, moyenne des classements)
player_rank = defaultdict(list)
for _, row in df.iterrows():
    player_rank[row["Joueur1_ID"]].append(row["Rank_Joueur_1"])
    player_rank[row["Joueur2_ID"]].append(row["Rank_Joueur_2"])

num_players = len(players)
player_features = np.zeros((num_players, 1))
for i in range(num_players):
    if player_rank[i]:
        player_features[i] = np.mean(player_rank[i])
    else:
        player_features[i] = 0.0

scaler = StandardScaler()
player_features = scaler.fit_transform(player_features)
player_features = torch.tensor(player_features, dtype=torch.float)

# Construction de l’historique de performance pour chaque joueur (séquence de rangs)
player_history = defaultdict(list)
for _, row in df.iterrows():
    player_history[row["Joueur1_ID"]].append([row["Rank_Joueur_1"]])
    player_history[row["Joueur2_ID"]].append([row["Rank_Joueur_2"]])
for player_id in player_history:
    player_history[player_id] = torch.tensor(player_history[player_id], dtype=torch.float)

###############################################
# 2. Calcul des features de match enrichies
###############################################

# Fonctions d'encodage pour les features catégorielles

def encode_surface(surface_str):
    """
    Encodage one-hot des surfaces en 5 catégories :
      "Clay", "Hard", "I. hard", "Grass", "Carpet"
    """
    mapping = {
        "Clay":    [1, 0, 0, 0, 0],
        "Hard":    [0, 1, 0, 0, 0],
        "I. hard": [0, 0, 1, 0, 0],
        "Grass":   [0, 0, 0, 1, 0],
        "Carpet":  [0, 0, 0, 0, 1]
    }
    return mapping.get(surface_str, [0, 0, 0, 0, 0])

def get_round_value(tour_str):
    """
    Mapping ordinal pour la colonne 'Tour'.
    Par exemple, on considère certaines valeurs comme officielles (avec une valeur croissante)
    et les autres comme non officielles (valeur 0).
    """
    round_mapping = {
        "1stround": 1,
        "2ndround": 2,
        "3rdround": 3,
        "4thround": 4,
        "1/4":      5,   # Quart de finale
        "1/2":      6,   # Demi-finale
        "fin":      7,   # Finale
        "qual.":    0,
        "q 1":      0,
        "q 2":      0,
        "Amical":   0,
        "Rubber 1": 0,
        "bronze":   0
    }
    return round_mapping.get(tour_str, 0)

# Définition des fenêtres pour la forme
WINDOW_IMMEDIATE = 3
WINDOW_LONG = 10

# Historiques de forme et head-to-head
player_recent_results = defaultdict(list)  # 1 pour victoire, 0 pour défaite
head_to_head_record = {}  # clé : (min_id, max_id), valeur : dict {id: nombre de victoires}

# Listes pour stocker les features de match et les labels
match_feature_list = []
match_list = []
labels_list = []

for idx, row in df.iterrows():
    p1 = row["Joueur1_ID"]
    p2 = row["Joueur2_ID"]
    # Label : 1 si Joueur 1 a gagné, 0 sinon
    label = 1 if row["Winner_ID"] == p1 else 0

    # --- Base features (29 dims) ---
    # Différences pour les features numériques
    rank_diff = row["Rank_Joueur_1"] - row["Rank_Joueur_2"]
    age_diff  = row["Age_Joueur_1"] - row["Age_Joueur_2"]
    pts_diff  = row["Points_Joueur_1"] - row["Points_Joueur_2"]
    
    # Encodage de la surface (vecteur one-hot à 5 dims)
    surface_vec = encode_surface(row["Surface"])
    
    # Valeur du tour
    round_val = get_round_value(row["Tour"])
    
    # Différences pour les statistiques basiques
    dfault_diff = row["prev_DOUBLE_FAULTS_player_1"] - row["prev_DOUBLE_FAULTS_player_2"]
    aces_diff   = row["prev_ACES_player_1"] - row["prev_ACES_player_2"]
    
    # 1st serve % (numérateur, dénominateur, pourcentage)
    first_serve_num_diff = row["prev_1st_SERVE_%_player_1_num"] - row["prev_1st_SERVE_%_player_2_num"]
    first_serve_den_diff = row["prev_1st_SERVE_%_player_1_den"] - row["prev_1st_SERVE_%_player_2_den"]
    first_serve_pct_diff = row["prev_1st_SERVE_%_player_1_pct"] - row["prev_1st_SERVE_%_player_2_pct"]
    
    # 1st serve points won
    first_serve_pw_num_diff = row["prev_1st_SERVE_POINTS_WON_player_1_num"] - row["prev_1st_SERVE_POINTS_WON_player_2_num"]
    first_serve_pw_den_diff = row["prev_1st_SERVE_POINTS_WON_player_1_den"] - row["prev_1st_SERVE_POINTS_WON_player_2_den"]
    first_serve_pw_pct_diff = row["prev_1st_SERVE_POINTS_WON_player_1_pct"] - row["prev_1st_SERVE_POINTS_WON_player_2_pct"]
    
    # 2nd serve points won
    second_serve_pw_num_diff = row["prev_2nd_SERVE_POINTS_WON_player_1_num"] - row["prev_2nd_SERVE_POINTS_WON_player_2_num"]
    second_serve_pw_den_diff = row["prev_2nd_SERVE_POINTS_WON_player_1_den"] - row["prev_2nd_SERVE_POINTS_WON_player_2_den"]
    second_serve_pw_pct_diff = row["prev_2nd_SERVE_POINTS_WON_player_1_pct"] - row["prev_2nd_SERVE_POINTS_WON_player_2_pct"]
    
    # Break points won
    break_points_pw_num_diff = row["prev_BREAK_POINTS_WON_player_1_num"] - row["prev_BREAK_POINTS_WON_player_2_num"]
    break_points_pw_den_diff = row["prev_BREAK_POINTS_WON_player_1_den"] - row["prev_BREAK_POINTS_WON_player_2_den"]
    break_points_pw_pct_diff = row["prev_BREAK_POINTS_WON_player_1_pct"] - row["prev_BREAK_POINTS_WON_player_2_pct"]
    
    # Total return points won
    total_return_pw_num_diff = row["prev_TOTAL_RETURN_POINTS_WON_player_1_num"] - row["prev_TOTAL_RETURN_POINTS_WON_player_2_num"]
    total_return_pw_den_diff = row["prev_TOTAL_RETURN_POINTS_WON_player_1_den"] - row["prev_TOTAL_RETURN_POINTS_WON_player_2_den"]
    total_return_pw_pct_diff = row["prev_TOTAL_RETURN_POINTS_WON_player_1_pct"] - row["prev_TOTAL_RETURN_POINTS_WON_player_2_pct"]
    
    # Total points won
    total_pw_num_diff = row["prev_TOTAL_POINTS_WON_player_1_num"] - row["prev_TOTAL_POINTS_WON_player_2_num"]
    total_pw_den_diff = row["prev_TOTAL_POINTS_WON_player_1_den"] - row["prev_TOTAL_POINTS_WON_player_2_den"]
    total_pw_pct_diff = row["prev_TOTAL_POINTS_WON_player_1_pct"] - row["prev_TOTAL_POINTS_WON_player_2_pct"]
    
    # Agrégation de toutes les features de base dans un vecteur (29 dims)
    base_features = [
        rank_diff,
        age_diff,
        pts_diff
    ] + surface_vec + [round_val, dfault_diff, aces_diff,
        first_serve_num_diff, first_serve_den_diff, first_serve_pct_diff,
        first_serve_pw_num_diff, first_serve_pw_den_diff, first_serve_pw_pct_diff,
        second_serve_pw_num_diff, second_serve_pw_den_diff, second_serve_pw_pct_diff,
        break_points_pw_num_diff, break_points_pw_den_diff, break_points_pw_pct_diff,
        total_return_pw_num_diff, total_return_pw_den_diff, total_return_pw_pct_diff,
        total_pw_num_diff, total_pw_den_diff, total_pw_pct_diff
    ]
    
    # --- Features additionnelles (7 dims) ---
    def compute_form(player, window):
        if player_recent_results[player]:
            recent = player_recent_results[player][-window:]
            return np.mean(recent)
        else:
            return 0.5  # valeur neutre
    
    form3_p1 = compute_form(p1, WINDOW_IMMEDIATE)
    form10_p1 = compute_form(p1, WINDOW_LONG)
    form3_p2 = compute_form(p2, WINDOW_IMMEDIATE)
    form10_p2 = compute_form(p2, WINDOW_LONG)
    
    # Head-to-head
    key = tuple(sorted((p1, p2)))
    if key in head_to_head_record:
        record = head_to_head_record[key]
        total_confrontations = record.get(key[0], 0) + record.get(key[1], 0)
        wins_p1 = record.get(p1, 0)
        win_ratio_p1 = wins_p1 / total_confrontations if total_confrontations > 0 else 0.5
    else:
        total_confrontations = 0
        win_ratio_p1 = 0.5
    
    # Court (si la colonne existe)
    if "Court" in df.columns:
        court_feature = 1 if row["Court"].lower().startswith("out") else 0
    else:
        court_feature = 0.5
    
    additional_features = [form3_p1, form10_p1, form3_p2, form10_p2,
                           total_confrontations, win_ratio_p1,
                           court_feature]
    
    # Vecteur final de features de match (29 + 7 = 36 dims)
    match_features = base_features + additional_features
    match_feature_list.append(match_features)
    match_list.append([p1, p2])
    labels_list.append(label)
    
    # --- Mise à jour des historiques ---
    player_recent_results[p1].append(1 if label == 1 else 0)
    player_recent_results[p2].append(1 if label == 0 else 0)
    
    key = tuple(sorted((p1, p2)))
    if key not in head_to_head_record:
        head_to_head_record[key] = {key[0]: 0, key[1]: 0}
    if label == 1:
        head_to_head_record[key][p1] += 1
    else:
        head_to_head_record[key][p2] += 1

# Conversion en tenseurs
match_data = torch.tensor(match_list, dtype=torch.long)                # [num_matches, 2]
match_features_tensor = torch.tensor(match_feature_list, dtype=torch.float)  # [num_matches, 36]
labels = torch.tensor(labels_list, dtype=torch.float).unsqueeze(1)       # [num_matches, 1]

###############################################
# 3. Division temporelle des données en train et test
###############################################
num_matches = len(match_data)
train_size = int(0.8 * num_matches)
train_indices = list(range(train_size))
test_indices = list(range(train_size, num_matches))

train_dataset = TensorDataset(match_data[train_indices],
                               match_features_tensor[train_indices],
                               labels[train_indices])
test_dataset = TensorDataset(match_data[test_indices],
                              match_features_tensor[test_indices],
                              labels[test_indices])
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

###############################################
# 4. Définition du modèle (GNN + LSTM + Attention + Dropout)
###############################################
# Ici, match_feat_dim est désormais 36
class TennisModel(nn.Module):
    def __init__(self, feature_dim, hidden_dim, lstm_dim, combined_dim, match_feat_dim, dropout_p=0.3):
        """
        feature_dim   : dimension des features statiques d'un joueur (ex: 1 pour le rang moyen)
        hidden_dim    : dimension cachée du GNN (ex: 64)
        lstm_dim      : dimension de sortie du LSTM pour l'historique (ex: 32)
        combined_dim  : dimension finale après fusion (ex: 32)
        match_feat_dim: dimension des features de match (ici 36)
        dropout_p     : taux de dropout
        """
        super(TennisModel, self).__init__()
        # Partie GNN
        self.conv1 = SAGEConv(feature_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        
        # Partie LSTM pour l'historique
        self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=lstm_dim, batch_first=True)
        self.lstm_proj = nn.Linear(lstm_dim, hidden_dim)  # Projection pour homogénéiser avec le GNN
        
        # Mécanisme d'attention pour fusionner GNN et LSTM
        self.att_layer = nn.Linear(2 * hidden_dim, 2)
        
        # Fusion finale des embeddings pondérés
        self.fc_player = nn.Linear(hidden_dim, combined_dim)
        
        # Dropout pour régularisation
        self.dropout = nn.Dropout(p=dropout_p)
        
        # Prédiction du match
        self.fc_match = nn.Sequential(
            nn.Linear(2 * combined_dim + match_feat_dim, combined_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(combined_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, edge_index, player_history, match_data, match_features):
        # --- Partie GNN ---
        x_gnn = self.conv1(x, edge_index).relu()       # [num_players, hidden_dim]
        x_gnn = self.conv2(x_gnn, edge_index).relu()     # [num_players, hidden_dim]
        
        # --- Partie LSTM ---
        num_players = x.size(0)
        hist_sequences = []
        for i in range(num_players):
            if i in player_history:
                seq = player_history[i]  # [seq_len, feature_dim]
            else:
                seq = torch.zeros((1, x.size(1)), device=x.device)
            hist_sequences.append(seq)
        padded_seqs = pad_sequence(hist_sequences, batch_first=True).to(x.device)
        _, (h_n, _) = self.lstm(padded_seqs)
        lstm_out = h_n.squeeze(0)  # [num_players, lstm_dim]
        lstm_out_proj = self.lstm_proj(lstm_out)  # [num_players, hidden_dim]
        
        # --- Fusion par attention ---
        fusion = torch.cat([x_gnn, lstm_out_proj], dim=1)  # [num_players, 2*hidden_dim]
        att_weights = F.softmax(self.att_layer(fusion), dim=1)  # [num_players, 2]
        fused_embedding = att_weights[:, 0].unsqueeze(1) * x_gnn + \
                          att_weights[:, 1].unsqueeze(1) * lstm_out_proj  # [num_players, hidden_dim]
        
        player_embedding = self.fc_player(fused_embedding)  # [num_players, combined_dim]
        player_embedding = self.dropout(player_embedding)
        
        # --- Prédiction du match ---
        p1_emb = player_embedding[match_data[:, 0]]  # [batch_size, combined_dim]
        p2_emb = player_embedding[match_data[:, 1]]  # [batch_size, combined_dim]
        match_input = torch.cat([p1_emb, p2_emb, match_features], dim=1)  # [batch_size, 2*combined_dim + match_feat_dim]
        out = self.fc_match(match_input)  # [batch_size, 1]
        return out

###############################################
# 5. Entraînement et évaluation avec barre de progression
###############################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TennisModel(feature_dim=player_features.size(1),
                    hidden_dim=64,
                    lstm_dim=32,
                    combined_dim=32,
                    match_feat_dim=36,  # Mise à jour de la dimension
                    dropout_p=0.3).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.BCELoss()

num_epochs = 20
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Époque {epoch+1}/{num_epochs}", leave=False)
    for match_batch, match_feat_batch, label_batch in progress_bar:
        match_batch = match_batch.to(device)
        match_feat_batch = match_feat_batch.to(device)
        label_batch = label_batch.to(device)
        
        optimizer.zero_grad()
        pred = model(player_features.to(device),
                     edge_index.to(device),
                     player_history,
                     match_batch,
                     match_feat_batch)
        loss = loss_fn(pred, label_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * match_batch.size(0)
        progress_bar.set_postfix(loss=loss.item())
    avg_loss = total_loss / len(train_loader.dataset)
    print(f"Époque {epoch+1}/{num_epochs}, Perte moyenne (train): {avg_loss:.4f}")

# Évaluation sur l'ensemble de test
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for match_batch, match_feat_batch, label_batch in test_loader:
        match_batch = match_batch.to(device)
        match_feat_batch = match_feat_batch.to(device)
        label_batch = label_batch.to(device)
        out = model(player_features.to(device),
                    edge_index.to(device),
                    player_history,
                    match_batch,
                    match_feat_batch)
        predicted = (out > 0.5).float()
        correct += (predicted == label_batch).sum().item()
        total += label_batch.size(0)
    accuracy = 100 * correct / total
    print(f"Précision du modèle sur test: {accuracy:.2f}%")
