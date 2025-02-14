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

# Encodage des joueurs en identifiants numériques
players = list(set(df["Player_1"].unique()) | set(df["Player_2"].unique()))
player_encoder = {name: i for i, name in enumerate(players)}
df["P1_ID"] = df["Player_1"].map(player_encoder)
df["P2_ID"] = df["Player_2"].map(player_encoder)
df["Winner_ID"] = df["Winner"].map(player_encoder)

# Construction du graphe : création d'arêtes bidirectionnelles
edges = []
for _, row in df.iterrows():
    edges.append([row["P1_ID"], row["P2_ID"]])
    edges.append([row["P2_ID"], row["P1_ID"]])
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # Shape: [2, num_edges]

# Création des features statiques pour chaque joueur (ici, moyenne des classements)
player_rank = defaultdict(list)
for _, row in df.iterrows():
    player_rank[row["P1_ID"]].append(row["Rank_1"])
    player_rank[row["P2_ID"]].append(row["Rank_2"])

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
    player_history[row["P1_ID"]].append([row["Rank_1"]])
    player_history[row["P2_ID"]].append([row["Rank_2"]])
for player_id in player_history:
    player_history[player_id] = torch.tensor(player_history[player_id], dtype=torch.float)

###############################################
# 2. Calcul des features de match enrichies
###############################################

# --- Base features (8 dims) ---
# [rank_diff, pts_diff, odd_diff, best_of, surface_Hard, surface_Clay, surface_Grass, round_val]

# Mapping du round en valeur ordinale
round_mapping = {
    "1st Round": 1,
    "2nd Round": 2,
    "Quarterfinals": 3,
    "Semifinals": 4,
    "The Final": 5,
    "Final": 5
}
def get_round_value(round_str):
    return round_mapping.get(round_str, 0)

# Encodage one-hot de la surface (Hard, Clay, Grass)
def encode_surface(surface_str):
    mapping = {"Hard": [1, 0, 0], "Clay": [0, 1, 0], "Grass": [0, 0, 1]}
    return mapping.get(surface_str, [0, 0, 0])

# Pour la forme, nous définissons deux fenêtres
WINDOW_IMMEDIATE = 3
WINDOW_LONG = 10

# Dictionnaires pour stocker :
player_recent_results = defaultdict(list)  # liste des résultats (1 pour victoire, 0 pour défaite)
head_to_head_record = {}  # clé : (min_id, max_id), valeur : dict {id: nombre de victoires}

# Listes pour stocker les features de match et les labels
match_feature_list = []
match_list = []
labels_list = []

# Itération chronologique sur les matchs
for idx, row in df.iterrows():
    p1 = row["P1_ID"]
    p2 = row["P2_ID"]
    label = 1 if row["Winner_ID"] == p1 else 0

    # --- Base features ---
    rank_diff = row["Rank_1"] - row["Rank_2"]
    pts_diff = row["Pts_1"] - row["Pts_2"]
    odd_diff = row["Odd_1"] - row["Odd_2"]
    best_of = row["Best of"]
    surface_vec = encode_surface(row["Surface"])
    round_val = get_round_value(row["Round"])
    base_features = [rank_diff, pts_diff, odd_diff, best_of] + surface_vec + [round_val]  # 8 dims

    # --- Forme (immediate et globale) ---
    def compute_form(player, window):
        if player_recent_results[player]:
            recent = player_recent_results[player][-window:]
            return np.mean(recent)
        else:
            return 0.5  # valeur par défaut neutre
    form3_p1 = compute_form(p1, WINDOW_IMMEDIATE)
    form10_p1 = compute_form(p1, WINDOW_LONG)
    form3_p2 = compute_form(p2, WINDOW_IMMEDIATE)
    form10_p2 = compute_form(p2, WINDOW_LONG)
    
    # --- Head-to-head ---
    key = tuple(sorted((p1, p2)))
    if key in head_to_head_record:
        record = head_to_head_record[key]
        total_confrontations = record.get(key[0], 0) + record.get(key[1], 0)
        wins_p1 = record.get(p1, 0)
        win_ratio_p1 = wins_p1 / total_confrontations if total_confrontations > 0 else 0.5
    else:
        total_confrontations = 0
        win_ratio_p1 = 0.5

    # --- Court (intérieur/extérieur) ---
    if "Court" in df.columns:
        # Considérer "Outdoor" comme 1 et "Indoor" comme 0 (insensible à la casse)
        court_feature = 1 if row["Court"].lower().startswith("out") else 0
    else:
        court_feature = 0.5  # valeur neutre si non spécifié

    # --- Combinaison des nouvelles features ---
    # Forme : 4 features (immediate et globale pour p1 et p2)
    # Head-to-head : 2 features (total confrontations, win ratio de p1)
    # Court : 1 feature
    additional_features = [form3_p1, form10_p1, form3_p2, form10_p2,
                           total_confrontations, win_ratio_p1,
                           court_feature]
    # Dimensions additionnelles = 4 + 2 + 1 = 7
    # Total dimensions = 8 (base) + 7 = 15
    match_features = base_features + additional_features
    
    match_feature_list.append(match_features)
    match_list.append([p1, p2])
    labels_list.append(label)
    
    # --- Mise à jour des historiques après le match courant ---
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
match_data = torch.tensor(match_list, dtype=torch.long)          # [num_matches, 2]
match_features_tensor = torch.tensor(match_feature_list, dtype=torch.float)  # [num_matches, 15]
labels = torch.tensor(labels_list, dtype=torch.float).unsqueeze(1)  # [num_matches, 1]

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
# 4. Définition du modèle amélioré (GNN + LSTM + Attention + Dropout)
###############################################

# Remarque : match_feat_dim est désormais 15.
class TennisModel(nn.Module):
    def __init__(self, feature_dim, hidden_dim, lstm_dim, combined_dim, match_feat_dim, dropout_p=0.3):
        """
        feature_dim   : dimension des features statiques d'un joueur (ex: 1 pour le rang moyen)
        hidden_dim    : dimension cachée du GNN (ex: 64)
        lstm_dim      : dimension de sortie du LSTM pour l'historique (ex: 32)
        combined_dim  : dimension finale après fusion (ex: 32)
        match_feat_dim: dimension des features de match (ici 15)
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
                    match_feat_dim=15,
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
