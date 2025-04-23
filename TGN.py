import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm  # barre de progression

#############################################
# 1. Définition des classes du modèle
#############################################

class PlayerTemporalEncoder(nn.Module):
    def __init__(self, seq_input_dim, gru_hidden_dim, num_layers=1):
        super(PlayerTemporalEncoder, self).__init__()
        self.gru = nn.GRU(seq_input_dim, gru_hidden_dim, num_layers, batch_first=True)
    
    def forward(self, x):
        # x: (num_nodes, seq_length, seq_input_dim)
        _, h_n = self.gru(x)  # h_n: (num_layers, num_nodes, gru_hidden_dim)
        return h_n[-1]       # (num_nodes, gru_hidden_dim)

class EdgeEnhancedConv(MessagePassing):
    def __init__(self, node_in_dim, edge_in_dim, out_dim):
        super(EdgeEnhancedConv, self).__init__(aggr='add')
        self.mlp = nn.Sequential(
            nn.Linear(node_in_dim + edge_in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
    
    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_j, edge_attr):
        m = torch.cat([x_j, edge_attr], dim=1)
        return self.mlp(m)

class GlobalMatchPredictor(nn.Module):
    def __init__(self, seq_input_dim, gru_hidden_dim, node_static_dim,
                 node_embed_dim, edge_feat_dim, mlp_hidden_dim):
        super(GlobalMatchPredictor, self).__init__()
        self.temporal_encoder = PlayerTemporalEncoder(seq_input_dim, gru_hidden_dim)
        self.node_input_dim = gru_hidden_dim + node_static_dim
        self.conv = EdgeEnhancedConv(self.node_input_dim, edge_feat_dim, node_embed_dim)
        self.link_predictor = nn.Sequential(
            nn.Linear(2 * node_embed_dim + edge_feat_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(mlp_hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, player_seq, node_static, edge_index, edge_attr):
        x_dyn = self.temporal_encoder(player_seq)
        x = torch.cat([x_dyn, node_static], dim=1)
        x_updated = self.conv(x, edge_index, edge_attr)
        src = x_updated[edge_index[0]]
        dst = x_updated[edge_index[1]]
        combined = torch.cat([src, dst, edge_attr], dim=1)
        pred = self.link_predictor(combined)
        return pred


def fonction_un_nom(df):
    # Paramètres Elo
    starting_elo = 1500
    K = 32  # Facteur de sensibilité

    # Nettoyer et normaliser la colonne 'surface'
    df['surface'] = df['surface'].str.strip().str.upper()

    # Mapping pour les surfaces
    surface_mapping = {
        "DUR": 1,
        "TERRE BATTUE": 2,
        "DUR (INDOOR)": 3,
        "GAZON": 4
    }
    df['surface_encoded'] = df['surface'].map(surface_mapping)

    # Dictionnaire global d'Elo
    elo_ratings = {}
    # Dictionnaires pour l'Elo par surface : 
    # Pour chaque surface, on initialise un dictionnaire pour stocker les Elo des joueurs.
    elo_surface_ratings = {surf: {} for surf in surface_mapping.keys()}

    # Listes pour stocker les valeurs d'Elo AVANT mise à jour pour chaque match
    elo_j1 = []
    elo_j2 = []
    elo_j1_surface = []
    elo_j2_surface = []

    # Fonctions d'accès aux ratings (en cas d'absence, retourne starting_elo)
    def get_elo(player):
        return elo_ratings.get(player, starting_elo)
    
    def get_elo_surface(player, surf):
        return elo_surface_ratings[surf].get(player, starting_elo)

    # Parcours du DataFrame match par match
    for idx, row in df.iterrows():
        # Récupération des identifiants et de la surface
        player1 = row["j1"]
        player2 = row["j2"]
        winner = row["winner"]
        surf = row["surface"]  # par exemple "DUR", "TERRE BATTUE", etc.

        # ---------------------------
        # Mise à jour de l'Elo global
        # ---------------------------
        current_R1 = get_elo(player1)
        current_R2 = get_elo(player2)
        # Stocker les ratings avant mise à jour pour ce match
        elo_j1.append(current_R1)
        elo_j2.append(current_R2)
        # Calcul des scores attendus
        E1 = 1 / (1 + 10 ** ((current_R2 - current_R1) / 400))
        E2 = 1 / (1 + 10 ** ((current_R1 - current_R2) / 400))
        # Scores réels : 1 pour la victoire, 0 pour la défaite
        S1 = 1 if winner == player1 else 0
        S2 = 1 if winner == player2 else 0
        # Mise à jour globale
        new_R1 = current_R1 + K * (S1 - E1)
        new_R2 = current_R2 + K * (S2 - E2)
        # Actualiser le dictionnaire
        elo_ratings[player1] = new_R1
        elo_ratings[player2] = new_R2

        # ---------------------------
        # Mise à jour de l'Elo par surface
        # ---------------------------
        current_R1_surf = get_elo_surface(player1, surf)
        current_R2_surf = get_elo_surface(player2, surf)
        elo_j1_surface.append(current_R1_surf)
        elo_j2_surface.append(current_R2_surf)
        # Calcul des scores attendus pour la surface
        E1_surf = 1 / (1 + 10 ** ((current_R2_surf - current_R1_surf) / 400))
        E2_surf = 1 / (1 + 10 ** ((current_R1_surf - current_R2_surf) / 400))
        # Mise à jour par surface
        new_R1_surf = current_R1_surf + K * (S1 - E1_surf)
        new_R2_surf = current_R2_surf + K * (S2 - E2_surf)
        # Actualiser le dictionnaire pour la surface correspondante
        elo_surface_ratings[surf][player1] = new_R1_surf
        elo_surface_ratings[surf][player2] = new_R2_surf

    # Ajout des colonnes d'Elo au DataFrame
    df["elo_j1"] = elo_j1
    df["elo_j2"] = elo_j2
    df["elo_j1_surface"] = elo_j1_surface
    df["elo_j2_surface"] = elo_j2_surface

    return df

#############################################
# 2. Préparation des données
#############################################

# Chargement et prétraitement
data = pd.read_csv("data/test2.csv", encoding="utf-8")
data = fonction_un_nom(data)

drop_cols = [col for col in data.columns if "Unnamed:" in col]
data.drop(columns=drop_cols, inplace=True)
data['date'] = pd.to_datetime(data['date'], errors='coerce')
data = data.sort_values('date')
data.reset_index(drop=True, inplace=True)
data['target'] = (data['winner'] == data['j1']).astype(int)
print(data)
# Exemple de DataFrame
# Assurez-vous que la colonne 'date' est au format datetime64[ns]
# Le DataFrame doit contenir au moins les colonnes 'date', 'j1', 'j2' et 'winner'
data = data.sort_values('date').reset_index(drop=True)

# Initialisation des colonnes pour les statistiques de chaque joueur dans le match courant
data['days_since_last_j1'] = 0
data['matches_played_j1'] = 0
data['winrate_j1'] = 0

data['days_since_last_j2'] = 0
data['matches_played_j2'] = 0
data['winrate_j2'] = 0

# Dictionnaires pour stocker l'historique global de chaque joueur, indépendamment du rôle
last_match_date = dict()          # Dernière date de match pour chaque joueur
match_counts = defaultdict(int)   # Nombre total de matchs joués par chaque joueur
win_counts = defaultdict(int)     # Nombre total de victoires pour chaque joueur

# Itération chronologique sur les matchs
for idx, row in data.iterrows():
    current_date = row['date']
    j1 = row['j1']
    j2 = row['j2']
    
    # Pour le joueur j1
    if j1 in last_match_date:
        days = (current_date - last_match_date[j1]).days
        winrate = win_counts[j1] / match_counts[j1] if match_counts[j1] > 0 else np.nan
        data.at[idx, 'days_since_last_j1'] = days
        data.at[idx, 'matches_played_j1'] = match_counts[j1]
        data.at[idx, 'winrate_j1'] = winrate
    else:
        data.at[idx, 'days_since_last_j1'] = 1000
        data.at[idx, 'matches_played_j1'] = 0
        data.at[idx, 'winrate_j1'] = 0

    # Pour le joueur j2
    if j2 in last_match_date:
        days = (current_date - last_match_date[j2]).days
        winrate = win_counts[j2] / match_counts[j2] if match_counts[j2] > 0 else np.nan
        data.at[idx, 'days_since_last_j2'] = days
        data.at[idx, 'matches_played_j2'] = match_counts[j2]
        data.at[idx, 'winrate_j2'] = winrate
    else:
        data.at[idx, 'days_since_last_j2'] = 1000
        data.at[idx, 'matches_played_j2'] = 0
        data.at[idx, 'winrate_j2'] = 0

    # Mise à jour globale pour le joueur j1
    last_match_date[j1] = current_date
    match_counts[j1] += 1
    if row['winner'] == j1:
        win_counts[j1] += 1

    # Mise à jour globale pour le joueur j2
    last_match_date[j2] = current_date
    match_counts[j2] += 1
    if row['winner'] == j2:
        win_counts[j2] += 1

# Affichage des premières lignes pour vérification

# Séparation chronologique train/test (80/20)
train_ratio = 0.8
split_idx = int(len(data) * train_ratio)
train_data = data.iloc[:split_idx].copy()
test_data = data.iloc[split_idx:].copy()

# Mapping des joueurs basé sur train_data
players_train = pd.concat([train_data['j1'], train_data['j2']]).unique()
player_to_idx = {player: idx for idx, player in enumerate(players_train)}
# Filtrer test_data pour garder uniquement des matchs avec joueurs connus
test_data = test_data[test_data['j1'].isin(player_to_idx) & test_data['j2'].isin(player_to_idx)].copy()
# Combiner les deux colonnes en une seule dimension
combined = train_data[['elo_j1', 'elo_j2']].values.ravel()

# Ajuster le scaler sur ces valeurs combinées
global_min = combined.min()
global_max = combined.max()

# Transformer chaque colonne avec le scaler ajusté globalement
train_data[['elo_j1', 'elo_j2']] = (train_data[['elo_j1', 'elo_j2']] - global_min) / (global_max - global_min)
test_data[['elo_j1', 'elo_j2']] = (test_data[['elo_j1', 'elo_j2']] - global_min) / (global_max - global_min)


combined = train_data[['elo_j1_surface', 'elo_j2_surface']].values.ravel()

# Ajuster le scaler sur ces valeurs combinées
global_min = combined.min()
global_max = combined.max()

# Transformer chaque colonne avec le scaler ajusté globalement
train_data[['elo_j1_surface', 'elo_j2_surface']] = (train_data[['elo_j1_surface', 'elo_j2_surface']] - global_min) / (global_max - global_min)
test_data[['elo_j1_surface', 'elo_j2_surface']] = (test_data[['elo_j1_surface', 'elo_j2_surface']] - global_min) / (global_max - global_min)


combined = train_data[['point1', 'point2']].values.ravel()

# Ajuster le scaler sur ces valeurs combinées
global_min = combined.min()
global_max = combined.max()

# Transformer chaque colonne avec le scaler ajusté globalement
train_data[['point1', 'point2']] = (train_data[['point1', 'point2']] - global_min) / (global_max - global_min)
test_data[['point1', 'point2']] = (test_data[['point1', 'point2']] - global_min) / (global_max - global_min)

combined = train_data[['rank1', 'rank2']].values.ravel()

# Ajuster le scaler sur ces valeurs combinées
global_min = combined.min()
global_max = combined.max()

# Transformer chaque colonne avec le scaler ajusté globalement
train_data[['rank1', 'rank2']] = (train_data[['rank1', 'rank2']] - global_min) / (global_max - global_min)
test_data[['rank1', 'rank2']] = (test_data[['rank1', 'rank2']] - global_min) / (global_max - global_min)

combined = train_data[['age1', 'age2']].values.ravel()

# Ajuster le scaler sur ces valeurs combinées
global_min = combined.min()
global_max = combined.max()

# Transformer chaque colonne avec le scaler ajusté globalement
train_data[['age1', 'age2']] = (train_data[['age1', 'age2']] - global_min) / (global_max - global_min)
test_data[['age1', 'age2']] = (test_data[['age1', 'age2']] - global_min) / (global_max - global_min)


combined = train_data[['days_since_last_j1', 'days_since_last_j2']].values.ravel()

# Ajuster le scaler sur ces valeurs combinées
global_min = combined.min()
global_max = combined.max()

# Transformer chaque colonne avec le scaler ajusté globalement
train_data[['days_since_last_j1', 'days_since_last_j2']] = (train_data[['days_since_last_j1', 'days_since_last_j2']] - global_min) / (global_max - global_min)
test_data[['days_since_last_j1', 'days_since_last_j2']] = (test_data[['days_since_last_j1', 'days_since_last_j2']] - global_min) / (global_max - global_min)



train_data['elo_diff'] = train_data['elo_j1'] - train_data['elo_j2']
test_data['elo_diff'] = test_data['elo_j1'] - test_data['elo_j2']

train_data['elo_diff_surface'] = train_data['elo_j1_surface'] - train_data['elo_j2_surface']
test_data['elo_diff_surface'] = test_data['elo_j1_surface'] - test_data['elo_j2_surface']

train_data['days_diff'] = train_data['days_since_last_j1'] - train_data['days_since_last_j2']
test_data['days_diff'] = test_data['days_since_last_j1'] - test_data['days_since_last_j2']

print(train_data[["age1","elo_j1", "rank1","age2","elo_j2", "rank2","point1","point2"]].describe())

#############################################
# 3. Préparation des features d'arête
#############################################
# Pour train_data
edge_features_train = train_data[['elo_diff','elo_diff_surface', 'surface_encoded', 'tour_encoded','days_diff']].values
edge_labels_train = train_data['target'].values.reshape(-1, 1)
train_data['j1_idx'] = train_data['j1'].map(player_to_idx)
train_data['j2_idx'] = train_data['j2'].map(player_to_idx)
edge_index_train = torch.tensor(train_data[['j1_idx', 'j2_idx']].values.T, dtype=torch.long)
edge_attr_train = torch.tensor(edge_features_train, dtype=torch.float)
edge_labels_train = torch.tensor(edge_labels_train, dtype=torch.float)

# Pour test_data
edge_features_test = test_data[['elo_diff','elo_diff_surface', 'surface_encoded', 'tour_encoded','days_diff']].values
edge_labels_test = test_data['target'].values.reshape(-1, 1)
test_data['j1_idx'] = test_data['j1'].map(player_to_idx)
test_data['j2_idx'] = test_data['j2'].map(player_to_idx)
edge_index_test = torch.tensor(test_data[['j1_idx', 'j2_idx']].values.T, dtype=torch.long)
edge_attr_test = torch.tensor(edge_features_test, dtype=torch.float)
edge_labels_test = torch.tensor(edge_labels_test, dtype=torch.float)

#############################################
# 4. Préparation des features dynamiques et statiques
#############################################

# Définir les colonnes de features dynamiques
dynamic_features_j1 = [
    "elo_j1","time","winrate_j1","days_since_last_j1","matches_played_j1"
]
dynamic_features_j2 = [
        "elo_j2","time","winrate_j2","days_since_last_j2","matches_played_j2"
]
feature_dim = len(dynamic_features_j1)
max_seq_len = 5

# Fonction pour obtenir la séquence historique avec padding
def get_history_sequence(player, history_dict, max_seq_len, feature_dim):
    hist = history_dict[player] if player in history_dict else []
    if len(hist) < max_seq_len:
        pad = [np.zeros(feature_dim, dtype=np.float32)] * (max_seq_len - len(hist))
        seq = pad + hist
    else:
        seq = hist[-max_seq_len:]
    return np.array(seq)

# Fonction pour extraire les features statiques (dernière valeur observée)
def build_static_features(player, data_subset, current_date):
    current_date = pd.Timestamp(current_date)
    # Filtrer les données pour ne garder que celles avant la date du match actuel
    data_subset = data_subset[data_subset['date'] <= current_date]
    
    data_j1 = data_subset[data_subset['j1'] == player]
    data_j2 = data_subset[data_subset['j2'] == player]
    
    if not data_j1.empty:
        row = data_j1.iloc[-1]
        age, elo, rank, point,winrate,elo_surface,number_match = row['age1'], row['elo_j1'], row['rank1'], row['point1'],row['winrate_j1'],row["elo_j1_surface"],row["matches_played_j1"]
    elif not data_j2.empty:
        row = data_j2.iloc[-1]
        age, elo, rank, point,winrate,elo_surface,number_match = row['age2'], row['elo_j2'], row['rank2'], row['point2'],row['winrate_j2'],row["elo_j2_surface"],row["matches_played_j2"]
    else:
        age, elo, rank, point,winrate,elo_surface,number_match = 0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0
        
    return np.array([age, elo, rank, point,winrate,elo_surface,number_match], dtype=np.float32)

#############################################
# 5. Création de batchs par date pour l'entraînement
#############################################

# Regrouper train_data par date
grouped = train_data.groupby(train_data['date'].dt.date)

# Initialiser l'historique "online"
player_history_online = defaultdict(list)

# Paramètres du modèle
seq_input_dim = feature_dim   # dimension dynamique
gru_hidden_dim = 32
node_static_dim = 7          # [age, elo, rank]
node_embed_dim = 128
edge_feat_dim = 5        # [elo_diff, surface_encoded, tour_encoded]
mlp_hidden_dim = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GlobalMatchPredictor(seq_input_dim, gru_hidden_dim, node_static_dim,
                             node_embed_dim, edge_feat_dim, mlp_hidden_dim).to(device)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Nombre total de paramètres :", total_params)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Entraînement sur plusieurs epochs avec batchs par date
num_epochs = 10
for epoch in range(num_epochs):
    total_loss = 0.0
    # Utilisation de tqdm pour la barre de progression sur les dates
    y_true_train = []
    y_pred_train = []
    for date, batch in tqdm(grouped, desc=f"Epoch {epoch+1}/{num_epochs}"):
        # Pour chaque match de la date
        for idx, row in batch.iterrows():
            j1 = row['j1']
            j2 = row['j2']
            # Construire l'historique dynamique pour j1 et j2 avant ce match
            seq_j1 = get_history_sequence(j1, player_history_online, max_seq_len, feature_dim)
            seq_j2 = get_history_sequence(j2, player_history_online, max_seq_len, feature_dim)
            player_seq_batch = torch.tensor(np.stack([seq_j1, seq_j2]), dtype=torch.float).to(device)
            # Récupérer les features statiques pour j1 et j2 (à partir de train_data)
            static_j1 = build_static_features(j1, train_data,date)
            static_j2 = build_static_features(j2, train_data,date)
            node_static_batch = torch.tensor(np.stack([static_j1, static_j2]), dtype=torch.float).to(device)
            # Dans ce mini-batch, les joueurs ont les indices 0 et 1
            edge_index_batch = torch.tensor([[0], [1]], dtype=torch.long).to(device)
            # Extraire les features d'arête pour ce match (en forçant le type numérique)
            edge_attr_batch = torch.tensor(
                row[['elo_diff','elo_diff_surface', 'surface_encoded', 'tour_encoded','days_diff']].astype(np.float32).values,
                dtype=torch.float
            ).unsqueeze(0).to(device)
            label = torch.tensor([[row['target']]], dtype=torch.float).to(device)
            
            model.train()
            optimizer.zero_grad()
            pred = model(player_seq_batch, node_static_batch, edge_index_batch, edge_attr_batch)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            # Mise à jour de l'historique après la prédiction
            if not row[dynamic_features_j1].isnull().any():
                player_history_online[j1].append(row[dynamic_features_j1].values.astype(np.float32))
            if not row[dynamic_features_j2].isnull().any():
                player_history_online[j2].append(row[[f.replace('_j1','_j2') for f in dynamic_features_j1]].values.astype(np.float32))
            y_true_train.append(label.item())
            y_pred_train.append(1 if pred.item() > 0.5 else 0)
    
    avg_loss = total_loss / len(train_data)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    acc = accuracy_score(y_true_train, y_pred_train)
    report = classification_report(y_true_train, y_pred_train)
    cm = confusion_matrix(y_true_train, y_pred_train)

    print("\nTest Accuracy:", acc)
    print("\nClassification Report:\n", report)
    print("\nConfusion Matrix:\n", cm)

#############################################
# 6. Évaluation sur l'ensemble test
#############################################

# Initialiser l'historique de test à partir de l'historique online du train
    player_history_test = defaultdict(list)
    for player, hist in player_history_online.items():
        player_history_test[player] = hist.copy()

    y_true_test = []
    y_pred_test = []

    for idx, row in test_data.iterrows():
        j1 = row['j1']
        j2 = row['j2']
        seq_j1 = get_history_sequence(j1, player_history_test, max_seq_len, feature_dim)
        seq_j2 = get_history_sequence(j2, player_history_test, max_seq_len, feature_dim)
        player_seq_batch = torch.tensor(np.stack([seq_j1, seq_j2]), dtype=torch.float).to(device)
        static_j1 = build_static_features(j1, test_data,date)
        static_j2 = build_static_features(j2, test_data,date)
        # Si une valeur manque, reprendre depuis train
        if np.any(np.isnan(static_j1)):
            print(1)
            static_j1 = build_static_features(j1, train_data,date)
        if np.any(np.isnan(static_j2)):
            print(2)
            static_j2 = build_static_features(j2, train_data)
        node_static_batch = torch.tensor(np.stack([static_j1, static_j2]), dtype=torch.float).to(device)
        edge_index_batch = torch.tensor([[0], [1]], dtype=torch.long).to(device)
        edge_attr_batch = torch.tensor(
            row[['elo_diff','elo_diff_surface', 'surface_encoded', 'tour_encoded','days_diff']].astype(np.float32).values,
            dtype=torch.float
        ).unsqueeze(0).to(device)
        label = torch.tensor([[row['target']]], dtype=torch.float).to(device)
        
        model.eval()
        with torch.no_grad():
            pred = model(player_seq_batch, node_static_batch, edge_index_batch, edge_attr_batch)
        y_true_test.append(label.item())
        y_pred_test.append(1 if pred.item() > 0.5 else 0)
        
        # Mise à jour de l'historique test
        if not row[dynamic_features_j1].isnull().any():
            player_history_test[j1].append(row[dynamic_features_j1].values.astype(np.float32))
        if not row[dynamic_features_j2].isnull().any():
            player_history_test[j2].append(row[[f.replace('_j1','_j2') for f in dynamic_features_j1]].values.astype(np.float32))

    # Calcul des métriques
    acc = accuracy_score(y_true_test, y_pred_test)
    report = classification_report(y_true_test, y_pred_test)
    cm = confusion_matrix(y_true_test, y_pred_test)

    print("\nTest Accuracy:", acc)
    print("\nClassification Report:\n", report)
    print("\nConfusion Matrix:\n", cm)