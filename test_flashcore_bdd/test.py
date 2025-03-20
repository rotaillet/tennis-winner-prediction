import torch
from utils import (
    normalize_columns, build_mappings, build_node_features, compute_player_differences,get_days_since_last_match,
    build_player_graph_with_weights, build_player_history, split_last_match,build_player_graph_with_weights_recent
)
from dataset import TennisMatchDataset2
import pandas as pd
from train import test_model2
from torch.utils.data import Dataset, DataLoader

# Charger le modèle pré-entraîné
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
    
STATIC_FEATURE_DIM = 34   
HIST_FEATURE_DIM = 9
d_model = 256
gnn_hidden = 256
num_gnn_layers = 12
dropout = 0.3
df = pd.read_csv("data/features.csv", parse_dates=["date"])

surface_mapping = {
    "DUR": 1,
    "TERRE BATTUE": 2,
    "DUR (INDOOR)": 3,
    "GAZON": 4
}

# Nettoyer la colonne "surface" pour uniformiser les valeurs
df['surface'] = df['surface'].str.strip().str.upper()

# Appliquer le mapping et créer une nouvelle colonne "surface_encoded"
df['surface_encoded'] = df['surface'].map(surface_mapping)


tour_mapping = {
    "1/16 DE FINALE":1,
    "DEMI-FINALES":2,
    "QUARTS DE FINALE":3,
    "1/8 DE FINALE":4,
    "1/32 DE FINALE":5,
    "1/64 DE FINALE":6,
    "FINALE":7,
    "3E PLACE":8
}

# Nettoyer la colonne "surface" pour uniformiser les valeurs
df['tour'] = df['tour'].str.strip().str.upper()

# Appliquer le mapping et créer une nouvelle colonne "surface_encoded"
df['tour_encoded'] = df['tour'].map(tour_mapping)


print("Nombre de matchs dans le dataset :", len(df))
player_to_idx, tournoi_to_idx = build_mappings(df)
history = build_player_history(df)
train_df, test_df = split_last_match(df)
# Supposons que df contient toutes les rencontres, player_to_idx et tournoi_to_idx sont vos mappings.
edge_index, edge_weight = build_player_graph_with_weights_recent(df, player_to_idx, lambda_=0.001)
node_features = build_node_features(df, player_to_idx)
num_players = len(player_to_idx)
num_tournois = len(tournoi_to_idx)

import numpy as np
import pandas as pd

# Supposons que compute_player_differences(row) et get_days_since_last_match(history, player, current_date)
# soient déjà définis.

# Listes pour accumuler les valeurs à normaliser

last_match_list = []

# Pour itérer sur chaque match du train_df
for idx, row in train_df.iterrows():
    current_date = row["date"]
    

    # 4. Temps depuis le dernier match pour chaque joueur
    last_match_p1 = get_days_since_last_match(history, row["j1"], current_date)
    last_match_p2 = get_days_since_last_match(history, row["j2"], current_date)
    last_match_list.append(last_match_p1)
    last_match_list.append(last_match_p2)

# Calcul des moyennes et écarts-types avec une petite constante pour éviter la division par zéro
epsilon = 1e-6
norm_params = {

    
    "last_match_mean": np.mean(last_match_list),
    "last_match_std": np.std(last_match_list) + epsilon,
}


from torch_geometric.data import Data
graph_data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_weight)


num_player = len(player_to_idx)
num_tournois = len(tournoi_to_idx)


train_dataset = TennisMatchDataset2(train_df,history,player_to_idx,tournoi_to_idx,norm_params)
test_dataset = TennisMatchDataset2(test_df,history,player_to_idx,tournoi_to_idx,norm_params)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False,num_workers=10)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False,num_workers=10)

model = TennisMatchPredictor(STATIC_FEATURE_DIM, num_players, num_tournois, d_model, gnn_hidden,HIST_FEATURE_DIM, num_gnn_layers, dropout,seq_length=5)

state_dict = torch.load('model/best_model84.pth', map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
device = "cuda" if torch.cuda.is_available() else "cpu"
criterion = nn.CrossEntropyLoss()
graph_data = graph_data.to(device)
model.to(device)
model.eval()

test_loss, test_acc,outputs = test_model2(model, test_dataloader, criterion,graph_data,device)

print(outputs)


preds_cpu = outputs.cpu().numpy()
import matplotlib.pyplot as plt

# Convert predictions to percentages
pourcentages = preds_cpu * 100

# Separate percentages for each class
classe0 = pourcentages[:, 0]
classe1 = pourcentages[:, 1]

# Calculate the percentage of samples with >80% confidence for each class
perc_over80_class0 = (classe0 > 80).sum() / len(classe0) * 100
perc_over80_class1 = (classe1 > 80).sum() / len(classe1) * 100

print("Percentage of matches with >80% confidence for Class 0: {:.2f}%".format(perc_over80_class0))
print("Percentage of matches with >80% confidence for Class 1: {:.2f}%".format(perc_over80_class1))

# Filter only values above 80% for each class
classe0_over80 = classe0[classe0 > 80]
classe1_over80 = classe1[classe1 > 80]

# Plot the histogram for the filtered predictions
plt.figure(figsize=(10, 6))
plt.hist(classe0_over80, bins=20, alpha=0.7, label='Classe 0')
plt.hist(classe1_over80, bins=20, alpha=0.7, label='Classe 1')
plt.xlabel('Pourcentage (%)')
plt.ylabel("Nombre d'échantillons")
plt.title("Distribution des prédictions >80% pour chaque classe")
plt.legend()
plt.savefig("test.png")
plt.show()
