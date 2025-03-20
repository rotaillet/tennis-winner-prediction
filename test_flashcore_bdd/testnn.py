import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
import torch_geometric
from scipy.cluster.hierarchy import linkage, fcluster
from rapidfuzz import fuzz
from tqdm import tqdm

# Importation des fonctions et classes depuis vos fichiers locaux
from utils import (
    normalize_columns, build_mappings, build_node_features, compute_player_differences,get_days_since_last_match,
    build_player_graph_with_weights, build_player_history, split_last_match,build_player_graph_with_weights_recent
)
from dataset import TennisMatchDataset2
from model import TennisMatchPredictor
from train import test_model2



      
STATIC_FEATURE_DIM = 34   
HIST_FEATURE_DIM = 9




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


# On crée un objet Data de PyTorch Geometric pour le graphe
from torch_geometric.data import Data
graph_data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_weight)


num_player = len(player_to_idx)
num_tournois = len(tournoi_to_idx)


train_dataset = TennisMatchDataset2(train_df,history,player_to_idx,tournoi_to_idx,norm_params)
test_dataset = TennisMatchDataset2(test_df,history,player_to_idx,tournoi_to_idx,norm_params)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False,num_workers=10)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False,num_workers=10)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
graph_data = graph_data.to(device)
d_model = 256
gnn_hidden = 256
num_gnn_layers = 12
dropout = 0.3

model = TennisMatchPredictor(STATIC_FEATURE_DIM, num_players, num_tournois, d_model, gnn_hidden,HIST_FEATURE_DIM, num_gnn_layers=12, dropout=0.3,seq_length=5)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Nombre total de paramètres du modèle : {total_params}")
model.load_state_dict(torch.load('model/best_model84.pth'))
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

from torch.optim.lr_scheduler import ReduceLROnPlateau
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
class EarlyStopping:
    """
    Arrête l'entraînement si la perte de validation ne s'améliore pas après un certain nombre d'époques.
    
    Args:
        patience (int): Nombre d'époques à attendre avant d'arrêter l'entraînement.
        min_delta (float): Amélioration minimale de la perte pour considérer qu'il y a une amélioration.
    """
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss < self.min_delta:
            self.best_loss = val_loss
            self.counter = 0  # Réinitialiser le compteur si amélioration
        else:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

max_test_acc = 0.0  
all_train_loss = []
all_test_loss = []
all_test_acc = []



early_stopping = EarlyStopping(patience=15, min_delta=0.001)  # par exemple
for epoch in tqdm(range(300), desc="Époques", leave=True):
    model.train()
    running_loss = 0.0
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/300", leave=False):

        static_feat = batch["static_feat"].to(device)
        player1_idx = batch["player1_idx"].to(device)
        player2_idx = batch["player2_idx"].to(device)
        tournoi_idx = batch["tournoi_idx"].to(device)
        player1_seq = batch["player1_seq"].to(device)
        player2_seq = batch["player2_seq"].to(device)
        targets = batch["target"].to(device)

        optimizer.zero_grad()
        outputs = model(static_feat, player1_idx, player2_idx, tournoi_idx, graph_data,player1_seq,player2_seq)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_loss = running_loss / len(train_dataloader)
    all_train_loss.append(train_loss)

    test_loss, test_acc,outputs = test_model2(model, test_dataloader,criterion,graph_data,device)
    if test_acc > max_test_acc:
        max_test_acc = test_acc
        torch.save(model.state_dict(), "model/best_model.pth")
    scheduler.step(test_loss)
    all_test_loss.append(test_loss)
    all_test_acc.append(test_acc)

    tqdm.write(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
    # Print details of mispredicted samples (if any)



import matplotlib.pyplot as plt

# Graphique des losses
plt.figure(figsize=(10, 6))
plt.plot(all_train_loss, label="Train Loss", marker="o")
plt.plot(all_test_loss, label="Validation Loss", marker="o")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training et Validation Loss vs Epochs")
plt.legend()
plt.grid(True)
plt.savefig("loss.png")

# Graphique de la précision (accuracy)
plt.figure(figsize=(10, 6))
plt.plot(all_test_acc, label="Validation Accuracy", marker="o")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Validation Accuracy vs Epochs")
plt.legend()
plt.grid(True)
plt.savefig("acc.png")
