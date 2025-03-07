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
    normalize_columns, build_mappings, build_node_features, 
    build_player_graph_with_weights, build_player_history, split_last_match,
    nettoyer_tournoi,extraire_prize_money
)
from dataset import TennisMatchDataset
from model import HybridTennisModel
from train import test_model



WINDOW_SIZE = 20            # Taille de la fentre glissante pour l'historique
HIST_FEATURE_DIM = 19       # Nombre de features historiques utilises
STATIC_FEATURE_DIM = 75    
D_MODEL = 256               # Dimension pour la fusion et le Transformer temporel
GAT_HIDDEN_DIM = 128         # Dimension cache pour le GAT
GAT_OUTPUT_DIM = 128        # Dimension de sortie du GAT
NUM_HEADS_DIM = 2
DROPOUT = 0.3


df = pd.read_csv("data/test2.csv", parse_dates=["Date"])
df = df[50000:]
df[["Tournoi_propre", "Prize_money"]] = df["Tournoi"].apply(lambda x: pd.Series(extraire_prize_money(x)))

# Liste des colonnes numériques à normaliser
columns_to_normalize = [
    
    "Age_Joueur_1", "Age_Joueur_2",
    "Points_Joueur_1", "Points_Joueur_2",
    "nb_sets",
    "prev_DOUBLE_FAULTS_p1", "prev_DOUBLE_FAULTS_p2",
    "prev_ACES_p1", "prev_ACES_p2",
    "prev_total_games_p1", "prev_total_games_p2",
    "prev_set_win_p1", "prev_set_win_p2",
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
    "prev_TOTAL_POINTS_WON_p2_num", "prev_TOTAL_POINTS_WON_p2_den", "prev_TOTAL_POINTS_WON_p2_pct","Prize_money"
]

df = normalize_columns(df, columns_to_normalize)


print(df[["Tournoi_propre", "Prize_money"]])
print(df["Tournoi_propre"].value_counts())

df["Tournoi_propre"] = df["Tournoi_propre"].apply(nettoyer_tournoi)

print(df["Tournoi_propre"].value_counts())
print("Nombre de matchs dans le dataset :", len(df))
player_to_idx, tournoi_to_idx = build_mappings(df)
history = build_player_history(df)
train_df, test_df = split_last_match(df)




train_dataset = TennisMatchDataset(train_df, history, player_to_idx, tournoi_to_idx)
test_dataset = TennisMatchDataset(test_df, history, player_to_idx, tournoi_to_idx)

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

num_players = len(player_to_idx)
num_tournois = len(tournoi_to_idx)

edge_index, edge_weight = build_player_graph_with_weights(df, player_to_idx, lambda_=0.01)
node_features = build_node_features(df, player_to_idx)
player_feature_dim = node_features.shape[1]

num_nodes = len(player_to_idx)  # Nombre de joueurs
num_edges = edge_index.size(1)  # Nombre d’arêtes (edges)

print(f"Nombre de nœuds (joueurs) : {num_nodes}")
print(f"Nombre d’arêtes (matchs pondérés) : {num_edges}")
print(f"Degré moyen par joueur : {num_edges / num_nodes:.2f}")



device = "cuda" if torch.cuda.is_available() else "cpu"

# Liste de combinaisons d'hyperparamètres à tester
hyperparams_list = [

    {"lr": 0.0001, "dropout": 0.4, "D_MODEL": 512, "GAT_HIDDEN_DIM": 128, "GAT_OUTPUT_DIM": 128, "weight_decay": 1e-4}
]

results = []
best_model_state = None  # Stockera le modèle avec la meilleure précision
best_hparams = None  # Stockera les hyperparamètres correspondants
best_test_acc = 0.0  # Meilleure précision trouvée

for i, hparams in enumerate(hyperparams_list):
    print("\n====== Expérience", i+1, "======")
    print("Hyperparamètres :", hparams)
    
    model = HybridTennisModel(
        player_feature_dim=player_feature_dim,
        gat_hidden_dim=hparams["GAT_HIDDEN_DIM"],
        gat_output_dim=hparams["GAT_OUTPUT_DIM"],
        hist_feature_dim=HIST_FEATURE_DIM,
        static_feature_dim=STATIC_FEATURE_DIM,
        d_model=hparams["D_MODEL"],
        num_players=num_players,
        num_tournois=num_tournois,
        num_heads=NUM_HEADS_DIM,
        dropout=hparams["dropout"]
    )
    
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams["lr"], weight_decay=hparams["weight_decay"])
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

    max_test_acc = 0.0  

    for epoch in tqdm(range(30), desc="Époques", leave=True):
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/30", leave=False):
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

        train_loss = running_loss / len(train_dataloader.dataset)
        test_loss, test_acc, missing = test_model(model, test_dataloader, criterion, node_features, edge_index, edge_weight, device)
        scheduler.step()

        tqdm.write(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

        # Sauvegarde du modèle si la précision est la meilleure obtenue
        if test_acc > max_test_acc:
            max_test_acc = test_acc

            if test_acc > best_test_acc:  # Vérifie si c'est le meilleur modèle global
                best_test_acc = test_acc
                # Sauvegarde des poids du modèle
                best_hparams = hparams  # Sauvegarde des hyperparamètres
        best_model_state = model.state_dict()  
        torch.save(best_model_state, f"model/model_{epoch}_{int(test_acc*100)}.pth")
    results.append((hparams, max_test_acc))
    print(">> Meilleure précision sur le test pour cette configuration :", max_test_acc)

print("\n===== Résultats des expériences =====")
for hparams, acc in results:
    print("Hyperparamètres :", hparams, "=> Précision max sur test :", acc)

# Enregistrer le meilleur modèle trouvé
if best_model_state:
    torch.save(best_model_state, "best_model.pth")
    print("\n>> Modèle avec la meilleure précision enregistré sous 'best_model.pth'")
    print("Meilleurs hyperparamètres :", best_hparams)

