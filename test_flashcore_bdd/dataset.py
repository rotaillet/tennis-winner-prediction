from utils import get_player_history,compute_experience,compute_static_features_max,get_days_since_last_match,compute_trend,compute_variance,compute_player_form,compute_player_differences,compute_weighted_player_form
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import pandas as pd

WINDOW_SIZE = 30            # Taille de la fentre glissante pour l'historique
HIST_FEATURE_DIM = 5       # Nombre de features historiques utilises
# 46 features statiques de base, 6 pour la forme, 4 pour le timing, 5 pour la surface, 2 pour head-to-head
STATIC_FEATURE_DIM = 6    

class TennisMatchDataset(Dataset):
    def __init__(self, df, history, player_to_idx, tournoi_to_idx):
        self.df = df.sort_values("date").reset_index(drop=True)
        self.history = history
        self.player_to_idx = player_to_idx
        self.tournoi_to_idx = tournoi_to_idx
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        current_date = row["date"]

        # Récupérer les historiques (déjà présents)
        p1_history = get_player_history(self.history, row["j1"], current_date,WINDOW_SIZE,HIST_FEATURE_DIM)
        p2_history = get_player_history(self.history, row["j2"], current_date,WINDOW_SIZE,HIST_FEATURE_DIM)
        static_feat = compute_static_features_max(row)
        
             
        # Fusionner les features statiques
        combined_static = np.concatenate([
            static_feat,
        ])
        
        # Cible
        if row["winner"] == row["j1"]:
            target = 0
        elif row["winner"] == row["j2"]:
            target = 1
        else:
            raise ValueError(f"Nom de gagnant inattendu : {row['winner']}")
        
        player1_idx = self.player_to_idx[row["j1"]]
        player2_idx = self.player_to_idx[row["j2"]]
        tournoi_idx = self.tournoi_to_idx[row["tournament"]]
                # Récupérer la performance historique sur la surface du match pour chaque joueur


        
        return {
            "p1_history": torch.tensor(p1_history, dtype=torch.float),  # (WINDOW_SIZE, HIST_FEATURE_DIM)
            "p2_history": torch.tensor(p2_history, dtype=torch.float),
            "static_feat": torch.tensor(combined_static, dtype=torch.float),
            "target": torch.tensor(target, dtype=torch.long),
            "player1_idx": torch.tensor(player1_idx, dtype=torch.long),
            "player2_idx": torch.tensor(player2_idx, dtype=torch.long),
            "tournoi_idx": torch.tensor(tournoi_idx, dtype=torch.long),
            "player1_name": row["Joueur 1"],
            "player2_name": row["Joueur 2"],
            "tournoi_name": row["Tournoi_propre"]
        }