from utils import get_player_history,compute_experience,compute_static_features_max,get_days_since_last_match,compute_trend,compute_variance,compute_player_form,compute_player_differences,compute_weighted_player_form
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import pandas as pd

WINDOW_SIZE = 20            # Taille de la fentre glissante pour l'historique
HIST_FEATURE_DIM = 19       # Nombre de features historiques utilises
# 46 features statiques de base, 6 pour la forme, 4 pour le timing, 5 pour la surface, 2 pour head-to-head
STATIC_FEATURE_DIM = 75    
D_MODEL = 256               # Dimension pour la fusion et le Transformer temporel
GAT_HIDDEN_DIM = 128         # Dimension cache pour le GAT
GAT_OUTPUT_DIM = 128        # Dimension de sortie du GAT
NUM_HEADS_DIM = 2
DROPOUT = 0.3

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

        # Récupérer les historiques (déjà présents)
        p1_history = get_player_history(self.history, row["Joueur 1"], current_date,WINDOW_SIZE,HIST_FEATURE_DIM)
        p2_history = get_player_history(self.history, row["Joueur 2"], current_date,WINDOW_SIZE,HIST_FEATURE_DIM)
        static_feat = compute_static_features_max(row)
        
        # --- Nouveaux calculs de features complémentaires ---
        # 1. Expérience : nombre total de matchs joués
        p1_experience = compute_experience(self.history, row["Joueur 1"], current_date)
        p2_experience = compute_experience(self.history, row["Joueur 2"], current_date)
        
        # 2. Tendance de performance sur la fenêtre (par exemple, pour le % de premiers services, aces et double fautes)
        # Utilisons les indices : 11 pour le % de premiers services, 3 pour les aces et 4 pour les double fautes.
        p1_trend_first_serve = compute_trend(self.history, row["Joueur 1"], current_date, WINDOW_SIZE, feature_index=11,hist_feature_dim=HIST_FEATURE_DIM)
        p2_trend_first_serve = compute_trend(self.history, row["Joueur 2"], current_date, WINDOW_SIZE, feature_index=11,hist_feature_dim=HIST_FEATURE_DIM)
        p1_trend_aces = compute_trend(self.history, row["Joueur 1"], current_date, WINDOW_SIZE, feature_index=3,hist_feature_dim=HIST_FEATURE_DIM)
        p2_trend_aces = compute_trend(self.history, row["Joueur 2"], current_date, WINDOW_SIZE, feature_index=3,hist_feature_dim=HIST_FEATURE_DIM)
        p1_trend_double_faults = compute_trend(self.history, row["Joueur 1"], current_date, WINDOW_SIZE, feature_index=4,hist_feature_dim=HIST_FEATURE_DIM)
        p2_trend_double_faults = compute_trend(self.history, row["Joueur 2"], current_date, WINDOW_SIZE, feature_index=4,hist_feature_dim=HIST_FEATURE_DIM)
        
        # 3. Variance (consistance) sur, par exemple, le % de premiers services
        p1_variance_first_serve = compute_variance(self.history, row["Joueur 1"], current_date, WINDOW_SIZE, feature_index=11,hist_feature_dim=HIST_FEATURE_DIM)
        p2_variance_first_serve = compute_variance(self.history, row["Joueur 2"], current_date, WINDOW_SIZE, feature_index=11,hist_feature_dim=HIST_FEATURE_DIM)
        
        # Vous pouvez regrouper ces features dans des vecteurs si nécessaire
        p1_trend_features = np.array([p1_trend_first_serve, p1_trend_aces, p1_trend_double_faults], dtype=np.float32)
        p2_trend_features = np.array([p2_trend_first_serve, p2_trend_aces, p2_trend_double_faults], dtype=np.float32)
        p1_variance_features = np.array([p1_variance_first_serve], dtype=np.float32)
        p2_variance_features = np.array([p2_variance_first_serve], dtype=np.float32)
        
        # Les autres features déjà existantes (forme, timing, etc.)
        p1_form_10 = compute_weighted_player_form(self.history, row["Joueur 1"], current_date, 10)
        p1_form_3  = compute_weighted_player_form(self.history, row["Joueur 1"], current_date, 3)
        p1_form_last = compute_player_form(self.history, row["Joueur 1"], current_date, 1)
        p2_form_10 = compute_weighted_player_form(self.history, row["Joueur 2"], current_date, 10)
        p2_form_3  = compute_weighted_player_form(self.history, row["Joueur 2"], current_date, 3)
        p2_form_last = compute_player_form(self.history, row["Joueur 2"], current_date, 1)
        form_features = np.array([p1_form_10, p1_form_3, p1_form_last,
                                p2_form_10, p2_form_3, p2_form_last], dtype=np.float32)
        
            # Calcul du nombre de jours écoulés depuis le dernier match pour chaque joueur
        last_match_p1 = get_days_since_last_match(self.history, row["Joueur 1"], current_date)
        last_match_p2 = get_days_since_last_match(self.history, row["Joueur 2"], current_date)
        # Créez un vecteur pour ces nouvelles features (2 dimensions)
        last_match_features = np.array([last_match_p1, last_match_p2], dtype=np.float32)


        
        # Timing et head-to-head restent inchangés
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
        diff_features = compute_player_differences(row)
        
        # Fusionner les features statiques
        combined_static = np.concatenate([
            static_feat,
            form_features,
            timing_features,
            surface_vec,
            head2head_features,
            # Ajout des nouvelles features : expérience, tendance et variance pour chaque joueur
            np.array([p1_experience, p2_experience], dtype=np.float32),
            p1_trend_features,
            p2_trend_features,
            p1_variance_features,
            p2_variance_features,
            last_match_features,
            diff_features    

        ])
        
        # Cible
        if row["winner"] == row["Joueur 1"]:
            target = 0
        elif row["winner"] == row["Joueur 2"]:
            target = 1
        else:
            raise ValueError(f"Nom de gagnant inattendu : {row['winner']}")
        
        player1_idx = self.player_to_idx[row["Joueur 1"]]
        player2_idx = self.player_to_idx[row["Joueur 2"]]
        tournoi_idx = self.tournoi_to_idx[row["Tournoi_propre"]]
        
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