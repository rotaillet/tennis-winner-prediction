from utils import (get_player_history,compute_experience,compute_static_features_max
                   ,get_days_since_last_match,compute_trend,compute_variance,
                   compute_player_form,compute_player_differences,
                   compute_weighted_player_form,get_weighted_stat_of_last_match,
                   get_last_performance_seq)
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import pandas as pd

class TennisMatchDataset2(Dataset):
    def __init__(self, df, history, player_to_idx, tournoi_to_idx, norm_params=None):
        """
        Args:
            df (pd.DataFrame): Le DataFrame contenant les matchs.
            history: Structure contenant l'historique des matchs (utilisé par get_days_since_last_match).
            player_to_idx (dict): Mapping des joueurs vers leurs indices.
            tournoi_to_idx (dict): Mapping des tournois vers leurs indices.
            norm_params (dict, optionnel): Dictionnaire contenant les moyennes et écarts-types 
                                           pour normaliser certaines features.
        """
        self.df = df.sort_values("date").reset_index(drop=True)
        self.history = history
        self.player_to_idx = player_to_idx
        self.tournoi_to_idx = tournoi_to_idx
        self.norm_params = norm_params

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        current_date = row["date"]
        
        # Calcul des différences entre joueurs (fonction définie ailleurs)
        diff = compute_player_differences(row)  # np.array, par exemple de forme (d,)

        # Calcul du head-to-head
        def compute_head_to_head(p1, p2):
            df_h2h = self.df[((self.df["j1"] == p1) & (self.df["j2"] == p2)) | 
                             ((self.df["j1"] == p2) & (self.df["j2"] == p1))]
            df_h2h = df_h2h[df_h2h["date"] < current_date]
            total = len(df_h2h)
            if total == 0:
                win_ratio = 0.5
            else:
                wins = (df_h2h["winner"] == p1).sum()
                win_ratio = wins / total
            return np.array([total, win_ratio], dtype=np.float32)
        
        head2head_features = compute_head_to_head(row["j1"], row["j2"])
        
        # Normalisation du head-to-head (seulement la première valeur, car le win_ratio est entre 0 et 1)


        combined_static = np.concatenate([head2head_features, diff])
        
        player1_idx = self.player_to_idx[row["j1"]]
        player2_idx = self.player_to_idx[row["j2"]]
        tournoi_idx = self.tournoi_to_idx[row["tournament"]]
        
        # Calcul des statistiques de performance sur une surface donnée
        def get_player_surface_performance(player, current_date, surface):
            """
            Calcule pour un joueur donné et pour une surface donnée (jusqu'à current_date) :
            - nb_matchs, nb_victoires et win_ratio.
            """
            df_player = self.df[
                (((self.df["j1"] == player) | (self.df["j2"] == player)) & 
                 (self.df["date"] < current_date) & 
                 (self.df["surface"] == surface))
            ]
            nb_matchs = len(df_player)
            if nb_matchs == 0:
                return np.array([0, 0, 0.5], dtype=np.float32)
            nb_victoires = df_player.apply(
                lambda row: 1 if row["winner"] == player else 0, axis=1
            ).sum()
            win_ratio = nb_victoires / nb_matchs
            return np.array([nb_matchs, nb_victoires, win_ratio], dtype=np.float32)
        
        p1_surface_stats = get_player_surface_performance(row["j1"], current_date, row["surface"])
        p2_surface_stats = get_player_surface_performance(row["j2"], current_date, row["surface"])
        
        def get_player_tour_performance(player, current_date, tour):
            """
            Calcule pour un joueur donné et pour une surface donnée, jusqu'à current_date :
            - nb_matchs : nombre de matchs joués,
            - nb_victoires : nombre de matchs gagnés,
            - win_ratio : taux de victoire (avec une valeur par défaut de 0.5 si aucun match).
            
            Args:
                df (pd.DataFrame): Le DataFrame complet contenant tous les matchs.
                player (str): Nom du joueur.
                current_date (pd.Timestamp): Date du match courant.
                surface (str): La surface du match (ex : "Clay", "Hard", etc.).
                
            Returns:
                np.array: Un vecteur de forme (3,) de type float32 contenant [nb_matchs, nb_victoires, win_ratio].
            """
            # Sélectionner les matchs où le joueur a participé et qui se sont déroulés avant current_date
            df_player = self.df[
                (((self.df["j1"] == player) | (self.df["j2"] == player)) & 
                (self.df["date"] < current_date) & 
                (self.df["tour"] == tour))
            ]
            nb_matchs = len(df_player)
            if nb_matchs == 0:
                # Si aucune donnée n'est disponible, on renvoie par défaut 0 match, 0 victoire, win_ratio=0.5 (indécision)
                return np.array([0, 0, 0.5], dtype=np.float32)
            # Calculer le nombre de victoires
            nb_victoires = df_player.apply(
                lambda row: 1 if row["winner"] == player else 0, axis=1
            ).sum()
            win_ratio = nb_victoires / nb_matchs
            return np.array([nb_matchs, nb_victoires, win_ratio], dtype=np.float32)

        p1_tour_stats = get_player_tour_performance(row["j1"], current_date, row["tour"])
        p2_tour_stats = get_player_tour_performance(row["j2"], current_date, row["tour"])
        # Normalisation des compteurs des stats de surface (seulement les deux premières valeurs)

        combined_static = np.concatenate([combined_static, p1_surface_stats])
        combined_static = np.concatenate([combined_static, p2_surface_stats])
        combined_static = np.concatenate([combined_static, p1_tour_stats])
        combined_static = np.concatenate([combined_static, p2_tour_stats])
        
        # Ajout de la surface encodée (supposée déjà sur une échelle adaptée ou à normaliser si nécessaire)
        combined_static = np.concatenate([combined_static, np.array([row['surface_encoded']])])
        combined_static = np.concatenate([combined_static, np.array([row['tour_encoded']])])
        # Calcul du temps depuis le dernier match pour chaque joueur
        last_match_p1 = get_days_since_last_match(self.history, row["j1"], current_date)
        last_match_p2 = get_days_since_last_match(self.history, row["j2"], current_date)
        
        # Normalisation des mesures temporelles
        if self.norm_params is not None:
            last_match_p1 = (last_match_p1 - self.norm_params["last_match_mean"]) / self.norm_params["last_match_std"]
            last_match_p2 = (last_match_p2 - self.norm_params["last_match_mean"]) / self.norm_params["last_match_std"]
        
        combined_static = np.concatenate([combined_static, np.array([last_match_p1])])
        combined_static = np.concatenate([combined_static, np.array([last_match_p2])])
        
        p1_weighted_aces,p1_weighted_dblf,p1_first_serv,p1_first_servpt,p1_scnd_serv,p1_break_save = get_weighted_stat_of_last_match(self.df, row["j1"], current_date)
        p2_weighted_aces,p2_weighted_dblf,p2_first_serv,p2_first_servpt,p2_scnd_serv,p2_break_save = get_weighted_stat_of_last_match(self.df, row["j2"], current_date)
        
        diff_weighted_stat_aces = p1_weighted_aces - p2_weighted_aces
        diff_weighted_stat_dblf = p1_weighted_dblf - p2_weighted_dblf
        combined_static = np.concatenate([combined_static, np.array([diff_weighted_stat_aces])])
        combined_static = np.concatenate([combined_static, np.array([diff_weighted_stat_dblf])])
        combined_static = np.concatenate([combined_static, np.array([p1_first_serv-p2_first_serv])])
        combined_static = np.concatenate([combined_static, np.array([p1_first_servpt-p2_first_servpt])])
        combined_static = np.concatenate([combined_static, np.array([p1_scnd_serv-p2_scnd_serv])])
        combined_static = np.concatenate([combined_static, np.array([p1_break_save-p2_break_save])])
        p1_form_10 = compute_weighted_player_form(self.history, row["j1"], current_date, 10)
        p1_form_3  = compute_weighted_player_form(self.history, row["j1"], current_date, 3)
        p1_form_last = compute_player_form(self.history, row["j1"], current_date, 1)
        p2_form_10 = compute_weighted_player_form(self.history, row["j2"], current_date, 10)
        p2_form_3  = compute_weighted_player_form(self.history, row["j2"], current_date, 3)
        p2_form_last = compute_player_form(self.history, row["j2"], current_date, 1)
        form_features = np.array([p1_form_10, p1_form_3, p1_form_last,
                                p2_form_10, p2_form_3, p2_form_last], dtype=np.float32)
        
        combined_static = np.concatenate([combined_static,form_features])

        player1_seq = get_last_performance_seq(self.df, row["j1"], current_date, seq_length=5)
        player2_seq = get_last_performance_seq(self.df, row["j2"], current_date, seq_length=5)

        # Détermination de la cible : 0 si j1 gagne, 1 si j2 gagne
        if row["winner"] == row["j1"]:
            target = 0
        elif row["winner"] == row["j2"]:
            target = 1
        else:
            raise ValueError(f"Nom de gagnant inattendu : {row['winner']}")
        
        return {
            "static_feat": torch.tensor(combined_static, dtype=torch.float),
            "player1_seq": torch.tensor(player1_seq, dtype=torch.float),  # shape: (5, 4)
            "player2_seq": torch.tensor(player2_seq, dtype=torch.float),  # shape: (5, 4)
            "target": torch.tensor(target, dtype=torch.long),
            "player1_idx": torch.tensor(player1_idx, dtype=torch.long),
            "player2_idx": torch.tensor(player2_idx, dtype=torch.long),
            "tournoi_idx": torch.tensor(tournoi_idx, dtype=torch.long),
        }

