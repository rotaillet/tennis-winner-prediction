import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from rapidfuzz import fuzz


def build_mappings(df):
    players = pd.concat([df["Joueur 1"], df["Joueur 2"]]).unique()
    player_to_idx = {player: idx for idx, player in enumerate(players)}
    tournois = df["Tournoi_propre"].unique()
    tournoi_to_idx = {tournoi: idx for idx, tournoi in enumerate(tournois)}
    return player_to_idx, tournoi_to_idx


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


def extract_history_features(row, player):
    if row["Joueur 1"] == player:
        return np.array([
            row["Rank_Joueur_1"],
            row["Age_Joueur_1"],
            row["Points_Joueur_1"],
            row["prev_ACES_p1"],
            row["prev_DOUBLE_FAULTS_p1"],
            row["prev_1st_SERVE_%_p1_num"],
            row["prev_1st_SERVE_POINTS_WON_p1_num"],
            row["prev_2nd_SERVE_POINTS_WON_p1_num"],
            row["prev_BREAK_POINTS_WON_p1_num"],
            row["prev_TOTAL_RETURN_POINTS_WON_p1_num"],
            row["prev_TOTAL_POINTS_WON_p1_num"],
            row["prev_1st_SERVE_%_p1_pct"],
            row["prev_1st_SERVE_POINTS_WON_p1_pct"],
            row["prev_2nd_SERVE_POINTS_WON_p1_pct"],
            row["prev_BREAK_POINTS_WON_p1_pct"],
            row["prev_TOTAL_RETURN_POINTS_WON_p1_pct"],
            row["prev_TOTAL_POINTS_WON_p1_pct"],
            row["prev_total_games_p1"],
            row["prev_set_win_p1"],
            row["prev_set_win_p1"]-row["prev_set_win_p2"]
        ], dtype=np.float32)
    else:
        return np.array([
            row["Rank_Joueur_2"],
            row["Age_Joueur_2"],
            row["Points_Joueur_2"],
            row["prev_ACES_p2"],
            row["prev_DOUBLE_FAULTS_p2"],
            row["prev_1st_SERVE_%_p2_num"],
            row["prev_1st_SERVE_POINTS_WON_p2_num"],
            row["prev_2nd_SERVE_POINTS_WON_p2_num"],
            row["prev_BREAK_POINTS_WON_p2_num"],
            row["prev_TOTAL_RETURN_POINTS_WON_p2_num"],
            row["prev_TOTAL_POINTS_WON_p2_num"],
            row["prev_1st_SERVE_%_p2_pct"],
            row["prev_1st_SERVE_POINTS_WON_p2_pct"],
            row["prev_2nd_SERVE_POINTS_WON_p2_pct"],
            row["prev_BREAK_POINTS_WON_p2_pct"],
            row["prev_TOTAL_RETURN_POINTS_WON_p2_pct"],
            row["prev_TOTAL_POINTS_WON_p2_pct"],
            row["prev_total_games_p2"],
            row["prev_set_win_p2"],
            row["prev_set_win_p2"]-row["prev_set_win_p1"]
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



def get_player_history(history, player, current_date, window_size,hist_feature_dim):
    matches = history.get(player, [])
    past_feats = [feat for (date, feat, win) in matches if date <= current_date]
    past_feats = past_feats[-window_size:]
    if len(past_feats) < window_size:
        pad = [np.zeros(hist_feature_dim, dtype=np.float32) for _ in range(window_size - len(past_feats))]
        past_feats = pad + past_feats
    return np.stack(past_feats)  # (window_size, HIST_FEATURE_DIM)

def compute_experience(history, player, current_date):
    """Nombre total de matchs joués par le joueur avant current_date."""
    matches = history.get(player, [])
    return sum(1 for (date, feat, win) in matches if date < current_date)

def compute_trend(history, player, current_date, window_size, feature_index,hist_feature_dim):
    """
    Calcule la pente (trend) de la série d'une statistique (feature_index)
    sur la fenêtre temporelle.
    """
    feats = get_player_history(history, player, current_date, window_size,hist_feature_dim)  # shape: (window_size, HIST_FEATURE_DIM)
    feature_series = feats[:, feature_index]
    x = np.arange(len(feature_series))
    # Si la variance est très faible, la pente est nulle.
    if np.std(feature_series) < 1e-6:
        return 0.0
    slope, _ = np.polyfit(x, feature_series, 1)
    return slope

def compute_variance(history, player, current_date, window_size, feature_index,hist_feature_dim):
    """
    Calcule la variance de la série d'une statistique (feature_index)
    sur la fenêtre temporelle.
    """
    feats = get_player_history(history, player, current_date, window_size,hist_feature_dim)
    feature_series = feats[:, feature_index]
    return float(np.var(feature_series))

def compute_static_features_max(row):
    cols = [
        "Rank_Joueur_1", "Rank_Joueur_2",
        "Age_Joueur_1", "Age_Joueur_2",
        "Points_Joueur_1", "Points_Joueur_2",
        "prev_DOUBLE_FAULTS_p1", "prev_DOUBLE_FAULTS_p2",
        "prev_ACES_p1", "prev_ACES_p2",
        "prev_1st_SERVE_%_p1_num",  "prev_1st_SERVE_%_p1_pct",
        "prev_1st_SERVE_%_p2_num",  "prev_1st_SERVE_%_p2_pct",
        "prev_1st_SERVE_POINTS_WON_p1_num",  "prev_1st_SERVE_POINTS_WON_p1_pct",
        "prev_1st_SERVE_POINTS_WON_p2_num",  "prev_1st_SERVE_POINTS_WON_p2_pct",
        "prev_2nd_SERVE_POINTS_WON_p1_num", "prev_2nd_SERVE_POINTS_WON_p1_pct",
        "prev_2nd_SERVE_POINTS_WON_p2_num",  "prev_2nd_SERVE_POINTS_WON_p2_pct",
        "prev_BREAK_POINTS_WON_p1_num",  "prev_BREAK_POINTS_WON_p1_pct",
        "prev_BREAK_POINTS_WON_p2_num",  "prev_BREAK_POINTS_WON_p2_pct",
        "prev_TOTAL_RETURN_POINTS_WON_p1_num",  "prev_TOTAL_RETURN_POINTS_WON_p1_pct",
        "prev_TOTAL_RETURN_POINTS_WON_p2_num",  "prev_TOTAL_RETURN_POINTS_WON_p2_pct",
        "prev_TOTAL_POINTS_WON_p1_num",  "prev_TOTAL_POINTS_WON_p1_pct",
        "prev_TOTAL_POINTS_WON_p2_num",  "prev_TOTAL_POINTS_WON_p2_pct",
        "prev_total_games_p1", "prev_total_games_p2", "prev_set_win_p1", "prev_set_win_p2","Prize_money"
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

def get_days_since_last_match(history, player, current_date):
    """
    Calcule le nombre de jours depuis le dernier match joué par 'player' avant 'current_date'.
    Si aucun match précédent n'existe, renvoie 0.
    """
    matches = history.get(player, [])
    past_dates = [m[0] for m in matches if m[0] < current_date]
    if past_dates:
        last_date = max(past_dates)
        return (current_date - last_date).days
    else:
        return 5000.0
    
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
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def normalize_columns(df, columns):
    """
    Normalise les colonnes spécifiées d'un DataFrame selon la formule:
        (x - mean) / std
    
    Args:
        df (pd.DataFrame): Le DataFrame à normaliser.
        columns (list): Liste des noms de colonnes à normaliser.
    
    Returns:
        pd.DataFrame: Un nouveau DataFrame avec les colonnes normalisées.
    """
    df_norm = df.copy()
    for col in columns:
        if pd.api.types.is_numeric_dtype(df_norm[col]):
            mean = df_norm[col].mean()
            std = df_norm[col].std()
            df_norm[col] = (df_norm[col] - mean) / std
        else:
            print(f"La colonne '{col}' n'est pas numérique et ne sera pas normalisée.")
    return df_norm

def nettoyer_tournoi(nom):
    # Uniformisation des majuscules/minuscules
    nom = nom.strip().lower()

    if "davis cup, w" in nom:
        return "Davis Cup World"
    if "davis cup, g" in nom:
        return "Davis Cup Group"
    if "davis cup, f" in nom:
        return "Davis Cup Finals"
    if "davis cup, q" in nom:
        return "Davis Cup Qualifiers"

    else:
        return nom


    
import re



import re
import pandas as pd

def extraire_prize_money(nom):
    # Supprimer les caractères non imprimables ou erronés (comme '�')
    nom = re.sub(r'[^\x00-\x7F]+', '', nom)
    
    # Expression régulière pour détecter "$15K", "$1.5M", "$15k", "$1.5m", "36K" (sans le $)
    match = re.search(r"(\$?)(\d+(\.\d+)?)([kmKM])", nom)  
    
    if match:
        montant = float(match.group(2))  # Extrait le nombre (ex: "1.5" ou "15")
        unite = match.group(4).upper()  # "K" ou "M", mis en majuscule pour uniformité

        if unite == "K":
            prize_money = int(montant * 1000)  # Convertit en milliers
        elif unite == "M":
            prize_money = int(montant * 1_000_000)  # Convertit en millions
    else:
        prize_money = 0  # Si pas de prize money détecté
    
    # Nettoyer le nom du tournoi en supprimant le prize money
    nom_propre = re.sub(r"/\s*\$?\d+(\.\d+)?[kmKM]", "", nom).strip()
    
    return nom_propre, prize_money


def compute_player_differences(row):
    """
    Calcule des features basées sur les différences et ratios entre Joueur 1 et Joueur 2.
    
    Retourne un vecteur contenant :
      - Différence de ranking (Rank_Joueur_1 - Rank_Joueur_2)
      - Différence d'âge (Age_Joueur_1 - Age_Joueur_2)
      - Différence de points (Points_Joueur_1 - Points_Joueur_2)
      - Ratio de ranking (Rank_Joueur_1 / Rank_Joueur_2)
      - Ratio d'âge (Age_Joueur_1 / Age_Joueur_2)
      - Ratio de points (Points_Joueur_1 / Points_Joueur_2)
    """
    rank_diff = row["Rank_Joueur_1"] - row["Rank_Joueur_2"]
    age_diff = row["Age_Joueur_1"] - row["Age_Joueur_2"]
    points_diff = row["Points_Joueur_1"] - row["Points_Joueur_2"]
    rank_ratio = row["Rank_Joueur_1"] / row["Rank_Joueur_2"] if row["Rank_Joueur_2"] != 0 else 0.0
    age_ratio = row["Age_Joueur_1"] / row["Age_Joueur_2"] if row["Age_Joueur_2"] != 0 else 0.0
    points_ratio = row["Points_Joueur_1"] / row["Points_Joueur_2"] if row["Points_Joueur_2"] != 0 else 0.0
    set_win_diff = row["prev_set_win_p2"]-row["prev_set_win_p1"]

    return np.array([rank_diff, age_diff, points_diff, rank_ratio, age_ratio, points_ratio,set_win_diff], dtype=np.float32)

import math
import numpy as np

def compute_weighted_player_form(history, player, current_date, window_size, decay_lambda=0.001):
    """
    Calcule la forme (win rate) d'un joueur sur les derniers matchs en appliquant
    une pondération exponentielle qui favorise les matchs récents.
    
    Args:
        history (dict): Historique du joueur (liste de tuples (date, features, win)).
        player (str): Nom du joueur.
        current_date (pd.Timestamp): Date du match actuel.
        window_size (int): Nombre de matchs récents à considérer.
        decay_lambda (float): Taux de décroissance pour le poids temporel.
        
    Returns:
        float: Forme pondérée, entre 0 et 1.
    """
    matches = history.get(player, [])
    # Sélectionner uniquement les matchs avant la date actuelle
    past_matches = [m for m in matches if m[0] < current_date]
    # Garder uniquement les derniers window_size matchs
    past_matches = past_matches[-window_size:]
    if not past_matches:
        return 0.5  # Valeur par défaut si aucune donnée n'est disponible
    weights = []
    outcomes = []
    for date, feat, win in past_matches:
        delta_days = (current_date - date).days
        weight = math.exp(-decay_lambda * delta_days)
        weights.append(weight)
        outcomes.append(win)
    weights = np.array(weights, dtype=np.float32)
    outcomes = np.array(outcomes, dtype=np.float32)
    weighted_average = np.sum(weights * outcomes) / np.sum(weights)
    return weighted_average
