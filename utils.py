import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from tqdm import tqdm
import math
from torch.optim.lr_scheduler import StepLR
import pandas as pd
import numpy as np


def build_mappings(df):
    players = pd.concat([df["j1"], df["j2"]]).unique()
    player_to_idx = {player: idx for idx, player in enumerate(players)}
    tournois = df["tournament"].unique()
    tournoi_to_idx = {tournoi: idx for idx, tournoi in enumerate(tournois)}
    return player_to_idx, tournoi_to_idx


def get_last_match_indices(df):
    last_match_idx = set()
    # Récupérer la liste de tous les joueurs
    players = pd.concat([df["j1"], df["j2"]]).unique()
    for player in players:
        # Filtrer les matchs du joueur
        df_player = df[(df["j1"] == player) | (df["j2"] == player)]
        # Ne considérer que les joueurs ayant au moins 3 matchs
        if len(df_player) < 2:
            continue
        # Sélectionner la date du dernier match
        last_date = df_player["date"].max()
        # Récupérer tous les indices correspondant à ce dernier match
        idxs = df_player[df_player["date"] == last_date].index.tolist()
        last_match_idx.update(idxs)
    return list(last_match_idx)

def split_last_match(df):
    test_indices = get_last_match_indices(df)
    train_df = df.drop(index=test_indices).reset_index(drop=True)
    test_df = df.loc[test_indices].reset_index(drop=True)
    return train_df, test_df


def extract_history_features(row, player):
    if row["j1"] == player:
        return np.array([
            row["rank1"],
            row["time"],
            row["age1"],
            row["point1"],
            row["Aces_j1"],


        ], dtype=np.float32)
    else:
        return np.array([
            row["rank2"],
            row["time"],
            row["age2"],
            row["point2"],
            row["Aces_j2"],
            
        ], dtype=np.float32)

def build_player_history(df):
    history = {}
    df_sorted = df.sort_values("date")
    for idx, row in df_sorted.iterrows():
        for player in [row["j1"], row["j2"]]:
            if player not in history:
                history[player] = []
        feat1 = extract_history_features(row, row["j1"])
        feat2 = extract_history_features(row, row["j2"])
        win1 = 1 if row["winner"] == row["j1"] else 0
        win2 = 1 if row["winner"] == row["j2"] else 0
        history[row["j1"]].append((row["date"], feat1, win1))
        history[row["j2"]].append((row["date"], feat2, win2))
    return history



def get_player_history(history, player, current_date, window_size,hist_feature_dim):
    matches = history.get(player, [])
    past_feats = [feat for (date, feat, win) in matches if date < current_date]
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
        "rank1", "rank2",
        "age1", "age2",
        "point1", "point2"
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
    reference_date = df["date"].max()
    edge_dict = {}
    for idx, row in df.iterrows():
        p1 = player_to_idx[row["j1"]]
        p2 = player_to_idx[row["j2"]]
        match_date = row["date"]
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

def build_player_graph_with_weights_recent(df, player_to_idx, lambda_=0.001, recent_factor=4.0):
    # Définir la date de référence comme la date la plus récente dans le DataFrame
    reference_date = df["date"].max()
    
    # Pré-calculer pour chaque joueur les indices de ses matchs triés par date
    # et extraire les 5 derniers matchs en excluant le dernier match (le plus récent)
    recent_matches = {}
    for player in player_to_idx.keys():
        # Récupérer les indices des matchs où le joueur a participé
        match_indices = df[((df["j1"] == player) | (df["j2"] == player))].sort_values("date").index.tolist()
        if len(match_indices) > 1:
            # Exclure le dernier match, puis prendre jusqu'aux 5 derniers
            recent_matches[player] = set(match_indices[-6:-1])
        else:
            recent_matches[player] = set()
    
    edge_dict = {}
    for idx, row in df.iterrows():
        p1 = player_to_idx[row["j1"]]
        p2 = player_to_idx[row["j2"]]
        match_date = row["date"]
        days_diff = (reference_date - match_date).days
        weight = np.exp(-lambda_ * days_diff)
        
        # Initialiser un multiplicateur (1.0 par défaut)
        multiplier = 1.0
        # Si le match fait partie des 5 derniers pour le joueur j1, appliquer le facteur
        if idx in recent_matches[row["j1"]]:
            multiplier *= recent_factor
        # Pareil pour le joueur j2
        if idx in recent_matches[row["j2"]]:
            multiplier *= recent_factor
        
        weight *= multiplier
        
        # Construire le graphe non orienté en ajoutant une arête dans chaque direction
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
        player_ranks[row["j1"]].append(row["rank1"])
        player_ranks[row["j2"]].append(row["rank2"])
    features = np.zeros((num_players, 2))
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
    rank_diff = row["rank1"] - row["rank2"]
    age_diff = row["age1"] - row["age2"]
    points_diff = row["point1"] - row["point2"]
    elo_diff = row["elo_j1"] - row["elo_j2"]
 
    return np.array([rank_diff, age_diff, points_diff,elo_diff], dtype=np.float32)

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


def get_weighted_stat_of_last_match(df, player, current_date):
    """
    Récupère la statistique d'un joueur lors de son dernier match avant 'current_date',
    puis la pondère par le rang de l'adversaire (exemple : stat * 1 / (rank_opponent + 1)).
    Retourne 0 si le joueur n'a pas de match précédent.
    """
    # Filtrer les matchs du joueur avant la date courante
    df_player = df[
        ((df["j1"] == player) | (df["j2"] == player)) &
        (df["date"] < current_date)
    ].sort_values("date", ascending=False)
    
    if len(df_player) == 0:
        # Aucun match précédent, on retourne (0.0, 0.0)
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    
    # Dernier match (le plus récent) avant 'current_date'
    last_match = df_player.iloc[0]
    
    # Déterminer si le joueur était j1 ou j2 dans ce match
    if last_match["j1"] == player:
        # Exemple de stat : (aces - double fautes) du joueur
        aces_stat = last_match.get("Aces_j1", 0) 
        dblf = last_match.get("double_faults1", 0)
        first_serv = last_match.get("_1er_Service_j1_perc", 0)
        first_serv_pt = last_match.get("Pts_au_1er_service_j1_perc", 0)
        second_serv_pt = last_match.get("Pts_au_2ème_service_j1_perc", 0)
        break_save = last_match.get("Breaks_sauvés_j1_perc", 0)
        opponent_rank = last_match.get("rank2", 9999)
    else:
        aces_stat = last_match.get("Aces_j2", 0)
        dblf = last_match.get("double_faults2", 0)
        first_serv = last_match.get("_1er_Service_j2_perc", 0)
        first_serv_pt = last_match.get("Pts_au_1er_service_j2_perc", 0)
        second_serv_pt = last_match.get("Pts_au_2ème_service_j2_perc", 0)
        break_save = last_match.get("Breaks_sauvés_j2_perc", 0)
        opponent_rank = last_match.get("rank1", 9999)
    
    # Pondération par la force de l’adversaire
    # Ici, on choisit une pondération simple : plus l’adversaire est bien classé (rank petit), plus on
    # "valorise" la stat. On peut adapter la formule à votre convenance.
    weighted_aces_stat = aces_stat * (1.0 / (opponent_rank + 1))
    weighted_dblf = dblf * (1.0 / (opponent_rank + 1))
    first_serv = first_serv * (1.0 / (opponent_rank + 1))
    first_serv_pt = first_serv_pt * (1.0 / (opponent_rank + 1))
    second_serv_pt = second_serv_pt * (1.0 / (opponent_rank + 1))
    break_save = break_save * (1.0 / (opponent_rank + 1))
    


    return weighted_aces_stat,weighted_dblf,first_serv,first_serv_pt,second_serv_pt,break_save


def get_last_performance_seq(df, player, current_date, seq_length=5):
    """
    Retourne une séquence (de longueur seq_length) des performances passées du joueur
    avant current_date. Chaque vecteur de performance contient 9 statistiques :
      - Aces, double fautes, 1er service (%), points au 1er service (%),
      - rank, age, elo, points,
      - résultat du match (1 si gagné, 0 sinon).
    Si le joueur a moins de seq_length matchs, on complète avec des zéros.
    """
    # Filtrer les matchs du joueur avant la date courante, triés par date décroissante
    df_player = df[((df["j1"] == player) | (df["j2"] == player)) & (df["date"] < current_date)].sort_values("date", ascending=False)
    
    seq = []
    for _, match in df_player.iterrows():
        if match["j1"] == player:
            won = 1 if match.get("winner", "") == player else 0
            perf = [
                match.get("Aces_j1", 0),
                match.get("double_faults1", 0),
                match.get("_1er_Service_j1_perc", 0),
                match.get("Pts_au_1er_service_j1_perc", 0),
                match.get("rank1", 0),
                match.get("age1", 0),
                match.get("elo_j1", 0),
                match.get("point1", 0),
                won
            ]
        else:
            won = 1 if match.get("winner", "") == player else 0
            perf = [
                match.get("Aces_j2", 0),
                match.get("double_faults2", 0),
                match.get("_1er_Service_j2_perc", 0),
                match.get("Pts_au_1er_service_j2_perc", 0),
                match.get("rank2", 0),
                match.get("age2", 0),
                match.get("elo_j2", 0),
                match.get("point2", 0),
                won
            ]
        seq.append(perf)
        if len(seq) >= seq_length:
            break
    # Si moins de matchs, on complète par des vecteurs de zéros
    while len(seq) < seq_length:
        seq.append([0, 0, 0, 0, 0, 0, 0, 0, 0])
    return np.array(seq, dtype=np.float32)
