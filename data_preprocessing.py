# data_preprocessing.py
import pandas as pd
import numpy as np
import torch
from collections import defaultdict
from sklearn.preprocessing import StandardScaler

# Fonctions d'encodage et mapping
def encode_surface(surface_str):
    mapping = {
        "Clay":    [1, 0, 0, 0, 0],
        "Hard":    [0, 1, 0, 0, 0],
        "I. hard": [0, 0, 1, 0, 0],
        "Grass":   [0, 0, 0, 1, 0],
        "Carpet":  [0, 0, 0, 0, 1]
    }
    return mapping.get(surface_str, [0, 0, 0, 0, 0])

def get_round_value(tour_str):
    round_mapping = {
        "1stround": 1, "2ndround": 2, "3rdround": 3, "4thround": 4,
        "1/4": 5, "1/2": 6, "fin": 7, "qual.": 0, "q 1": 0,
        "q 2": 0, "Amical": 0, "Rubber 1": 0, "bronze": 0
    }
    return round_mapping.get(tour_str, 0)

# Paramètres pour la forme
WINDOW_IMMEDIATE = 3
WINDOW_LONG = 10

def compute_form(player, window, player_recent_results):
    if player_recent_results[player]:
        return np.mean(player_recent_results[player][-window:])
    else:
        return 0.5

def compute_weighted_form(player, window, player_recent_results, alpha=0.7):
    if player_recent_results[player]:
        weighted_sum = 0.0
        weight_total = 0.0
        for i, result in enumerate(player_recent_results[player][-window:]):
            weight = alpha ** (window - i - 1)
            weighted_sum += weight * result
            weight_total += weight
        return weighted_sum / weight_total if weight_total > 0 else 0.5
    else:
        return 0.5

def compute_decayed_win_rate(player, surface, current_date, surface_stats, decay_lambda=0.01):
    events = surface_stats.get((player, surface), [])
    weighted_sum = 0.0
    weight_total = 0.0
    for match_date, result in events:
        days_diff = (current_date - match_date).days
        weight = np.exp(-decay_lambda * days_diff)
        weighted_sum += weight * result
        weight_total += weight
    return weighted_sum / weight_total if weight_total > 0 else 0.5

def load_data(file_path="test.csv"):
    df = pd.read_csv(file_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df

def build_data(df):
    # Encodage des joueurs
    players = list(set(df["Joueur 1"].unique()) | set(df["Joueur 2"].unique()))
    player_encoder = {name: i for i, name in enumerate(players)}
    df["Joueur1_ID"] = df["Joueur 1"].map(player_encoder)
    df["Joueur2_ID"] = df["Joueur 2"].map(player_encoder)
    df["Winner_ID"]  = df["winner"].map(player_encoder)

    # Construction du graphe
    edges = []
    for _, row in df.iterrows():
        edges.append([row["Joueur1_ID"], row["Joueur2_ID"]])
        edges.append([row["Joueur2_ID"], row["Joueur1_ID"]])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # Features statiques pour chaque joueur (moyenne des classements)
    player_rank = defaultdict(list)
    for _, row in df.iterrows():
        player_rank[row["Joueur1_ID"]].append(row["Rank_Joueur_1"])
        player_rank[row["Joueur2_ID"]].append(row["Rank_Joueur_2"])
    num_players = len(players)
    player_features = np.zeros((num_players, 1))
    for i in range(num_players):
        player_features[i] = np.mean(player_rank[i]) if player_rank[i] else 0.0
    scaler = StandardScaler()
    player_features = scaler.fit_transform(player_features)
    player_features = torch.tensor(player_features, dtype=torch.float)

    # Initialisations pour les features dynamiques et additionnelles
    player_recent_results = defaultdict(list)
    head_to_head_record = {}
    dynamic_player_history = defaultdict(list)
    surface_stats = defaultdict(list)
    last_match_date = {}
    head_to_head_last_date = {}

    match_feature_list = []
    match_list = []
    labels_list = []
    match_dynamic_histories = []

    for idx, row in df.iterrows():
        p1 = row["Joueur1_ID"]
        p2 = row["Joueur2_ID"]
        label = 1 if row["Winner_ID"] == p1 else 0

        # Base features (exemple – adaptez selon vos données)
        rank_diff = row["Rank_Joueur_1"] - row["Rank_Joueur_2"]
        age_diff  = row["Age_Joueur_1"] - row["Age_Joueur_2"]
        pts_diff  = row["Points_Joueur_1"] - row["Points_Joueur_2"]
        surface_vec = encode_surface(row["Surface"])
        round_val = get_round_value(row["Tour"])
        base_features = [rank_diff, age_diff, pts_diff] + surface_vec + [
            round_val,
            row["prev_DOUBLE_FAULTS_p1"] - row["prev_DOUBLE_FAULTS_p2"],
            row["prev_ACES_p1"] - row["prev_ACES_p2"],
            row["prev_1st_SERVE_%_p1_num"] - row["prev_1st_SERVE_%_p2_num"],
            row["prev_1st_SERVE_%_p1_den"] - row["prev_1st_SERVE_%_p2_den"],
            row["prev_1st_SERVE_%_p1_pct"] - row["prev_1st_SERVE_%_p2_pct"],
            row["prev_1st_SERVE_POINTS_WON_p1_num"] - row["prev_1st_SERVE_POINTS_WON_p2_num"],
            row["prev_1st_SERVE_POINTS_WON_p1_den"] - row["prev_1st_SERVE_POINTS_WON_p2_den"],
            row["prev_1st_SERVE_POINTS_WON_p1_pct"] - row["prev_1st_SERVE_POINTS_WON_p2_pct"],
            row["prev_2nd_SERVE_POINTS_WON_p1_num"] - row["prev_2nd_SERVE_POINTS_WON_p2_num"],
            row["prev_2nd_SERVE_POINTS_WON_p1_den"] - row["prev_2nd_SERVE_POINTS_WON_p2_den"],
            row["prev_2nd_SERVE_POINTS_WON_p1_pct"] - row["prev_2nd_SERVE_POINTS_WON_p2_pct"],
            row["prev_BREAK_POINTS_WON_p1_num"] - row["prev_BREAK_POINTS_WON_p2_num"],
            row["prev_BREAK_POINTS_WON_p1_den"] - row["prev_BREAK_POINTS_WON_p2_den"],
            row["prev_BREAK_POINTS_WON_p1_pct"] - row["prev_BREAK_POINTS_WON_p2_pct"],
            row["prev_TOTAL_RETURN_POINTS_WON_p1_num"] - row["prev_TOTAL_RETURN_POINTS_WON_p2_num"],
            row["prev_TOTAL_RETURN_POINTS_WON_p1_den"] - row["prev_TOTAL_RETURN_POINTS_WON_p2_den"],
            row["prev_TOTAL_RETURN_POINTS_WON_p1_pct"] - row["prev_TOTAL_RETURN_POINTS_WON_p2_pct"],
            row["prev_TOTAL_POINTS_WON_p1_num"] - row["prev_TOTAL_POINTS_WON_p2_num"],
            row["prev_TOTAL_POINTS_WON_p1_den"] - row["prev_TOTAL_POINTS_WON_p2_den"],
            row["prev_TOTAL_POINTS_WON_p1_pct"] - row["prev_TOTAL_POINTS_WON_p2_pct"]
        ]

        # Additional features
        current_date = row["Date"]
        form3_p1  = compute_form(p1, WINDOW_IMMEDIATE, player_recent_results)
        form10_p1 = compute_form(p1, WINDOW_LONG, player_recent_results)
        form3_p2  = compute_form(p2, WINDOW_IMMEDIATE, player_recent_results)
        form10_p2 = compute_form(p2, WINDOW_LONG, player_recent_results)
        weighted_form_immediate_p1 = compute_weighted_form(p1, WINDOW_IMMEDIATE, player_recent_results)
        weighted_form_long_p1      = compute_weighted_form(p1, WINDOW_LONG, player_recent_results)
        weighted_form_immediate_p2 = compute_weighted_form(p2, WINDOW_IMMEDIATE, player_recent_results)
        weighted_form_long_p2      = compute_weighted_form(p2, WINDOW_LONG, player_recent_results)
        
        delay_p1 = (current_date - last_match_date[p1]).days if p1 in last_match_date else 0
        delay_p2 = (current_date - last_match_date[p2]).days if p2 in last_match_date else 0
        last_match_date[p1] = current_date
        last_match_date[p2] = current_date

        key = tuple(sorted((p1, p2)))
        if key in head_to_head_record:
            record = head_to_head_record[key]
            total_confrontations = record.get(key[0], 0) + record.get(key[1], 0)
            wins_p1 = record.get(p1, 0)
            win_ratio_p1 = wins_p1 / total_confrontations if total_confrontations > 0 else 0.5
        else:
            total_confrontations = 0
            win_ratio_p1 = 0.5
        head_to_head_delay = (current_date - head_to_head_last_date[key]).days if key in head_to_head_last_date else 0
        head_to_head_last_date[key] = current_date

        decay_lambda = 0.01
        win_rate_surface_p1 = compute_decayed_win_rate(p1, row["Surface"], current_date, surface_stats, decay_lambda)
        win_rate_surface_p2 = compute_decayed_win_rate(p2, row["Surface"], current_date, surface_stats, decay_lambda)
        surface_stats[(p1, row["Surface"])].append((current_date, 1 if label == 1 else 0))
        surface_stats[(p2, row["Surface"])].append((current_date, 1 if label == 0 else 0))

        if "Court" in df.columns:
            court_feature = 1 if row["Court"].lower().startswith("out") else 0
        else:
            court_feature = 0.5

        additional_features = [
             form3_p1, form10_p1, weighted_form_immediate_p1, weighted_form_long_p1,
             form3_p2, form10_p2, weighted_form_immediate_p2, weighted_form_long_p2,
             total_confrontations, win_ratio_p1,
             delay_p1, delay_p2,
             win_rate_surface_p1, win_rate_surface_p2,
             head_to_head_delay,
             court_feature
        ]

        match_features = base_features + additional_features
        match_feature_list.append(match_features)
        match_list.append([p1, p2])
        labels_list.append(label)

        if dynamic_player_history[p1]:
            p1_history_seq = torch.tensor(dynamic_player_history[p1], dtype=torch.float)
        else:
            p1_history_seq = torch.zeros((1, 1))
        if dynamic_player_history[p2]:
            p2_history_seq = torch.tensor(dynamic_player_history[p2], dtype=torch.float)
        else:
            p2_history_seq = torch.zeros((1, 1))
        match_dynamic_histories.append((p1_history_seq, p2_history_seq))

        player_recent_results[p1].append(1 if label == 1 else 0)
        player_recent_results[p2].append(1 if label == 0 else 0)
        if key not in head_to_head_record:
            head_to_head_record[key] = {key[0]: 0, key[1]: 0}
        if label == 1:
            head_to_head_record[key][p1] += 1
        else:
            head_to_head_record[key][p2] += 1

        dynamic_player_history[p1].append([row["Rank_Joueur_1"]])
        dynamic_player_history[p2].append([row["Rank_Joueur_2"]])

    taille = len(additional_features) + len(base_features)
    match_data_tensor = torch.tensor(match_list, dtype=torch.long)
    scaler_match = StandardScaler()
    match_features_normalized = scaler_match.fit_transform(match_feature_list)
    match_features_tensor = torch.tensor(match_features_normalized, dtype=torch.float)
    labels_tensor = torch.tensor(labels_list, dtype=torch.float).unsqueeze(1)

    return {
        "edge_index": edge_index,
        "player_features": player_features,
        "match_data_tensor": match_data_tensor,
        "match_features_tensor": match_features_tensor,
        "labels_tensor": labels_tensor,
        "match_dynamic_histories": match_dynamic_histories
    }
