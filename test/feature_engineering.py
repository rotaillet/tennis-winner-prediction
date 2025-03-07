#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script de feature engineering pour prédire le gagnant d’un match de tennis.
Ce script réalise :
    - Le calcul de variables différentielles (classement, âge, statistiques)
    - L'extraction de caractéristiques temporelles à partir de la date
    - L'encodage des variables catégorielles (Surface, Tour, Tournoi, saison)
    - Le calcul de moyennes mobiles (rolling averages) pour certaines statistiques
"""

import pandas as pd
import numpy as np

def load_data(filepath):
    """
    Charge les données depuis un fichier CSV.
    On convertit la colonne "Date" en datetime.
    """
    df = pd.read_csv(filepath, parse_dates=["Date"])
    return df

def create_differential_features(df):
    """
    Crée des variables différentielles entre Joueur 1 et Joueur 2 pour les colonnes suivantes, puis supprime 
    les colonnes d'origine utilisées pour calculer ces différences :
    
      - Rank_Joueur_1, Rank_Joueur_2         -> diff_rank
      - Age_Joueur_1, Age_Joueur_2           -> diff_age
      - Points_Joueur_1, Points_Joueur_2     -> diff_points
      - prev_DOUBLE_FAULTS_p1, prev_DOUBLE_FAULTS_p2
                                             -> diff_prev_double_faults
      - prev_ACES_p1, prev_ACES_p2           -> diff_prev_aces
      - prev_1st_SERVE_%_p1_pct, prev_1st_SERVE_%_p2_pct
                                             -> diff_prev_1st_serve_pct
      - prev_1st_SERVE_POINTS_WON_p1_pct, prev_1st_SERVE_POINTS_WON_p2_pct
                                             -> diff_prev_1st_serve_points_won_pct
      - prev_2nd_SERVE_POINTS_WON_p1_pct, prev_2nd_SERVE_POINTS_WON_p2_pct
                                             -> diff_prev_2nd_serve_points_won_pct
      - prev_BREAK_POINTS_WON_p1_pct, prev_BREAK_POINTS_WON_p2_pct
                                             -> diff_prev_break_points_won_pct
      - prev_TOTAL_RETURN_POINTS_WON_p1_pct, prev_TOTAL_RETURN_POINTS_WON_p2_pct
                                             -> diff_prev_total_return_points_won_pct
      - prev_TOTAL_POINTS_WON_p1_pct, prev_TOTAL_POINTS_WON_p2_pct
                                             -> diff_prev_total_points_won_pct
    """
    # Calcul des différences
    df["diff_rank"] = df["Rank_Joueur_1"] - df["Rank_Joueur_2"]
    df["diff_age"] = df["Age_Joueur_1"] - df["Age_Joueur_2"]
    df["diff_points"] = df["Points_Joueur_1"] - df["Points_Joueur_2"]
    df["diff_prev_double_faults"] = df["prev_DOUBLE_FAULTS_p1"] - df["prev_DOUBLE_FAULTS_p2"]
    df["diff_prev_aces"] = df["prev_ACES_p1"] - df["prev_ACES_p2"]
    df["diff_prev_1st_serve_pct"] = df["prev_1st_SERVE_%_p1_pct"] - df["prev_1st_SERVE_%_p2_pct"]
    df["diff_prev_1st_serve_points_won_pct"] = (df["prev_1st_SERVE_POINTS_WON_p1_pct"] - 
                                                df["prev_1st_SERVE_POINTS_WON_p2_pct"])
    df["diff_prev_2nd_serve_points_won_pct"] = (df["prev_2nd_SERVE_POINTS_WON_p1_pct"] - 
                                                df["prev_2nd_SERVE_POINTS_WON_p2_pct"])
    df["diff_prev_break_points_won_pct"] = df["prev_BREAK_POINTS_WON_p1_pct"] - df["prev_BREAK_POINTS_WON_p2_pct"]
    df["diff_prev_total_return_points_won_pct"] = (df["prev_TOTAL_RETURN_POINTS_WON_p1_pct"] - 
                                                   df["prev_TOTAL_RETURN_POINTS_WON_p2_pct"])
    df["diff_prev_total_points_won_pct"] = df["prev_TOTAL_POINTS_WON_p1_pct"] - df["prev_TOTAL_POINTS_WON_p2_pct"]



    return df


def extract_temporal_features(df):
    """
    Extrait des informations temporelles à partir de la colonne "Date" :
        - Mois, Année, Jour de la semaine
        - Saison (winter, spring, summer, fall) et encodage en one-hot
    """
    df["month"] = df["Date"].dt.month
    df["year"] = df["Date"].dt.year
    df["day_of_week"] = df["Date"].dt.dayofweek  # 0 = lundi, 6 = dimanche

    # Définition d'une fonction pour déterminer la saison à partir du mois
    def month_to_season(month):
        if month in [12, 1, 2]:
            return "winter"
        elif month in [3, 4, 5]:
            return "spring"
        elif month in [6, 7, 8]:
            return "summer"
        else:
            return "fall"
    
    df["season"] = df["month"].apply(month_to_season)
    # Encodage one-hot pour la saison
    df = pd.get_dummies(df, columns=["season"], drop_first=True)
    
    return df

def encode_categorical_features(df):
    """
    Encode les variables catégorielles en utilisant l'encodage one-hot :
        - Surface, Tour, Tournoi
    """
    categorical_cols = ["Surface", "Tour"]
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df

def compute_moving_averages(df, player_col, stats_cols, window=5):
    """
    Calcule la moyenne mobile (sur 'window' matchs) pour une liste de statistiques pour un joueur.
    Le calcul se fait en regroupant par le joueur (nom) et en triant par date.
    
    Args:
        df (DataFrame): DataFrame contenant les données.
        player_col (str): Nom de la colonne identifiant le joueur (ex: "Joueur 1" ou "Joueur 2").
        stats_cols (list): Liste des colonnes statistiques à lisser.
        window (int): Taille de la fenêtre de la moyenne mobile.
    """
    for stat in stats_cols:
        col_name = f"{player_col}_moving_avg_{stat}"
        # On suppose que pour chaque joueur, les matchs sont triés par date.
        df[col_name] = df.sort_values("Date").groupby(player_col)[stat].transform(lambda x: x.rolling(window, min_periods=1).mean())

    return df

def main():
    # Chemin vers votre fichier CSV
    filepath = "test.csv"
    df = load_data(filepath)
    
    # Création de variables différentielles
    df = create_differential_features(df)
    
    # Extraction des caractéristiques temporelles
    df = extract_temporal_features(df)
    
    # Encodage des variables catégorielles
    df = encode_categorical_features(df)
    
    # Calcul de moyennes mobiles pour certaines statistiques
    # Exemple pour Joueur 1
    stats_p1 = ["prev_ACES_p1", "prev_DOUBLE_FAULTS_p1", "prev_1st_SERVE_%_p1_pct"]
    # Exemple pour Joueur 2
    stats_p2 = ["prev_ACES_p2", "prev_DOUBLE_FAULTS_p2", "prev_1st_SERVE_%_p2_pct"]
    
    df = compute_moving_averages(df, "Joueur 1", stats_p1, window=5)
    df = compute_moving_averages(df, "Joueur 2", stats_p2, window=5)

    print(df.columns)
    
    # Sauvegarder le nouveau dataset avec les features créées
    df.to_csv("data_engineered.csv", index=False)
    print("Feature engineering terminé et fichier sauvegardé sous 'data_engineered.csv'.")

if __name__ == "__main__":
    main()
