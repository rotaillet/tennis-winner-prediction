import pandas as pd
import os
import datetime
import numpy as np

# Fonction pour trier le DataFrame par date
def sort_dataframe(df):
    df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%y')
    df = df.sort_values(by="Date").reset_index(drop=True)
    return df

# Fonction pour trouver le fichier le plus proche et l'ajouter au DataFrame
def ranking(df, dos="ranking_dataframe"):
    # Conversion des dates du DataFrame en datetime
    df["Date"] = pd.to_datetime(df["Date"], format='%d.%m.%y')

    # Récupérer toutes les dates des fichiers CSV dans le dossier
    file_dates = {}
    for filename in os.listdir(dos):
        if filename.endswith(".csv"):
            try:
                file_date = datetime.datetime.strptime(filename.split('_')[-1].replace(".csv", ""), "%Y-%m-%d")
                file_dates[filename] = file_date
            except ValueError:
                continue  # Ignore les fichiers avec un format incorrect
    
    if not file_dates:
        print("Aucun fichier valide trouvé.")
        return df

    # Trier les fichiers par date pour faciliter la recherche
    sorted_files = sorted(file_dates.items(), key=lambda x: x[1])

    # Stocker les fichiers les plus proches pour chaque date
    closest_files = []
    
    for i in range(len(df)):
        current_date = df["Date"][i]
        closest_file = None

        for filename, file_date in sorted_files:
            if file_date <= current_date:
                closest_file = filename  # Mettre à jour avec le fichier le plus proche
            else:
                break  # Arrêter dès qu'on dépasse la date du match
        
        closest_files.append(closest_file if closest_file else "Aucun fichier trouvé")

    # Ajouter la colonne "Closest_File" au DataFrame
    df["Closest_File"] = closest_files

    # Trier le DataFrame à nouveau après ajout de la colonne (au cas où)
    df = df.sort_values(by="Date").reset_index(drop=True)

    return df

import pandas as pd


def extract(df, dos="ranking_dataframe"):
    rank1, rank2, age1, age2, point1, point2 = [], [], [], [], [], []
    
    for i in tqdm(range(len(df)), desc="Extraction en cours"):  # Ajout de tqdm
        file = df["Closest_File"].iloc[i]
        joueur1 = df["Joueur 1"].iloc[i]
        joueur2 = df["Joueur 2"].iloc[i]

        # Charger le fichier de classement correspondant
        try:
            df2 = pd.read_csv(f"{dos}/{file}")
        except FileNotFoundError:
            print(f"Fichier non trouvé : {file}")
            rank1.append(None)
            rank2.append(None)
            age1.append(None)
            age2.append(None)
            point1.append(None)
            point2.append(None)
            continue

        # Recherche des informations des joueurs
        ra1 = df2.loc[df2["Player"] == joueur1, "Rank"].values
        ra2 = df2.loc[df2["Player"] == joueur2, "Rank"].values
        ag1 = df2.loc[df2["Player"] == joueur1, "Age"].values
        ag2 = df2.loc[df2["Player"] == joueur2, "Age"].values
        p1 = df2.loc[df2["Player"] == joueur1, "Points"].values
        p2 = df2.loc[df2["Player"] == joueur2, "Points"].values

        # Ajout des valeurs aux listes
        rank1.append(ra1[0] if len(ra1) > 0 else 1000)
        rank2.append(ra2[0] if len(ra2) > 0 else 1000)
        age1.append(ag1[0] if len(ag1) > 0 else None)
        age2.append(ag2[0] if len(ag2) > 0 else None)
        point1.append(p1[0] if len(p1) > 0 else 1)
        point2.append(p2[0] if len(p2) > 0 else 1)

    # Ajouter les données au DataFrame
    df["Rank_Joueur_1"] = rank1
    df["Rank_Joueur_2"] = rank2
    df["Age_Joueur_1"] = age1
    df["Age_Joueur_2"] = age2
    df["Points_Joueur_1"] = point1
    df["Points_Joueur_2"] = point2
    
    print(f"Classements récupérés pour {len(rank1)} matchs.")
    df.to_csv("tennis_matches.csv")
    return df



import pandas as pd
from tqdm import tqdm  # Ajout de la barre de progression

import pandas as pd
import numpy as np
import re

def parse_individual_set(set_str):
    """
    Parse un score de set individuel.
    Si le score est mal formaté, par exemple "6-76",
    on considère le score de base comme "6-7" et on extrait 
    l'extra (ici "6") à ajouter aux deux côtés, ce qui donne (6+6, 7+6).
    
    Pour un set normal comme "7-5", retourne (7, 5).
    """
    parts = set_str.split('-')
    if len(parts) != 2:
        return None  # Format inattendu
    try:
        base_left = int(parts[0].strip())
    except ValueError:
        base_left = 0
    right_part = parts[1].strip()
    
    # Si la partie droite comporte plusieurs chiffres, on applique la règle spéciale.
    if len(right_part) > 1:
        try:
            base_right = int(right_part[0])
            extra = int(right_part[1:])
        except ValueError:
            base_right = 0
            extra = 0
        final_left = base_left + extra
        final_right = base_right + extra
        return final_left, final_right
    else:
        try:
            return base_left, int(right_part)
        except ValueError:
            return base_left, 0

def parse_score_string_detailed(score_string):
    """
    Découpe la chaîne de score contenant plusieurs sets (ex: "3-6, 7-5, 6-1")
    et retourne la liste des tuples (score_j1, score_j2) pour chaque set.
    """
    sets = score_string.split(',')
    sets = [s.strip() for s in sets if s.strip() != ""]
    set_scores = []
    for s in sets:
        parsed = parse_individual_set(s)
        if parsed is not None:
            set_scores.append(parsed)
    return set_scores

def extract_score_features(score_string):
    """
    Extrait de nombreuses informations issues de la chaîne de score.
    
    Renvoie un dictionnaire contenant :
      - nb_sets : nombre de sets joués
      - set_win_j1 : nombre de sets remportés par le joueur 1
      - set_win_j2 : nombre de sets remportés par le joueur 2
      - total_games_j1 : score total (agrégé) pour le joueur 1
      - total_games_j2 : score total (agrégé) pour le joueur 2
      - diff_moy : différence moyenne (j1 - j2) par set
      - diff_min : différence minimale par set
      - diff_max : différence maximale par set
      - diff_std : écart-type de la différence par set
    """
    set_scores = parse_score_string_detailed(score_string)
    nb_sets = len(set_scores)
    set_win_j1 = 0
    set_win_j2 = 0
    total_games_j1 = 0
    total_games_j2 = 0
    diffs = []
    
    for (s1, s2) in set_scores:
        total_games_j1 += s1
        total_games_j2 += s2
        diffs.append(s1 - s2)
        if s1 > s2:
            set_win_j1 += 1
        elif s2 > s1:
            set_win_j2 += 1
        # En cas d'égalité, vous pouvez décider d'ignorer ou de répartir

    return {
        "nb_sets": nb_sets,
        "set_win_player_1": set_win_j1,
        "set_win_player_2": set_win_j2,
        "total_games_player_1": total_games_j1,
        "total_games_player_2": total_games_j2
    }

def preprocess_scores_detailed(df, score_col='Score'):
    """
    Prend un DataFrame et ajoute plusieurs colonnes issues du prétraitement
    de la chaîne de score.
    
    Les nouvelles colonnes ajoutées sont :
      - score1 : score total pour le joueur 1
      - score2 : score total pour le joueur 2
      - nb_sets : nombre de sets joués
      - set_win_j1 : nombre de sets remportés par le joueur 1
      - set_win_j2 : nombre de sets remportés par le joueur 2
      - diff_moy : différence moyenne de jeux par set (j1 - j2)
      - diff_min : différence minimale
      - diff_max : différence maximale
      - diff_std : écart-type de la différence
    """
    # Appliquer l'extraction pour chaque ligne
    features = df[score_col].apply(lambda x: pd.Series(extract_score_features(x)))
    # Calculer les scores totaux séparément pour faciliter la lecture (agrégation sur tous les sets)
    df = pd.concat([df, features], axis=1)
    return df

def process_score_string(score_string):
    """
    Agrège les scores de tous les sets pour retourner le score total.
    Cette fonction utilise parse_score_string_detailed pour accumuler les scores.
    """
    set_scores = parse_score_string_detailed(score_string)
    total1 = sum(s[0] for s in set_scores)
    total2 = sum(s[1] for s in set_scores)
    return total1, total2




def get_previous_match_stats(df):
    # Convertir la colonne Date en datetime et trier le DataFrame par date
    df["Date"] = pd.to_datetime(df["Date"])
    
    # Liste des statistiques de base à récupérer (sans la partie _player_X)
    base_stats = [
        "1st_SERVE_%",
        "1st_SERVE_POINTS_WON",
        "2nd_SERVE_POINTS_WON",
        "BREAK_POINTS_WON",
        "TOTAL_RETURN_POINTS_WON",
        "TOTAL_POINTS_WON",
        "DOUBLE_FAULTS",
        "ACES",
        "total_games",
        "set_win"
    ]
    
    # Dictionnaire pour stocker le dernier match de chaque joueur
    previous_stats = {}
    new_data = []
    
    # Trier les matchs par date
    df = df.sort_values(by="Date")
    
    # Parcourir les matchs chronologiquement
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing matches"):
        j1 = row["Joueur 1"]
        j2 = row["Joueur 2"]
        
        # Copier la ligne courante pour ajouter les colonnes "prev_..."
        row_copy = row.copy()
        for stat in base_stats:
            row_copy["prev_" + stat + "_p1"] = None  # Pour le Joueur 1
            row_copy["prev_" + stat + "_p2"] = None  # Pour le Joueur 2
        
        # Récupérer les stats du match précédent pour Joueur 1 (s'il existe)
        if j1 in previous_stats:
            prev_row = previous_stats[j1]
            # Déterminer si j1 était Joueur 1 ou Joueur 2 dans le match précédent
            if prev_row["Joueur 1"] == j1:
                for stat in base_stats:
                    row_copy["prev_" + stat + "_p1"] = prev_row[stat + "_player_1"]
            elif prev_row["Joueur 2"] == j1:
                for stat in base_stats:
                    row_copy["prev_" + stat + "_p1"] = prev_row[stat + "_player_2"]
        
        # Récupérer les stats du match précédent pour Joueur 2 (s'il existe)
        if j2 in previous_stats:
            prev_row = previous_stats[j2]
            if prev_row["Joueur 1"] == j2:
                for stat in base_stats:
                    row_copy["prev_" + stat + "_p2"] = prev_row[stat + "_player_1"]
            elif prev_row["Joueur 2"] == j2:
                for stat in base_stats:
                    row_copy["prev_" + stat + "_p2"] = prev_row[stat + "_player_2"]
        
        # Ajouter la ligne modifiée à la nouvelle liste
        new_data.append(row_copy)
        
        # Mettre à jour le dictionnaire des statistiques précédentes pour les deux joueurs
        previous_stats[j1] = row.copy()
        previous_stats[j2] = row.copy()
    
    new_df = pd.DataFrame(new_data)
    
    # Conserver uniquement les lignes où les deux joueurs ont des stats précédentes
    new_df = new_df.dropna(subset=["prev_" + stat + "_p1" for stat in base_stats] +
                                     ["prev_" + stat + "_p2" for stat in base_stats])
    
    # Optionnel : supprimer les colonnes inutiles
    new_df.drop(["Closest_File", "No_stats_available", "Match_details",
     "Unnamed: 0", "Player_1", "Player_2","1st_SERVE_%_player_1","1st_SERVE_%_player_2",
     "1st_SERVE_POINTS_WON_player_1","1st_SERVE_POINTS_WON_player_2","2nd_SERVE_POINTS_WON_player_1","2nd_SERVE_POINTS_WON_player_2",
     "BREAK_POINTS_WON_player_1","BREAK_POINTS_WON_player_2","TOTAL_RETURN_POINTS_WON_player_1","TOTAL_RETURN_POINTS_WON_player_2",
     "TOTAL_POINTS_WON_player_1","TOTAL_POINTS_WON_player_2","DOUBLE_FAULTS_player_1","DOUBLE_FAULTS_player_2","ACES_player_1","ACES_player_2",
     "set_win_player_1","set_win_player_2","total_games_player_1","total_games_player_2"], axis=1, inplace=True, errors='ignore')
    
    return new_df



def melange(df):




    # 1. Ajout d'une colonne 'winner'
    # Dans la base, le Joueur 2 perd toujours, donc le gagnant est initialement Joueur 1
    df['winner'] = df['Joueur 1']

    # 2. Liste des colonnes à échanger (les colonnes liées aux joueurs et à leurs stats)
    columns_to_swap = [
        ('Joueur 1', 'Joueur 2'),
        ('Rank_Joueur_1', 'Rank_Joueur_2'),
        ('Age_Joueur_1', 'Age_Joueur_2'),
        ('Points_Joueur_1', 'Points_Joueur_2'),
        ('prev_1st_SERVE_%_p1', 'prev_1st_SERVE_%_p2'),
        ('prev_1st_SERVE_POINTS_WON_p1', 'prev_1st_SERVE_POINTS_WON_p2'),
        ('prev_2nd_SERVE_POINTS_WON_p1', 'prev_2nd_SERVE_POINTS_WON_p2'),
        ('prev_BREAK_POINTS_WON_p1', 'prev_BREAK_POINTS_WON_p2'),
        ('prev_TOTAL_RETURN_POINTS_WON_p1', 'prev_TOTAL_RETURN_POINTS_WON_p2'),
        ('prev_TOTAL_POINTS_WON_p1', 'prev_TOTAL_POINTS_WON_p2'),
        ('prev_DOUBLE_FAULTS_p1', 'prev_DOUBLE_FAULTS_p2'),
        ('prev_ACES_p1', 'prev_ACES_p2'),
        ("prev_total_games_p1","prev_total_games_p2"),
        ("prev_set_win_p1","prev_set_win_p2")
    ]

    # 3. Sélectionner aléatoirement 50% des lignes pour procéder à l'inversion
    mask = np.random.rand(len(df)) < 0.5

    # 4. Pour les lignes sélectionnées, échanger les colonnes de Joueur 1 et Joueur 2 ainsi que leurs stats
    for col1, col2 in columns_to_swap:
        temp = df.loc[mask, col1].copy()
        df.loc[mask, col1] = df.loc[mask, col2]
        df.loc[mask, col2] = temp

    # 5. Mise à jour de la colonne 'winner'
    # Pour les lignes échangées, le gagnant (initialement en 'Joueur 1') se retrouve désormais en 'Joueur 2'
    df.loc[mask, 'winner'] = df.loc[mask, 'Joueur 2']

    # Sauvegarder la base modifiée dans un nouveau fichier CSV
    return df

def features(df):

    # Liste des colonnes à transformer
    cols_to_transform = [
        'prev_1st_SERVE_%_p1',
        'prev_1st_SERVE_%_p2',
        'prev_1st_SERVE_POINTS_WON_p1',
        'prev_1st_SERVE_POINTS_WON_p2',
        'prev_2nd_SERVE_POINTS_WON_p1',
        'prev_2nd_SERVE_POINTS_WON_p2',
        'prev_BREAK_POINTS_WON_p1',
        'prev_BREAK_POINTS_WON_p2',
        'prev_TOTAL_RETURN_POINTS_WON_p1',
        'prev_TOTAL_RETURN_POINTS_WON_p2',
        'prev_TOTAL_POINTS_WON_p1',
        'prev_TOTAL_POINTS_WON_p2'
    ]

    # Pour chaque colonne, extraire les trois valeurs à l'aide d'une expression régulière
    for col in cols_to_transform:
        # L'expression régulière capture : 
        # - le numérateur (\d+)
        # - le dénominateur (\d+) après le caractère '/'
        # - le pourcentage (\d+) entre parenthèses suivi du signe '%'
        extracted = df[col].str.extract(r'(?P<num>\d+)/(?P<den>\d+)\s*\((?P<pct>\d+)%\)')
        
        # Convertir les valeurs extraites aux types souhaités
        df[col + '_num'] = extracted['num'].astype(int)
        df[col + '_den'] = extracted['den'].astype(int)
        df[col + '_pct'] = extracted['pct'].astype(float) / 100.0

        # Optionnel : supprimer la colonne d'origine
        df.drop(columns=[col], inplace=True)

    # Sauvegarder le DataFrame modifié dans un nouveau fichier CSV
    return df


    # Charger le DataFrame (remplace 'data.csv' par ton fichier)


def nan_traitement(df):
    # Remplacer les valeurs manquantes dans la colonne "Tour" par "Amical"
    df.loc[df["Tour"].isna(), "Tour"] = "Amical"
    
    # Remplacer les NaN dans la colonne "Age" par la médiane de cette colonne
# Calculer la médiane globale à partir des deux colonnes d'âge
    global_median_age = pd.concat([df["Age_Joueur_1"], df["Age_Joueur_2"]]).median()

    # Remplacer les NaN dans chacune des colonnes avec la médiane globale
    df["Age_Joueur_1"].fillna(global_median_age, inplace=True)
    df["Age_Joueur_2"].fillna(global_median_age, inplace=True)
    
    return df

df = pd.read_csv("tennis_matchs.csv")


df= preprocess_scores_detailed(df, score_col='Score')
df = get_previous_match_stats(df)
df = melange(df)
df = features(df)
df = nan_traitement(df)

df.to_csv('test3.csv')