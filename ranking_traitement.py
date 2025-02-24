import pandas as pd
import os
import datetime

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
    rank1 = []
    rank2 = []
    
    for i in range(len(df)):  # Prend uniquement les 100 premières lignes
        print(i)
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
            continue

        # Recherche des classements des joueurs
        ra1 = df2.loc[df2["Player"] == joueur1, "Rank"].values
        ra2 = df2.loc[df2["Player"] == joueur2, "Rank"].values

        # Ajout des classements dans les listes
        rank1.append(ra1[0] if len(ra1) > 0 else None)
        rank2.append(ra2[0] if len(ra2) > 0 else None)

    # Ajouter les classements au DataFrame d'origine
    df["Rank_Joueur_1"] = rank1
    df["Rank_Joueur_2"] = rank2

    print(f"Classements récupérés pour {len(rank1)} matchs.")

    return df

# Charger les données
df = pd.read_csv("tennis_matches_merged_clean.csv")

# Trier et attribuer le fichier le plus proche
df = sort_dataframe(df)
df = ranking(df)
df = extract(df)
print(df)

