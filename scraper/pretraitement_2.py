import pandas as pd


def merge_and_clean(df1,df2):

    # Fusionner sur la colonne 'Match_details'
    df_merged = pd.merge(df1, df2, on='Match_details', how='inner')  # 'inner' garde seulement les correspondances

    # Afficher les premières lignes du DataFrame fusionné
    print(df_merged.head())

    # Sauvegarder le fichier fusionné si besoin
    df_merged.to_csv('tennis_matches_merged.csv', index=False)

    df_merged_clean = df_merged.dropna(subset=['1st_SERVE_%_player_1'])

    # Afficher les premières lignes du DataFrame nettoyé
    print(df_merged_clean.head())

    # Sauvegarder le fichier nettoyé si besoin
    df_merged_clean.to_csv('tennis_matches_merged_clean.csv', index=False)

import os
import pandas as pd
import re

def extract_name_age(s):
    """
    Extrait le nom et l'âge d'une chaîne de type 
    "Rafael Nadal (ESP) (33 years)".
    Renvoie un tuple (nom, age).
    """
    pattern = r"^(.*?)\s*\([^)]*\)\s*\((\d+)\s*years\)$"
    match = re.match(pattern, s)
    if match:
        name = match.group(1).strip()
        age = int(match.group(2))
        return name, age
    else:
        return s, None

# Dossier source contenant les fichiers CSV
source_folder = "rankin_dataframe"
# Dossier destination pour enregistrer les fichiers traités
dest_folder = "preprocess_ranking_dataframe"
os.makedirs(dest_folder, exist_ok=True)

# Parcourir chaque fichier CSV dans le dossier source
for filename in os.listdir(source_folder):
    if filename.endswith(".csv"):
        filepath = os.path.join(source_folder, filename)
        df = pd.read_csv(filepath)

        # Si vous souhaitez ignorer la valeur de rang initiale et 
        # lui attribuer un rang séquentiel basé sur l'ordre du DataFrame :
        df['Rank'] = range(1, len(df) + 1)
        
        # Sinon, si vous souhaitez nettoyer la valeur existante :
        # df['Rank'] = df['Rank'].astype(str).str.replace(".", "", regex=False).astype(int)

        # Extraire le nom et l'âge à partir de la colonne "Name"
        df[['Player', 'Age']] = df['Name'].apply(lambda x: pd.Series(extract_name_age(x)))

        # Sauvegarder le DataFrame modifié dans le dossier destination
        output_filename = f"processed_{filename}"
        output_path = os.path.join(dest_folder, output_filename)
        df.to_csv(output_path, index=False)
        print(f"Fichier traité et enregistré dans : {output_path}")
