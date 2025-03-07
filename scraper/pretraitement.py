import pandas as pd

# Liste des fichiers CSV
fichiers = [
    "tennis_matches_2020.csv",
    "tennis_matches_2021.csv",
    "tennis_matches_2022.csv",
    "tennis_matches_2023.csv",
    "tennis_matches_2024.csv",
    "tennis_matches_2025.csv"
]
def filter_frequent_players(df, min_matches=5):
    """
    Filtre les joueurs ayant disputé au moins `min_matches` matchs.
    :param df: DataFrame contenant les matchs de tennis.
    :param min_matches: Nombre minimum de matchs requis pour garder un joueur.
    :return: DataFrame filtré.
    """
    player_counts = df['Joueur 1'].value_counts().add(df['Joueur 2'].value_counts(), fill_value=0)

    print(player_counts)
    valid_players = player_counts[player_counts >= min_matches].index
    return df[df['Joueur 1'].isin(valid_players) & df['Joueur 2'].isin(valid_players)]


# Lire et concaténer tous les fichiers
df_final = pd.concat([pd.read_csv(f) for f in fichiers], ignore_index=True)



df_final = df_final.drop_duplicates(subset="Match_details")


def filter_frequent_players(df, min_matches=5):
    """
    Filtre les joueurs ayant disputé au moins `min_matches` matchs.
    :param df: DataFrame contenant les matchs de tennis.
    :param min_matches: Nombre minimum de matchs requis pour garder un joueur.
    :return: DataFrame filtré.
    """
    player_counts = df['Joueur 1'].value_counts().add(df['Joueur 2'].value_counts(), fill_value=0)

    valid_players = player_counts[player_counts >= min_matches].index
    return df[df['Joueur 1'].isin(valid_players) & df['Joueur 2'].isin(valid_players)]


df = filter_frequent_players(df_final,18)
df2 = filter_frequent_players(df,9)
player_counts = df2['Joueur 1'].value_counts().add(df2['Joueur 2'].value_counts(), fill_value=0)

print(player_counts.min())



# Sauvegarder le fichier final
df2.to_csv("tennis_matches_all_years_pre.csv", index=False)



print("Fusion terminée avec succès !")
