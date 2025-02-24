import pandas as pd

def filter_frequent_players(df, min_matches=5):
    """
    Filtre les joueurs ayant disputé au moins `min_matches` matchs.
    :param df: DataFrame contenant les matchs de tennis.
    :param min_matches: Nombre minimum de matchs requis pour garder un joueur.
    :return: DataFrame filtré.
    """
    player_counts = df['player_1'].value_counts().add(df['player_2'].value_counts(), fill_value=0)

    print(player_counts)
    valid_players = player_counts[player_counts >= min_matches].index
    return df[df['player_1'].isin(valid_players) & df['player_2'].isin(valid_players)]

# Exemple d'utilisation avec un DataFrame `df_matches`
# df_filtered = filter_frequent_players(df_matches)

df = pd.read_csv("atp_tennis_pretraite.csv")

df_filtered = filter_frequent_players(df,30)
df_filtered_2 = filter_frequent_players(df_filtered,15)
df_filtered_3 = filter_frequent_players(df_filtered_2,8)
df_filtered_4 = filter_frequent_players(df_filtered_3,4)
df_filtered_5 = filter_frequent_players(df_filtered_4,2)

print(df_filtered_5)

df_filtered_5.to_csv("tennis_matchs.csv")