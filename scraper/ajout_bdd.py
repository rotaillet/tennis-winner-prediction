import pandas as pd
import os
import multiprocessing
from selenium import webdriver
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from tqdm import tqdm

# Fichier CSV final
output_file = "tennis_matches_with_stats_3.csv"

# Charger les matchs
df = pd.read_csv("tennis_matches_all_years_pre.csv")

# Vérifier si un fichier partiel existe déjà
if os.path.exists(output_file):
    df_existing = pd.read_csv(output_file)
    processed_matches = set(df_existing["Match_details"].dropna())  # Matches déjà traités
    print(f"🔄 Reprise : {len(processed_matches)} matchs déjà traités.")
else:
    df_existing = None
    processed_matches = set()
    print("🚀 Nouveau scraping.")

# Filtrer les matchs restants
df_to_scrape = df[~df["Match_details"].isin(processed_matches)]

# Définir le nombre de navigateurs à ouvrir en parallèle
num_processes = min(10, len(df_to_scrape))  # Évite d'ouvrir trop de processus


def init_driver():
    """Initialise un navigateur Selenium sans affichage pour plus d'efficacité."""
    options = webdriver.ChromeOptions()
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    return webdriver.Chrome(options=options)


def scrape_batch(batch):
    """Scrape un lot de matchs et garde uniquement les stats des deux joueurs."""
    driver = init_driver()
    match_data = []

    for index, row in tqdm(batch.iterrows(), total=len(batch)):
        match_url = row["Match_details"]
        player_1 = row.get("Player_1", "Unknown")  # Nom du joueur 1
        player_2 = row.get("Player_2", "Unknown")  # Nom du joueur 2

        try:
            driver.get(match_url)

            # Attendre la table des stats (on continue même si elle ne charge pas)
            try:
                WebDriverWait(driver, 1).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "table.table_stats_match"))
                )
            except:
                pass  # Si la table n'est pas trouvée, on continue quand même

            soup = BeautifulSoup(driver.page_source, 'html.parser')

            # Extraire les stats
            table = soup.find('table', class_='table_stats_match')
            match_stats = {"Match_details": match_url, "Player_1": player_1, "Player_2": player_2}

            if table:
                for row in table.find_all('tr'):
                    cols = row.find_all('td')

                    # Vérifier qu'on a bien 3 colonnes et que la première est une vraie statistique
                    if len(cols) == 3:
                        stat_name = cols[0].get_text(strip=True)
                        player_1_stat = cols[1].get_text(strip=True)
                        player_2_stat = cols[2].get_text(strip=True)

                        # Vérifier que les valeurs sont bien des statistiques (évite les noms, images, etc.)
                        if (
                            stat_name and player_1_stat and player_2_stat  # Éviter les valeurs vides
                            and any(char.isdigit() for char in player_1_stat)  # Vérifier qu'on a des chiffres
                            and any(char.isdigit() for char in player_2_stat)  
                        ):
                            stat_name = stat_name.replace(" ", "_")
                            match_stats[f"{stat_name}_player_1"] = player_1_stat
                            match_stats[f"{stat_name}_player_2"] = player_2_stat

            else:
                match_stats["No_stats_available"] = True  # Indiquer qu'il n'y a pas de stats
        
            match_data.append(match_stats)

        except Exception as e:
            print(f"❌ Erreur sur {match_url}: {e}")
            continue

    driver.quit()
    return match_data


def process_batches():
    """Divise les matchs en sous-groupes et lance le scraping en parallèle."""
    batch_size = 200
    batches = [df_to_scrape.iloc[i * batch_size:(i + 1) * batch_size] for i in range(num_processes)]

    with multiprocessing.Pool(num_processes) as pool:
        results = pool.map(scrape_batch, batches)

    return [item for sublist in results for item in sublist]  # Aplatir les résultats


if __name__ == "__main__":
    scraped_data = process_batches()
    df_stats = pd.DataFrame(scraped_data)

    # Fusionner avec l'existant et sauvegarder
    if df_existing is not None:
        df_final = pd.concat([df_existing, df_stats], ignore_index=True)
    else:
        df_final = df_stats

    df_final.to_csv(output_file, index=False)
    print("✅ Scraping terminé avec succès !")
