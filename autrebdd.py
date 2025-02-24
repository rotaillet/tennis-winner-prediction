import os
import pickle
import pandas as pd
import time
from selenium import webdriver
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from tqdm import tqdm  # Pour la barre de progression

# Fichier dans lequel nous enregistrons l'état
STATE_FILE = "state.pkl"

# Chargement ou initialisation de l'état
if os.path.exists(STATE_FILE):
    with open(STATE_FILE, "rb") as f:
        processed_urls, urls_to_process = pickle.load(f)
    print("État chargé depuis", STATE_FILE)
else:
    processed_urls = set()   # URLs déjà traitées
    urls_to_process = set()   # URLs à traiter
    print("Aucun état sauvegardé trouvé, initialisation.")

def save_state():
    """Enregistre l'état courant dans le fichier STATE_FILE."""
    with open(STATE_FILE, "wb") as f:
        pickle.dump((processed_urls, urls_to_process), f)
    print("État sauvegardé.")

def scraping_match(url, year, driver):
    """
    Scrape la page 'url' pour l'année 'year' en utilisant le driver passé en paramètre.
    Met à jour globalement l'ensemble urls_to_process en ajoutant uniquement les nouveaux liens.
    Retourne un DataFrame avec les données extraites.
    """
    print(f"Scraping de {url}?y={year} ...")
    driver.get(url + f"?y={year}")
    
    WebDriverWait(driver, 10).until(
         EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.player_matches"))
    )
    
    page_source = driver.page_source
    soup = BeautifulSoup(page_source, 'html.parser')
    
    match_data = []
    divs_player_matches = soup.find_all('div', class_='player_matches')
    
    for div in divs_player_matches:
        h2_tag = div.find('h2')
        if h2_tag and str(year) in h2_tag.get_text():
            table = div.find('table', class_='table_pmatches')
            if table:
                tbody = table.find('tbody')
                rows = tbody.find_all('tr')
                
                # Variables pour gérer les rowspans
                rowspan_data = None  
                rowspan_count = 0  
                rowspan_data_2 = None  
                rowspan_count_2 = 0  
                
                for row in rows:
                    cols = row.find_all('td')
                    
                    # Gestion du rowspan pour "Tournoi" (indice 7)
                    if len(cols) > 7 and cols[7].has_attr("rowspan"):
                        rowspan_count = int(cols[7]["rowspan"])
                        rowspan_data = cols[7].find("a")["title"] if cols[7].find("a") else ""
                    title_value = rowspan_data if rowspan_count > 0 else ""
                    if rowspan_count > 0:
                        rowspan_count -= 1
                        
                    # Gestion du rowspan pour "Surface" (indice 8)
                    if len(cols) > 8 and cols[8].has_attr("rowspan"):
                        rowspan_count_2 = int(cols[8]["rowspan"])
                        rowspan_data_2 = cols[8].get_text(strip=True)
                    value_surface = rowspan_data_2 if rowspan_count_2 > 0 else ""
                    if rowspan_count_2 > 0:
                        rowspan_count_2 -= 1
                        
                    if len(cols) >= 5:
                        date = cols[0].get_text(strip=True)
                        tour = cols[1].get_text(strip=True)
                        joueur1 = cols[2].get_text(strip=True)
                        joueur2 = cols[3].get_text(strip=True)
                        score = cols[4].get_text(strip=True)
                        match_details = cols[6].find("a")["href"] if cols[6].find("a") else ""
                        
                        match_data.append([
                            date, tour, joueur1, joueur2, score,
                            title_value, value_surface, match_details
                        ])
                        
                        # Récupérer les liens des pages des joueurs (colonnes 2 et 3)
                        for col_index in [2, 3]:
                            a_tag = cols[col_index].find("a")
                            if a_tag:
                                page = a_tag.get("href")
                                if page and (page not in urls_to_process) and (page not in processed_urls):
                                    urls_to_process.add(page)
    
    # Marquer l'URL actuelle comme traitée
    processed_urls.add(url)
    
    df = pd.DataFrame(match_data, columns=[
        "Date", "Tour", "Joueur 1", "Joueur 2", "Score", "Tournoi", "Surface", "Match_details"
    ])
    return df

def main():
    driver = webdriver.Chrome()
    year = 2020
    csv_filename = f"tennis_matches_{year}.csv"

    
    total_expected_matches = 75000  # Nombre total de matchs attendu dans le CSV
    total_matches_scraped = 0         # Compteur du nombre de matchs déjà récupérés
    total_time_spent = 0.0            # Temps total passé sur le scraping (en secondes)
    
    # Initialisation de la barre de progression avec tqdm
    progress_bar = tqdm(total=total_expected_matches, desc="Progression", unit="match")
    
    # Traitement de l'URL initiale
    url_initial = "https://www.tennislive.net/atp/alexander-zverev/"
    if url_initial not in processed_urls:
        start_time = time.time()
        df_initial = scraping_match(url_initial, year, driver)
        duration = time.time() - start_time
        num_matches = len(df_initial)
        total_matches_scraped += num_matches
        total_time_spent += duration
        
        df_initial.to_csv(csv_filename, index=False, encoding="utf-8-sig")
        print(f"Fichier CSV créé avec les données de {url_initial}.")
        progress_bar.update(num_matches)
        save_state()
    
    # Traitement des URLs restantes
    while urls_to_process:
        print(len(urls_to_process))
        next_url = urls_to_process.pop()
        print(f"Traitement du lien suivant : {next_url}")
        try:
            start_time = time.time()
            df_new = scraping_match(next_url, year, driver)
            duration = time.time() - start_time
            num_matches = len(df_new)
            
            if num_matches > 0:
                total_matches_scraped += num_matches
                total_time_spent += duration
                # Calcul du temps moyen par match global
                avg_time_per_match = total_time_spent / total_matches_scraped
                # Estimation du temps total et temps restant
                estimated_total_time = avg_time_per_match * total_expected_matches
                remaining_time = estimated_total_time - total_time_spent
                
                # Mise à jour de la barre de progression avec quelques informations
                progress_bar.set_postfix({
                    "Matches": total_matches_scraped,
                    "ETA (s)": int(remaining_time)
                })
                
                df_new.to_csv(csv_filename, mode='a', header=False, index=False, encoding="utf-8-sig")
                progress_bar.update(num_matches)
            else:
                print("Aucun match trouvé sur ce lien.")
        except Exception as e:
            print(f"Erreur lors du traitement de {next_url} : {e}")
        
        # Sauvegarder l'état après chaque itération
        save_state()
    
    progress_bar.close()
    print("Tous les liens ont été traités.")
    driver.quit()

if __name__ == '__main__':
    main()
