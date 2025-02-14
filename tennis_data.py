# -*- coding: utf-8 -*-
import csv
import os
import time
import random
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# --- Configuration de Selenium ---
options = webdriver.ChromeOptions()
# Optionnel : lancer en mode headless
# options.add_argument('--headless')
options.add_argument('--disable-blink-features=AutomationControlled')
driver = webdriver.Chrome(options=options)

# --- Définition des fichiers d'entrée et de sortie ---
input_csv = "players.csv"              # Fichier d'entrée (avec au moins une colonne "Player_Name")
output_csv = "players_full_names.csv"  # Fichier de sortie

# --- Charger la liste des joueurs déjà traités ---
processed = set()
if os.path.exists(output_csv):
    with open(output_csv, newline='', encoding='utf-8') as out_file:
        reader = csv.DictReader(out_file)
        for row in reader:
            processed.add(row["Player_Name"])

# --- Ouvrir le fichier de sortie en mode append ---
out_file_exists = os.path.exists(output_csv)
with open(output_csv, 'a', newline='', encoding='utf-8') as out_file:
    fieldnames = ["Player_Name", "Full_Name", "Player_Page"]
    writer = csv.DictWriter(out_file, fieldnames=fieldnames)
    if not out_file_exists:
        writer.writeheader()

    # --- Traitement du fichier d'entrée ---
    with open(input_csv, newline='', encoding='utf-8') as in_file:
        reader = csv.DictReader(in_file)
        for row in reader:
            player_name = row["Player_Name"]

            # Vérifier si ce joueur a déjà été traité
            if player_name in processed:
                print(f"Déjà traité : {player_name}. Passage au suivant.")
                continue

            # Construction de la requête de recherche (ici sur DuckDuckGo)
            query = f"tennisendirect {player_name} tennis player"
            print(f"Recherche pour : {query}")

            # Accéder à DuckDuckGo
            driver.get("https://duckduckgo.com/")
            try:
                search_box = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.ID, "search_form_input_homepage"))
                )
                search_box.clear()
                search_box.send_keys(query)
                search_box.send_keys(Keys.RETURN)
            except Exception as e:
                print(f"Erreur lors de la saisie de la requête pour {player_name} : {e}")
                continue

            # Pause aléatoire après la soumission de la recherche (entre 5 et 15 secondes)
            time.sleep(random.uniform(5, 15))

            # Récupérer le premier résultat
            try:
                result_link = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "a.result__a"))
                )
                result_url = result_link.get_attribute("href")
                print(f"URL trouvée pour {player_name} : {result_url}")
            except Exception as e:
                print(f"Aucun résultat trouvé pour {player_name} : {e}")
                result_url = None

            full_name = ""
            if result_url:
                # Accéder à la page du joueur
                driver.get(result_url)
                # Pause pour laisser le temps à la page de se charger (entre 5 et 10 secondes)
                time.sleep(random.uniform(5, 10))
                try:
                    h1_element = WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.TAG_NAME, "h1"))
                    )
                    full_name = h1_element.text.strip()
                    print(f"Nom complet trouvé pour {player_name} : {full_name}")
                except Exception as e:
                    print(f"Nom complet non trouvé pour {player_name} : {e}")
                    full_name = "Nom complet non trouvé sur la page"
            else:
                full_name = "Aucun résultat trouvé"

            # Écrire le résultat dans le fichier de sortie
            writer.writerow({
                "Player_Name": player_name,
                "Full_Name": full_name,
                "Player_Page": result_url if result_url else ""
            })
            out_file.flush()  # Sauvegarder immédiatement

            # Ajouter le joueur à l'ensemble des joueurs traités
            processed.add(player_name)
            print(f"Terminé pour {player_name}.\n")

            # Pause aléatoire avant de passer au joueur suivant (entre 10 et 20 secondes)
            time.sleep(random.uniform(1, 2))

# Fermer le navigateur Selenium
driver.quit()
print(f"Traitement terminé. Le fichier '{output_csv}' a été généré.")
