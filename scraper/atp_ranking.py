from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
import time
import pandas as pd
import os 
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
# 1) Lancement du navigateur
driver = webdriver.Chrome()
driver.get("https://www.tennislive.net/atp/ranking/EN/")  # URL à adapter si besoin

# 2) Récupérer l'élément <select> (par son ID 'rankD' dans votre exemple)
select_element = driver.find_element(By.ID, "rankD")

# 3) Transformer l'élément en un objet "Select" Selenium
select_obj = Select(select_element)

# 4) Récupérer toutes les <option>
all_options = select_obj.options  # liste de WebElement

    
# 5) Sélectionner la première date (par exemple l'option [0])
#    Dans votre capture, ça pourrai
# t être "2020-10-19" (en 'value' ou 'text').
option_values = [opt.get_attribute("value") for opt in all_options]

for i in range(220):
    first_option = option_values[i]
    csv_filename = f"ranking_{first_option}.csv"
    csv_path = os.path.join("ranking_dataframe", csv_filename)

    # Vérifier si le fichier existe déjà
    if os.path.exists(csv_path):
        print(f"Le fichier {csv_filename} existe déjà, passage à la prochaine itération.")
        break  # Passer à la prochaine itération pour éviter de re-télécharger les mêmes données


    driver.get(f"https://www.tennislive.net/atp/ranking/EN/{first_option}")

    # 7) Attendre un peu le rechargement de la page (selon la vitesse du site)
    time.sleep(1)

    # 8) Récupérer le tableau avec la classe 'table_pranks'
    table = driver.find_element(By.CSS_SELECTOR, "table.table_pranks")

    # 9) Récupérer toutes les lignes <tr> dans le <tbody> (sauf la ligne d'en-tête si besoin)
    rows = table.find_elements(By.TAG_NAME, "tr")

    # 10) Parcourir les lignes pour extraire les données
    #     D'après votre exemple, on voit que :
    #       - la première ligne a la classe "header" (et contient "points" en 3e colonne)
    #       - puis les lignes suivantes sont "pair" ou "unpair" avec : 
    #            1er <td> = rang
    #            2e <td> = (nom ? vide ?)
    #            3e <td> = points
    #     Adaptez selon la structure réelle !
    data = []
    for row in rows:
        row_class = row.get_attribute("class")
        # On ignore la ligne d'entête "header"
        if "header" in row_class:
            continue

        # On ne traite que les lignes "pair" ou "unpair"
        if "pair" in row_class or "unpair" in row_class:
            cols = row.find_elements(By.TAG_NAME, "td")
            # Selon votre capture : 
            #   - col[0] = # (classement)
            #   - col[1] = ?
            #   - col[2] = points
            rank = cols[0].text.strip()
            # Parfois le nom du joueur est dans col[1], parfois c’est une autre structure...
            # À vérifier selon la page réelle
            name = cols[1].text.strip()
            points = cols[2].text.strip()
            
            data.append((rank, name, points))

    # 11) Afficher les données récupérées
    print("\nTableau pour la date :", first_option)
    for row_data in data:
        print(row_data)
    # 12) Construire un DataFrame pandas
    df = pd.DataFrame(data, columns=["Rank", "Name", "Points"])



    # 14) Enregistrer le DataFrame dans un fichier CSV, 
    #     en incluant la date dans le nom du fichier
    
    df.to_csv(csv_path, index=False)

    print(f"\nDataFrame enregistré dans : {csv_path}")
    print(df.head())  # Afficher un aperçu du DataFrame

# 12) Fermer le navigateur
driver.quit()

# Dossier source contenant les fichiers CSV
source_folder = "ranking_dataframe"
# Dossier destination pour enregistrer les fichiers traitésZ
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
