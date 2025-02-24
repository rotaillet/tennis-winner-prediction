from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
import time
import pandas as pd
import os 

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

for i in range(200,220):
    first_option = option_values[i]

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

    # 13) Créer le dossier "rankin_dataframe" s'il n'existe pas
    os.makedirs("rankin_dataframe", exist_ok=True)

    # 14) Enregistrer le DataFrame dans un fichier CSV, 
    #     en incluant la date dans le nom du fichier
    csv_filename = f"ranking_{first_option}.csv"
    csv_path = os.path.join("rankin_dataframe", csv_filename)
    df.to_csv(csv_path, index=False)

    print(f"\nDataFrame enregistré dans : {csv_path}")
    print(df.head())  # Afficher un aperçu du DataFrame

# 12) Fermer le navigateur
driver.quit()
