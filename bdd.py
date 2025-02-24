import pandas as pd
from selenium import webdriver
import time
from bs4 import BeautifulSoup


def recuperer_contenu_table_avec_selenium(driver,url):
    """
    Lance un navigateur Chrome via Selenium, va sur l'URL spécifiée,
    et renvoie le contenu du tableau <tbody> sous forme de liste de listes.
    """

    # 1. Définir le chemin vers votre chromedriver
 # À adapter

    # 2. Configurer le service et le driver

    
    
    try:
        # 3. Aller sur la page
        driver.get(url)
        
        # 4. Attendre quelques secondes que la page se charge (ou utiliser WebDriverWait)
        time.sleep(2)

        # 5. Récupérer le HTML complet de la page
        page_source = driver.page_source
    except:
        print("ça marche pas")
    
    # 6. Parser le HTML avec BeautifulSoup
    soup = BeautifulSoup(page_source, 'html.parser')
    
    # 7. Trouver le tableau ou le <tbody> qui nous intéresse
    #    (Ici, on cherche le premier <tbody> trouvé, mais on peut être plus précis)
    tableau = soup.find('tbody')
    if not tableau:
        print("Aucun <tbody> trouvé.")
        return []
    
    # 8. Récupérer toutes les lignes <tr> et leurs cellules <td>
    donnees = []
    lignes = tableau.find_all('tr')
    
    for ligne in lignes:
        cellules = ligne.find_all('td')
        valeurs_ligne = [cellule.get_text(strip=True) for cellule in cellules]
        donnees.append(valeurs_ligne)
    
    return donnees


def ajouter_stats_joueur(player_name, data,player_stats_dict):
    """
    Ajoute les statistiques d'un joueur dans le dictionnaire global.
    
    Parameters:
        player_name (str): Le nom du joueur (ex: "nicolas-kiefer").
        data (list of lists): La liste de listes contenant les statistiques.
                              La première ligne doit contenir les en-têtes.
    
    Returns:
        pd.DataFrame: Le DataFrame créé à partir des données.
    """
    # Création du DataFrame en utilisant la première ligne comme en-têtes
    df_stats = pd.DataFrame(data[1:], columns=data[0])
    
    # Ajout du DataFrame dans le dictionnaire, indexé par le nom du joueur
    player_stats_dict[player_name] = df_stats
    
    # Retourner le DataFrame créé (optionnel)
    return player_stats_dict

