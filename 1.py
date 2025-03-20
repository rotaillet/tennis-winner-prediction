from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException
import time
import pandas as pd
from tqdm import tqdm
import re
import numpy as np
from multiprocessing import Pool
import datetime
import os
from urllib.parse import urlparse



# --- Variables globales pour chaque processus ---
global global_driver
global_driver= webdriver.Chrome()

def convert_time_to_minutes(time_string):
    # time_string est au format "H:MM"
    hours_str, minutes_str = time_string.split(":")
    hours = int(hours_str)
    minutes = int(minutes_str)
    total_minutes = hours * 60 + minutes
    return total_minutes

def init_driver():
    """Initialise le driver une fois par processus."""
    global global_driver
    global_driver = webdriver.Chrome()

def all_tounament():
    driver = webdriver.Chrome()
    driver.get("https://www.flashscore.fr/tennis/")  # Remplace par l'URL de la page qui contient tes lmc__template
    time.sleep(2)

    # On attend que tous les <span class="lmc__template"> soient présents
    template_spans = WebDriverWait(driver, 10).until(
        EC.presence_of_all_elements_located((By.CSS_SELECTOR, "span.lmc__template"))
    )

    # Pour chaque <span class="lmc__template">, on cherche l’<a> qui contient le href
    hrefs = []
    for span in template_spans:
        try:
            # Soit on récupère directement le <a> par sa classe
            link_element = span.find_element(By.CSS_SELECTOR, "a.lmc__templateHref")
            link = link_element.get_attribute("href")
            hrefs.append(link)
        except:
            # Si jamais il n’y a pas de <a> à l’intérieur (ou autre cas)
            pass

    # Vérification ou affichage
    for h in hrefs:
        print(h)
    

    # Conversion de la liste en DataFrame pandas
    df = pd.DataFrame(hrefs, columns=["href"])

    # Enregistrement dans un fichier CSV
    df.to_csv("tournois.csv", index=False, encoding="utf-8")
    driver.quit()

    return df

def all_tournament_since_2020(df,min_year):
    driver = webdriver.Chrome()
    links = []
    for i in range(len(df)):
        print(i)
        href = df["href"][i]
        driver.get(href+"archives/")
        time.sleep(0.75)
        rows = driver.find_elements(By.CSS_SELECTOR, "div.archive__row")

        
        for row in rows:
            try:
                # À l'intérieur de chaque row, on cherche la div "archive__season"
                season_div = row.find_element(By.CSS_SELECTOR, "div.archive__season")
                
                # Puis on récupère le <a> qui contient le texte (ex: "ATP Belgrade 2022")
                anchor = season_div.find_element(By.CSS_SELECTOR, "a.archive__text.archive__text--clickable")
                
                # Extrait le texte, ex : "ATP Belgrade 2022"
                link_text = anchor.text.strip()
                # Extrait l'href
                link_href = anchor.get_attribute("href")
                
                # Méthode 1 : on suppose que l'année est le dernier mot
                # tokens = link_text.split()
                # year_str = tokens[-1]  # "2022" dans "ATP Belgrade 2022"

                # Méthode 2 : expression régulière pour trouver un nombre 4 chiffres
                match = re.search(r"\b(\d{4})\b", link_text)
                if match:
                    year = int(match.group(1))
                    if year >= min_year:
                        links.append(link_href)

            except Exception as e:
                # Si on ne trouve pas la div, l'a, etc., on passe
                pass

    driver.quit()
    df = pd.DataFrame(links, columns=["links"])
    df.to_csv("all_tournois.csv", index=False, encoding="utf-8")
    return links







def links(tournois,df1):
    """
    Récupère les liens pour chaque saison et les enregistre dans un CSV.
    """
    existing_links = set(df1['href'].tolist())
    
    all_hrefs = []
    driver = webdriver.Chrome()
    # Parcours des saisons de 2012-2013 à 2024-2025
    for tour in tournois:
        
        
        
        url = f"{tour}resultats/"

        
        driver.get(url)
        time.sleep(0.5)  # Laisser le temps au chargement
        actions = ActionChains(driver)
        wait = WebDriverWait(driver, 5)
        
        # Cliquer sur "Montrer plus de matchs" tant que possible
        while True:
            try:
                button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "a.event__more.event__more--static")))
                actions.move_to_element(button).perform()
                time.sleep(0.25)
                button.click()
                time.sleep(0.25)
            except Exception:
                print("Plus de bouton 'Montrer plus de matchs' disponible pour cette saison.")
                break

        # Récupérer tous les liens de matchs
        matches = driver.find_elements(By.CSS_SELECTOR, "div.event__match")
        for match in matches:
            anchors = match.find_elements(By.TAG_NAME, "a")
            for a in anchors:
                link = a.get_attribute("href")
                if link and link not in existing_links:
                    all_hrefs.append((tour, link))
                    
    driver.quit()
    df = pd.DataFrame(all_hrefs, columns=["tour", "href"])
    df.to_csv("test.csv", index=False)
    print("Fin de l'extraction des liens. Le fichier CSV a été enregistré.")
    return df

def process_match(match_tuple):
    """
    Pour un match donné (saison, href), cette fonction :
      - Ouvre une instance Selenium
      - Extrait le score complet et les statistiques via l'URL "/statistiques-du-match/1"
      - Extrait le score mi-temps via l'URL de base
      - Ferme le driver et renvoie un dictionnaire des résultats.
    """
    tour, href = match_tuple
    result = {"tournois": tour, "href": href,
              "j1": None, "j2": None,"time":None,
              "score_j1":None,"score_j2":None,"sets":None,"date":None,"tour":None,"surface":None}
    global global_driver    
    wait = WebDriverWait(global_driver, 5)
    try:
        # --- Extraction des statistiques et score complet ---
        try:
            global_driver.get(href + "/statistiques-du-match/0")
            time.sleep(0.75)
        
            try:
                stat_rows = global_driver.find_elements(By.CSS_SELECTOR, 'div[data-testid="wcl-statistics"]')
                for row in stat_rows:
                    try:
                        tab = row.find_elements(By.CSS_SELECTOR, '[data-testid="wcl-scores-simpleText-01"]')
                        if len(tab) >= 3:
                            category_name = tab[1].text  # ex: "Possession"
                            home_value = tab[0].text     # ex: "52%"
                            away_value = tab[2].text     # ex: "48%"
                            # Création dynamique des colonnes selon la catégorie
                            col_name_home = category_name.replace(" ", "_") + "_j1"
                            col_name_away = category_name.replace(" ", "_") + "_j2"
                            result[col_name_home] = home_value
                            result[col_name_away] = away_value
                    except NoSuchElementException:
                        pass
            except NoSuchElementException:
                print(f"Tableau des statistiques introuvable pour {href}")

            try :
                a_element = global_driver.find_elements(By.CSS_SELECTOR, "a.participant__participantName.participant__overflow")
                hrefs = a_element[0].get_attribute("href")  
                parts = hrefs.split("/")  
                slug = parts[4]          
                player_name = slug.replace("-", " ").title()  
                result["j1"] = player_name
                
                hrefs = a_element[1].get_attribute("href") 
                parts = hrefs.split("/")  
                slug = parts[4]          
                player_name = slug.replace("-", " ").title()  
                result["j2"] = player_name

            except:
                print(f"Erreur lors du chargement des nom pour {href}: {e}")

            
            try :
                scores = global_driver.find_elements(By.CSS_SELECTOR, "div.detailScore__wrapper")
                
                score = scores[0].text
                result["score_j1"] = score[0]
                result["score_j2"] = score[4]


            except:
                print(f"Erreur lors du chargement des scores pour {href}: {e}")


        except Exception as e:
            print(f"Erreur lors du chargement des stats pour {href}: {e}")

        try :
            global_driver.get(href)
            time.sleep(0.75)
            times = global_driver.find_element(By.CSS_SELECTOR, "div.smh__time.smh__time--overall")
            times_txt = times.text
            times_txt = convert_time_to_minutes(times_txt)
            result["time"] = times_txt

            try :
                
                home_sets = global_driver.find_elements(By.CSS_SELECTOR, "div.smh__part.smh__home")
                away_sets = global_driver.find_elements(By.CSS_SELECTOR, "div.smh__part.smh__away")
                # Nombre total de sets disponibles
                points_str = ""
                for i in range (1,11,2):
                    
                    element = home_sets[i]
                    # Le score de base est dans le div (en ignorant le sup)
                    base_score = element.get_attribute("innerText").strip()
                    # Souvent, base_score ressemblera à "6\n4" si le sup est renvoyé sur une autre ligne.
                    
                    # On récupère le texte exact de la balise <sup>
                    sup_element = element.find_element(By.TAG_NAME, "sup")
                    tie_break = sup_element.text.strip()
                    
                    # On enlève la partie tie-break du base_score
                    # Généralement, le texte avant le premier saut de ligne correspond au score principal
                    base_score = base_score.split("\n")[0].strip()  # "6"
                    
                    final_score = f"{base_score}({tie_break})"      # "6(4)"
                    if tie_break =='':
                        final_score = element.text.strip()  # par ex. "6"
                    
                   
                    element2 = away_sets[i]
                    # Le score de base est dans le div (en ignorant le sup)
                    base_score2 = element2.get_attribute("innerText").strip()
                    # Souvent, base_score ressemblera à "6\n4" si le sup est renvoyé sur une autre ligne.
                    
                    # On récupère le texte exact de la balise <sup>
                    sup_element2 = element2.find_element(By.TAG_NAME, "sup")
                    tie_break2 = sup_element2.text.strip()
                    
                    # On enlève la partie tie-break du base_score
                    # Généralement, le texte avant le premier saut de ligne correspond au score principal
                    base_score2 = base_score2.split("\n")[0].strip()  # "6"
                    
                    final_score2 = f"{base_score2}({tie_break2})"      # "6(4)"
                    if tie_break2 =='':
                        # S'il n'y a pas de tie-break, on prend le texte entier
                        final_score2 = element2.text.strip()  # par ex. "6"
                    if final_score!='':
                        points_str += f"{final_score}-{final_score2} "

                result["sets"] = points_str


                    

            except:
                print(f"Erreur lors du chargement des scores pour {href}: {e}")

            try :
                date = global_driver.find_element(By.CSS_SELECTOR, "div.duelParticipant__startTime").text
                result["date"] = date
            except:
                print(f"Erreur lors du chargement des scores pour {href}: {e}")

            try :
                element = global_driver.find_element(By.CSS_SELECTOR, "span.tournamentHeader__country a")
                link_text = element.text
                pattern = r".*?,\s*(.*?)\s*-\s*(.*)"
                match = re.match(pattern, link_text)
                if match:
                    surface = match.group(1)       # correspond à (.*?) entre la virgule et le tiret
                    match_round = match.group(2)   # correspond à (.*) après le tiret
                    result["surface"] = surface
                    result["tour"] = match_round


            except:
                print(f"Erreur lors du chargement des scores pour {href}: {e}")

        except Exception as e:
            print(f"Erreur lors du chargement des stats pour {href}: {e}")

   



    
        
        # --- Extraction du score mi-temps ---
    
    finally:
        print("fin")
        
    return result

def merge(df1, df2):
    """
    Concatène verticalement deux DataFrames et enregistre le résultat dans un CSV.
    """
    df_merged = pd.concat([df1, df2], axis=0, ignore_index=True)
    df_merged.to_csv("merged_data.csv", index=False)
    return df_merged

def nettoyage(df):
    """
    Nettoyage du DataFrame : suppression de colonnes inutiles, remplissage et suppression des valeurs manquantes.
    """
    cols_to_drop = ["Vitesse_moyenne_du_second_service_j2","Distance_couverte_(mètres)_j1","Distance_couverte_(mètres)_j2",
                    "Vitesse_moyenne_du_second_service_j1","Vitesse_moyenne_du_1er_service_j2",
                    "Vitesse_moyenne_du_1er_service_j1","Dix_derniers_points_j1","Dix_derniers_points_j2","Points_Gagnants_j2","Points_Gagnants_j1",
                    "Fautes_Directes_j2","Fautes_Directes_j1","Points_Gagnés_au_Filet_j1","Points_Gagnés_au_Filet_j2","Balles_de_match_sauvées_j1","Balles_de_match_sauvées_j2"]
    df.drop(columns=cols_to_drop, axis=1, inplace=True, errors='ignore')
    
    df.dropna(inplace=True)
    df.reset_index(inplace=True, drop=True)
    df.to_csv("matchs_utilisable.csv", index=False)
    return df

def ranking(df, dos="ranking_dataframe"):
    # Conversion des dates du DataFrame en datetime
    df["date"] = pd.to_datetime(df["date"], format='%d.%m.%Y %H:%M')
    df["date"] = df["date"].apply(lambda dt: datetime.datetime(dt.year, dt.month, dt.day))



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
        current_date = df["date"][i]
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
    df = df.sort_values(by="date").reset_index(drop=True)

    return df

def extract_tournament(url):
    # Extraction du chemin de l'URL
    parsed = urlparse(url)
    path_parts = parsed.path.strip('/').split('/')
    
    # Structure attendue : tennis/atp-simples/tournoi (ex: adelaide-2020 ou adelaide)
    if len(path_parts) >= 3:
        tournament = path_parts[2]
        # Supprimer le suffixe "-année" s'il existe (ex: "-2020")
        tournament = re.sub(r'-\d+$', '', tournament)
        return tournament
    return None
def pretraitement(df):
    
    rank1, rank2, age1, age2, point1, point2 = [], [], [], [], [], []
    
    for i in tqdm(range(len(df)), desc="Extraction en cours"):  # Ajout de tqdm
        df_rank = pd.read_csv(f"ranking_dataframe/{df["Closest_File"][i]}")

        ra1,ra2,a1,a2,p1,p2 = None,None,None,None,None,None
        for j in range(len(df_rank)):
            if set(df["j1"][i].lower().split()) == set(df_rank['Player'][j].lower().split()):
                ra1 = df_rank['Rank'][j]
                a1 = df_rank['Age'][j]
                p1 = df_rank['Points'][j]

            if set(df["j2"][i].lower().split()) == set(df_rank['Player'][j].lower().split()):
                ra2 = df_rank['Rank'][j]
                a2 = df_rank['Age'][j]
                p2 = df_rank['Points'][j]

                
        rank1.append(ra1)
        rank2.append(ra2)
        age1.append(a1)
        age2.append(a2)
        point1.append(p1)
        point2.append(p2)
    df["rank1"]=rank1
    df["rank2"]=rank2
    df["age1"]=age1
    df["age2"]=age2
    df["point1"]=point1
    df["point2"]=point2
    max_values = df[["rank1", "rank2"]].max()
    df[["rank1", "rank2"]] = df[["rank1", "rank2"]].fillna(max_values.max()+1)

    df[["point1", "point2"]] = df[["point1", "point2"]].fillna(0)

    mean_values = df[["age1", "age2"]].mean()
    df[["age1", "age2"]] = df[["age1", "age2"]].fillna(mean_values.mean())
    df["tournament"] = df["tournois"].apply(extract_tournament)
    df.to_csv("matchs_rank.csv", index=False)
    return df

def parse_value(val):
    """
    Extrait la valeur en pourcentage et, si présente, le numérateur et le dénominateur.
    Exemple :
      - "84% (48/57)" -> (84, 48, 57)
      - "74%" -> (74, None, None)
    """
    pattern = r'(\d+)%(?:\s*\((\d+)/(\d+)\))?'
    match = re.match(pattern, str(val))
    if match:
        perc = int(match.group(1))
        num = int(match.group(2)) if match.group(2) is not None else None
        den = int(match.group(3)) if match.group(3) is not None else None
        return perc, num, den
    return None, None, None

def process_percent_column(series):
    """
    Traite une colonne de pourcentages sous forme de chaîne.
    Si au moins une cellule comporte une fraction, on renvoie trois colonnes.
    Sinon, on renvoie une seule colonne avec le pourcentage.
    """
    # Appliquer la fonction d'extraction sur chaque cellule de la série
    parsed = series.apply(lambda x: parse_value(x))
    perc = parsed.apply(lambda t: t[0])
    num = parsed.apply(lambda t: t[1])
    den = parsed.apply(lambda t: t[2])
    
    # Si au moins une cellule comporte une fraction (num ou den non null),
    # on crée trois colonnes. Sinon, une seule.
    if num.notnull().any() or den.notnull().any():
        df_new = pd.DataFrame({
            series.name + '_perc': perc,
            series.name + '_num': num,
            series.name + '_den': den
        })
    else:
        df_new = pd.DataFrame({
            series.name + '_perc': perc
        })
    return df_new

def drop_elements(df):
    columns_with_percent = [col for col in df.columns if df[col].astype(str).str.contains('%').any()]
    
    

    # Appliquer le traitement sur chaque colonne et récupérer les nouveaux DataFrames
    new_columns = [process_percent_column(df[col]) for col in columns_with_percent]

    # Concaténer le résultat avec le DataFrame original si besoin
    df_final = pd.concat([df] + new_columns, axis=1)
    
    df_final.drop(['tournois',"Closest_File","sets"], axis=1, inplace=True)
    df_final.drop(columns_with_percent, axis=1, inplace=True)
    df['score_j1'] = df['score_j1'].astype(float)

    df_final['winner'] = np.where(df['score_j1'] > df['score_j2'], df["j1"],
                        np.where(df['score_j1'] < df['score_j2'], df["j2"], 'égalité'))
    df_final.to_csv('all_features_bonus.csv')

    return df_final

def fonction_un_nom(df):
    

# Paramètres Elo
    starting_elo = 1500
    K = 32  # Facteur de sensibilité

    # Dictionnaire pour stocker les ratings de chaque joueur
    elo_ratings = {}

    def get_elo(player):
        # Retourne le rating actuel d'un joueur, ou 1500 s'il n'a pas encore joué
        return elo_ratings.get(player, starting_elo)

    # Listes pour stocker l'Elo AVANT mise à jour pour chaque match
    elo_j1 = []
    elo_j2 = []

    # Parcours du dataset match par match
    for idx, row in df.iterrows():
        player1 = row["j1"]
        player2 = row["j2"]
        winner = row["winner"]

        # Récupérer les ratings actuels (avant le match)
        current_R1 = get_elo(player1)
        current_R2 = get_elo(player2)

        # Stocker ces valeurs pour le match courant (pour éviter le leak)
        elo_j1.append(current_R1)
        elo_j2.append(current_R2)

        # Calcul des scores attendus
        E1 = 1 / (1 + 10 ** ((current_R2 - current_R1) / 400))
        E2 = 1 / (1 + 10 ** ((current_R1 - current_R2) / 400))

        # Scores réels : 1 pour la victoire, 0 pour la défaite
        S1 = 1 if winner == player1 else 0
        S2 = 1 if winner == player2 else 0

        # Mise à jour des ratings Elo
        new_R1 = current_R1 + K * (S1 - E1)
        new_R2 = current_R2 + K * (S2 - E2)

        # Mise à jour du dictionnaire pour le prochain match
        elo_ratings[player1] = new_R1
        elo_ratings[player2] = new_R2

    # Ajout des colonnes dans le DataFrame
    df["elo_j1"] = elo_j1
    df["elo_j2"] = elo_j2


    surface_mapping = {
        "DUR": 1,
        "TERRE BATTUE": 2,
        "DUR (INDOOR)": 3,
        "GAZON": 4
    }

    # Nettoyer la colonne "surface" pour uniformiser les valeurs
    df['surface'] = df['surface'].str.strip().str.upper()

    # Appliquer le mapping et créer une nouvelle colonne "surface_encoded"
    df['surface_encoded'] = df['surface'].map(surface_mapping)


    tour_mapping = {
        "1/16 DE FINALE":1,
        "DEMI-FINALES":2,
        "QUARTS DE FINALE":3,
        "1/8 DE FINALE":4,
        "1/32 DE FINALE":5,
        "1/64 DE FINALE":6,
        "FINALE":7,
        "3E PLACE":8
    }

    # Nettoyer la colonne "surface" pour uniformiser les valeurs
    df['tour'] = df['tour'].str.strip().str.upper()

    # Appliquer le mapping et créer une nouvelle colonne "surface_encoded"
    df['tour_encoded'] = df['tour'].map(tour_mapping)


    df.to_csv("all_features_bonus.csv",index=False)



if __name__ == '__main__':
    
    driver = webdriver.Chrome()
   
    try:
        driver.get("https://www.flashscore.fr/tennis/")
        time.sleep(5)  # Laissez le temps à la page de charger
        b = False

        # 1) Identifier le conteneur global (ou un conteneur pertinent)
        #    Sur Flashscore, la structure peut varier, adaptez si besoin :
        container = driver.find_element(By.CLASS_NAME, "sportName")  
        # Par exemple, "sportName" contient généralement la liste des matchs

        # 2) Récupérer tous les <div> enfants directs dans ce conteneur
        all_divs = container.find_elements(By.XPATH, "./div")

        # 3) Parcourir chaque <div> dans l'ordre
        for div in all_divs:
            # Vérifier si on est tombé sur l'élément "header admin v-leagueheader collapsed"
            # On s'arrête alors.
            classes = div.get_attribute("class")
            if "wcl-header_uBhYi wclLeagueHeader wclLeagueHeader--collapsed" in classes and b==False:
                b = True  # On ne va pas plus loin
                continue
            if "wcl-header_uBhYi wclLeagueHeader wclLeagueHeader--collapsed" in classes and b==True:
                break  # On ne va pas plus loin

            # 4) Sinon, si l'id commence par "g_2_", on récupère tous les <a> qu'il contient
            div_id = div.get_attribute("id")
            if div_id and div_id.startswith("g_2_"):
                # Extraire tous les liens
                links = div.find_elements(By.TAG_NAME, 'a')
                for link in links:
                    href_value = link.get_attribute('href')
                    if href_value:
                        print(href_value)

    finally:
        driver.quit()
            