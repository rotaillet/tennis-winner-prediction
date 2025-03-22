from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException
import pandas as pd
from multiprocessing import Pool
import time
from tqdm import tqdm

def odds():
    driver = webdriver.Chrome()
    odd = []
    df = pd.read_csv("href.csv")
    for href in tqdm(df["href"], desc="Traitement des liens"):
        driver.get(href)
        time.sleep(0.25)

        try:
            # Récupère un élément dont la classe contient 'cellWinner'
            cell = driver.find_element(By.CSS_SELECTOR, "[class*='cellWinner']")
            # À l’intérieur, récupère l’élément avec la classe "oddsValueInner"
            odds_element = cell.find_element(By.CSS_SELECTOR, ".oddsValueInner")
            odds_text = odds_element.text
        except:
            odds_text = None

        odd.append(odds_text)

            
        
    driver.quit()

    df["odd"] = odd
    print(df)

    df.to_csv("href_odd.csv")

df = pd.read_csv("href_odd.csv")
na_par_colonne = df.isna().sum()
print("Nombre de NA par colonne :")
print(na_par_colonne)
df.dropna(inplace=True)
print(df[["odd","predicted_probability"]].describe())