import re
import pandas as pd

def normalize_name(name):
    """
    Convertit un nom en une version normalisée :
    - Tout en minuscules
    - Suppression de la ponctuation (par exemple, le point)
    - Suppression des espaces superflus
    """
    name = str(name).lower()                  # tout en minuscules
    name = re.sub(r'[^\w\s]', '', name)         # suppression de la ponctuation
    name = " ".join(name.split())              # suppression des espaces en trop
    return name
