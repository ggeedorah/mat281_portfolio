import pandas as pd 
import numpy as np 
from lectura import lecture
from sklearn.feature_extraction.text import TfidfVectorizer

def get_director(x):
    """
    
    get_director(x)
    
    Obtiene el nombre del director desde la columna 'crew', en caso
    de que el nombre no esté listado retorna un NaN.
    
    Parameters
    ----------
    x : pd.Series
        Serie donde se encuentra el nombre del director.
    
    Returns
    ----------
    i['name'] : str
        String con el nombre del director, si no existe retorna un np.NaN
        
    """
    
    for i in x:
        if i['job'] == 'Director':
            return i['name']
        
    return np.nan


def get_list(x):
    """
    
    get_list(x)
    
    Obtiene una lista con los 3 primeros elementos de una lista, o bien,
    la lista completa en caso de que tenga 3 o menos elementos.
    
    Parameters
    ----------
    x : list
        Lista cualquiera.
        
    Returns
    ----------
    names : list
        Lista con los primeros 3 elementos de la lista o una lista vacía.
    
    """
    
    if isinstance(x, list): # Vemos si es una lista
        names = [i['name'] for i in x]
        if len(names) > 3:     # Vemos si existen más de 3 elementos
            names = names[:3]  # Si se cumple retorna los 3 primeros
        return names           # De otro modo retorna la lista entera
    
    return [] # Retorna una lista vacía en caso de datos perdidos


def clean_data(x):
    """
    
    clean_data(x)
    
    Función encargada de convertir todos los string en minúsculas y
    remueve los espacios (realiza un strip).
    
    Parameters
    ----------
    x : list, str
        Lista con strings o un string.
    
    Returns
    ---------
    list : list
        Retorna una lista con strings en minúsculas y sin espacios.
    str : str
        Si no es una lista, retorna un string en minúsculas y sin espacios.
        
    """
    
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        # Revisa si existe el director
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        # En caso de que no, retorna un string vacío
        else:
            return '' 
        
        
def create_soup(x):
    """
    
    create_soup(df)
    
    A partir de un DataFrame se crea un string con información de
    la película como palabras claves, elenco, director y géneros.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con la información de las películas.
    
    Returns
    ----------
    str : str
        String con la información de la película.
    
    """
    
    str = ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])
    
    return str
