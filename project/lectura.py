import pandas as pd 
import numpy as np 
    
def lecture() -> pd.DataFrame:
    """
    
    lecture()
    
    Transforma los archivos "tmdb_500_credits.csv" y "tmbd_5000_movies.csv"
    en pd.DataFrame, para luego unirlos a partir de la columna 'id' y retornar
    un pd.DataFrame.
    
    Parameters
    ----------
    None
    
    Returns
    ----------
    df : pd.Dataframe
        Dataframe producto de la unión de los DataFrame de créditos y películas

    """
    
    df1 = pd.read_csv('data/tmdb_5000_credits.csv')
    df2 = pd.read_csv('data/tmdb_5000_movies.csv')
    df1.columns = [
        'id',
        'tittle',
        'cast',
        'crew'
    ]
    df = df2.merge(df1, on = 'id').drop(['tittle'], axis = 1)
    
    return df
    
def num_features(df) -> pd.DataFrame:
    """
    
    num_features(df)
    
    Obtiene un DataFrame solo con los elementos númericos de un
    DataFrame dado y elimina la columna "id" de éste.
    
    Parameters
    ----------
    df: pd.DataFrame
        DataFrame con distintos tipos de datos.
        
    Returns
    ----------
    df_num: pd.DataFrame
        DataFrame sólo con tipos de datos int y float.
    
    """
    
    df_num = df[list(df.dtypes[df.dtypes != "object"].index)]
    df_num = df_num.drop(["id"], axis = 1)
    
    return df_num