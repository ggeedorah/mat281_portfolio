from lectura import lecture
from preprocesamiento import get_list,get_director,clean_data,create_soup
import pandas as pd 
import numpy as np 
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity 

def get_recommendations(df, title, cosine_sim, n):
    """
    
    get_recommendations(df, title, cosine_sim)
    
    Función que recibe un DataFrame con información de las películas, un string
    con el título de una película y la similitud coseno a utilizar. Retorna las
    n películas más similares de la película de título 'title'.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con información de las películas.
    title : str
        String con el nombre de la película a buscar similitud.
    cosine_sim : np.ndarray
        Similitud coseno a utilizar.
    n : int
        Entero que determina la cantidad de películas similares a obtener.
        
    Returns
    ----------
    n_sim : pd.Series
        Serie con las n películas más similares y respectivo índice.
        
    """
    df = df.reset_index()                                                 # Reiniciamos el índice del DataFrame.
    indices = pd.Series(df.index, index = df['title']).drop_duplicates()  # Se construye un mapeo inverso de índices y títulos.
    idx = indices[title]                                                  # Se obtiene el índice de la película de título 'title'.
    sim_scores = list(enumerate(cosine_sim[idx]))                         # Se obtienen las puntuaciones de similaridad de la película con las otras en pares.
    sim_scores = sorted(sim_scores, key = lambda x: x[1], reverse = True) # Se ordenan las películas según su puntuación de similaridad.
    sim_scores = sim_scores[1:(n+1)]                                      # Se obtienen las n más similares.
    movie_indices = [i[0] for i in sim_scores]                            # Se obtienen los índices de estas n películas.
    n_sim = df['title'].iloc[movie_indices]   

    return n_sim


def plot_description_based_recommender(df, title, n):
    """
    
    plot_description_based_recommender(df, title, n)
    
    Función que aplica el modelo "Plot description based Recommender",
    para esto crea un TF-IDF Vectorizer Object para crear una matriz
    TF-IDF a partir del DataFrame dado, luego se calcula la matriz 
    similitud coseno y se procede a generar una recomendación de las n
    películas más similares.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con información de las películas.
    title : str
        String con el nombre de la película a buscar similitud.
    n : int
        Entero que determina la cantidad de películas similares a obtener.
        
    Returns
    ----------
    recommendations : pd.Series
        Serie con las n películas más similares y respectivo índice.
    
    """
    
    # Se define un TF-IDF Vectorizer Object.
    tfidf = TfidfVectorizer(stop_words = 'english')        # Se remueven todas las "stop words" del inglés tales como 'the' o 'a'. 
    df['overview'] = df['overview'].fillna('')             # Se reemplazan todos los NaN con un string vacío.
    tfidf_matrix = tfidf.fit_transform(df['overview'])     # Se construye la matriz TF-IDF ajustando y transformando el DataFrame.
 
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix) # Se calcula la matriz similitud coseno

    #Retornamos una serie de n recomendaciones
    recommendations = get_recommendations(df, title, cosine_sim, n)
    
    return recommendations


def credits_genres_and_keywords_based_recommender(df, title, n):
    """
    
    credits_genres_and_keywords_based_recommender(df, title, n)
    
    Función que aplica el modelo "Plot description based Recommender",
    para esto crea un TF-IDF Vectorizer Object para crear una matriz
    TF-IDF a partir del DataFrame dado, luego se calcula la matriz 
    similitud coseno y se procede a generar una recomendación de las n
    películas más similares.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con información de las películas.
    title : str
        String con el nombre de la película a buscar similitud.
    n : int
        Entero que determina la cantidad de películas similares a obtener.
        
    Returns
    ----------
    recommendations : pd.Series
        Serie con las n películas más similares y respectivo índice.
    
    """
    
    df_copy = df.copy()
    
    # Se define una lista de carasterísticas.
    features = [
        'cast', 
        'crew', 
        'keywords', 
        'genres'
    ] 
    for feature in features: 
        df_copy[feature] = df_copy[feature].apply(literal_eval) # Se aplica "literal_eval" para analizarlas gramaticalmente.
    
    df_copy['director'] = df_copy['crew'].apply(get_director)   # Se define una nueva columan 'director' a partir de aplicar "get_director" a la columna 'crew'.

    # Se redefine una lista de características.
    features = [
        'cast', 
        'keywords', 
        'genres'
    ]
    for feature in features:
        df_copy[feature] = df_copy[feature].apply(get_list)     # Se aplica "get_list" para obtener una lista de estas características.
    
    # Se redefine una lista de características.
    features = [
        'cast', 
        'keywords', 
        'director', 
        'genres'
    ]
    for feature in features:
        df_copy[feature] = df_copy[feature].apply(clean_data)   # Se aplica "clean_data" para transformar los string en minúsculas y eliminar espacios.
        
    df_copy['soup'] = df_copy.apply(create_soup, axis = 1)      # Se crea la columna 'soup' aplicando la función "create_soup".
    
    # Se define un Count Vectorizer Object.
    count = CountVectorizer(stop_words='english')     # Se remueven todas las "stop words" del inglés tales como 'the' o 'a'. 
    count_matrix = count.fit_transform(df_copy['soup'])    # Se construye la matriz Count ajustando y transformando el DataFrame.

    cosine_sim2 = cosine_similarity(count_matrix, count_matrix) # Se calcula la matriz Count.
    
    # Se retorna una serie de n recomendaciones.
    recommendations = get_recommendations(df_copy, title, cosine_sim2, n)
    
    return recommendations
    