import matplotlib.pyplot as plt
import seaborn as sns 
    
def heatmap(df):
    """
    
    heatmap(df)
    
    Crea un mapa de calor a partir de un DataFrame que sólo 
    posee elementos numéricos.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con elementos numéricos.
    
    Returns
    ----------
    None
    
    """
    
    plt.figure(figsize = (12, 12))
    
    corr = df.corr()
    sns.heatmap(corr.round(2), cmap = "viridis", annot = True)
    
    plt.show()
    
    
def histogram(df):
    """
    
    histogram(df)
    
    Crea un histograma para cada columna de un DataFrame que sólo 
    posee elementos numéricos.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con elementos numéricos.
    
    Returns
    ----------
    None
    
    """
    
    fig, axes = plt.subplots(3, 2, figsize = (18, 12))
    colors = [
        'royalblue',
        'green',
        'crimson',
        'orangered',
        'rebeccapurple',
        'gold'
    ]
    for i in range(len(df.columns)):
        sns.distplot(
            df[df.columns[i]], 
            hist = True, 
            kde = True,
            ax = axes[i%3, i%2], 
            label = df.columns[i],
            color = colors[i]
        )
        
    plt.show()
    
    
def top_pop(df, n):
    """
    
    top_pop()
    
    Crea un gráfico de barras de las n mejores películas según
    popularidad.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con la columna 'popularity'.
    n : int
        Cantidad de películas en el gráfico de barras.
    
    Returns
    ----------
    None
    
    """
    
    df_pop = df.sort_values(by = 'popularity', ascending = False)
    plt.figure(figsize = (10,10))
    plt.barh(
        df_pop['title'].head(n),
        df_pop['popularity'].head(n), 
        align = 'center',
        color = 'crimson'
    )
    plt.gca().invert_yaxis()
    plt.xlabel("Popularidad")
    
    plt.show()
 
