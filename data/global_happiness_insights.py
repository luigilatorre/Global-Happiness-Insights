# %% [markdown]
# # Final Assignment

# %%
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
sns.set(rc={'figure.figsize':(15, 9)})
sns.set(font_scale=1.5) 

# %% [markdown]
# Il dataset `world_happiness.csv` contenuto nella cartella `data` presenta una serie di variabili che possono essere utilizzate come proxy per la valutazione del benessere di un paese. La metrica `happiness_score` cerca di riassumere quanto sia "felice" ciascun paese. Carica e salva il dataset in un oggetto DataFrame chiamato `happy`. Come sempre, familiarizzati con il suo contenuto. 

# %%
happy = pd.read_csv(r"C:\Program Files\Python312\python-homework\myspace\homework\final-assignment-ML-webscr\data\world_happiness.csv")
happy.info()

# %%
nan_values_transpose = happy.isna().transpose().copy()
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(nan_values_transpose, ax=ax, cmap="viridis")
plt.show()

# %%
# Cerco quali valori nulli sono presenti nel DataFrame, essendo un massimo di 8 righe che li contengono mostro con un .head(10)
nulls = happy[happy.isnull().any(axis=1)]
nulls.head(10)

# %%
# BONUS - Ho 8 valori vuoti di "corruption", 1 di "social_support", 1 di "freedom", 1 di "generosity"
# Voglio riempire i valori vuoti con medie più attinenti ed effettuo un data augmentation raggruppando i paesi per regione
# cosi effettuerò una media più precisa per paesi limitrofi a quelli vuoti.
# I NaN da trattare saranno racchiusi in una colonna region e saranno "Medio Oriente", "Asia"
regioni_geografiche = {
    'Europa': [
        'Finland', 'Denmark', 'Norway', 'Iceland', 'Netherlands', 'Switzerland', 'Sweden', 'Austria',
        'Luxembourg', 'United Kingdom', 'Ireland', 'Germany', 'Belgium', 'Czech Republic', 'Malta',
        'France', 'Spain', 'Italy', 'Poland', 'Lithuania', 'Slovenia', 'Romania', 'Cyprus', 'Latvia',
        'Estonia', 'Portugal', 'Hungary', 'Serbia', 'Moldova', 'Montenegro', 'Croatia', 'Turkey',
        'Belarus', 'Greece', 'Bulgaria', 'Albania', 'Armenia', 'Georgia', 'Ukraine'
    ],
    'Stati Uniti e Canada': [
        'United States', 'Canada'
    ],
    'America Latina': [
        'Costa Rica', 'Chile', 'Guatemala', 'Panama', 'Brazil', 'Uruguay', 'El Salvador',
        'Trinidad and Tobago', 'Argentina', 'Ecuador', 'Bolivia', 'Honduras', 'Paraguay',
        'Peru', 'Colombia', 'Nicaragua', 'Jamaica'
    ],
    'Medio Oriente': [
        'Israel', 'United Arab Emirates', 'Saudi Arabia', 'Qatar', 'Bahrain', 'Kuwait', 'Lebanon'
    ],
    'Asia': [
        'New Zealand', 'Singapore', 'South Korea', 'Japan', 'Kazakhstan', 'Thailand', 'Malaysia',
        'Uzbekistan', 'Philippines', 'Mongolia', 'Pakistan', 'Vietnam', 'China', 'Bangladesh',
        'Sri Lanka', 'Myanmar', 'Azerbaijan', 'Iraq', 'Iran', 'Syria', 'Jordan', 'India',
        'Nepal', 'Bhutan', 'Cambodia', 'Indonesia', 'Turkmenistan', 'Afghanistan'
    ],
    'Africa': [
        'South Africa', 'Nigeria', 'Algeria', 'Morocco', 'Libya', 'Tunisia', 'Egypt', 'Kenya',
        'Ethiopia', 'Ghana', 'Cameroon', 'Uganda', 'Tanzania', 'Mozambique', 'Mauritius', 'Mauritania',
        'Mali', 'Senegal', 'Zimbabwe', 'Zambia', 'Namibia', 'Rwanda', 'Burundi', 'Botswana',
        'Malawi', 'Somalia', 'Niger', 'Burkina Faso', 'Gabon', 'Benin', 'Guinea', 'Sierra Leone',
        'Liberia', 'Madagascar', 'Comoros', 'Lesotho', 'Eswatini', 'Central African Republic',
        'Gambia', 'Chad', 'Togo', 'South Sudan'
    ]
}

# Funzione per mappare ciascun paese alla sua regione geografica
def mappa_regione(paese):
    for regione, paesi in regioni_geografiche.items():
        if paese in paesi:
            return regione
    return 'Altro'  # Genererà ciò che non è stato ben filtrato ma non interessa i dati NaN

# Applico la funzione per creare una nuova colonna 'region'
happy['region'] = happy['country'].apply(mappa_regione)

# Se si vuole spostare la colonna region prima di country (comodità visiva). Commento e lancio solo per me
# col_name = happy.columns.tolist()
# happy = happy[[col_name[-1]] + col_name[:-1]]

happy.head()

# %%
# Calcola la media per la regione "Medio Oriente" delle colonne con valori nulli
media_medio_oriente = happy[happy['region'] == 'Medio Oriente'][['social_support', 'freedom', 'corruption', 'generosity']].mean()
media_medio_oriente.head()

# %%
# Calcola la media per la regione "Asia" delle colonne con valori nulli (solo corruption 3 valori)
media_asia = happy[happy['region'] == 'Asia'][['corruption']].mean()
media_asia.head()

# %%
# Sostituisci i valori mancanti con la media calcolata dei paesi che fanno parte del Medio Oriente.
happy.loc[happy['region'] == 'Medio Oriente', 'social_support'] = happy.loc[happy['region'] == 'Medio Oriente', 'social_support'].fillna(64.00)
happy.loc[happy['region'] == 'Medio Oriente', 'freedom'] = happy.loc[happy['region'] == 'Medio Oriente', 'freedom'].fillna(62.00)
happy.loc[happy['region'] == 'Medio Oriente', 'corruption'] = happy.loc[happy['region'] == 'Medio Oriente', 'corruption'].fillna(103.00)
happy.loc[happy['region'] == 'Medio Oriente', 'generosity'] = happy.loc[happy['region'] == 'Medio Oriente', 'generosity'].fillna(41.50)

# Sostituisci i valori mancanti con la media calcolata dei paesi che fanno parte dell'Asia. Arrotondo per intero il valore 66.4 in 66.0
happy.loc[happy['region'] == 'Asia', 'corruption'] = happy.loc[happy['region'] == 'Asia', 'corruption'].fillna(66.00)
happy.head()

# %%
# 1. Utilizzando un istogramma, tracciamo la distribuzione della variabile `happiness_score`. **Quale distribuzione somiglia?**

happiness_score = happy['happiness_score']

plt.figure(figsize=(10, 6))
sns.histplot(happiness_score, bins=20, color='crimson', kde=True)
plt.title('Distribuzione dei punteggi di felicità')
plt.xlabel('Happiness Score')
plt.ylabel('Frequenza')
plt.grid(True)
plt.show()

# Sembra una distribuzione uniforme continua ma con una piccola deviazione della frequenza sui valori minimi e massimi schiacciando un po' quelli centrali
happiness_score.mean()

# %% [markdown]
# 2. Traccia una matrice di correlazione (o una heatmap di correlazione) tra tutte le variabili numeriche del dataset. **Quale variabile è meno correlata con la metrica `happiness_score`?**

# %%
cols_numeriche = ['social_support', 'freedom', 'corruption', 'generosity', 'gdp_per_cap', 'life_exp', 'happiness_score']
subset = happy[cols_numeriche]
correlation_matrix = subset.corr()

plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()

# Le variabili più correlate con "happiness_score" sono:
# 1) life_exp --> 0.78
# 2) gdp_per_cap --> 0.72

# %%
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Test di multicollinearità tra le variabili numeriche
cols_numeriche = ['social_support', 'freedom', 'corruption', 'generosity', 'gdp_per_cap', 'life_exp', 'h
