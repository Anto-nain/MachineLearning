# Introduction

Pandas est un module Python utilisé pour le traitement de donnés sous forme de tableaux. Sous Pandas, les données (ou data) son stockés dans des objets de type ```DataFrame```. Pour les objets de données en 1 dimension (liste) on utilise le type ```Series```. Il est souvent important d'importer `numpy` pour pouvoir utiliser les fonctions de Pandas :


```python
import numpy as np
import pandas as pd
```
# Création des données
## Création de Series et de DataFrames par des tableaux

Une `Series` se créé à partir d'une liste (type `list`ou `np.array`) :

```python
serie = pd.Series([1, 3, 5, np.nan, 6, 8])
```
Ici `np.nan` créé une valeur manquante (not a number). 
Pour créer un `DataFrame` on peut utiliser un tableau `numpy` :

```python
data = pd.DataFrame(np.ones((3, 4))) # 3 lignes, 4 colonnes
```

Par défaut, les `Series` sont des colones. Les `Series` et les `DataFrame` possèdent deux attributs : `index` (numéros de lignes) et `columns` (numéros de colones inexistant pour `Series`). Par défaut, les `index` sont des entiers de 0 à n-1 et les `columns` sont des entiers de 0 à m-1. On peut changer ces valeurs par des listes pour obtenir de labels plus explicites :

```python
data = pd.DataFrame(np.ones((3, 4)), index=['a', 'b', 'c'], columns=['A', 'B', 'C', 'D'])
serie = pd.Series([1, 3, 5, np.nan, 6, 8], index=['a', 'b', 'c', 'd', 'e', 'f'])
```

## Création de Series et de DataFrames par des dictionnaires
Comme pour les tableaux, on peut créer des `Series` et des `DataFrame` à partir de dictionnaires. Pour les `Series`, les clés du dictionnaire deviennent les labels des lignes :

```python
serie = pd.Series({'a': 1, 'b': 3, 'c': 5, 'd': np.nan, 'e': 6, 'f': 8})
```
Pour les `DataFrame`, les clés du dictionnaire deviennent les labels des colonnes :

```python
data = pd.DataFrame(
    {
        'A': [1, 1, 1]
        'B': [1, 1, 1]
        'C': [1, 1, 1]
        'D': pd.Series([1, 1, 1])
    }, index=['a', 'b', 'c']
)
```
Bien sûr, on peut construire des `DataFrame` à partir de `Series` :

# Visualisation des données

```python	
data.head() # donne les 5 premières lignes
data.tail() # donne les 5 dernières lignes
data.index # donne les labels des lignes
data.columns # donne les labels des colonnes
data.describe() # donne des statistiques de base sur les données
```
# Sélection des données

Que l'on sélectionne des lignes ou des colonnes le résultat est du type ```Series``` ce qui s'apparente à une colonne (même si on sélectionnait une ligne donc il faut faire attention à la confusion). Si la sélection des données est un tableau alors on aura le type ```DataFrame```.

## Sélection avec les colonnes
Pour sélectionner une colonne, on va utiliser la notation dictionnaire (ou liste) :

```python
data['A'] # donne la colonne A
data.A # donne la colonne A et ne marche pas pour les colonnes avec des espaces ou des valeurs numériques
```

## Sélection avec les noms des lignes et colonnes

Pour sélectionner avec les labels, on utilise ```pd.loc``` que l'on va slicer en ligne puis en colonne que l'on souhaite obtenir:

```python
data.loc['a'] # donne la ligne a
data.loc['a', 'A'] # donne la valeur de la ligne a et de la colonne A
data.loc['a':'c'] # donne les lignes a, b et c
data.loc['a':'c', 'A':'C'] # donne les lignes a, b et c et les colonnes A, B et C
data.loc[:, 'A':'C'] # donne toutes les lignes et les colonnes A, B et C
```

## Sélection avec les numéros des lignes et colonnes
Pour sélectionner avec les index, on utilise ```pd.iloc``` on passe en argument les "slices" de numéros de ligne et de colonne que l'on souhaite obtenir:

```python
data.iloc[0] # donne la ligne 0
data.iloc[0, 0] # donne la valeur de la ligne 0 et de la colonne 0
data.iloc[0:3] # donne les lignes 0, 1 et 2
data.iloc[0:3, 0:3] # donne les lignes 0, 1 et 2 et les colonnes 0, 1 et 2
data.iloc[:, 0:3] # donne toutes les lignes et les colonnes 0, 1 et 2
```

## Obtention rapide d'une seule valeur
Pour obtenir une seule valeur, on peut utiliser ```pd.at``` ou ```pd.iat``` :

```python
data.at['a', 'A'] # donne la valeur de la ligne a et de la colonne A
data.iat[0, 0] # donne la valeur de la ligne 0 et de la colonne 0
```

## Sélection avec des tables de booléens
```python	
data["A"] > 1 # donne une table de bolléens
```
Cette opération donne la colonne A en remplaçant les valeurs plus petites ou égales à 1 par False et les autres par True. Lorsque l'on passe cette table de bolléens en argument, on obtient les lignes pour lesquelles la condition est vraie :

```python
data[data.A > 1] # donne les lignes pour lesquelles la valeur en A est plus grande que 1
```
De plus, on peut appliquer se résonnement à un tableau entier :

```python
data[data > 1] # retourne le tableau avec les valeurs plus grandes que 1 et les autres sont remplacées par NaN
```
On peut ausso utiliser la méthode ````.isin()```` pour sélectionner des valeurs dans une colonne :

```python
data["A"].isin(['nice',4]) # retourne la colonne A avec avec true si la valeur est 'nice'ou 4 et false sinon
```
**Remarque :**

Pour le données manquante, la fonction ```pd.isna(data)``` transforme le tableau en un tableau de bolléens avec True pour les valeurs manquantes et False pour les autres. On peut aussi utiliser ```pd.notna(data)``` pour obtenir le tableau inverse.

# Opérations sur les données

## Opérations "numpy"
 Si `op` est une opération sur les tableaux numpy (`.mean()` ou `.std()` ...), alors on peut l'appliquer sur les `Series` et les `DataFrame`. Pour savoir dans quel dimension :
 * `None` ou 0 : applique l'opération sur les colonnes
 * 1 : applique l'opération sur les lignes
 ## Opération quelconques
 Pour appliquer une opération quelconque, on peut utiliser la méthode `.apply(func)`. On peut préciser dans quel dimension on veut appliquer l'opération :
* `axis=0` ou `axis='index'`: applique l'opération sur les colonnes (par défaut)
* `axis=1` ou `axis='columns'`: applique l'opération sur les lignes
* `args = (arg1, arg2, ...)` : permet de passer des arguments à la fonction `func`

# Suppressions de données

Pour supprimer des données, on utilise la méthode `.drop()` qui prend en argument une liste de labels des lignes et des colonnes que l'on souhaite supprimer. par défaut, `axis=0` donc on supprime des lignes. Pour supprimer des colonnes, il faut préciser `axis=1` ou `axis='columns'`.


# Plot des données
On peut simplement utiliser la librairie matplotlib pour faire des plots (et utiliser tout ce qu'elle propose comme des histogrammes, des nuages de points etc ...):

```python
import matplotlib.pyplot as plt
plt.plot(data)
plt.show()
```

# importer des données
Pour importer des données on utilise :
```python
data = pd.read_csv('path_to_my_data.csv')
```