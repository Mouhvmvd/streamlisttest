# -*- coding: utf-8 -*-
"""
TP_1

@author: AHIDOTE Miguel SIRI
"""
import urllib.request
import pandas as pd
import matplotlib.pyplot as plt

# 1. Télécharger la base de données Iris
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
urllib.request.urlretrieve(url, 'iris.csv')
print("Base de données Iris téléchargée.")

# 2. Charger la base de données
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
iris_data = pd.read_csv('iris.csv', header=None, names=columns)
print("Base de données Iris chargée.")
print(iris_data.head())  # Afficher les premières lignes du DataFrame pour vérifier

# 3. Manipuler les données
# Afficher les statistiques descriptives des données
print(iris_data.describe())

# Filtrer les données pour la classe 'Iris-setosa'
setosa_data = iris_data[iris_data['class'] == 'Iris-setosa']

# Calculer la moyenne des longueurs de sépales pour chaque classe
mean_sepal_length = iris_data.groupby('class')['sepal_length'].mean()
print(mean_sepal_length)

# Ajouter une nouvelle colonne pour le rapport longueur/largeur des sépales
iris_data['sepal_ratio'] = iris_data['sepal_length'] / iris_data['sepal_width']

print("Manipulations des données terminées.")
print(iris_data.head())  # Afficher les premières lignes pour vérifier

# 4. Faire des graphes
# Histogramme de la longueur des sépales
plt.figure(figsize=(10, 6))
plt.hist(iris_data['sepal_length'], bins=30, color='blue', edgecolor='black')
plt.title('Histogramme de la longueur des sépales')
plt.xlabel('Longueur des sépales')
plt.ylabel('Fréquence')
plt.show()

# Graphique en ligne de la longueur et largeur des sépales
plt.figure(figsize=(10, 6))
for species in iris_data['class'].unique():
    subset = iris_data[iris_data['class'] == species]
    plt.scatter(subset['sepal_length'], subset['sepal_width'], label=species)
plt.title('Longueur vs Largeur des Sépales')
plt.xlabel('Longueur des sépales')
plt.ylabel('Largeur des sépales')
plt.legend()
plt.show()

# Box plot des longueurs de pétales par classe
plt.figure(figsize=(10, 6))
iris_data.boxplot(column='petal_length', by='class', grid=False)
plt.title('Box plot des longueurs de pétales par classe')
plt.xlabel('Classe')
plt.ylabel('Longueur des pétales')
plt.suptitle('')  # Supprimer le titre par défaut du suplot
plt.show()
