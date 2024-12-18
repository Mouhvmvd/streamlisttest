import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Titre de l'application
st.title("Analyse des données Iris")

# Charger les données
try:
    df = pd.read_csv("Iris.csv")
except FileNotFoundError:
    st.error("Le fichier Iris.csv est introuvable. Assurez-vous qu'il est présent dans le dossier de travail.")
    st.stop()

# Vérification de colonnes numériques
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

if numeric_columns.empty:
    st.error("Aucune colonne numérique disponible dans le fichier.")
    st.stop()

# Aperçu des données
st.subheader("Aperçu des données")
st.dataframe(df)

# Affichage des lignes personnalisées
st.subheader("Affichage personnalisé des données")
rows_to_display = st.slider(
    "Nombre de lignes à afficher :", 
    min_value=5, max_value=len(df), value=10, step=5
)
st.write(df.head(rows_to_display))

# Statistiques descriptives
st.subheader("Statistiques descriptives")
st.write(df.describe())

# Sélection des colonnes pour visualisation
st.subheader("Visualisations interactives")
x_axis = st.selectbox("Sélectionnez la colonne pour l'axe X", options=numeric_columns, key="scatter_x")
y_axis = st.selectbox("Sélectionnez la colonne pour l'axe Y", options=numeric_columns, key="scatter_y")

# Création d'un graphique interactif
st.subheader("Graphique interactif")
fig, ax = plt.subplots()
sns.scatterplot(data=df, x=x_axis, y=y_axis, ax=ax)
ax.set_title(f"Graphique de {x_axis} vs {y_axis}")
st.pyplot(fig)

# Histogramme avec une taille réduite
st.subheader("Histogramme")
hist_column = st.selectbox("Sélectionnez une colonne pour l'histogramme", options=numeric_columns, key="hist_column")

# Taille de l'histogramme plus petite
fig_hist, ax_hist = plt.subplots(figsize=(6, 4))  # Modifier la taille ici
sns.histplot(df[hist_column].dropna(), bins=20, kde=True, ax=ax_hist)
ax_hist.set_title(f"Histogramme de {hist_column}")
st.pyplot(fig_hist)

# Matrice de corrélation
st.subheader("Matrice de corrélation")
corr_data = df[numeric_columns].corr()  # Utiliser uniquement les colonnes numériques
if not corr_data.empty:
    fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_data, annot=True, cmap="coolwarm", fmt=".2f", ax=ax_corr)
    ax_corr.set_title("Matrice de corrélation")
    st.pyplot(fig_corr)
else:
    st.error("Aucune corrélation ne peut être calculée (colonnes numériques manquantes).")
