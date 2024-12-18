import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Titre de l'application
st.title("Analyse des données Iris")

# Charger les données directement depuis le fichier dans le dossier de travail
df = pd.read_csv("Iris.csv")

# Aperçu des données
st.subheader("Aperçu des données")
st.dataframe(df)

# Ajouter une méthode pour afficher les données
st.subheader("Affichage personnalisé des données")
rows_to_display = st.slider(
    "Nombre de lignes à afficher :", 
    min_value=5, max_value=len(df), value=10, step=5
)
st.write(df.head(rows_to_display))

# Options de visualisation
st.subheader("Statistiques descriptives")
st.write(df.describe())

# Sélectionner les colonnes pour la visualisation
st.subheader("Visualisations")
columns = df.columns
x_axis = st.selectbox("Sélectionnez la colonne pour l'axe X", options=columns)
y_axis = st.selectbox("Sélectionnez la colonne pour l'axe Y", options=columns)

# Création du graphique interactif
st.subheader("Graphique interactif")
fig, ax = plt.subplots()
sns.scatterplot(data=df, x=x_axis, y=y_axis, ax=ax)
ax.set_title(f"Graphique de {x_axis} vs {y_axis}")
st.pyplot(fig)

# Visualisation avec Seaborn : Matrice de corrélation
st.subheader("Matrice de corrélation")
fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax_corr)
ax_corr.set_title("Matrice de corrélation")
st.pyplot(fig_corr)
