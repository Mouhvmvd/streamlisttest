import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Titre de l'application
st.title("Analyse des données Iris")

# Chargement du fichier Iris.csv
uploaded_file = st.file_uploader("Téléchargez votre fichier Iris.csv", type=["csv"])

if uploaded_file is not None:
    # Charger les données dans un DataFrame
    df = pd.read_csv(uploaded_file)
    st.subheader("Aperçu des données")
    st.dataframe(df)

    # Options de visualisation
    st.subheader("Statistiques descriptives")
    st.write(df.describe())

    # Sélectionner les colonnes pour la visualisation
    st.subheader("Visualisations")
    columns = df.columns
    x_axis = st.selectbox("Sélectionnez la colonne pour l'axe X", options=columns)
    y_axis = st.selectbox("Sélectionnez la colonne pour l'axe Y", options=columns)
    
    # Création du graphique interactif
    st.subheader("Graphique :")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x=x_axis, y=y_axis, ax=ax)
    st.pyplot(fig)

    # Visualisation avec Seaborn : Matrice de corrélation
    st.subheader("Matrice de corrélation")
    fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax_corr)
    st.pyplot(fig_corr)
else:
    st.info("Veuillez télécharger un fichier CSV pour continuer.")
