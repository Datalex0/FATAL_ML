import pandas as pd
import numpy as np
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt

#suppression de la colonne 'Unnamed: 0' créee lors de l'initialisation du Dataframe
def unnamed_drop() :
    df.drop(columns = ['Unnamed: 0'], inplace=True)

#affichage des titres de colonnes et des valeurs vides dessous
def affichage_blanc() :
    valeur_manquantes = df.isna().sum()
    print("la liste de vos colonnes : chaque colonne contient X valeurs manquantes ")
    print(valeur_manquantes)

    # selection manuelle des colonnes
def selection_colonnes() :
    liste_colonnes = df.columns.tolist()
    selection_col = st.multiselect("Sélectionnez les colonnes", liste_colonnes)
    return selection_col

#suppression des colonnes vides
def suppression_colonne_vide() :
    empty_columns = df.columns[df.isna().all()].tolist()
    df.dropna(how='all', axis=1, inplace=True)
    print(f'Vous venez de supprimer les colonnes {empty_columns } ' )

    # suppression des lignes ou des cases sont vides
def suppression_blanc() :
    long_init_df = len(df)
    df.dropna(how='any', axis=0, inplace=True)
    print(f'Vous venez de supprimer { long_init_df - len(df)} lignes' )

    #selection colonne target
def selection_target() :
    liste_colonnes = df.columns.tolist()
    colonne_target = st.selectbox("Selectionnez la colonne cible : ", liste_colonnes)
    return colonne_target

#encodage manuel pour les classifications
def encodage(colonne_target):
    unique_values_colonne_target = colonne_target.unique()
    x=0
    for i in unique_values_colonne_target : 
        df.replace(to_replace = i, value =x, inplace=True)
        print('Les valeurs de votre cibles ont été remplacées' )
        print(f'remplacement de : {i} par {x}' )
        x+=1

        #afficher colinearités
def colinearite() :
    mask = np.triu(df.select_dtypes("number").corr())
    fig, ax = plt.subplots(figsize=(10, 10))
    cmap = sns.diverging_palette(15, 160, n=11, s=100)

    sns.heatmap(
        df.select_dtypes("number").corr(),
        mask=mask,
        annot=True,
        cmap=cmap,
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax
    )
colinearite()