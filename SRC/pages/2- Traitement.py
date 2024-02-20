import streamlit as st
import numpy as np
import pandas as pd
import csv
from modules import state_write
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from lazypredict.Supervised import LazyClassifier
from lazypredict.Supervised import LazyRegressor
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
import warnings
from ydata_profiling import ProfileReport
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.model_selection import KFold
import statsmodels.api as sm
warnings.filterwarnings("ignore")
import itertools
from modules import *


# Configuration des dimensions & affichage de la page
st.set_page_config(page_title="Traitement", 
                   page_icon=":mortar_board:", 
                   layout='wide')
 
df = st.session_state['df']

for colonne in df.columns:
    if colonne=='Unnamed: 0':
        # Suppression de la colonne Unnamed créée par pandas
        unnamed_drop(df)


# liste_df =['DF Diabète', 'DF Vin', 'Autre DF selon import']
# choix = st.sidebar.selectbox(
# 'Quel df voulez-vous traiter ?', liste_df
# )


# Suppression des colonnes vides
empty_col_box = st.sidebar.checkbox('Supprimer les colonnes vides')
if empty_col_box:
    (suppression_colonne_vide(df))

# Affichage des None
button = st.sidebar.button('Affichage des valeurs manquantes')
if button:
    (affichage_blanc(df))
dropna_box = st.sidebar.checkbox('Supprimer les valeurs manquantes')
if dropna_box:
    suppression_blanc(df)

# Colonne cible
def print_target(colonne_target):
    st.sidebar.write(f"La colonne cible est {colonne_target}")
if 'target' in df.columns:
    colonne_target = 'target'
    st.session_state['colonne_target'] = colonne_target
target_box = st.sidebar.checkbox('Changer la colonne cible')
if target_box:
    colonne_target = selection_target(df)
    st.session_state['colonne_target'] = colonne_target
print_target(colonne_target)

if isinstance(df[colonne_target][0], (int, float)):
    type_model = 'reg'
else:
    type_model = 'class'    
    # Encodage pour les classifications
    encod_box = st.sidebar.checkbox('Encoder')
    if encod_box:
        encodage(df, colonne_target)
st.session_state['type_model'] = type_model
st.session_state['df'] = df
    
# Affichage des colinarités
button = st.sidebar.button('Affichage des colinéarités')
if button:  
    colinearite(df)

st.dataframe(df, height=730)



# # Diviser l'espace d'affichage en 2 colonnes
# col1, col2 = st.columns(2)
# # Afficher le df dans la première colonne
# with col1: 
# # Afficher la heatmap dans la deuxième colonne
# with col2: