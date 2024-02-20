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

#Configuration des dimensions & affichage de la page
st.set_page_config(page_title="Import", 
                   page_icon=":mortar_board:", 
                   layout='wide')

liste_df =['DF Diabète', 'DF Vin', 'Autre DF selon import']
choix = st.sidebar.selectbox(
'Quel df voulez-vous traiter ?', liste_df
)

if choix == 'DF Diabète':
    df = pd.read_csv("SRC/diabete.csv")
    # df = pd.read_csv("C:/Users/murai/OneDrive/Documents/GitHub/FATAL_ML/SRC/diabete.csv")
    # if 'df' not in st.session_state:
    state_write(df)

elif choix == 'DF Vin':
    df = pd.read_csv("SRC/vin.csv")
    # df = pd.read_csv("C:/Users/murai/OneDrive/Documents/GitHub/FATAL_ML/SRC/vin.csv")
    # if 'df' not in st.session_state:
    state_write(df)

else:
    file = st.file_uploader("Uploader un fichier", type="csv")
    button = st.button('importer')
    if button:

        content = file.getvalue().decode("utf-8")
        
        # Utiliser csv.Sniffer pour identifier automatiquement le séparateur
        dialect = csv.Sniffer().sniff(content)
        separator = dialect.delimiter

        # Lire le fichier CSV dans un DataFrame pandas en utilisant le séparateur identifié
        df = pd.read_csv(file, sep=separator)
        state_write(df)


