import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve

# Configuration des dimensions & affichage de la page
st.set_page_config(page_title="Graphiques", 
                   page_icon=":chart_with_upwards_trend:", 
                   layout='wide')
 
df = st.session_state['df']
type_model = st.session_state['type_model']
model_ML = st.session_state['model_ML']
X = st.session_state['X']
y = st.session_state['y']
selected_model_ML = st.session_state['selected_model_ML']



hyperparam_reg = {
    'LinearRegression': {
        'alpha': [0.1, 1.0, 10.0]
    },
    'Ridge': {
        'alpha': [0.1, 1.0, 10.0]
    },
    'Lasso': {
        'alpha': [0.1, 1.0, 10.0],
        'selection': ['cyclic', 'random']
    },
    'ElasticNet': {
        'alpha': [0.1, 1.0, 10.0],
        'l1_ratio': [0.1, 0.5, 0.9]
    },
    'DecisionTreeRegressor': {
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'RandomForestRegressor': {
        'n_estimators': [50, 100, 200],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'SVR': {
        'C': [0.1, 1.0, 10.0],
        'epsilon': [0.01, 0.1, 0.2]
    }
}
    


hyperparam_class = {
    'LogisticRegression': {
        'penalty': ['l1', 'l2'],
        'C': [0.001, 0.01, 0.1, 1, 10],
        'solver': ['liblinear', 'saga']
    },
    'DecisionTreeClassifier': {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'RandomForestClassifier': {
        'n_estimators': [50, 100, 200],
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt']
    },
    'SVC': {
        'C': [0.1, 1.0, 10.0],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'gamma': ['scale', 'auto']
    }
}


search = GridSearchCV(
		model_ML,
		[hyperparam_reg[selected_model_ML] if type_model=='reg' else hyperparam_class[selected_model_ML] if type_model=='class' else None]	
)

search.fit(X,y)
st.write('Meilleurs hyperparamètres : ', search.best_params_)
st.write('Meilleur score de cross-validation : ', search.best_score_)

# Ligne de séparation
st.write("***")

dico_mean = {'Moyenne des targets : ':round(df['target'].mean(),2), 'Somme des targets : ':round(df['target'].sum(),2)}
col_1, col_2 = st.columns(2)
col_1.metric(label='Moyenne des targets', value=round(df['target'].mean(),2))
col_2.metric(label='Somme des targets', value=round(df['target'].sum(),2))

# Ligne de séparation
st.write("***")

# N, train_score, val_score =  learning_curve(model, X_train, y_train, train_sizes=np.linspace(0.2, 1.0, 5))
# st.write("N : ",N)
# st.write("train_score : ", train_score)
# st.write("val_score : ", val_score)