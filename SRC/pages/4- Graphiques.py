import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import learning_curve
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import classification_report,confusion_matrix

# Configuration des dimensions & affichage de la page
st.set_page_config(page_title="Graphiques", 
                   page_icon=":chart_with_upwards_trend:", 
                   layout='wide')
 
df = st.session_state['df']
type_model = st.session_state['type_model']
model_ML = st.session_state['model_ML']
new_X = st.session_state['new_X']
y = st.session_state['y']
selected_model_ML = st.session_state['selected_model_ML']
new_X_train = st.session_state['new_X_train']
new_X_test = st.session_state['new_X_test']
y_train = st.session_state['y_train']
y_test = st.session_state['y_test']


model_ML.fit(new_X_train, y_train)
y_pred = model_ML.predict(new_X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
st.write(f"{selected_model_ML} MSE :", mse)
st.write(f"{selected_model_ML} R2 Score :", r2)

if selected_model_ML in ["LogisticRegression", "DecisionTreeClassifier", "RandomForestClassifier", "SVC"]:
    cm = confusion_matrix(y_test, y_pred) #MATRICE CONFUSION
    cr = classification_report(y_test, y_pred) #MATRICE CONFUSION
    st.write(f'\n -------------\\\ Matrice de confusion  ///-------------\n')
    st.write(cm[:15, :15])
            # Extraire les 10 premières et 10 dernières lignes du rapport de classification
    st.write(cr)



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