import streamlit as st
import pandas as pd


# Configuration des dimensions & affichage de la page
st.set_page_config(page_title="Traitement", 
                   page_icon=":mortar_board:", 
                   layout='wide')
 
df = st.session_state['df']
print(st.session_state)
st.write(df)

