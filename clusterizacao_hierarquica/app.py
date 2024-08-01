import streamlit as st
import pandas as pd

# Carregar Dados e colocar no Cache do streamlit

@st.cache_data
def carregar_dados():
    return pd.read_csv('./datasets/laptops_clustered.csv') 

df = carregar_dados()

# Sidebar para Filtro
st.sidebar.title('Filtro')

# Selecionar modelos
model = st.sidebar.selectbox('Selecionar o Modelo', df['model'].unique())

# Filtrar modelo
df_laptops_modelo = df[df['model'] == model]

# Filtrar cluster do modelo escolhido
df_laptops_final = df[df['cluster'] == df_laptops_modelo.iloc[0]['cluster']]

# Mostrar os dados
st.write("Recomendações de Modelos")
st.table(df_laptops_final)