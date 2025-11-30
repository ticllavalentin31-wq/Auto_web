import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Configuración básica
st.set_page_config(page_title="Predicción Autos", layout="centered")
st.title("🚗 Tasador de Autos Inteligente")
st.write("Sistema de predicción basado en Random Forest.")

# --- CARGA DE ARCHIVOS ---
@st.cache_resource
def cargar_todo():
    model = joblib.load('modelo_precio_autos.pkl')
    scaler = joblib.load('scaler_autos.pkl')
    encoder = joblib.load('encoder_autos.pkl')
    col_num = joblib.load('columnas_numericas.pkl')
    col_cat = joblib.load('columnas_categoricas.pkl')
    dic_unicos = joblib.load('valores_unicos.pkl')
    return model, scaler, encoder, col_num, col_cat, dic_unicos

try:
    model, scaler, encoder, col_num, col_cat, dic_unicos = cargar_todo()
except Exception as e:
    st.error(f"Error cargando archivos: {e}")
    st.stop()

# --- INTERFAZ (Formulario) ---
st.sidebar.header("Características del Auto")
user_inputs = {}

# 1. Inputs Numéricos (Año, Millas, etc)
for col in col_num:
    nombre = col.replace('_', ' ').capitalize()
    # Si es año, ponemos rango lógico, si no, numérico estándar
    if 'year' in col.lower():
        val = st.sidebar.slider(nombre, 1990, 2025, 2015)
    else:
        val = st.sidebar.number_input(nombre, min_value=0.0, value=10000.0)
    user_inputs[col] = [val]

# 2. Inputs Categóricos (Marca, Transmisión, etc)
for col in col_cat:
    nombre = col.replace('_', ' ').capitalize()
    opciones = dic_unicos.get(col, [])
    # Seleccionamos la primera opción por defecto
    val = st.sidebar.selectbox(nombre, opciones)
    user_inputs[col] = [val]

# --- PREDICCIÓN ---
# Mostrar lo que el usuario seleccionó
st.subheader("Datos Ingresados")
df_usuario = pd.DataFrame(user_inputs)
st.dataframe(df_usuario)

if st.button("CALCULAR PRECIO 💰"):
    try:
        # 1. Separar numéricas y categóricas (mismo orden que entrenamiento)
        X_num = df_usuario[col_num]
        X_cat = df_usuario[col_cat]
        
        # 2. Escalar numéricos
        X_num_scaled = scaler.transform(X_num)
        
        # 3. Codificar categóricos
        # El encoder devuelve una matriz sparse, la convertimos a array
        X_cat_encoded = encoder.transform(X_cat).toarray()
        
        # 4. Unir (Concatenar)
        X_final = np.concatenate([X_num_scaled, X_cat_encoded], axis=1)
        
        # 5. Predecir
        prediccion = model.predict(X_final)[0]
        
        st.success(f"### Precio Estimado: ${prediccion:,.2f} USD")
        st.balloons()
        
    except Exception as e:
        st.error(f"Error en el cálculo: {e}")