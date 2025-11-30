import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Configuración de página
st.set_page_config(page_title="Predicción Venta Autos", layout="centered")

# Título y descripción
st.title("🚗 Predicción de Precio de Venta")
st.markdown("""
Esta aplicación estima el precio de venta de un vehículo usado basándose en sus características 
y su precio actual de mercado (nuevo).
""")

# --- 1. CARGA DE RECURSOS ---
@st.cache_resource
def cargar_archivos():
    # Cargamos los 6 archivos generados
    model = joblib.load('modelo_precio_autos.pkl')
    scaler = joblib.load('scaler_autos.pkl')
    encoder = joblib.load('encoder_autos.pkl')
    col_num = joblib.load('columnas_numericas.pkl')
    col_cat = joblib.load('columnas_categoricas.pkl')
    dic_unicos = joblib.load('valores_unicos.pkl')
    return model, scaler, encoder, col_num, col_cat, dic_unicos

try:
    model, scaler, encoder, col_num, col_cat, dic_unicos = cargar_archivos()
    st.success("✅ Sistema cargado correctamente (Modelo: Random Forest)")
except Exception as e:
    st.error(f"Error crítico cargando archivos: {e}")
    st.stop()

# --- 2. INTERFAZ DE USUARIO (SIDEBAR) ---
st.sidebar.header("📝 Ingrese los datos")

user_inputs = {}

# A) Generación automática de inputs NUMÉRICOS
# Detectamos qué columnas son para poner sliders o inputs adecuados
for col in col_num:
    nombre_mostrar = col.replace('_', ' ').capitalize()
    
    if 'year' in col.lower():
        # Slider para el año
        val = st.sidebar.slider("Año del Vehículo", 2000, 2025, 2017)
    elif 'present_price' in col.lower():
        # Input para precio actual
        st.sidebar.markdown("---")
        val = st.sidebar.number_input(f"{nombre_mostrar} (Precio Nuevo)", min_value=0.0, value=5.0, step=0.1, help="Precio en miles o la moneda del dataset")
    elif 'owner' in col.lower():
        # Input para dueños anteriores
        val = st.sidebar.selectbox(f"{nombre_mostrar} (Dueños previos)", [0, 1, 2, 3])
    else:
        # Kilometraje u otros
        val = st.sidebar.number_input(f"{nombre_mostrar}", min_value=0, value=10000)
        
    user_inputs[col] = [val]

# B) Generación automática de inputs CATEGÓRICOS
for col in col_cat:
    nombre_mostrar = col.replace('_', ' ').capitalize()
    opciones = dic_unicos.get(col, [])
    
    # Selectbox con las opciones aprendidas
    val = st.sidebar.selectbox(nombre_mostrar, opciones)
    user_inputs[col] = [val]

# --- 3. PROCESAMIENTO Y PREDICCIÓN ---
st.subheader("Resumen de Características")
df_usuario = pd.DataFrame(user_inputs)
st.dataframe(df_usuario)

if st.button("CALCULAR PRECIO ESTIMADO 💰", type="primary"):
    try:
        # Paso 1: Separar en numéricos y categóricos
        X_num = df_usuario[col_num]
        X_cat = df_usuario[col_cat]
        
        # Paso 2: Escalar numéricos (MinMax)
        X_num_scaled = scaler.transform(X_num)
        
        # Paso 3: Codificar categóricos (OneHot)
        X_cat_encoded = encoder.transform(X_cat).toarray()
        
        # Paso 4: Unir todo
        X_final = np.concatenate([X_num_scaled, X_cat_encoded], axis=1)
        
        # Paso 5: Predecir
        prediccion = model.predict(X_final)[0]
        
        # --- MOSTRAR RESULTADO ---
        st.balloons()
        st.markdown("---")
        st.markdown(f"### 💎 Precio de Venta Estimado: **{prediccion:,.2f}**")
        st.info("Nota: La moneda depende de los datos de entrada (ej. USD, Soles, Lakhs).")
        
    except Exception as e:
        st.error(f"Ocurrió un error en el cálculo: {e}")