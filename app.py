import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Cotizador de Autos", page_icon="🚗", layout="centered")

# Estilos CSS para limpiar la interfaz
st.markdown("""
    <style>
    .stDeployButton {display:none;}
    div[data-testid="stToolbar"] {display: none;}
    </style>
""", unsafe_allow_html=True)

st.title("🚗 Cotizador de Vehículos Usados")
st.markdown("##### Complete el formulario para obtener una valoración instantánea.")
st.markdown("---")

# --- CARGA DE ARCHIVOS ---
@st.cache_resource
def cargar_archivos():
    try:
        model = joblib.load('modelo_precio_autos.pkl')
        scaler = joblib.load('scaler_autos.pkl')
        encoder = joblib.load('encoder_autos.pkl')
        col_num = joblib.load('columnas_numericas.pkl')
        col_cat = joblib.load('columnas_categoricas.pkl')
        dic_unicos = joblib.load('valores_unicos.pkl')
        return model, scaler, encoder, col_num, col_cat, dic_unicos
    except:
        return None, None, None, None, None, None

model, scaler, encoder, col_num, col_cat, dic_unicos = cargar_archivos()

if model is None:
    st.error("⚠️ Error: No se encontraron los archivos del modelo. Asegúrese de que estén en la misma carpeta.")
    st.stop()

# --- DICCIONARIO DE TRADUCCIÓN (LA SOLUCIÓN) ---
# Aquí definimos qué texto mostrar por cada columna técnica
NOMBRES_AMIGABLES = {
    # Categóricas
    'Car_Name': 'Seleccionar Marca / Modelo',
    'Fuel_Type': 'Seleccionar Tipo de Combustible',
    'Seller_Type': 'Seleccionar Vendedor',
    'Transmission': 'Seleccionar Transmisión',
    
    # Numéricas
    'Year': 'Seleccionar Año de Fabricación',
    'Present_Price': 'Ingrese Precio de Lista (Nuevo)',
    'Kms_Driven': 'Ingrese Kilometraje',
    'Owner': 'Seleccionar Dueños Anteriores'
}

# --- FORMULARIO ---
col1, col2 = st.columns(2)
input_data = {}

# Lógica para mostrar inputs
with col1:
    st.subheader("Datos Básicos")
    
    for col in col_num:
        # Buscamos el nombre bonito, si no existe, usamos el original
        etiqueta = NOMBRES_AMIGABLES.get(col, col)
        
        # Detectamos nombres clave para dar el input correcto
        if 'year' in col.lower():
            val = st.slider(etiqueta, 2000, 2025, 2018)
            input_data[col] = [val]
            
        elif 'present_price' in col.lower():
            val = st.number_input(etiqueta, min_value=0.0, value=5.0, step=0.5, 
                                help="Precio del auto cuando era nuevo (en miles)")
            input_data[col] = [val]
            
        elif 'driven' in col.lower() or 'kms' in col.lower():
            val = st.number_input(etiqueta, min_value=0, value=20000, step=1000)
            input_data[col] = [val]
            
        elif 'owner' in col.lower():
            pass # Lo pasamos a la columna derecha

with col2:
    st.subheader("Detalles")
    
    # 1. Poner el "Owner" aquí si existe
    for col in col_num:
        if 'owner' in col.lower():
            etiqueta = NOMBRES_AMIGABLES.get(col, "Seleccionar Dueños")
            val = st.selectbox(etiqueta, [0, 1, 2, 3])
            input_data[col] = [val]

    # 2. Poner las Categóricas (Marca, Combustible, etc.)
    for col in col_cat:
        etiqueta = NOMBRES_AMIGABLES.get(col, col) # Obtiene el nombre bonito
        opciones = dic_unicos.get(col, [])
        
        # Selectbox con etiqueta clara
        val = st.selectbox(etiqueta, opciones)
        input_data[col] = [val]

# --- BOTÓN Y RESULTADO ---
st.markdown("<br>", unsafe_allow_html=True)

_, col_btn, _ = st.columns([1, 2, 1])
with col_btn:
    boton = st.button("CALCULAR PRECIO", type="primary", use_container_width=True)

if boton:
    try:
        df_usuario = pd.DataFrame(input_data)
        
        # Procesamiento
        X_num = df_usuario[col_num]
        X_cat = df_usuario[col_cat]
        X_num_scaled = scaler.transform(X_num)
        X_cat_encoded = encoder.transform(X_cat).toarray()
        X_final = np.concatenate([X_num_scaled, X_cat_encoded], axis=1)
        
        # Predicción
        prediccion = model.predict(X_final)[0]
        
        st.markdown("---")
        st.success("✅ Estimación Completada")
        st.markdown(f"<h3 style='text-align: center; color: gray;'>Precio Sugerido de Venta:</h3>", unsafe_allow_html=True)
        st.markdown(f"<h1 style='text-align: center; color: #007bff;'>{prediccion:,.2f}</h1>", unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error en el cálculo. Verifique los datos. Detalle: {e}")
