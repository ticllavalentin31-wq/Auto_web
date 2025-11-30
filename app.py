import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. CONFIGURACIÓN VISUAL (ICONO Y TÍTULO) ---
st.set_page_config(page_title="Cotizador de Autos", page_icon="🚗", layout="centered")

# CSS para ocultar elementos técnicos y limpiar la vista
st.markdown("""
    <style>
    .stDeployButton {display:none;}
    div[data-testid="stToolbar"] {display: none;}
    </style>
""", unsafe_allow_html=True)

st.title("🚗 Cotizador de Vehículos Usados")
st.markdown("##### Complete el formulario para obtener una valoración instantánea de mercado.")
st.markdown("---")

# --- 2. CARGA DEL CEREBRO (MODELO) ---
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

# Si falla la carga, mostramos mensaje amigable
if model is None:
    st.error("⚠️ El sistema se está iniciando o actualizando. Por favor espere unos segundos y recargue la página.")
    st.stop()

# --- 3. FORMULARIO DE USUARIO (LIMPIO) ---

# Usamos columnas para que no se vea una lista eterna hacia abajo
col1, col2 = st.columns(2)

input_data = {}

# --- COLUMNA IZQUIERDA: DATOS BÁSICOS ---
with col1:
    st.subheader("Datos del Vehículo")
    
    # Buscamos y mostramos inputs numéricos con nombres amigables
    for col in col_num:
        # TRADUCCIÓN DE VARIABLES TÉCNICAS A ESPAÑOL AMIGABLE
        if 'year' in col.lower():
            val = st.slider("Año de Fabricación", 2000, 2025, 2018)
            input_data[col] = [val]
            
        elif 'present_price' in col.lower():
            # Explicación clara para el usuario
            val = st.number_input("Precio de Lista (Nuevo)", min_value=0.0, value=0.0, step=0.5, 
                                help="¿Cuánto costaba este auto cuando era nuevo? (Use la misma moneda que sus datos, ej: miles)")
            input_data[col] = [val]
            
        elif 'driven' in col.lower() or 'kms' in col.lower():
            val = st.number_input("Kilometraje (Recorrido)", min_value=0, value=0, step=1000)
            input_data[col] = [val]
            
        elif 'owner' in col.lower():
            pass # Lo ponemos en la otra columna para ordenar

# --- COLUMNA DERECHA: DETALLES ---
with col2:
    st.subheader("Características")
    
    # Input de dueños (si existe en numéricos)
    for col in col_num:
        if 'owner' in col.lower():
            val = st.selectbox("Cantidad de Dueños Anteriores", [0, 1, 2, 3])
            input_data[col] = [val]

    # Inputs Categóricos (Marca, Transmisión, etc.)
    for col in col_cat:
        # Limpieza del nombre (ej: Fuel_Type -> Tipo de Combustible)
        nombre_amigable = col.replace('_', ' ').capitalize()
        if 'fuel' in nombre_amigable.lower(): nombre_amigable = "Tipo de Combustible"
        if 'seller' in nombre_amigable.lower(): nombre_amigable = "Tipo de Vendedor"
        if 'transmission' in nombre_amigable.lower(): nombre_amigable = "Transmisión"
        if 'name' in nombre_amigable.lower() or 'car' in nombre_amigable.lower(): nombre_amigable = "Marca / Modelo"

        opciones = dic_unicos.get(col, [])
        val = st.selectbox(nombre_amigable, opciones)
        input_data[col] = [val]

# --- 4. BOTÓN DE ACCIÓN Y RESULTADO ---
st.markdown("<br>", unsafe_allow_html=True) # Espacio

# Botón grande y centrado
col_centrada = st.columns([1, 2, 1])
with col_centrada[1]:
    boton_calcular = st.button("🔍 CALCULAR VALOR AHORA", use_container_width=True, type="primary")

if boton_calcular:
    try:
        # Validar que no haya ceros ilógicos (opcional, para guiar al usuario)
        # (Si el usuario dejó todo en 0, le avisamos)
        df_usuario = pd.DataFrame(input_data)
        
        # Procesamiento interno (Técnico pero oculto)
        X_num = df_usuario[col_num]
        X_cat = df_usuario[col_cat]
        X_num_scaled = scaler.transform(X_num)
        X_cat_encoded = encoder.transform(X_cat).toarray()
        X_final = np.concatenate([X_num_scaled, X_cat_encoded], axis=1)
        
        # Predicción
        prediccion = model.predict(X_final)[0]
        
        # --- RESULTADO FINAL ---
        st.markdown("---")
        st.success("✅ ¡Cálculo Exitoso!")
        
        # Mostramos el precio en grande
        st.markdown(f"<h2 style='text-align: center; color: #2E86C1;'>Valor Estimado de Mercado:</h2>", unsafe_allow_html=True)
        st.markdown(f"<h1 style='text-align: center;'>{prediccion:,.2f}</h1>", unsafe_allow_html=True)
        st.caption(f"*Este valor es una estimación basada en inteligencia artificial y las características ingresadas.")
        
    except Exception as e:
        st.error("Hubo un problema con los datos ingresados. Por favor verifique e intente nuevamente.")
