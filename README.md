# Auto_web
# 🚗 Sistema de Predicción de Precios de Autos Usados

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)](https://scikit-learn.org/)

## 📄 Descripción del Proyecto
Este proyecto consiste en el desarrollo y despliegue de un sistema de Inteligencia Artificial capaz de estimar el precio de venta de vehículos usados. Utilizando un algoritmo de **Random Forest Regressor**, el modelo analiza características clave como el año de fabricación, kilometraje, tipo de combustible, transmisión y precio original de lista.

El objetivo es proporcionar una herramienta accesible vía web para apoyar la toma de decisiones en la compra-venta de automóviles.

### 🔗 Demo en Vivo
Haz clic aquí para probar la aplicación:
👉 **(https://proyectoestadis.streamlit.app/)**

---

## 🛠️ Tecnologías Utilizadas

* **Lenguaje:** Python
* **Machine Learning:** Scikit-Learn (Random Forest)
* **Web Framework:** Streamlit
* **Procesamiento de Datos:** Pandas, NumPy
* **Persistencia:** Joblib

---

## 📂 Estructura del Repositorio

Este repositorio contiene los siguientes archivos esenciales para la ejecución del modelo en la nube:

| Archivo | Descripción |
| :--- | :--- |
| `app.py` | Código fuente de la aplicación web (Frontend y Backend). |
| `requirements.txt` | Lista de dependencias para la instalación en el servidor. |
| `modelo_precio_autos.pkl` | Modelo entrenado (Random Forest). |
| `scaler_autos.pkl` | Objeto MinMaxScaler para normalización de datos numéricos. |
| `encoder_autos.pkl` | Objeto OneHotEncoder para transformación de variables categóricas. |
| `*.pkl` (varios) | Archivos auxiliares para mapeo de columnas y listas desplegables. |
