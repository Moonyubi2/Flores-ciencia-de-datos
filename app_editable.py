
import streamlit as st
import joblib
import numpy as np
import os

# Título de la aplicación
st.title('Clasificador de Especies de Flores Iris (con entradas editables)')

st.write('Introduce las dimensiones de la flor (en cm) para predecir su especie.')

# Cargar el modelo y el label encoder
model_path = '/content/mlp_model.joblib'
encoder_path = '/content/label_encoder.joblib'

# Verificar si los archivos existen
if not os.path.exists(model_path):
    st.error(f"Error: El archivo del modelo no se encontró en {model_path}")
    st.stop()
if not os.path.exists(encoder_path):
    st.error(f"Error: El archivo del codificador de etiquetas no se encontró en {encoder_path}")
    st.stop()

try:
    model = joblib.load(model_path)
    label_encoder = joblib.load(encoder_path)
except Exception as e:
    st.error(f"Error al cargar los archivos: {e}")
    st.stop()

# Entradas del usuario con cuadros de texto editables
sepal_length_str = st.text_input('Longitud del sépalo (cm)', '4.0')
sepal_width_str = st.text_input('Ancho del sépalo (cm)', '3.8')
petal_length_str = st.text_input('Longitud del pétalo (cm)', '1.6')
petal_width_str = st.text_input('Ancho del pétalo (cm)', '0.1')

# Botón para realizar la predicción
if st.button('Predecir especie'):
    try:
        # Convertir las entradas de texto a números flotantes
        sepal_length = float(sepal_length_str)
        sepal_width = float(sepal_width_str)
        petal_length = float(petal_length_str)
        petal_width = float(petal_width_str)

        # Crear el array de características
        flower_dimensions = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

        # Realizar la predicción
        prediction = model.predict(flower_dimensions)

        # Decodificar la predicción
        decoded_prediction = label_encoder.inverse_transform(prediction)

        st.success(f"La especie de la flor predicha es: **{decoded_prediction[0]}**")
    except ValueError:
        st.error("Por favor, introduce valores numéricos válidos para las dimensiones de la flor.")
