import streamlit as st
import pandas as pd
import joblib
import numpy as np
import sklearn
import sys

st.title("Predicción de Suscripción a Depósito Bancario")
st.write("Esta aplicación predice si un cliente suscribirá o no un depósito a plazo fijo basado en sus características")


# 1. CARGAMOS DEL MODELO
modelo = None
try:
    # ¡¡¡IMPORTANTE!!! el archivo .pkl debe estar en la misma carpeta que mystreamlit.py
    modelo = joblib.load('modelo_final.pkl')
    st.success("Modelo cargado correctamente")  # mostramos un mensaje verde de éxito
except Exception as e:
    st.error(f"Error cargando el modelo: {e}")  # si falla mostramos un aviso en rojo

# debug por el fallo de distinta versión en scikit-learn
st.warning(f"Versión actual de scikit-learn: {sklearn.__version__}")
st.warning(f"Ruta de Python: {sys.executable}")


# 2. FORMULARIO PARA LOS DATOS DE ENTRADA
st.sidebar.header("Datos del Cliente")  # titulo de la barra lateral de los datos

# Variables numéricas
# slider es una barra deslizable de minimo 18 y maximo 100, con un valor por defecto de 35
# number_input es una caja con un valor numerico
age = st.sidebar.slider("Edad", 18, 100, 35)
balance = st.sidebar.number_input("Balance anual medio", value=1000)
day = st.sidebar.slider("Día de contacto", 1, 31, 15)
duration = st.sidebar.number_input("Duración del contacto (segundos)", value=200)
campaign = st.sidebar.number_input("Número de contactos en campaña", value=1, min_value=1)
previous = st.sidebar.number_input("Contactos previos", value=0, min_value=0)

# Input del usuario para pdays original
pdays_input = st.sidebar.number_input("Días desde el último contacto (-1 si no hubo)", value=-1)

# Variables categóricas
# selectbox crea un menu con opciones en una lista y solo se pude elegir una
# radio crea botones de una unica opcion (como un test)
job = st.sidebar.selectbox("Trabajo", ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown'])
marital = st.sidebar.selectbox("Estado civil", ['divorced', 'married', 'single', 'unknown'])
education = st.sidebar.selectbox("Educación", ['primary', 'secondary', 'tertiary', 'unknown'])
default = st.sidebar.radio("¿Tiene crédito en mora?", ['no', 'yes'])
housing = st.sidebar.radio("¿Tiene hipoteca?", ['yes', 'no'])
loan = st.sidebar.radio("¿Tiene préstamo personal?", ['no', 'yes'])
contact = st.sidebar.selectbox("Medio de contacto", ['cellular', 'telephone', 'unknown'])
month = st.sidebar.selectbox("Mes de contacto", ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
poutcome = st.sidebar.selectbox("Resultado campaña anterior", ['unknown', 'failure', 'other', 'success'])


# 3. TRANSFORMACIÓN DE VARIABLES
# como el modelo no entiende el -1 de pdays, lo traducimos como en el cuaderno
# si pdays = -1 (no contactado) --> contactado previamente = 0
# si pdays != -1 --> contactado previamente = 1
was_previously_contacted = 0 if pdays_input == -1 else 1
pdays_clean = 0 if pdays_input == -1 else pdays_input


# 4. CREAMOS EL DATAFRAME
# metemos todas las respuestas del usuario en una tabla de una sola fila
input_data = pd.DataFrame({
    'age': [age],
    'job': [job],
    'marital': [marital],
    'education': [education],
    'default': [default],
    'balance': [balance],
    'housing': [housing],
    'loan': [loan],
    'contact': [contact],
    'day': [day],
    'month': [month],
    'duration': [duration],
    'campaign': [campaign],
    'previous': [previous],
    'poutcome': [poutcome],
    'was_previously_contacted': [was_previously_contacted],
    'pdays_clean': [pdays_clean]
})


# 5. MOSTRAMOS TABLA RESUMEN EN LA PANTALLA 
st.write("### Resumen de los datos del cliente:")

# mostramos la tabla en la pantalla
tabla_resumen = input_data.style.set_properties()
st.dataframe(tabla_resumen)


# 6. PREDICCIÓN
if st.button("Predecir"):
    if modelo is not None:
        try:
            # hacemos la prediccion con el pipeline cargado
            prediccion = modelo.predict(input_data)
            
            # mostramos el resultado
            if prediccion[0] == 1 or prediccion[0] == 'yes':
                st.success("El cliente SÍ suscribirá el depósito")
            else:
                st.warning("El cliente NO suscribirá el depósito")
                
        except Exception as e:
            st.error(f"Error al realizar la predicción: Verifica que el nombre y formato de todas las columnas coincidan. Detalles: {e}")
    else:
        st.error("No se puede predecir porque el modelo no esta cargado")
