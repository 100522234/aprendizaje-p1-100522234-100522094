# Predicción de Suscripción a Depósitos Bancarios

Este proyecto utiliza tecnicas de Aprendizaje Automático para predecir si un cliente bancario contratará un deposito a plazo fijo o no


## Estructura del Proyecto

* `bank_07.pkl`: Dataset principal del proyecto. Contiene los datos de los clientes y la variable objetivo (deposit)
* `bank_competition.pkl`: Dataset de evaluación final. Contiene información de nuevos clientes sin la variable objetivo, sobre el cual nuestro modelo final debe inferir si contratarán o no el depósito
* `parte1_100522094_100522234.ipynb`: Cuaderno principal con el ciclo  de vida completo del dato: EDA, Preprocesamiento, Selección de modelos (KNN, Árboles, SVM) y HPO. Se incluyen explicaciones detalladas de cada ciclo
* `parte2_100522094_100522234.ipynb`: Cuaderno dedicado a la predicción sobre datos nuevos (simulación real). Carga el modelo serializado y genera las predicciones a ciegas sobre el dataset de competición (bank_competition.pkl)
* `modelo_final.pkl`: Modelo final entrenado y serializado
* `mystreamlit.py`: Codigo fuente de la aplicacion web desarrollada con Streamlit
* `predicciones.csv`: Archivo de salida generado por el segundo cuaderno (parte2_100522094_100522234.ipynb) que contiene las predicciones finales (valores yes o no para la variable objetivo deposit) de los clientes del dataset de competición
* `Verificacion_Streamlit_Grupo82_Equipo05`: Documento de comprobación que demuestra que las predicciones generadas por Streamlit coinciden con las del entorno de desarrollo para distintos clientes de prueba, validando el despliegue del modelo


## Instalación y ejecución

1. asegurate de tener instalada la version de scikit-learn utilizada en el entrenamiento:
    pip install scikit-learn==1.8.0 streamlit pandas joblib

2. ejecuta la aplicacion de streamlit: 
    python -m streamlit run mystreamlit.py


## Autores

* Carmen Peláez Martín - 100522094
* Ana Sanz del Collado - 100522234

Grupo 82 - Equipo 05 | Grado en ingeniería informática - UC3M