from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import numpy as np
import logging
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

app = Flask(__name__)

# Configurar el registro
logging.basicConfig(level=logging.DEBUG)

# Cargar el modelo entrenado y el scaler
def create_model():
    model = Sequential()
    model.add(Input(shape=(6,)))  # Cambiar el tamaño de entrada a 6 para las características seleccionadas
    model.add(Dense(64, activation='relu'))  # Capa de entrada
    model.add(Dense(32, activation='relu'))  # Capa oculta
    model.add(Dense(8, activation='relu'))   # Capa oculta
    model.add(Dense(1))  # Capa de salida
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

model = joblib.load('modelo.pkl')
app.logger.debug('Modelo cargado correctamente.')

@app.route('/')
def home():
    return render_template('formulario.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos enviados en el request
        age = float(request.form['age'])
        km = float(request.form['km'])
        hp = float(request.form['hp'])
        doors = float(request.form['doors'])
        weight = float(request.form['weight'])
        fuelType_Petrol = int(request.form.get('fuelType_Petrol', 0))

        # Crear un DataFrame con los datos
        data_df = pd.DataFrame([[age, km, hp, doors, weight, fuelType_Petrol]], 
                               columns=['Age', 'KM', 'HP', 'Doors', 'Weight', 'FuelType_Petrol'])
        app.logger.debug(f'DataFrame creado: {data_df}')

        # Realizar predicciones
        prediction = model.predict(data_df)
        app.logger.debug(f'Predicción: {prediction[0]}')

        # Devolver las predicciones como respuesta JSON
        return jsonify({'prediction': float(prediction[0])})
    except Exception as e:
        app.logger.error(f'Error en la predicción: {str(e)}')
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
