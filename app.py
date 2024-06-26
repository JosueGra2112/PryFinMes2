from flask import Flask, request, render_template, jsonify
import pandas as pd
import joblib
import logging

app = Flask(__name__)

# Configurar el registro
logging.basicConfig(level=logging.DEBUG)

# Ruta del archivo CSV
csv_path = 'ToyotaCorolla.csv'

# Cargar los datos
try:
    data = pd.read_csv(csv_path)
    app.logger.debug('Datos cargados correctamente.')
    data = data[['Age', 'KM', 'HP', 'MetColor', 'Weight', 'FuelType', 'Price']].dropna()
    app.logger.debug('Datos limpios, NaN eliminados.')
except FileNotFoundError as e:
    app.logger.error(f'Error al cargar los datos: {str(e)}')
    data = None

# Definir el scaler
scaler = joblib.load('scaler.pkl')

# Cargar el modelo
model = joblib.load('modelo.pkl')

@app.route('/')
def home():
    return render_template('formulario.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Imprimir los datos recibidos
        app.logger.debug(f'Datos recibidos: {request.form}')

        # Obtener los datos enviados en el request
        age = float(request.form['age'])
        km = float(request.form['km'])
        hp = float(request.form['hp'])
        metcolor = float(request.form['metcolor'])
        weight = float(request.form['weight'])
        fuel_type = int(request.form['fuel_type'])

        # Crear un DataFrame con los datos
        input_data = pd.DataFrame([[age, km, hp, metcolor, weight, fuel_type]], 
                                  columns=['Age', 'KM', 'HP', 'MetColor', 'Weight', 'FuelType'])

        app.logger.debug(f'Datos de entrada: {input_data}')

        # Escalar los datos
        input_data_scaled = scaler.transform(input_data)
        app.logger.debug(f'Datos escalados: {input_data_scaled}')

        # Realizar predicciones
        prediction = model.predict(input_data_scaled)
        app.logger.debug(f'Predicción: {prediction}')

        # Devolver las predicciones como respuesta JSON
        return jsonify({'prediction': float(prediction[0])})
    except Exception as e:
        app.logger.error(f'Error en la predicción: {str(e)}')
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
