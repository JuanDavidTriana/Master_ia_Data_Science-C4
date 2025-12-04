from io import StringIO, BytesIO
from flask import Flask, jsonify, request, send_file
import pandas as pd

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "API de procesamiento de CSV"}), 200

@app.route('/procesar_csv_limpiar_1', methods=['POST'])
def procesar_csv_limpiar_1():
    if 'file' not in request.files:
        return jsonify({"error": "No se encontró el archivo"}), 400
    
    archivo = request.files['file']

    contenido = archivo.read().decode('utf-8')
    df = pd.read_csv(StringIO(contenido))

    df["Society"] = df["Society"].fillna("SinSociedad")
    df = df.drop(columns=["Plot Area","Dimensions"])

    output = StringIO()
    df.to_csv(output, index=False)
    output.seek(0)

    men_file = BytesIO(output.getvalue().encode('utf-8'))

    return send_file(
        men_file,
        mimetype='text/csv',
        as_attachment=True,
        download_name='archivo_procesado.csv'
    )

@app.route('/procesar_csv_limpiar_1B', methods=['POST'])
def procesar_csv_limpiar_1B():
    if 'file' not in request.files:
        return jsonify({"error": "No se encontró el archivo"}), 400
    
    archivo = request.files['file']

    contenido = archivo.read().decode('utf-8')
    df = pd.read_csv(StringIO(contenido))

    df["Society"] = df["Society"].fillna("SinSociedad")
    df = df.drop(columns=["Plot Area","Dimensions"])

    return jsonify({"message": "Archivo procesado y limpiado exitosamente",
                    "data": df.to_dict(orient='records')}), 200
    

if __name__ == "__main__":
    app.run(debug=True)
