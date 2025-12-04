from flask import Flask, jsonify, request

app = Flask(__name__)


clientes = {
    1: {
        "nombre": "Juan David",
        "edad": 28,
        "ciudad": "Ibagué"
    }
    
}

contador = 1

@app.route('/clientes', methods=['GET']) # / = Root URL
def listar_clientes():
    return jsonify({"clientes": clientes})

@app.route("/clientes",methods=['POST'])
def agregar_clientes():
    global contador

    data = request.get_json()

    contador +=1
    clientes[contador] = data

    return jsonify({
        "mensaje": "Clientes agregado correctamente"
    })


app.run(debug=True)
