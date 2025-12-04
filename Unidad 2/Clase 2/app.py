from flask import Flask, jsonify, request
from flasgger import Swagger

app = Flask(__name__)
swagger = Swagger(app)

clientes = {
    1: {
        "nombre": "Juan David",
        "edad": 28,
        "ciudad": "Ibagué"
    }
}

contador = 1  # Va a ir incrementando


@app.route('/clientes', methods=['GET'])
def listar_clientes():
    """
    Endpoint para obtener la lista de clientes
    ---
    tags:
      - Clientes
    responses:
      200:
        description: Lista de clientes
        examples:
          application/json:
            clientes:
              "1":
                nombre: "name"
                edad: 20
                ciudad: "ciudad"
    """
    return jsonify({"clientes": clientes})


@app.route("/clientes", methods=['POST'])
def agregar_clientes():
    """
    Endpoint para agregar un cliente
    ---
    tags:
      - Clientes
    parameters:
        - name: body
        in: body
        required: true
        schema:
            id: Cliente
            required:
            - nombre
            - edad
            - ciudad
          properties:
            nombre:
              type: string
              example: "Carlos"
            edad:
              type: integer
              example: 30
            ciudad:
              type: string
              example: "Bogotá"
    responses:
      201:
        description: Cliente agregado
    """
    global contador

    data = request.get_json()

    contador += 1
    clientes[contador] = data

    return jsonify({
        "mensaje": "Cliente agregado correctamente",
        "cliente_id": contador,
        "cliente": data
    }), 201


if __name__ == "__main__":
    app.run(debug=True)
