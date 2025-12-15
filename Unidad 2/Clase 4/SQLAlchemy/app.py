from flask import Flask, jsonify
from database import engine, Base
from models import User  # Importar los modelos ANTES de crear las tablas

app = Flask(__name__)

# Create all tables in the database
Base.metadata.create_all(bind=engine)

@app.route('/')
def home():
    return jsonify({"message": "Welcome to the SQLAlchemy Flask App!"})

@app.route('/users/model')
def get_user_model():
    return jsonify({
        "table_name": User.__tablename__,
        "columns": [column.name for column in User.__table__.columns]
    })

if __name__ == '__main__':
    app.run(debug=True)