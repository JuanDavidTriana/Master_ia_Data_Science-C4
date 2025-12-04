from fastapi import FastAPI

app = FastAPI()

@app.get("/saludo", tags=["Saludo"])
def saludo():
    return {"message": "Welcome to the FastAPI API!"}
