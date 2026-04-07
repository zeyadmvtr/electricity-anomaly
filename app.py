from fastapi import FastAPI
import numpy as np
import pickle

app = FastAPI()

model = pickle.load(open("model.pkl", "rb"))

@app.get("/")
def home():
    return {"message": "API running"}

@app.post("/predict")
def predict(data: list):
    data = np.array(data).reshape(1, -1)
    prediction = model.predict(data)
    return {"prediction": int(prediction[0])}