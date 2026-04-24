from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI()

# ✅ أضف ده
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

class Readings(BaseModel):
    reading_1: float
    reading_2: float
    reading_3: float
    reading_4: float
    reading_5: float
    reading_6: float

@app.get("/")
def home():
    return {"status": "Model is running!"}

@app.post("/predict")
def predict(data: Readings):
    features = np.array([[
        data.reading_1,
        data.reading_2,
        data.reading_3,
        data.reading_4,
        data.reading_5,
        data.reading_6
    ]])
    
    prediction = model.predict(features)[0]
    result = "normal" if prediction == 1 else "not normal"
    
    return {
        "result": result,
        "raw_prediction": int(prediction)
    }
    
    result = "normal" if prediction == 1 else "not normal"
    
    return {
        "result": result,
        "raw_prediction": int(prediction)
    }
