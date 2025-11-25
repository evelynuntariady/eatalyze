# app.py
from fastapi import FastAPI
import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel

app = FastAPI()

model = joblib.load("rf_baseline.pkl")



class Nutrition(BaseModel):
    energy_kcal_1g: float
    fat_1g: float
    carbohydrates_1g: float
    proteins_1g: float
    saturated_fat_1g: float
    trans_fat_1g: float
    sugars_1g: float
    added_sugars_1g: float
    sodium_1g: float
    salt_1g: float
    fiber_1g: float

@app.get("/")
def read_root():
       return {"message": "Welcome to the ML Model API"}

@app.post('/predict')

def predict(nutrition: Nutrition):
    df = pd.DataFrame([nutrition.dict()])
    prediction = model.predict(df)
    return {'prediction': prediction.tolist()}

