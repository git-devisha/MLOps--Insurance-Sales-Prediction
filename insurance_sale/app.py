from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app= FastAPI()

class InputData(BaseModel):
    Gender: str
    Age: int
    HasDrvingLicense: int
    RegionID: int
    PastAccident : str
    AnnualPremium: float

model= joblib.load("insurance_sale/models/model.pkl")
@app.get("/")
async def read_root():
    return {"health check": "ok", "model_version": "v1.0"}

@app.post("insurance_sale/predict")
async def predict(input_data: InputData):
    df= pd.DataFrame([input_data.model_dump().values()], columns= input_data.model_dump().keys())
    pred= model.prediction(df)
    return {"prediction": int(pred[0])}