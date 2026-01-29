from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import mlflow.pyfunc
import pandas as pd
import os

from schemas import DiabetesInput, PredictionResponse

model = None


def load_model():
    global model
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_uri)
    try:
        model = mlflow.pyfunc.load_model("models:/BestRiskDiabetModel@champion")
        return True
    except Exception:
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(title="Diabetes Risk Prediction API", lifespan=lifespan)


@app.get("/")
def health_check():
    return {"status": "running", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(data: DiabetesInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    input_data = pd.DataFrame([data.model_dump()])
    prediction = int(model.predict(input_data)[0])
    
    risk_mapping = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
    risk_level = risk_mapping.get(prediction, "Unknown")
    
    return PredictionResponse(
        prediction=prediction,
        risk_level=risk_level,
        message=f"Patient classified as: {risk_level}"
    )