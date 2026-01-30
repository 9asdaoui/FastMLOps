from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from contextlib import asynccontextmanager
import mlflow.pyfunc
import pandas as pd
import time
import os

from src.schemas import DiabetesInput, PredictionResponse

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


# Metrics
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total number of HTTP requests",
    ["method", "endpoint", "status"]
)

REQUEST_LATENCY = Histogram(
    "http_request_latency_seconds",
    "Latency of HTTP requests",
    ["endpoint"]
)

INFERENCE_TIME = Histogram(
    "model_inference_seconds",
    "Time spent during model inference"
)

ERROR_COUNT = Counter(
    "http_errors_total",
    "Total number of errors",
    ["endpoint"]
)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    latency = time.time() - start_time

    endpoint = request.url.path
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=endpoint,
        status=response.status_code
    ).inc()

    REQUEST_LATENCY.labels(endpoint=endpoint).observe(latency)

    if response.status_code >= 400:
        ERROR_COUNT.labels(endpoint=endpoint).inc()

    return response







@app.get("/")
def health_check():
    return {"status": "running", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(data: DiabetesInput):
    start = time.time()

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    input_data = pd.DataFrame([data.model_dump()])
    prediction = int(model.predict(input_data)[0])
    
    risk_mapping = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
    risk_level = risk_mapping.get(prediction, "Unknown")
    
    INFERENCE_TIME.observe(time.time() - start)

    return PredictionResponse(
        prediction=prediction,
        risk_level=risk_level,
        message=f"Patient classified as: {risk_level}"
    )


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)