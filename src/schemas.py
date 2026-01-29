from pydantic import BaseModel


class DiabetesInput(BaseModel):
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float


class PredictionResponse(BaseModel):
    prediction: int
    risk_level: str
    message: str
