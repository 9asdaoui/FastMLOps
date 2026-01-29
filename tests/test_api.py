from fastapi.testclient import TestClient
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert "status" in response.json()


def test_predict_endpoint():
    data = {
        "Pregnancies": 6,
        "Glucose": 148,
        "BloodPressure": 72,
        "SkinThickness": 35,
        "Insulin": 0,
        "BMI": 33.6,
        "DiabetesPedigreeFunction": 0.627,
        "Age": 50
    }
    response = client.post("/predict", json=data)
    assert response.status_code in [200, 503]


def test_predict_invalid_data():
    response = client.post("/predict", json={})
    assert response.status_code == 422
