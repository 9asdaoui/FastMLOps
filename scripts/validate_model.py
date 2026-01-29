import mlflow
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(mlflow_uri)

print(f"Loading model from MLflow: {mlflow_uri}")
model = mlflow.pyfunc.load_model("models:/BestRiskDiabetModel@champion")
print("Model loaded successfully")

data_path = os.path.join(project_root, "src/data/raw_data.csv")
df = pd.read_csv(data_path)
df = df.drop(columns=['Unnamed: 0'], errors='ignore')

cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_with_zeros] = df[cols_with_zeros].replace(0, np.nan)

imputer = KNNImputer(n_neighbors=5)
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

X = df_imputed.drop(columns=['Cluster', 'risk_category'], errors='ignore')

predictions = model.predict(X)
unique, counts = np.unique(predictions, return_counts=True)

print(f"Predictions distribution: {dict(zip(unique, counts))}")
print(f"Total samples: {len(predictions)}")

assert len(predictions) == len(X), "Prediction count mismatch"
assert len(unique) >= 2, "Model predicting only one class"

print("Model validation passed")
