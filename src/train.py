import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    confusion_matrix,
    recall_score,
    f1_score,
)

# Set experiment name
experiment_name = "risk_diabet"

# Create experiment if it doesn't exist
mlflow.set_experiment(experiment_name)


##### Define target y (risk_category) and features X.
df = pd.read_csv("./data/clustered_data.csv")
y = df["Cluster"]
X = df.drop(columns=["Cluster", "risk_category"])


##### Split data (train/test with train_test_split).
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


##### Handle class imbalance
ros = RandomOverSampler(random_state=42)
X_train_res, y_train_res = ros.fit_resample(X_train, y_train)


##### Model Training & Evaluation
models = {
    "Random Forest": {
        "model": RandomForestClassifier(random_state=42, max_depth=None, min_samples_leaf=1, min_samples_split=5, n_estimators=100),
        "hyperparameters": {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 100}
    },
    "SVM": {
        "model": SVC(random_state=42, C=10, gamma='scale', kernel='linear'),
        "hyperparameters": {'C': 10, 'gamma': 'scale', 'kernel': 'linear'}
    },
    "Gradient Boosting": {
        "model": GradientBoostingClassifier(random_state=42, learning_rate=0.05, max_depth=3, n_estimators=100),
        "hyperparameters": {'learning_rate': 0.05, 'max_depth': 3, 'n_estimators': 100}
    },
    "Decision Tree": {
        "model": DecisionTreeClassifier(random_state=42, criterion='entropy', max_depth=None, min_samples_split=2),
        "hyperparameters": {'criterion': 'entropy', 'max_depth': None, 'min_samples_split': 2}
    },
    "Logistic Regression": {
        "model": LogisticRegression(random_state=42, C=10, solver='lbfgs'),
        "hyperparameters": {'C': 10, 'solver': 'lbfgs'}
    },
    # "XGBoost": {
    #     "model": XGBClassifier(random_state=42, learning_rate=0.2, max_depth=3, n_estimators=200, subsample=0.8),
    #     "hyperparameters": {'learning_rate': 0.2, 'max_depth': 3, 'n_estimators': 200, 'subsample': 0.8}
    # }
}

best_acc = 0
best_model = None
best_model_name = ""
best_run_id = None

for name, details in models.items():
    with mlflow.start_run(run_name=name) as run:

        model = details["model"]

        # Log dataset info
        mlflow.log_param("X_train_shape", X_train_res.shape)
        mlflow.log_param("y_train_classes", len(np.unique(y_train_res)))

        # Log hyperparameters
        for h_name, h_value in details["hyperparameters"].items():    
           mlflow.log_param(h_name, h_value)

        # Train & predict
        model.fit(X_train_res, y_train_res)
        y_pred = model.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1", f1)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        cv_mean = np.mean(cv_scores)
        mlflow.log_metric("cv_std_accuracy", np.std(cv_scores))
        mlflow.log_metric("cv_mean_accuracy", cv_mean)

        # Save model as artifact
        mlflow.sklearn.save_model(model, 'models/' + name)
        mlflow.log_artifact('models/' + name)

         # Track best model
        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_model_name = name
            best_run_id = run.info.run_id



# Register only the best model
if best_model is not None:
    mlflow.sklearn.log_model(
        sk_model=best_model,
        name="best_model",
        registered_model_name="BestRiskDiabetModel"
    )


from mlflow.tracking import MlflowClient

client = MlflowClient()
client.transition_model_version_stage(
    name="BestRiskDiabetModel",
    version=1,
    stage="Production",
    archive_existing_versions=True
)

# model = mlflow.pyfunc.load_model("models:/BestRiskDiabetModel/Production")