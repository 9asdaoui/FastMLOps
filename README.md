# Fast MLOps – End-to-End Machine Learning Service

## Project Overview

This project implements a complete MLOps pipeline for deploying and operating a Machine Learning prediction service in production.

The goal is to design a reliable, fast, and maintainable REST API backed by a Machine Learning model, while managing the full lifecycle of the model: training, tracking, validation, deployment, monitoring, and continuous improvement.

The project combines `MLflow`, `FastAPI`, `Docker`, `GitHub Actions`, `Prometheus`, and `Grafana` to deliver a production-ready MLOps architecture.



## Architecture Overview

The system is composed of the following main components:

- **Training pipeline**
Trains multiple ML models, logs experiments, and selects the best model.

- **MLflow Tracking & Model Registry**
Tracks experiments, stores artifacts, versions models, and manages lifecycle stages.

- **Prediction API (FastAPI)**
Exposes the trained model via a REST API with input validation and Swagger UI.

- **CI/CD Pipeline (GitHub Actions)**
Automates testing, validation, quality checks, and deployment.

- **Monitoring Stack**
Prometheus collects metrics, Grafana visualizes dashboards, and alerts monitor system health.



## Features

### MLflow – Model Lifecycle Management

- MLflow Tracking integrated into the training script
- Automatic logging of:
    - Model hyperparameters
    - Training & validation metrics
    - Model artifacts
- Comparison of runs via MLflow UI
- Best model registered in the Model Registry
- Model promotion workflow:
    - Development → Staging → Production
- Only the Production model is deployed to the API

### Prediction API

- REST API built with FastAPI
- Model loaded from MLflow at API startup
- Pydantic schema for input validation
- Interactive API testing via Swagger UI
- Dedicated `/predict` endpoint
- `/metrics` endpoint exposing Prometheus metrics

### CI/CD with GitHub Actions

A complete automated pipeline triggered on every push:

- Code quality checks (linting, formatting)
- Data quality validation
- Model performance validation
- Automated tests
- Docker image build (API + production model)
- Docker image versioning
- Continuous deployment after successful validation

### Monitoring & Observability

- Prometheus
    - Scrapes application metrics from /metrics
    - Collects container metrics via cAdvisor
- Grafana
    - Connected to Prometheus as a data source
    - Preconfigured dashboards
    - Real-time visualization of:
        - Request volume
        - API latency
        - Error rate
        - Model inference time
        - CPU, RAM, and network usage
    - Dashboards provisioned automatically at startup
- Alerting
    - Alerts on critical metrics (high latency, error rate, resource usage)
    - Alert rules managed via Prometheus / Alertmanager

### Containerized Deployment

The entire stack runs using Docker & Docker Compose:

- FastAPI prediction service
- MLflow server (PostgreSQL backend)
- Prometheus
- Grafana
- cAdvisor
- PostgreSQL database

This ensures:

- Reproducibility
- Easy local development
- Production-ready deployment

### Key Metrics Monitored

- Total number of requests
- Request latency
- HTTP error rate
- Model inference time
- CPU usage
- Memory usage
- Network traffic



## Tech Stack

- Python 3.10
- FastAPI
- Scikit-learn / XGBoost
- MLflow
- Docker & Docker Compose
- GitHub Actions
- Prometheus
- Grafana
- PostgreSQL



## Project Structure

```
├── .github
│   └── workflows               → GitHub Actions CI/CD pipelines (tests, build, deploy)
│
├── docker                      → Docker & infrastructure configuration
│   ├── grafana                 → Grafana configuration and provisioning
│   │   ├── dashboards          → Prebuilt Grafana dashboards (JSON)
│   │   └── provisioning        → Auto-configuration for Grafana at startup
│   │       ├── dashboards      → Dashboard provisioning definitions
│   │       └── datasources     → Prometheus datasource configuration
│   └── mlflow_data             → Persistent MLflow storage
│       └── artifacts           → Logged models, metrics, parameters, artifacts
│
├── scripts                     → Utility scripts
│
├── src                         → Main application & ML source code
│   ├── data                    → Datasets
│   └── processing              → Data preprocessing, feature engineering & model trainning
│
└── tests                       → Automated tests
```

## Installation & Initialization

Follow these steps to set up the Fast MLOps project.

**1. Clone the repository**

```Bash
git clone https://github.com/9asdaoui/FastMLOps.git
cd FastMLOps
```

**2. Build Docker images**

```Bash
docker-compose build
```

This will build all necessary images:

- FastAPI service
- MLflow server
- Prometheus
- Grafana
- cAdvisor
- PostgreSQL
- Alert Manager

**3. Start services**

```Bash
docker-compose up -d
```

Verify all containers are running:

```Bash
docker ps
```

**4. MLflow**

- Access the MLflow UI: `http://localhost:5000`

- Train your model using the provided training script:

```Bash
docker-compose exec -it api bash
cd src/processing
python model_trainning.py
```

- The model will be logged automatically to MLflow Tracking and registered in the Model Registry.

**5. FastAPI**

- API endpoint: `http://localhost:8000`
- Interactive Swagger UI: `http://localhost:8000/docs`
- Test prediction endpoint `/predict` with sample inputs.

**6. Prometheus & Grafana**

- Prometheus UI: `http://localhost:9090`
- Grafana UI: `http://localhost:3000`
    - Default credentials: **admin** / **admin** (change via env variable `GF_SECURITY_ADMIN_PASSWORD`)
    - Dashboards and data sources are provisioned automatically.

**7. Alerts**

- Alertmanager UI: `http://localhost:9093/`
- Alerts will trigger based on thresholds set for latency, error rate, or resource usage.



## Project Goals

- Demonstrate an industrial-grade MLOps workflow
- Ensure traceability and reproducibility of ML experiments
- Enable safe and controlled model deployment
- Provide full observability of the ML service in production
- Automate quality checks and deployments



## Conclusion

This project showcases how to build, deploy, monitor, and maintain a Machine Learning system using modern MLOps practices.
It serves as a strong foundation for scalable, observable, and production-ready AI services.