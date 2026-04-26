# Homework 4

## Architecture

- Airflow: orchestrates the pipeline as a DAG
- MLflow: tracks experiments, logs metrics, and stores the model registry
- SQLite: lightweight metadata DB for Airflow (dev/test only)

## Project Structure

```
.
├── dags/
│   └── wine_preprocessing_dag.py
├── src/
│   ├── config.py
│   ├── data_download.py
│   ├── data_merge.py
│   ├── cleaning.py
│   ├── splitting.py
│   ├── preprocessing.py
│   ├── training.py
│   ├── inference.py
├── mlflow_server/
│   └── Dockerfile
├── data/                           # (automatically created)
├── mlflow_data/                    # (automatically created)
├── airflow_db/                     # (automatically created)
├── docker-compose.yaml
├── requirements.txt
└── run_all.sh
```

## How to Run

### 1. Clone the repository and navigate to the project folder

```bash
cd module_4
```

### 2. Start all services

```bash
chmod +x run_all.sh && ./run_all.sh
```

This script will:
- Detect your local user ID and write it to `.env`
- Create required directories (`mlflow_data/`, `data/`, `airflow_db/`)
- Build the MLflow Docker image and start all containers

NOTE: You may be prompted for your system password for `chown`

### 3. Wait for services to be ready (approx. 60 seconds)

Check that all 3 containers are running:

```bash
docker compose ps
```

If `airflow-scheduler` is not listed, restart it:

```bash
docker compose restart airflow-scheduler
```

### 4. Open the Airflow UI

URL: **http://localhost:8080**  
Login: `airflow` / `airflow`

### 5. Trigger the DAG

- Find `wine_preprocessing_mlflow_pipeline` in the DAG list
- Toggle it on using the switch on the left
- Click the play button to trigger DAG

The pipeline takes approximately 1–2 minutes to complete

### 6. View results in MLflow

URL: http://localhost:5001

You will see:
- Experiment `airflow-wine-preprocessing` with 3 runs (logreg, random_forest, svc)
- Metrics: `mean_cv_f1`, `mean_cv_accuracy`, `test_f1`, `test_accuracy`
- Registered model: `wine_best_model v1`

## Stopping the Project

```bash
docker compose down
```

To also remove all data and start fresh:

```bash
docker compose down
rm -rf mlflow_data/* airflow_db/* data/*
```