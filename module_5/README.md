# HW5: Model Deployment

The project implements model deployment in two modes: batch prediction as an Airflow pipeline and online prediction as a Flask REST API

## Project Structure

```text
.
├── dags/
│   └── wine_deployment_dag.py
├── src/
│   └── wine_deployment/
│       ├── app.py
│       ├── batch_image.py
│       ├── cleaning.py
│       ├── config.py
│       ├── data_download.py
│       ├── data_merge.py
│       ├── inference.py
│       ├── preprocessing.py
│       ├── splitting.py
│       └── training.py
├── tests/
│   ├── test_pipeline.py
│   ├── test_data_steps.py
│   └── test_api.py
├── docker-compose.yaml
├── Dockerfile
├── online.Dockerfile
├── pyproject.toml
├── requirements.txt
└── run_all.sh
```

## Services used

- PostgreSQL for Airflow metadata
- Airflow webserver
- Airflow scheduler
- MLflow tracking server
- Cron trigger for scheduled batch execution
- Online Flask API container

## How to Run

### 1. Start all services

```bash
bash run_all.sh
```

After startup:

```text
Airflow UI: http://localhost:8080
Login: airflow / airflow

MLflow UI: http://localhost:5001
```

### 2. Open Airflow

Go to:

```text
http://localhost:8080
```

Login:

```text
airflow / airflow
```

Find DAG:

```text
wine_batch_deployment_pipeline
```

### 3. Trigger batch pipeline manually

You can trigger the DAG from Airflow UI or from terminal:

```bash
docker compose exec cron-trigger curl -i -X POST \
  -u airflow:airflow \
  -H "Content-Type: application/json" \
  -d '{"conf": {}}' \
  http://airflow-webserver:8080/api/v1/dags/wine_batch_deployment_pipeline/dagRuns
```

Expected response:

```text
HTTP/1.1 200 OK
```

### 4. Check batch outputs

After the DAG finishes, check:

```bash
ls -lah data/reports
```

Expected files include:

```text
batch_predictions.csv
batch_latency.txt
experiment_results.csv
best_model_info.json
model_uri.txt
```

Preview predictions:

```bash
head data/reports/batch_predictions.csv
cat data/reports/batch_latency.txt
```

### 5. Run online API

After the DAG builds the online Docker image, start the REST API:

```bash
docker compose --profile online up -d online-api
```

Check health endpoint:

```bash
curl http://localhost:8000/health
```

Expected:

```json
{"status":"ok"}
```

### 6. Test online prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "fixed acidity": 7.4,
    "volatile acidity": 0.70,
    "citric acid": 0.00,
    "residual sugar": 1.9,
    "chlorides": 0.076,
    "free sulfur dioxide": 11.0,
    "total sulfur dioxide": 34.0,
    "density": 0.9978,
    "pH": 3.51,
    "sulphates": 0.56,
    "alcohol": 9.4,
    "alcohol_segment": "low_alcohol"
  }'
```

Expected response format:

```json
{
  "success": true,
  "predictions": [0],
  "n_rows": 1,
  "latency_ms": 10.5
}
```

## Running Tests

Run tests locally:

```bash
pytest -v
```

Run tests with coverage:

```bash
pytest -v --cov=wine_deployment --cov-report=term-missing
```

Or inside the Airflow container:

```bash
docker compose exec airflow-scheduler bash -c "cd /opt/airflow/project && pytest -v"
```

Run tests with coverage inside the Airflow container:

```bash
docker compose exec airflow-scheduler bash -c "cd /opt/airflow/project && pytest -v --cov=wine_deployment --cov-report=term-missing"
```

## Checking Cron Trigger

The DAG is triggered externally by the cron container

To check that cron can access Airflow REST API:

```bash
docker compose exec cron-trigger curl -i -u airflow:airflow \
  http://airflow-webserver:8080/api/v1/dags/wine_batch_deployment_pipeline
```

Expected:

```text
HTTP/1.1 200 OK
```

To trigger the DAG through the same API call used by cron:

```bash
docker compose exec cron-trigger curl -i -X POST \
  -u airflow:airflow \
  -H "Content-Type: application/json" \
  -d '{"conf": {}}' \
  http://airflow-webserver:8080/api/v1/dags/wine_batch_deployment_pipeline/dagRuns
```

Expected:

```text
HTTP/1.1 200 OK
```

## Stop Services

```bash
docker compose down
```

To remove volumes as well:

```bash
docker compose down -v
```