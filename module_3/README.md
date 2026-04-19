# Homework 3: MLflow Wine Project

Simple ML pipeline with MLflow using Docker

## Setup and run

From the project root:

```bash
chmod +x run_all.sh
./run_all.sh
```

This script will:

- generate .env with your local UID/GID
- prepare local directories
- build Docker images
- start the MLflow server
- run the training and inference container

## MLflow UI

After startup, open:

[http://localhost:5001](http://localhost:5001)

## What it does

* compares models (LogReg, RF, SVC)
* uses different feature sets
* selects best model via cross-validation
* logs everything to MLflow
* saves best model and runs inference

## Outputs

* `artifacts/` - results, plots, model
* `mlflow_data/` - MLflow storage
* `tmp/` - temporary files

## Cleanup

To stop and remove containers and volumes:

```bash
docker compose down -v
```

To remove generated local files:

```bash
rm -rf artifacts mlflow_data tmp .env
```