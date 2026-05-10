set -euo pipefail

LOCAL_UID="$(id -u)"
LOCAL_GID="$(id -g)"

cat > .env <<EOF
LOCAL_UID=${LOCAL_UID}
LOCAL_GID=${LOCAL_GID}
EOF

mkdir -p data logs build mlflow_data postgres_data postgres data/.pytest_cache

sudo chown -R "${LOCAL_UID}:0" data logs build mlflow_data
sudo chown -R "${LOCAL_UID}:${LOCAL_GID}" postgres postgres_data

chmod -R 775 data logs build mlflow_data

docker compose up -d --build

echo "Airflow UI: http://localhost:8080"
echo "Login: airflow / airflow"
echo "MLflow UI: http://localhost:5001"
echo ""
echo "After DAG finishes:"
echo "docker compose --profile online up -d online-api"
echo ""
echo "Test REST endpoint:"
echo "curl http://localhost:8000/health"