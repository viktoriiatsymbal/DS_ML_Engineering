set -e
LOCAL_UID=$(id -u)

cat > .env <<EOF
LOCAL_UID=$LOCAL_UID
EOF

mkdir -p mlflow_data data airflow_db
sudo chown -R "$LOCAL_UID:0" mlflow_data data airflow_db
chmod -R 775 mlflow_data

docker compose up -d --build

echo "Airflow UI: http://localhost:8080"
echo "Login: airflow / airflow"
echo "MLflow UI: http://localhost:5001"