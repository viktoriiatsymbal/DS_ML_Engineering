FROM python:3.11-slim
WORKDIR /server
RUN pip install --no-cache-dir mlflow==3.0.1
RUN mkdir -p /server/mlruns && chmod -R 777 /server
EXPOSE 5000
CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--backend-store-uri", "sqlite:////server/mlflow.db", \
     "--default-artifact-root", "mlflow-artifacts:/", \
     "--artifacts-destination", "/server/mlruns", \
     "--serve-artifacts", \
     "--workers", "1"]