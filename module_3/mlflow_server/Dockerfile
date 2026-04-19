FROM python:3.11-slim
WORKDIR /server
RUN pip install --no-cache-dir mlflow==3.0.1
EXPOSE 5000
CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--backend-store-uri", "sqlite:///mlflow.db", \
     "--default-artifact-root", "/server/mlruns", \
     "--workers", "1"]