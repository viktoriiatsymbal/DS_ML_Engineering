import pandas as pd
import wine_deployment.app as api_app

def sample_payload():
    return {
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
        "alcohol_segment": "low_alcohol",
    }

class DummyModel:
    def predict(self, input_df):
        assert isinstance(input_df, pd.DataFrame)
        assert len(input_df) > 0
        return [1 for _ in range(len(input_df))]

def test_payload_to_dataframe_accepts_single_object():
    payload = sample_payload()
    df = api_app.payload_to_dataframe(payload)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert "fixed acidity" in df.columns
    assert "alcohol_segment" in df.columns

def test_payload_to_dataframe_accepts_list_of_objects():
    payload = [sample_payload(), sample_payload()]
    df = api_app.payload_to_dataframe(payload)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "alcohol" in df.columns

def test_payload_to_dataframe_accepts_instances_format():
    payload = {
        "instances": [
            sample_payload(),
            sample_payload(),
            sample_payload()]}
    df = api_app.payload_to_dataframe(payload)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert "pH" in df.columns

def test_payload_to_dataframe_rejects_invalid_input():
    invalid_payload = "not a valid JSON object"

    try:
        api_app.payload_to_dataframe(invalid_payload)
        assert False, "Expected ValueError for invalid payload"
    except ValueError as exc:
        assert "Input must be" in str(exc)

def test_health_endpoint_returns_ok():
    client = api_app.app.test_client()
    response = client.get("/health")

    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}

def test_predict_endpoint_returns_predictions(monkeypatch):
    monkeypatch.setattr(api_app, "model", DummyModel())
    client = api_app.app.test_client()
    response = client.post("/predict", json=sample_payload())

    assert response.status_code == 200

    body = response.get_json()

    assert body["success"] is True
    assert body["predictions"] == [1]
    assert body["n_rows"] == 1
    assert "latency_ms" in body
    assert body["latency_ms"] >= 0

def test_predict_endpoint_accepts_batch_instances(monkeypatch):
    monkeypatch.setattr(api_app, "model", DummyModel())
    client = api_app.app.test_client()
    response = client.post(
        "/predict",
        json={
            "instances": [
                sample_payload(),
                sample_payload(),
            ]})

    assert response.status_code == 200

    body = response.get_json()

    assert body["success"] is True
    assert body["predictions"] == [1, 1]
    assert body["n_rows"] == 2
    assert "latency_ms" in body
