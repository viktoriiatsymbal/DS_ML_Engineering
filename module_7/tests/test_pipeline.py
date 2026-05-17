import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from wine_deployment.preprocessing import build_preprocessor

def build_sample_dataframe():
    X = pd.DataFrame(
        [
            {
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
            },
            {
                "fixed acidity": 7.8,
                "volatile acidity": 0.58,
                "citric acid": 0.02,
                "residual sugar": 2.0,
                "chlorides": 0.073,
                "free sulfur dioxide": 9.0,
                "total sulfur dioxide": 18.0,
                "density": 0.9968,
                "pH": 3.36,
                "sulphates": 0.57,
                "alcohol": 12.1,
                "alcohol_segment": "high_alcohol",
            },
            {
                "fixed acidity": 8.5,
                "volatile acidity": 0.28,
                "citric acid": 0.56,
                "residual sugar": 1.8,
                "chlorides": 0.092,
                "free sulfur dioxide": 35.0,
                "total sulfur dioxide": 103.0,
                "density": 0.9969,
                "pH": 3.30,
                "sulphates": 0.75,
                "alcohol": 10.5,
                "alcohol_segment": "medium_alcohol",
            },
            {
                "fixed acidity": 6.7,
                "volatile acidity": 0.32,
                "citric acid": 0.44,
                "residual sugar": 2.4,
                "chlorides": 0.061,
                "free sulfur dioxide": 24.0,
                "total sulfur dioxide": 34.0,
                "density": 0.9948,
                "pH": 3.29,
                "sulphates": 0.80,
                "alcohol": 11.6,
                "alcohol_segment": "medium_alcohol"}])
    y = pd.Series([0, 1, 0, 1], name="quality_class")
    return X, y

def test_preprocessing_and_model_prediction_work():
    X, y = build_sample_dataframe()
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(list(X.columns))),
            ("model", LogisticRegression(max_iter=1000, solver="liblinear")),
        ]
    )
    pipeline.fit(X, y)
    predictions = pipeline.predict(X)

    assert len(predictions) == len(X)
    assert set(predictions).issubset({0, 1})
