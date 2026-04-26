import pickle
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.config import PROCESSED_DIR

CATEGORICAL_COLS = ["alcohol_segment"]

def fit_preprocessor(split_paths):
    X_train = pd.read_csv(split_paths["X_train_raw"])

    numeric_cols = [col for col in X_train.columns if col not in CATEGORICAL_COLS]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_COLS,
            ),
        ]
    )
    preprocessor.fit(X_train)

    output_path = PROCESSED_DIR/"preprocessor.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(preprocessor, f)
    return str(output_path)

def transform_data(split_paths, preprocessor_path):
    X_train = pd.read_csv(split_paths["X_train_raw"])
    X_test = pd.read_csv(split_paths["X_test_raw"])

    with open(preprocessor_path, "rb") as f:
        preprocessor = pickle.load(f)

    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()

    X_train_df = pd.DataFrame(X_train_transformed, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_transformed, columns=feature_names)

    paths = {
        "X_train": str(PROCESSED_DIR/"X_train_transformed.csv"),
        "X_test": str(PROCESSED_DIR/"X_test_transformed.csv"),
        "y_train": split_paths["y_train"],
        "y_test": split_paths["y_test"],
        "preprocessor": preprocessor_path}
    X_train_df.to_csv(paths["X_train"], index=False)
    X_test_df.to_csv(paths["X_test"], index=False)
    return paths
