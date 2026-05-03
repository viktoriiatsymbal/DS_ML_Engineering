from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from wine_deployment.config import CATEGORICAL_COLS

def build_preprocessor(feature_columns):
    numeric_cols = [col for col in feature_columns if col not in CATEGORICAL_COLS]
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
    return preprocessor
