import json
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from src.config import PROCESSED_DIR, FOLDS_DIR

TARGET_COL = "quality_class"
DROP_COLS = ["quality", "quality_class", "row_id"]

def split_train_test(clean_path):
    df = pd.read_csv(clean_path)

    X = df.drop(columns=DROP_COLS)
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    paths = {
        "X_train_raw": str(PROCESSED_DIR / "X_train_raw.csv"),
        "X_test_raw": str(PROCESSED_DIR / "X_test_raw.csv"),
        "y_train": str(PROCESSED_DIR / "y_train.csv"),
        "y_test": str(PROCESSED_DIR / "y_test.csv"),
    }

    X_train.to_csv(paths["X_train_raw"], index=False)
    X_test.to_csv(paths["X_test_raw"], index=False)
    y_train.to_frame(TARGET_COL).to_csv(paths["y_train"], index=False)
    y_test.to_frame(TARGET_COL).to_csv(paths["y_test"], index=False)

    return paths


def create_cv_folds(split_paths):
    X_train = pd.read_csv(split_paths["X_train_raw"])
    y_train = pd.read_csv(split_paths["y_train"])[TARGET_COL]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    folds = []
    for fold_id, train_idx, val_idx in [
        (fold_id, train_idx, val_idx)
        for fold_id, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train))]:
        folds.append(
            {
                "fold_id": fold_id,
                "train_idx": train_idx.tolist(),
                "val_idx": val_idx.tolist(),
            }
        )

    output_path = FOLDS_DIR/"cv_folds.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(folds, f, indent=2)
    return str(output_path)
