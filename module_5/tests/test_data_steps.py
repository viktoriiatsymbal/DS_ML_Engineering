from pathlib import Path
import pandas as pd
import wine_deployment.cleaning as cleaning
import wine_deployment.data_merge as data_merge
import wine_deployment.splitting as splitting

WINE_COLUMNS = [
    "fixed acidity", "volatile acidity", "citric acid",
    "residual sugar", "chlorides", "free sulfur dioxide",
    "total sulfur dioxide", "density", "pH",
    "sulphates", "alcohol", "quality"]

def build_raw_wine_df(n_rows=20):
    rows = []
    for i in range(n_rows):
        quality = 5 if i < n_rows // 2 else 6
        rows.append(
            {
                "fixed acidity": 7.0 + (i % 3) * 0.1,
                "volatile acidity": 0.50 + (i % 4) * 0.01,
                "citric acid": 0.10 + (i % 5) * 0.01,
                "residual sugar": 2.0 + (i % 3) * 0.1,
                "chlorides": 0.070 + (i % 4) * 0.001,
                "free sulfur dioxide": 10.0 + i,
                "total sulfur dioxide": 30.0 + i,
                "density": 0.996 + (i % 3) * 0.0001,
                "pH": 3.20 + (i % 4) * 0.01,
                "sulphates": 0.55 + (i % 5) * 0.01,
                "alcohol": 9.5 + (i % 6) * 0.5,
                "quality": quality,
            }
        )
    return pd.DataFrame(rows)

def test_create_extra_source_and_merge(tmp_path, monkeypatch):
    monkeypatch.setattr(data_merge, "INTERIM_DIR", tmp_path)
    raw_df = build_raw_wine_df(n_rows=10)
    raw_path = tmp_path / "winequality-red.csv"
    raw_df.to_csv(raw_path, sep=";", index=False)

    extra_path = data_merge.create_extra_source(str(raw_path))
    merged_path = data_merge.merge_sources(str(raw_path), extra_path)
    extra_df = pd.read_csv(extra_path)
    merged_df = pd.read_csv(merged_path)

    assert Path(extra_path).exists()
    assert Path(merged_path).exists()

    assert "row_id" in extra_df.columns
    assert "alcohol_segment" in extra_df.columns

    assert "row_id" in merged_df.columns
    assert "alcohol_segment" in merged_df.columns
    assert len(merged_df) == len(raw_df)

    assert merged_df["alcohol_segment"].notna().all()


def test_clean_data_creates_quality_class_and_removes_bad_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(cleaning, "INTERIM_DIR", tmp_path)

    valid_df = build_raw_wine_df(n_rows=6)
    valid_df["row_id"] = range(len(valid_df))
    valid_df["alcohol_segment"] = ["low_alcohol", "medium_alcohol", "high_alcohol"] * 2

    duplicate_row = valid_df.iloc[[0]].copy()
    invalid_quality_row = valid_df.iloc[[1]].copy()
    invalid_quality_row["row_id"] = 999
    invalid_quality_row["quality"] = 10

    missing_value_row = valid_df.iloc[[2]].copy()
    missing_value_row["row_id"] = 1000
    missing_value_row["alcohol_segment"] = None

    merged_df = pd.concat(
        [valid_df, duplicate_row, invalid_quality_row, missing_value_row],
        ignore_index=True)
    merged_path = tmp_path / "merged_wine_data.csv"
    merged_df.to_csv(merged_path, index=False)

    clean_path = cleaning.clean_data(str(merged_path))
    clean_df = pd.read_csv(clean_path)

    assert Path(clean_path).exists()
    assert "quality_class" in clean_df.columns

    assert clean_df.duplicated().sum() == 0
    assert clean_df["quality"].between(3, 8).all()
    assert clean_df[cleaning.NUMERIC_COLUMNS + ["alcohol_segment"]].notna().all().all()

    expected_classes = (clean_df["quality"] >= 6).astype(int)
    assert clean_df["quality_class"].equals(expected_classes)

def test_split_train_test_creates_expected_files(tmp_path, monkeypatch):
    monkeypatch.setattr(splitting, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(splitting, "FOLDS_DIR", tmp_path / "folds")

    splitting.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    splitting.FOLDS_DIR.mkdir(parents=True, exist_ok=True)

    clean_df = build_raw_wine_df(n_rows=20)
    clean_df["row_id"] = range(len(clean_df))
    clean_df["alcohol_segment"] = ["low_alcohol", "medium_alcohol"] * 10
    clean_df["quality_class"] = (clean_df["quality"] >= 6).astype(int)

    clean_path = tmp_path / "clean_wine_data.csv"
    clean_df.to_csv(clean_path, index=False)

    split_paths = splitting.split_train_test(str(clean_path))

    for path in split_paths.values():
        assert Path(path).exists()

    X_train = pd.read_csv(split_paths["X_train_raw"])
    X_test = pd.read_csv(split_paths["X_test_raw"])
    y_train = pd.read_csv(split_paths["y_train"])
    y_test = pd.read_csv(split_paths["y_test"])

    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)

    assert "quality" not in X_train.columns
    assert "quality_class" not in X_train.columns
    assert "row_id" not in X_train.columns

    assert "alcohol_segment" in X_train.columns
    assert set(y_train["quality_class"]).issubset({0, 1})
    assert set(y_test["quality_class"]).issubset({0, 1})

def test_create_cv_folds_creates_valid_folds(tmp_path, monkeypatch):
    monkeypatch.setattr(splitting, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(splitting, "FOLDS_DIR", tmp_path / "folds")

    splitting.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    splitting.FOLDS_DIR.mkdir(parents=True, exist_ok=True)

    clean_df = build_raw_wine_df(n_rows=30)
    clean_df["row_id"] = range(len(clean_df))
    clean_df["alcohol_segment"] = ["low_alcohol", "medium_alcohol", "high_alcohol"] * 10
    clean_df["quality_class"] = (clean_df["quality"] >= 6).astype(int)

    clean_path = tmp_path / "clean_wine_data.csv"
    clean_df.to_csv(clean_path, index=False)

    split_paths = splitting.split_train_test(str(clean_path))
    folds_path = splitting.create_cv_folds(split_paths)

    assert Path(folds_path).exists()

    import json

    with open(folds_path, "r", encoding="utf-8") as file:
        folds = json.load(file)

    assert len(folds) == 5

    for fold in folds:
        assert "fold_id" in fold
        assert "train_idx" in fold
        assert "val_idx" in fold
        assert len(fold["train_idx"]) > 0
        assert len(fold["val_idx"]) > 0
