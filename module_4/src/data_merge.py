import pandas as pd
from src.config import INTERIM_DIR

def create_extra_source(raw_path):
    df = pd.read_csv(raw_path, sep=";")
    df = df.reset_index().rename(columns={"index": "row_id"})

    df["alcohol_segment"] = pd.cut(
        df["alcohol"],
        bins=[0, 10, 12, 100],
        labels=["low_alcohol", "medium_alcohol", "high_alcohol"])
    extra = df[["row_id", "alcohol_segment"]].copy()

    output_path = INTERIM_DIR/"extra_alcohol_segments.csv"
    extra.to_csv(output_path, index=False)
    return str(output_path)

def merge_sources(raw_path, extra_path):
    main_df = pd.read_csv(raw_path, sep=";")
    main_df = main_df.reset_index().rename(columns={"index": "row_id"})
    extra_df = pd.read_csv(extra_path)
    merged = main_df.merge(extra_df, on="row_id", how="left")

    output_path = INTERIM_DIR/"merged_wine_data.csv"
    merged.to_csv(output_path, index=False)
    return str(output_path)
