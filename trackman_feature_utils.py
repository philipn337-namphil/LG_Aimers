import os

import pandas as pd


def load_trackman_feature_tables(feature_dir: str = "output/trackman_features") -> dict:
    tables = {}
    pitcher_path = os.path.join(feature_dir, "pitcher_trackman_features.csv")
    batter_path = os.path.join(feature_dir, "batter_trackman_features.csv")
    if os.path.exists(pitcher_path):
        tables["pitcher"] = pd.read_csv(pitcher_path)
    if os.path.exists(batter_path):
        tables["batter"] = pd.read_csv(batter_path)
    return tables


def add_trackman_features(df: pd.DataFrame, tables: dict | None) -> pd.DataFrame:
    if not tables:
        return df
    out = df.copy()
    if "pitcher" in tables:
        out = out.merge(tables["pitcher"], on="pitcher_id", how="left")
    if "batter" in tables:
        out = out.merge(tables["batter"], on="batter_id", how="left")
    for col in out.columns:
        if col.startswith("tm_") and col.endswith("_matched"):
            out[col] = out[col].fillna(0)
    return out
