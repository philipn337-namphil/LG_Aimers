import argparse
import os
import tempfile
import zipfile

import joblib
import pandas as pd

from calibration_utils import apply_calibration, fit_calibrators
from hierarchical_utils import (
    ALPHA_CONTEXT,
    ALPHA_PITCHER,
    BLEND_K,
    TARGET_COL,
    build_hierarchy_tables,
    candidate_a_native,
    predict_hierarchy,
    target_rate_for_year,
)


def load_v2_artifact(path):
    path = str(path)
    if path.endswith(".zip"):
        with tempfile.TemporaryDirectory() as tmp:
            with zipfile.ZipFile(path) as z:
                z.extract("model/model.pkl", tmp)
            return joblib.load(os.path.join(tmp, "model", "model.pkl"))
    return joblib.load(path)


def build_artifact(train_path, v2_model_path, out_path):
    train = pd.read_csv(train_path, encoding="utf-8-sig")
    v2_artifact = load_v2_artifact(v2_model_path)
    builder = v2_artifact["builder"]
    model = v2_artifact["model"]
    v2_calibration = v2_artifact["calibration"]
    v2_calibration_method = v2_artifact["calibration_method"]

    cal_year = 2024
    cal = train[train["season"] == cal_year].copy()
    x_cal = builder.transform(cal.drop(columns=[TARGET_COL]))
    raw_cal = model.predict_proba(x_cal)[:, 1]
    v2_platt_cal = apply_calibration(raw_cal, cal[["game_type"]], v2_calibration, v2_calibration_method)

    cal_tables = build_hierarchy_tables(train, cal_year, ALPHA_PITCHER, ALPHA_CONTEXT)
    # Source-of-truth from validate_v3_hierarchical_robustness.py:
    # validation rows are soft-blended with V2 Platt, but calibration-year Platt
    # is fit on the pure hierarchy score returned by context_builder.
    cal_hierarchy, _, _ = predict_hierarchy(cal, cal_tables)
    hierarchy_calibration = fit_calibrators(cal_hierarchy, cal[TARGET_COL].astype("int8"), cal[["game_type"]])

    prediction_year = 2025
    inference_tables = build_hierarchy_tables(train, prediction_year, ALPHA_PITCHER, ALPHA_CONTEXT)
    year_rates = train.groupby("season")[TARGET_COL].mean().sort_index().to_dict()
    target_rate_2025 = target_rate_for_year(train, prediction_year)

    artifact = {
        "candidate_name": "C_PITCHER_GAME_probblend100",
        "alpha_pitcher": ALPHA_PITCHER,
        "alpha_context": ALPHA_CONTEXT,
        "blend_k": BLEND_K,
        "blend_space": "probability",
        "context": "pitcher_id|game_type",
        "hierarchy_tables": inference_tables,
        "calibration": hierarchy_calibration,
        "calibration_method": "platt",
        "calibration_year": cal_year,
        "prediction_year": prediction_year,
        "target_rate_2025": float(target_rate_2025),
        "year_rates": {int(k): float(v) for k, v in year_rates.items()},
        "source_history": "train seasons 2019-2024 for 2025 hierarchy; train seasons <2024 for hierarchy calibration",
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump(artifact, out_path)
    return artifact


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", default="data/train.csv")
    parser.add_argument("--v2-model-path", default="catboost_submit_v2.zip")
    parser.add_argument("--out-path", default="model/hierarchy.pkl")
    args = parser.parse_args()
    artifact = build_artifact(args.train_path, args.v2_model_path, args.out_path)
    tables = artifact["hierarchy_tables"]
    print(f"saved={args.out_path}")
    print(f"candidate={artifact['candidate_name']}")
    print(f"target_rate_2025={artifact['target_rate_2025']:.15f}")
    print(f"pitcher_rows={len(tables['pitcher'])} pitcher_game_rows={len(tables['pitcher_game'])}")
    print(f"history_rows={tables['history_rows']} history={tables['history_min_season']}-{tables['history_max_season']}")


if __name__ == "__main__":
    main()
