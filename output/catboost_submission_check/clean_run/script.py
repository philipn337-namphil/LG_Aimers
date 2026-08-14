import os
import time

import joblib
import pandas as pd

from calibration_utils import apply_calibration


ID_COL = "row_id"
TARGET_COL = "control_success"


def load_test(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    if ID_COL not in df.columns:
        raise ValueError(f"{ID_COL} column is missing from test data.")
    return df


def load_sample_submission(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    if list(df.columns[:2]) != [ID_COL, TARGET_COL]:
        raise ValueError(f"sample_submission columns are invalid: {list(df.columns)}")
    return df


def main():
    started = time.time()
    test_dir = "./data"
    model_dir = "./model"
    out_dir = "./output"

    test = load_test(os.path.join(test_dir, "test.csv"))
    sub = load_sample_submission(os.path.join(test_dir, "sample_submission.csv"))
    artifact = joblib.load(os.path.join(model_dir, "model.pkl"))

    builder = artifact["builder"]
    model = artifact["model"]
    calibration = artifact["calibration"]
    calibration_method = artifact["calibration_method"]
    prior_weight = artifact["prior_blend_weight"]
    trend_prior = artifact["trend_prior_rate"]

    X = builder.transform(test)
    pred = model.predict_proba(X)[:, 1]
    pred = apply_calibration(pred, test[["game_type"]], calibration, calibration_method)
    pred = (1.0 - prior_weight) * pred + prior_weight * trend_prior

    pred_map = dict(zip(test[ID_COL], pred))
    sub[TARGET_COL] = sub[ID_COL].map(pred_map).fillna(sub[TARGET_COL]).clip(0.0, 1.0)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "submission.csv")
    sub.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Saved: {out_path} rows={len(sub)} seconds={time.time() - started:.3f}")


if __name__ == "__main__":
    main()
