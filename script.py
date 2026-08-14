import os

import joblib
import pandas as pd

from calibration_utils import apply_calibration
from trackman_feature_utils import add_trackman_features


ID_COL = "row_id"
TARGET_COL = "control_success"


def load_test(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    if ID_COL not in df.columns:
        raise ValueError(f"{ID_COL} column is missing from test data.")
    return df


def load_sample_submission(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    expected = [ID_COL, TARGET_COL]
    if list(df.columns[:2]) != expected:
        raise ValueError(f"sample_submission columns must start with {expected}.")
    return df


def main():
    test_dir = "./data"
    model_dir = "./model"
    out_dir = "./output"

    test_path = os.path.join(test_dir, "test.csv")
    sample_path = os.path.join(test_dir, "sample_submission.csv")
    model_path = os.path.join(model_dir, "model.pkl")
    out_path = os.path.join(out_dir, "submission.csv")

    print("Load model...")
    artifact = joblib.load(model_path)
    builder = artifact["builder"]
    model = artifact["model"]
    trackman_tables = artifact.get("trackman_tables", {})
    calibration = artifact.get("calibration")
    calibration_method = artifact.get("calibration_method")
    prior_blend_weight = artifact.get("prior_blend_weight", 0.0)
    trend_prior_rate = artifact.get("trend_prior_rate")

    print("Load test...")
    test = load_test(test_path)
    sub = load_sample_submission(sample_path)
    ids = test[ID_COL].copy()

    print("Build features...")
    test = add_trackman_features(test, trackman_tables)
    X = builder.transform(test)

    print("Predict...")
    preds = model.predict_proba(X)[:, 1]
    preds = apply_calibration(preds, test[["game_type"]], calibration, calibration_method)
    if trend_prior_rate is not None and prior_blend_weight:
        preds = (1.0 - prior_blend_weight) * preds + prior_blend_weight * trend_prior_rate
    pred_map = dict(zip(ids, preds))
    sub[TARGET_COL] = sub[ID_COL].map(pred_map).fillna(sub[TARGET_COL]).clip(0.0, 1.0)

    os.makedirs(out_dir, exist_ok=True)
    sub.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Saved: {out_path} rows={len(sub)}")


if __name__ == "__main__":
    main()
