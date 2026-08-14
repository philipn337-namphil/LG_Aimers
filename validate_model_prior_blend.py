import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from calibration_utils import apply_calibration, fit_calibrators
from model_utils import FeatureBuilder, TARGET_COL, make_model
from validate_prior_strategies import linear_forecast_one


def evaluate(y, pred):
    return {
        "brier": float(brier_score_loss(y, pred)),
        "logloss": float(log_loss(y, pred)),
        "pred_mean": float(pred.mean()),
        "target_mean": float(y.mean()),
        "bias": float(pred.mean() - y.mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", default="data/train.csv")
    parser.add_argument("--output-path", default="output/model_prior_blend_results.csv")
    parser.add_argument("--base-end-season", type=int, default=2022)
    parser.add_argument("--calibration-season", type=int, default=2023)
    parser.add_argument("--validation-season", type=int, default=2024)
    args = parser.parse_args()

    df = pd.read_csv(args.train_path, encoding="utf-8-sig")
    base_df = df[df["season"] <= args.base_end_season].copy()
    cal_df = df[df["season"] == args.calibration_season].copy()
    valid_df = df[df["season"] == args.validation_season].copy()
    print(f"base_rows={len(base_df)} cal_rows={len(cal_df)} valid_rows={len(valid_df)}")

    y_base = base_df[TARGET_COL].astype("int8")
    builder = FeatureBuilder(alpha=80.0)
    X_base = builder.fit_transform(base_df.drop(columns=[TARGET_COL]), y_base)
    model = make_model()
    model.fit(X_base, y_base)

    X_cal = builder.transform(cal_df.drop(columns=[TARGET_COL]))
    y_cal = cal_df[TARGET_COL].astype("int8")
    cal_pred = model.predict_proba(X_cal)[:, 1]
    calibration = fit_calibrators(cal_pred, y_cal, cal_df[["game_type"]])

    X_valid = builder.transform(valid_df.drop(columns=[TARGET_COL]))
    y_valid = valid_df[TARGET_COL].astype("int8")
    raw_valid = model.predict_proba(X_valid)[:, 1]
    platt_valid = apply_calibration(raw_valid, valid_df[["game_type"]], calibration, "platt")

    prior_rate = np.clip(linear_forecast_one(df[df["season"] <= args.calibration_season].groupby("season")[TARGET_COL].mean()), 0.42, 0.62)
    prior_valid = np.full(len(valid_df), prior_rate)
    print(f"prior_rate={prior_rate:.6f}")

    rows = []
    for w in np.linspace(0, 1, 21):
        pred = (1.0 - w) * platt_valid + w * prior_valid
        row = {"prior_weight": round(float(w), 2), **evaluate(y_valid, pred)}
        rows.append(row)
        print(
            f"prior_weight={w:.2f} "
            f"brier={row['brier']:.6f} "
            f"logloss={row['logloss']:.6f} "
            f"pred_mean={row['pred_mean']:.6f} "
            f"target_mean={row['target_mean']:.6f} "
            f"bias={row['bias']:.6f}"
        )

    result = pd.DataFrame(rows).sort_values("brier")
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    result.to_csv(args.output_path, index=False, encoding="utf-8")
    print("Best")
    print(result.head(5).to_string(index=False))
    print(f"Saved: {args.output_path}")


if __name__ == "__main__":
    main()
