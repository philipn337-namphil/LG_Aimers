import argparse
import os
import time

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from calibration_utils import apply_calibration, fit_calibrators
from model_utils import FeatureBuilder, TARGET_COL


CATBOOST_PARAMS = {
    "loss_function": "Logloss",
    "iterations": 220,
    "learning_rate": 0.045,
    "depth": 6,
    "l2_leaf_reg": 5.0,
    "random_seed": 42,
    "verbose": False,
    "allow_writing_files": False,
    "thread_count": -1,
}


def linear_forecast(year_rates: pd.Series) -> float:
    s = year_rates.dropna().sort_index()
    if len(s) == 1:
        return float(s.iloc[-1])
    if len(s) > 4:
        s = s.tail(4)
    x = s.index.to_numpy(dtype=np.float64)
    y = s.to_numpy(dtype=np.float64)
    slope, intercept = np.polyfit(x, y, 1)
    return float(np.clip(intercept + slope * (x[-1] + 1), 0.42, 0.62))


def score(y, pred):
    return {
        "brier": float(brier_score_loss(y, pred)),
        "logloss": float(log_loss(y, pred)),
        "auc": float(roc_auc_score(y, pred)),
        "pred_mean": float(np.mean(pred)),
        "pred_std": float(np.std(pred)),
        "actual_rate": float(np.mean(y)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", default="data/train.csv")
    parser.add_argument("--model-path", default="catboost_model/model.pkl")
    parser.add_argument("--calibration-season", type=int, default=2024)
    parser.add_argument("--prior-weight", type=float, default=0.75)
    args = parser.parse_args()

    print("Load train...")
    df = pd.read_csv(args.train_path, encoding="utf-8-sig")
    fit_df = df[df["season"] < args.calibration_season].copy()
    cal_df = df[df["season"] == args.calibration_season].copy()
    print(f"fit_rows={len(fit_df)} calibration_rows={len(cal_df)}")

    y_fit = fit_df[TARGET_COL].astype("int8")
    builder = FeatureBuilder(alpha=80.0)
    X_fit = builder.fit_transform(fit_df.drop(columns=[TARGET_COL]), y_fit)
    model = CatBoostClassifier(**CATBOOST_PARAMS)

    started = time.time()
    model.fit(X_fit, y_fit)
    train_seconds = time.time() - started

    X_cal = builder.transform(cal_df.drop(columns=[TARGET_COL]))
    y_cal = cal_df[TARGET_COL].astype("int8").to_numpy()
    started = time.time()
    raw_cal = model.predict_proba(X_cal)[:, 1]
    calibration_inference_seconds = time.time() - started
    calibration = fit_calibrators(raw_cal, y_cal, cal_df[["game_type"]])
    platt_cal = apply_calibration(raw_cal, cal_df[["game_type"]], calibration, "platt")
    trend_prior_rate = linear_forecast(df.groupby("season")[TARGET_COL].mean())
    final_cal = (1.0 - args.prior_weight) * platt_cal + args.prior_weight * trend_prior_rate

    print("calibration_raw", score(y_cal, raw_cal))
    print("calibration_platt", score(y_cal, platt_cal))
    print("calibration_final_blend", score(y_cal, final_cal))
    print(f"trend_prior_rate={trend_prior_rate:.12f} prior_weight={args.prior_weight}")
    print(f"train_seconds={train_seconds:.3f} calibration_inference_seconds={calibration_inference_seconds:.3f}")

    artifact = {
        "builder": builder,
        "model": model,
        "calibration": calibration,
        "calibration_method": "platt",
        "trend_prior_rate": trend_prior_rate,
        "prior_blend_weight": args.prior_weight,
        "catboost_params": CATBOOST_PARAMS,
        "calibration_season": args.calibration_season,
        "trackman_tables": {},
    }
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    joblib.dump(artifact, args.model_path)
    print(f"Saved: {args.model_path}")


if __name__ == "__main__":
    main()
