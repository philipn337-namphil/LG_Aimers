import argparse
import os

import joblib
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from calibration_utils import apply_calibration, fit_calibrators
from model_utils import FeatureBuilder, TARGET_COL, make_model
from trackman_feature_utils import add_trackman_features, load_trackman_feature_tables
from validate_prior_strategies import linear_forecast_one


def train_base(train_df: pd.DataFrame):
    y_train = train_df[TARGET_COL].astype("int8")
    builder = FeatureBuilder(alpha=80.0)
    X_train = builder.fit_transform(train_df.drop(columns=[TARGET_COL]), y_train)
    model = make_model()
    model.fit(X_train, y_train)
    return builder, model


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
    parser.add_argument("--model-path", default="model/model.pkl")
    parser.add_argument("--calibration-season", type=int, default=2024)
    parser.add_argument("--trackman-feature-dir", default="output/trackman_features")
    parser.add_argument("--no-trackman", action="store_true")
    parser.add_argument("--prior-blend-weight", type=float, default=0.75)
    parser.add_argument(
        "--save-method",
        choices=["raw", "platt", "isotonic", "global_logit_offset", "game_type_logit_offset", "best"],
        default="best",
    )
    args = parser.parse_args()

    print("Load train...")
    df = pd.read_csv(args.train_path, encoding="utf-8-sig")
    trackman_tables = {} if args.no_trackman else load_trackman_feature_tables(args.trackman_feature_dir)
    if trackman_tables:
        df = add_trackman_features(df, trackman_tables)
        print(
            "trackman_features="
            f"pitcher:{len(trackman_tables.get('pitcher', []))} "
            f"batter:{len(trackman_tables.get('batter', []))}"
        )

    train_df = df[df["season"] < args.calibration_season].copy()
    cal_df = df[df["season"] == args.calibration_season].copy()
    print(f"base_train_rows={len(train_df)} calibration_rows={len(cal_df)}")

    builder, model = train_base(train_df)
    X_cal = builder.transform(cal_df.drop(columns=[TARGET_COL]))
    y_cal = cal_df[TARGET_COL].astype("int8")
    raw_pred = model.predict_proba(X_cal)[:, 1]
    calibration = fit_calibrators(raw_pred, y_cal, cal_df[["game_type"]])

    methods = ["raw", "platt", "isotonic", "global_logit_offset", "game_type_logit_offset"]
    rows = []
    for method in methods:
        pred = apply_calibration(raw_pred, cal_df[["game_type"]], calibration, method)
        row = {"method": method, **evaluate(y_cal, pred)}
        rows.append(row)
        print(
            f"{method:22s} "
            f"brier={row['brier']:.6f} "
            f"logloss={row['logloss']:.6f} "
            f"pred_mean={row['pred_mean']:.6f} "
            f"target_mean={row['target_mean']:.6f} "
            f"bias={row['bias']:.6f}"
        )

    result = pd.DataFrame(rows).sort_values("brier")
    os.makedirs("output", exist_ok=True)
    result.to_csv("output/calibration_2024_results.csv", index=False, encoding="utf-8")

    selected_method = result.iloc[0]["method"] if args.save_method == "best" else args.save_method
    trend_prior_rate = float(
        max(0.42, min(0.62, linear_forecast_one(df.groupby("season")[TARGET_COL].mean())))
    )
    artifact = {
        "builder": builder,
        "model": model,
        "trackman_tables": trackman_tables,
        "calibration": calibration,
        "calibration_method": selected_method,
        "calibration_season": args.calibration_season,
        "prior_blend_weight": args.prior_blend_weight,
        "trend_prior_rate": trend_prior_rate,
    }
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    joblib.dump(artifact, args.model_path)
    print(f"selected_method={selected_method}")
    print(f"trend_prior_rate={trend_prior_rate:.6f} prior_blend_weight={args.prior_blend_weight:.2f}")
    print(f"Saved calibrated model: {args.model_path}")


if __name__ == "__main__":
    main()
