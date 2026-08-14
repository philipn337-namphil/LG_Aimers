import argparse
import os

import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from calibration_utils import apply_calibration, fit_calibrators
from model_utils import FeatureBuilder, TARGET_COL, make_model
from trackman_feature_utils import add_trackman_features, load_trackman_feature_tables


METHODS = [
    "raw",
    "platt",
    "isotonic",
    "global_logit_offset",
    "game_type_logit_offset",
    "blend_global_0.15",
    "blend_global_0.30",
    "blend_global_0.50",
    "blend_game_type_0.15",
    "blend_game_type_0.30",
    "blend_game_type_0.50",
]


def score(y, pred):
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
    parser.add_argument("--trackman-feature-dir", default="output/trackman_features")
    parser.add_argument("--no-trackman", action="store_true")
    parser.add_argument("--base-end-season", type=int, default=2022)
    parser.add_argument("--calibration-season", type=int, default=2023)
    parser.add_argument("--validation-season", type=int, default=2024)
    parser.add_argument("--output-path", default="output/temporal_calibration_results.csv")
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

    rows = []
    for method in METHODS:
        pred = apply_calibration(raw_valid, valid_df[["game_type"]], calibration, method)
        row = {"method": method, **score(y_valid, pred)}
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
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    result.to_csv(args.output_path, index=False, encoding="utf-8")
    print(f"Saved: {args.output_path}")


if __name__ == "__main__":
    main()
