import argparse
import os

import joblib
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from model_utils import FeatureBuilder, TARGET_COL, make_model
from trackman_feature_utils import add_trackman_features, load_trackman_feature_tables


def train_once(train_df: pd.DataFrame, valid_df: pd.DataFrame | None):
    y_train = train_df[TARGET_COL].astype("int8")
    builder = FeatureBuilder(alpha=80.0)
    X_train = builder.fit_transform(train_df.drop(columns=[TARGET_COL]), y_train)
    model = make_model()
    model.fit(X_train, y_train)

    if valid_df is not None and len(valid_df):
        X_valid = builder.transform(valid_df.drop(columns=[TARGET_COL]))
        y_valid = valid_df[TARGET_COL].astype("int8")
        pred = model.predict_proba(X_valid)[:, 1]
        print(f"valid_rows={len(valid_df)}")
        print(f"valid_brier={brier_score_loss(y_valid, pred):.6f}")
        print(f"valid_logloss={log_loss(y_valid, pred):.6f}")
        print(f"valid_pred_mean={pred.mean():.6f} valid_y_mean={y_valid.mean():.6f}")

    return {"builder": builder, "model": model}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", default="data/train.csv")
    parser.add_argument("--model-path", default="model/model.pkl")
    parser.add_argument("--valid-season", type=int, default=2024)
    parser.add_argument("--trackman-feature-dir", default="output/trackman_features")
    parser.add_argument("--skip-valid", action="store_true")
    parser.add_argument("--no-trackman", action="store_true")
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
    print(f"train_shape={df.shape} target_mean={df[TARGET_COL].mean():.6f}")

    if not args.skip_valid:
        fit_df = df[df["season"] != args.valid_season].copy()
        valid_df = df[df["season"] == args.valid_season].copy()
        print(f"Holdout: train seasons != {args.valid_season}, valid season = {args.valid_season}")
        _ = train_once(fit_df, valid_df)

    print("Fit final model on all train...")
    artifact = train_once(df, None)
    artifact["trackman_tables"] = trackman_tables
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    joblib.dump(artifact, args.model_path)
    print(f"Saved model: {args.model_path}")


if __name__ == "__main__":
    main()
