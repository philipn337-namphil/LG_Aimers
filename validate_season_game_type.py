import argparse
import os
import time

import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from model_utils import FeatureBuilder, TARGET_COL, make_model


def evaluate_split(df: pd.DataFrame, season: int, game_type: str) -> dict:
    valid_mask = (df["season"] == season) & (df["game_type"] == game_type)
    train_df = df.loc[~valid_mask].copy()
    valid_df = df.loc[valid_mask].copy()

    started = time.time()
    y_train = train_df[TARGET_COL].astype("int8")
    y_valid = valid_df[TARGET_COL].astype("int8")

    builder = FeatureBuilder(alpha=80.0)
    X_train = builder.fit_transform(train_df.drop(columns=[TARGET_COL]), y_train)
    X_valid = builder.transform(valid_df.drop(columns=[TARGET_COL]))

    model = make_model()
    model.fit(X_train, y_train)
    pred = model.predict_proba(X_valid)[:, 1]

    return {
        "valid_season": season,
        "valid_game_type": game_type,
        "train_rows": len(train_df),
        "valid_rows": len(valid_df),
        "train_target_mean": float(y_train.mean()),
        "valid_target_mean": float(y_valid.mean()),
        "valid_pred_mean": float(pred.mean()),
        "pred_minus_target": float(pred.mean() - y_valid.mean()),
        "brier": float(brier_score_loss(y_valid, pred)),
        "logloss": float(log_loss(y_valid, pred)),
        "seconds": round(time.time() - started, 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", default="data/train.csv")
    parser.add_argument("--output-path", default="output/season_game_type_holdout.csv")
    parser.add_argument("--season", type=int, default=None, help="Evaluate one season only.")
    parser.add_argument("--game-type", default=None, help="Evaluate one game_type only, e.g. R or F.")
    parser.add_argument("--max-splits", type=int, default=None, help="Debug option: stop after N splits.")
    args = parser.parse_args()

    print("Load train...")
    df = pd.read_csv(args.train_path, encoding="utf-8-sig")
    print(f"train_shape={df.shape} target_mean={df[TARGET_COL].mean():.6f}")

    splits = (
        df[["season", "game_type"]]
        .drop_duplicates()
        .sort_values(["season", "game_type"])
        .itertuples(index=False, name=None)
    )
    splits = list(splits)
    if args.season is not None:
        splits = [s for s in splits if s[0] == args.season]
    if args.game_type is not None:
        splits = [s for s in splits if s[1] == args.game_type]
    if args.max_splits is not None:
        splits = splits[: args.max_splits]

    if not splits:
        raise ValueError("No matching season/game_type splits.")

    rows = []
    for i, (season, game_type) in enumerate(splits, start=1):
        print(f"[{i}/{len(splits)}] Holdout season={season} game_type={game_type}")
        result = evaluate_split(df, int(season), str(game_type))
        rows.append(result)
        print(
            "  "
            f"rows={result['valid_rows']} "
            f"brier={result['brier']:.6f} "
            f"logloss={result['logloss']:.6f} "
            f"pred_mean={result['valid_pred_mean']:.6f} "
            f"target_mean={result['valid_target_mean']:.6f} "
            f"bias={result['pred_minus_target']:.6f} "
            f"seconds={result['seconds']}"
        )

        out = pd.DataFrame(rows)
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        out.to_csv(args.output_path, index=False, encoding="utf-8")

    print(f"Saved: {args.output_path}")


if __name__ == "__main__":
    main()
