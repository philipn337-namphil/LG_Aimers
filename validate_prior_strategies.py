import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss


TARGET_COL = "control_success"
EPS = 1e-6


def clip_prob(x):
    return np.clip(np.asarray(x, dtype=np.float64), EPS, 1.0 - EPS)


def logit(p):
    p = clip_prob(p)
    return np.log(p / (1.0 - p))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def smoothed_rate(train, group_cols, valid, alpha=200):
    global_rate = train[TARGET_COL].mean()
    stats = train.groupby(group_cols, dropna=False)[TARGET_COL].agg(["sum", "count"])
    rates = (stats["sum"] + alpha * global_rate) / (stats["count"] + alpha)
    if len(group_cols) == 1:
        keys = valid[group_cols[0]]
    else:
        keys = list(map(tuple, valid[group_cols].to_numpy()))
    return pd.Series(keys, index=valid.index).map(rates.to_dict()).fillna(global_rate).to_numpy()


def recent_weighted_global(train):
    years = sorted(train["season"].unique())
    weights = np.arange(1, len(years) + 1, dtype=np.float64)
    rates = np.array([train.loc[train["season"] == y, TARGET_COL].mean() for y in years])
    return float(np.average(rates, weights=weights))


def linear_forecast_rate(train, group_cols, valid, min_n=3000, clamp=(0.42, 0.62)):
    global_pred = linear_forecast_one(train.groupby("season")[TARGET_COL].mean())
    stats = train.groupby([*group_cols, "season"], dropna=False)[TARGET_COL].agg(["mean", "count"]).reset_index()
    forecasts = {}
    for key, sub in stats.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        if len(sub) >= 2 and sub["count"].sum() >= min_n:
            rate = linear_forecast_one(sub.set_index("season")["mean"])
        else:
            rate = global_pred
        forecasts[key] = float(np.clip(rate, *clamp))

    if len(group_cols) == 1:
        keys = [(x,) for x in valid[group_cols[0]].to_numpy()]
    else:
        keys = list(map(tuple, valid[group_cols].to_numpy()))
    return np.asarray([forecasts.get(k, global_pred) for k in keys])


def linear_forecast_one(year_rate):
    s = year_rate.dropna().sort_index()
    if len(s) < 2:
        return float(s.iloc[-1])
    x = s.index.to_numpy(dtype=np.float64)
    y = s.to_numpy(dtype=np.float64)
    if len(s) > 4:
        x = x[-4:]
        y = y[-4:]
    slope, intercept = np.polyfit(x, y, 1)
    return float(intercept + slope * (x[-1] + 1))


def evaluate(y, pred):
    pred = clip_prob(pred)
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
    parser.add_argument("--output-path", default="output/prior_strategy_results.csv")
    parser.add_argument("--start-valid-season", type=int, default=2021)
    args = parser.parse_args()

    df = pd.read_csv(args.train_path, encoding="utf-8-sig")
    rows = []
    for valid_season in sorted(y for y in df["season"].unique() if y >= args.start_valid_season):
        train = df[df["season"] < valid_season].copy()
        valid = df[df["season"] == valid_season].copy()
        y = valid[TARGET_COL].astype("int8")

        strategies = {
            "global_all_history": np.full(len(valid), train[TARGET_COL].mean()),
            "global_recent_weighted": np.full(len(valid), recent_weighted_global(train)),
            "global_linear_trend": np.full(
                len(valid), np.clip(linear_forecast_one(train.groupby("season")[TARGET_COL].mean()), 0.42, 0.62)
            ),
            "game_type_smoothed": smoothed_rate(train, ["game_type"], valid, alpha=500),
            "game_type_month_smoothed": smoothed_rate(train, ["game_type", "game_month"], valid, alpha=800),
            "game_type_count_smoothed": smoothed_rate(
                train, ["game_type", "balls_before", "strikes_before"], valid, alpha=800
            ),
            "game_type_linear_trend": linear_forecast_rate(train, ["game_type"], valid, min_n=10000),
        }

        for name, pred in strategies.items():
            row = {"valid_season": int(valid_season), "method": name, "valid_rows": len(valid), **evaluate(y, pred)}
            rows.append(row)
            print(
                f"{valid_season} {name:26s} "
                f"brier={row['brier']:.6f} "
                f"pred_mean={row['pred_mean']:.6f} "
                f"target_mean={row['target_mean']:.6f} "
                f"bias={row['bias']:.6f}"
            )

    result = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    result.to_csv(args.output_path, index=False, encoding="utf-8")
    summary = result.groupby("method")["brier"].mean().sort_values().reset_index()
    summary.to_csv(args.output_path.replace(".csv", "_summary.csv"), index=False, encoding="utf-8")
    print("Summary")
    print(summary.to_string(index=False))
    print(f"Saved: {args.output_path}")


if __name__ == "__main__":
    main()
