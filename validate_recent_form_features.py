import argparse
import os
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from calibration_utils import apply_calibration, fit_calibrators
from model_utils import FeatureBuilder, TARGET_COL


FOLDS = [
    (2019, 2020, 2021, 2022),
    (2019, 2021, 2022, 2023),
    (2019, 2022, 2023, 2024),
]
TREND_WEIGHT = 0.75
WINDOWS = [5, 10, 20, 50, 100]
ALPHAS = [5, 10, 20, 50, 100]
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

FEATURE_SETS = {
    "baseline": [],
    "recent_5": [
        "rf_pitcher_recent_5_control_rate",
        "rf_pitcher_recent_5_n",
    ],
    "recent_10": [
        "rf_pitcher_recent_10_control_rate",
        "rf_pitcher_recent_10_n",
    ],
    "recent_20": [
        "rf_pitcher_recent_20_control_rate",
        "rf_pitcher_recent_20_n",
    ],
    "recent_50": [
        "rf_pitcher_recent_50_control_rate",
        "rf_pitcher_recent_50_n",
    ],
    "recent_20_50_100": [
        "rf_pitcher_recent_20_control_rate",
        "rf_pitcher_recent_20_n",
        "rf_pitcher_recent_50_control_rate",
        "rf_pitcher_recent_50_n",
        "rf_pitcher_recent_100_control_rate",
        "rf_pitcher_recent_100_n",
    ],
    "recent_delta": [
        "rf_pitcher_recent_20_control_rate",
        "rf_pitcher_recent_50_control_rate",
        "rf_pitcher_long_term_rate",
        "rf_pitcher_form_delta_20",
        "rf_pitcher_form_delta_50",
        "rf_pitcher_form_delta_20_vs_global",
        "rf_pitcher_form_delta_50_vs_global",
    ],
    "recent_20_50_100_delta": [
        "rf_pitcher_recent_20_control_rate",
        "rf_pitcher_recent_20_n",
        "rf_pitcher_recent_50_control_rate",
        "rf_pitcher_recent_50_n",
        "rf_pitcher_recent_100_control_rate",
        "rf_pitcher_recent_100_n",
        "rf_pitcher_long_term_rate",
        "rf_pitcher_form_delta_20",
        "rf_pitcher_form_delta_50",
        "rf_pitcher_form_delta_100",
        "rf_pitcher_form_delta_20_vs_global",
        "rf_pitcher_form_delta_50_vs_global",
        "rf_pitcher_form_delta_100_vs_global",
    ],
    "context_recent": [
        "rf_pitcher_recent_50_control_rate",
        "rf_pitcher_recent_100_control_rate",
        "rf_pitcher_game_type_recent_50_control_rate",
        "rf_pitcher_count_recent_50_control_rate",
        "rf_pitcher_batter_hand_recent_50_control_rate",
        "rf_pitcher_game_type_recent_50_n",
        "rf_pitcher_count_recent_50_n",
        "rf_pitcher_batter_hand_recent_50_n",
    ],
    "best_candidate": [
        "rf_pitcher_recent_20_control_rate",
        "rf_pitcher_recent_50_control_rate",
        "rf_pitcher_recent_100_control_rate",
        "rf_pitcher_long_term_rate",
        "rf_pitcher_form_delta_20",
        "rf_pitcher_form_delta_50",
        "rf_pitcher_form_delta_100",
    ],
}


def clip_prob(pred):
    return np.clip(np.asarray(pred, dtype=np.float64), 1e-6, 1.0 - 1e-6)


def linear_forecast(year_rates: pd.Series) -> float:
    s = year_rates.dropna().sort_index()
    if len(s) > 4:
        s = s.tail(4)
    if len(s) == 1:
        return float(s.iloc[-1])
    x = s.index.to_numpy(dtype=np.float64)
    y = s.to_numpy(dtype=np.float64)
    slope, intercept = np.polyfit(x, y, 1)
    return float(np.clip(intercept + slope * (x[-1] + 1), 0.42, 0.62))


def metric(y, pred):
    pred = clip_prob(pred)
    return {
        "brier": float(brier_score_loss(y, pred)),
        "logloss": float(log_loss(y, pred)),
        "auc": float(roc_auc_score(y, pred)),
        "pred_mean": float(pred.mean()),
        "pred_std": float(pred.std()),
        "actual_rate": float(np.mean(y)),
    }


def add_basic(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_order"] = np.arange(len(out), dtype=np.int64)
    out["count_state"] = out["balls_before"].astype(str) + "-" + out["strikes_before"].astype(str)
    return out


def rolling_previous_sum_count(df: pd.DataFrame, group_cols, window: int):
    y_shifted = df.groupby(group_cols, dropna=False)[TARGET_COL].shift(1)
    success = (
        y_shifted.groupby([df[c] for c in group_cols], dropna=False)
        .rolling(window=window, min_periods=1)
        .sum()
        .reset_index(level=list(range(len(group_cols))), drop=True)
    )
    count = (
        y_shifted.notna()
        .astype("float32")
        .groupby([df[c] for c in group_cols], dropna=False)
        .rolling(window=window, min_periods=1)
        .sum()
        .reset_index(level=list(range(len(group_cols))), drop=True)
    )
    return success.fillna(0).to_numpy(dtype=np.float64), count.fillna(0).to_numpy(dtype=np.float64)


def history_tail_stats(history_df: pd.DataFrame, group_cols, window: int):
    rows = []
    for keys, group in history_df.groupby(group_cols, dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        tail = group[TARGET_COL].tail(window)
        rows.append((*keys, float(tail.sum()), int(tail.count())))
    if not rows:
        return {}
    key_cols = [f"k{i}" for i in range(len(group_cols))]
    stats = pd.DataFrame(rows, columns=key_cols + ["sum", "count"])
    if len(group_cols) == 1:
        return {
            "sum": dict(zip(stats["k0"], stats["sum"])),
            "count": dict(zip(stats["k0"], stats["count"])),
        }
    keys = list(map(tuple, stats[key_cols].to_numpy()))
    return {
        "sum": dict(zip(keys, stats["sum"])),
        "count": dict(zip(keys, stats["count"])),
    }


def history_total_stats(history_df: pd.DataFrame, group_cols):
    stats = history_df.groupby(group_cols, dropna=False)[TARGET_COL].agg(["sum", "count"])
    return stats["sum"].to_dict(), stats["count"].to_dict()


def map_group(df: pd.DataFrame, group_cols, mapping: dict):
    if len(group_cols) == 1:
        keys = df[group_cols[0]]
    else:
        keys = list(map(tuple, df[list(group_cols)].to_numpy()))
    return pd.Series(keys, index=df.index).map(mapping)


def add_rate(out, prefix, success, count, prior, alpha):
    count = np.asarray(count, dtype=np.float64)
    success = np.asarray(success, dtype=np.float64)
    prior = np.asarray(prior, dtype=np.float64)
    rate = (success + alpha * prior) / (count + alpha)
    out[f"{prefix}_n"] = count.astype("float32")
    out[f"{prefix}_control_rate"] = rate.astype("float32")
    out[f"{prefix}_fallback"] = (count == 0).astype("int8")
    return rate


def add_train_recent_features(df: pd.DataFrame, alpha: int) -> pd.DataFrame:
    out = df.sort_values("_order").copy()
    y = out[TARGET_COL].astype(float)
    global_n = np.arange(len(out), dtype=np.float64)
    global_s = y.cumsum().shift(1).fillna(0).to_numpy(dtype=np.float64)
    global_prior = np.divide(global_s, global_n, out=np.full(len(out), 0.5), where=global_n > 0)

    pitcher_prev_n = out.groupby("pitcher_id", dropna=False).cumcount().to_numpy(dtype=np.float64)
    pitcher_prev_s = (out.groupby("pitcher_id", dropna=False)[TARGET_COL].cumsum() - y).to_numpy(dtype=np.float64)
    long_rate = (pitcher_prev_s + alpha * global_prior) / (pitcher_prev_n + alpha)
    out["rf_pitcher_long_term_n"] = pitcher_prev_n.astype("float32")
    out["rf_pitcher_long_term_rate"] = long_rate.astype("float32")
    out["rf_global_historical_rate"] = global_prior.astype("float32")

    for window in WINDOWS:
        s, n = rolling_previous_sum_count(out, ["pitcher_id"], window)
        rate = add_rate(out, f"rf_pitcher_recent_{window}", s, n, long_rate, alpha)
        out[f"rf_pitcher_form_delta_{window}"] = (rate - long_rate).astype("float32")
        out[f"rf_pitcher_form_delta_{window}_vs_global"] = (rate - global_prior).astype("float32")

    for name, cols in [
        ("pitcher_game_type", ["pitcher_id", "game_type"]),
        ("pitcher_count", ["pitcher_id", "count_state"]),
        ("pitcher_batter_hand", ["pitcher_id", "batter_hand"]),
    ]:
        s, n = rolling_previous_sum_count(out, cols, 50)
        add_rate(out, f"rf_{name}_recent_50", s, n, long_rate, max(alpha, 50))
    return out.sort_values("_order")


def add_static_recent_from_history(df: pd.DataFrame, history_df: pd.DataFrame, alpha: int) -> pd.DataFrame:
    out = df.copy()
    global_prior = float(history_df[TARGET_COL].mean()) if len(history_df) else 0.5
    prior_arr = np.full(len(out), global_prior, dtype=np.float64)
    p_sum, p_count = history_total_stats(history_df, ["pitcher_id"])
    pitcher_s = map_group(out, ["pitcher_id"], p_sum).fillna(0).to_numpy(dtype=np.float64)
    pitcher_n = map_group(out, ["pitcher_id"], p_count).fillna(0).to_numpy(dtype=np.float64)
    long_rate = (pitcher_s + alpha * prior_arr) / (pitcher_n + alpha)
    out["rf_pitcher_long_term_n"] = pitcher_n.astype("float32")
    out["rf_pitcher_long_term_rate"] = long_rate.astype("float32")
    out["rf_global_historical_rate"] = prior_arr.astype("float32")

    for window in WINDOWS:
        stats = history_tail_stats(history_df, ["pitcher_id"], window)
        s = map_group(out, ["pitcher_id"], stats.get("sum", {})).fillna(0).to_numpy(dtype=np.float64)
        n = map_group(out, ["pitcher_id"], stats.get("count", {})).fillna(0).to_numpy(dtype=np.float64)
        rate = add_rate(out, f"rf_pitcher_recent_{window}", s, n, long_rate, alpha)
        out[f"rf_pitcher_form_delta_{window}"] = (rate - long_rate).astype("float32")
        out[f"rf_pitcher_form_delta_{window}_vs_global"] = (rate - prior_arr).astype("float32")

    for name, cols in [
        ("pitcher_game_type", ["pitcher_id", "game_type"]),
        ("pitcher_count", ["pitcher_id", "count_state"]),
        ("pitcher_batter_hand", ["pitcher_id", "batter_hand"]),
    ]:
        stats = history_tail_stats(history_df, cols, 50)
        s = map_group(out, cols, stats.get("sum", {})).fillna(0).to_numpy(dtype=np.float64)
        n = map_group(out, cols, stats.get("count", {})).fillna(0).to_numpy(dtype=np.float64)
        add_rate(out, f"rf_{name}_recent_50", s, n, long_rate, max(alpha, 50))
    return out


def select_feature_set(df: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    keep = set(FEATURE_SETS[feature_set])
    drop = [c for c in df.columns if c.startswith("rf_") and c not in keep]
    return df.drop(columns=drop, errors="ignore")


def fit_eval(train_df, cal_df, valid_df, feature_set):
    tr = select_feature_set(train_df, feature_set)
    ca = select_feature_set(cal_df, feature_set)
    va = select_feature_set(valid_df, feature_set)

    y_train = tr[TARGET_COL].astype("int8")
    y_cal = ca[TARGET_COL].astype("int8").to_numpy()
    y_valid = va[TARGET_COL].astype("int8").to_numpy()

    builder = FeatureBuilder(alpha=80.0)
    X_train = builder.fit_transform(tr.drop(columns=[TARGET_COL]), y_train)
    X_cal = builder.transform(ca.drop(columns=[TARGET_COL]))
    X_valid = builder.transform(va.drop(columns=[TARGET_COL]))

    model = CatBoostClassifier(**CATBOOST_PARAMS)
    started = time.time()
    model.fit(X_train, y_train)
    train_seconds = time.time() - started

    raw_cal = model.predict_proba(X_cal)[:, 1]
    raw_valid = model.predict_proba(X_valid)[:, 1]
    calibration = fit_calibrators(raw_cal, y_cal, ca[["game_type"]])
    platt = apply_calibration(raw_valid, va[["game_type"]], calibration, "platt")
    trend_prior = linear_forecast(pd.concat([tr, ca]).groupby("season")[TARGET_COL].mean())
    trend = (1.0 - TREND_WEIGHT) * platt + TREND_WEIGHT * trend_prior

    rows = []
    for stage, pred in [("raw", raw_valid), ("platt", platt), ("platt_plus_trend_w075", trend)]:
        rows.append({"stage": stage, "trend_prior": trend_prior, "train_seconds": train_seconds, **metric(y_valid, pred)})

    try:
        fi = pd.DataFrame({"feature": X_train.columns, "importance": model.get_feature_importance()})
    except Exception:
        fi = pd.DataFrame(columns=["feature", "importance"])
    return rows, fi


def rolling_feature_stats(df: pd.DataFrame, feature_cols):
    rows = []
    y = df[TARGET_COL].astype(float)
    for col in feature_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        low = s <= s.quantile(0.2) if s.nunique(dropna=True) > 2 else pd.Series(False, index=df.index)
        high = s >= s.quantile(0.8) if s.nunique(dropna=True) > 2 else pd.Series(False, index=df.index)
        rows.append(
            {
                "feature": col,
                "missing_rate": float(s.isna().mean()),
                "fallback_rate": float(pd.to_numeric(df.get(f"{col.rsplit('_control_rate', 1)[0]}_fallback", 0), errors="coerce").mean())
                if col.endswith("_control_rate")
                else np.nan,
                "unique_count": int(s.nunique(dropna=True)),
                "mean": float(s.mean()),
                "std": float(s.std()),
                "min": float(s.min()),
                "max": float(s.max()),
                "target_corr": float(s.corr(y)) if s.notna().sum() > 2 and s.nunique(dropna=True) > 1 else np.nan,
                "bottom20_target_rate": float(y[low].mean()) if low.any() else np.nan,
                "top20_target_rate": float(y[high].mean()) if high.any() else np.nan,
                "top_bottom_target_gap": float(y[high].mean() - y[low].mean()) if low.any() and high.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", default="data/train.csv")
    parser.add_argument("--output-dir", default="output/recent_form")
    parser.add_argument("--alphas", default=",".join(map(str, ALPHAS)))
    parser.add_argument("--feature-sets", default=",".join(FEATURE_SETS.keys()))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df = add_basic(pd.read_csv(args.train_path, encoding="utf-8-sig"))
    alphas = [int(x) for x in args.alphas.split(",") if x]
    feature_sets = [x for x in args.feature_sets.split(",") if x]

    all_fold_metrics = []
    all_importance = []
    all_stats = []

    for alpha in alphas:
        print(f"alpha={alpha}")
        for fold in FOLDS:
            train_start, train_end, cal_year, valid_year = fold
            raw_train = df[(df["season"] >= train_start) & (df["season"] <= train_end)].copy()
            raw_cal = df[df["season"] == cal_year].copy()
            raw_valid = df[df["season"] == valid_year].copy()
            print(f"  fold train={train_start}-{train_end} cal={cal_year} valid={valid_year}")

            rf_train = add_train_recent_features(raw_train, alpha)
            rf_cal = add_static_recent_from_history(raw_cal, raw_train, alpha)
            rf_valid = add_static_recent_from_history(raw_valid, pd.concat([raw_train, raw_cal], ignore_index=True), alpha)
            rf_cols = [c for c in rf_train.columns if c.startswith("rf_") and not c.endswith("_fallback")]
            st = rolling_feature_stats(rf_valid, rf_cols)
            st["alpha"] = alpha
            st["valid_year"] = valid_year
            all_stats.append(st)

            for feature_set in feature_sets:
                rows, fi = fit_eval(rf_train, rf_cal, rf_valid, feature_set)
                for row in rows:
                    all_fold_metrics.append(
                        {
                            "alpha": alpha,
                            "feature_set": feature_set,
                            "train_start": train_start,
                            "train_end": train_end,
                            "cal_year": cal_year,
                            "valid_year": valid_year,
                            **row,
                        }
                    )
                if len(fi):
                    fi["alpha"] = alpha
                    fi["feature_set"] = feature_set
                    fi["valid_year"] = valid_year
                    all_importance.append(fi)

                pd.DataFrame(all_fold_metrics).to_csv(
                    os.path.join(args.output_dir, "fold_metrics_partial.csv"), index=False, encoding="utf-8"
                )

    fold_metrics = pd.DataFrame(all_fold_metrics)
    baseline = fold_metrics[
        (fold_metrics["feature_set"] == "baseline")
        & (fold_metrics["stage"] == "platt_plus_trend_w075")
        & (fold_metrics["alpha"] == alphas[0])
    ][["valid_year", "brier", "auc"]].rename(columns={"brier": "baseline_brier", "auc": "baseline_auc"})
    fold_metrics = fold_metrics.merge(baseline, on="valid_year", how="left")
    fold_metrics["improvement_vs_recent_baseline"] = fold_metrics["baseline_brier"] - fold_metrics["brier"]
    fold_metrics["auc_delta_vs_recent_baseline"] = fold_metrics["auc"] - fold_metrics["baseline_auc"]
    fold_metrics.to_csv(os.path.join(args.output_dir, "fold_metrics.csv"), index=False, encoding="utf-8")

    pd.concat(all_stats, ignore_index=True).to_csv(os.path.join(args.output_dir, "rolling_feature_stats.csv"), index=False, encoding="utf-8")
    if all_importance:
        pd.concat(all_importance, ignore_index=True).to_csv(os.path.join(args.output_dir, "feature_importance.csv"), index=False, encoding="utf-8")
    else:
        pd.DataFrame(columns=["feature", "importance", "alpha", "feature_set", "valid_year"]).to_csv(
            os.path.join(args.output_dir, "feature_importance.csv"), index=False, encoding="utf-8"
        )

    trend = fold_metrics[fold_metrics["stage"] == "platt_plus_trend_w075"]
    summary = (
        trend.groupby(["alpha", "feature_set"])
        .agg(
            mean_brier=("brier", "mean"),
            std_brier=("brier", "std"),
            worst_brier=("brier", "max"),
            folds_improved=("improvement_vs_recent_baseline", lambda s: int((s > 0).sum())),
            mean_improvement=("improvement_vs_recent_baseline", "mean"),
            mean_auc=("auc", "mean"),
            mean_auc_delta=("auc_delta_vs_recent_baseline", "mean"),
            mean_logloss=("logloss", "mean"),
            mean_pred_std=("pred_std", "mean"),
            mean_train_seconds=("train_seconds", "mean"),
        )
        .reset_index()
        .sort_values(["mean_brier", "std_brier"])
    )
    summary.to_csv(os.path.join(args.output_dir, "feature_set_metrics.csv"), index=False, encoding="utf-8")
    summary.to_csv(os.path.join(args.output_dir, "smoothing_grid.csv"), index=False, encoding="utf-8")

    print("\nTop recent-form feature sets")
    print(summary.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
