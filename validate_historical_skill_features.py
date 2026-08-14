import argparse
import os
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from calibration_utils import apply_calibration, fit_calibrators, logit
from model_utils import FeatureBuilder, TARGET_COL


FOLDS = [
    (2019, 2020, 2021, 2022),
    (2019, 2021, 2022, 2023),
    (2019, 2022, 2023, 2024),
]
ALPHAS = [20, 50, 100, 300, 1000]
TREND_WEIGHT = 0.75
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

GROUPS = {
    "pitcher": ["pitcher_id"],
    "pitcher_game_type": ["pitcher_id", "game_type"],
    "pitcher_batter_hand": ["pitcher_id", "batter_hand"],
    "pitcher_count": ["pitcher_id", "count_state"],
    "batter": ["batter_id"],
    "pitcher_batter": ["pitcher_id", "batter_id"],
}

FEATURE_SETS = {
    "baseline": [],
    "pitcher_abs": ["hs_pitcher_n", "hs_pitcher_rate"],
    "pitcher_rel": ["hs_pitcher_n", "hs_pitcher_rate", "hs_pitcher_rel", "hs_pitcher_logit_rel"],
    "pitcher_count": [
        "hs_pitcher_n",
        "hs_pitcher_rate",
        "hs_pitcher_rel",
        "hs_pitcher_logit_rel",
        "hs_pitcher_count_n",
        "hs_pitcher_count_rate",
        "hs_pitcher_count_rel",
        "hs_pitcher_count_logit_rel",
    ],
    "pitcher_batter_hand": [
        "hs_pitcher_n",
        "hs_pitcher_rate",
        "hs_pitcher_rel",
        "hs_pitcher_logit_rel",
        "hs_pitcher_batter_hand_n",
        "hs_pitcher_batter_hand_rate",
        "hs_pitcher_batter_hand_rel",
        "hs_pitcher_batter_hand_logit_rel",
    ],
    "batter_env": [
        "hs_pitcher_n",
        "hs_pitcher_rate",
        "hs_pitcher_rel",
        "hs_pitcher_logit_rel",
        "hs_batter_n",
        "hs_batter_rate",
        "hs_batter_rel",
        "hs_batter_logit_rel",
    ],
    "combo_core": [
        "hs_pitcher_n",
        "hs_pitcher_rate",
        "hs_pitcher_rel",
        "hs_pitcher_logit_rel",
        "hs_pitcher_game_type_n",
        "hs_pitcher_game_type_rate",
        "hs_pitcher_game_type_rel",
        "hs_pitcher_game_type_logit_rel",
        "hs_pitcher_count_n",
        "hs_pitcher_count_rate",
        "hs_pitcher_count_rel",
        "hs_pitcher_count_logit_rel",
        "hs_pitcher_batter_hand_n",
        "hs_pitcher_batter_hand_rate",
        "hs_pitcher_batter_hand_rel",
        "hs_pitcher_batter_hand_logit_rel",
        "hs_batter_n",
        "hs_batter_rate",
        "hs_batter_rel",
        "hs_batter_logit_rel",
    ],
    "combo_with_pitcher_batter": [
        "hs_pitcher_n",
        "hs_pitcher_rate",
        "hs_pitcher_rel",
        "hs_pitcher_logit_rel",
        "hs_pitcher_count_n",
        "hs_pitcher_count_rate",
        "hs_pitcher_count_rel",
        "hs_pitcher_count_logit_rel",
        "hs_pitcher_batter_n",
        "hs_pitcher_batter_rate",
        "hs_pitcher_batter_rel",
        "hs_pitcher_batter_logit_rel",
    ],
}


def clip_prob(x):
    return np.clip(np.asarray(x, dtype=np.float64), 1e-6, 1 - 1e-6)


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


def add_basic(df):
    out = df.copy()
    out["count_state"] = out["balls_before"].astype(str) + "-" + out["strikes_before"].astype(str)
    return out


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


def add_rate_columns(out, name, n, s, prior, alpha):
    prior = np.asarray(prior, dtype=np.float64)
    rate = (s + alpha * prior) / (n + alpha)
    out[f"hs_{name}_n"] = n.astype("float32")
    out[f"hs_{name}_rate"] = rate.astype("float32")
    out[f"hs_{name}_rel"] = (rate - prior).astype("float32")
    out[f"hs_{name}_logit_rel"] = (logit(rate) - logit(prior)).astype("float32")


def train_historical_features(train_df, alpha):
    out = train_df.copy()
    y = out[TARGET_COL].astype(float)
    global_n = np.arange(len(out), dtype=np.float64)
    global_s = y.cumsum().shift(1).fillna(0).to_numpy(dtype=np.float64)
    global_prior = np.divide(global_s, global_n, out=np.full(len(out), 0.5), where=global_n > 0)

    for name, cols in GROUPS.items():
        grp = out.groupby(cols, dropna=False)[TARGET_COL]
        n_prev = grp.cumcount().to_numpy(dtype=np.float64)
        s_prev = (grp.cumsum() - y).to_numpy(dtype=np.float64)
        add_rate_columns(out, name, n_prev, s_prev, global_prior, alpha)
    return out


def apply_historical_from_history(df, history_df, alpha):
    out = df.copy()
    prior = float(history_df[TARGET_COL].mean()) if len(history_df) else 0.5
    prior_arr = np.full(len(out), prior, dtype=np.float64)
    for name, cols in GROUPS.items():
        stats = history_df.groupby(cols, dropna=False)[TARGET_COL].agg(["sum", "count"])
        if len(cols) == 1:
            keys = out[cols[0]]
        else:
            keys = list(map(tuple, out[cols].to_numpy()))
        s = pd.Series(keys, index=out.index).map(stats["sum"].to_dict()).fillna(0).to_numpy(dtype=np.float64)
        n = pd.Series(keys, index=out.index).map(stats["count"].to_dict()).fillna(0).to_numpy(dtype=np.float64)
        add_rate_columns(out, name, n, s, prior_arr, alpha)
    return out


def select_feature_set(df, feature_set):
    if feature_set == "baseline":
        return df.drop(columns=[c for c in df.columns if c.startswith("hs_")], errors="ignore")
    keep = set(FEATURE_SETS[feature_set])
    drop = [c for c in df.columns if c.startswith("hs_") and c not in keep]
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
    trend = (1 - TREND_WEIGHT) * platt + TREND_WEIGHT * trend_prior

    rows = []
    for stage, pred in [("raw", raw_valid), ("platt", platt), ("platt_plus_trend_w075", trend)]:
        rows.append({"stage": stage, "trend_prior": trend_prior, "train_seconds": train_seconds, **metric(y_valid, pred)})

    try:
        importance = model.get_feature_importance()
        fi = pd.DataFrame({"feature": X_train.columns, "importance": importance})
    except Exception:
        fi = pd.DataFrame(columns=["feature", "importance"])
    return rows, fi


def feature_stats(df, feature_cols):
    rows = []
    y = df[TARGET_COL].astype(float)
    for col in feature_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        corr = float(s.corr(y)) if s.notna().sum() > 2 and s.nunique(dropna=True) > 1 else np.nan
        rows.append(
            {
                "feature": col,
                "missing_rate": float(s.isna().mean()),
                "unique_count": int(s.nunique(dropna=True)),
                "mean": float(s.mean()),
                "std": float(s.std()),
                "min": float(s.min()),
                "max": float(s.max()),
                "target_corr": corr,
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", default="data/train.csv")
    parser.add_argument("--output-dir", default="output/historical_skill")
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

            hist_train = train_historical_features(raw_train, alpha)
            hist_cal = apply_historical_from_history(raw_cal, raw_train, alpha)
            hist_valid = apply_historical_from_history(raw_valid, pd.concat([raw_train, raw_cal], ignore_index=True), alpha)
            hs_cols = [c for c in hist_train.columns if c.startswith("hs_")]
            st = feature_stats(hist_valid, hs_cols)
            st["alpha"] = alpha
            st["valid_year"] = valid_year
            all_stats.append(st)

            for feature_set in feature_sets:
                rows, fi = fit_eval(hist_train, hist_cal, hist_valid, feature_set)
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

                pd.DataFrame(all_fold_metrics).to_csv(os.path.join(args.output_dir, "fold_metrics_partial.csv"), index=False, encoding="utf-8")

    fold_metrics = pd.DataFrame(all_fold_metrics)
    if "baseline" in feature_sets:
        baseline = fold_metrics[
            (fold_metrics["feature_set"] == "baseline")
            & (fold_metrics["stage"] == "platt_plus_trend_w075")
            & (fold_metrics["alpha"] == alphas[0])
        ][["valid_year", "brier"]].rename(columns={"brier": "baseline_brier"})
    else:
        prior = pd.read_csv("output/model_comparison/model_fold_metrics.csv")
        baseline = prior[
            (prior["model"] == "catboost")
            & (prior["stage"] == "platt_plus_trend_w075")
        ][["valid_year", "brier"]].rename(columns={"brier": "baseline_brier"})
    fold_metrics = fold_metrics.merge(baseline, on="valid_year", how="left")
    fold_metrics["improvement_vs_catboost_baseline"] = fold_metrics["baseline_brier"] - fold_metrics["brier"]

    fold_metrics.to_csv(os.path.join(args.output_dir, "fold_metrics.csv"), index=False, encoding="utf-8")
    pd.concat(all_stats, ignore_index=True).to_csv(os.path.join(args.output_dir, "historical_feature_stats.csv"), index=False, encoding="utf-8")
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
            folds_improved=("improvement_vs_catboost_baseline", lambda s: int((s > 0).sum())),
            mean_improvement=("improvement_vs_catboost_baseline", "mean"),
            mean_auc=("auc", "mean"),
            mean_logloss=("logloss", "mean"),
            mean_pred_std=("pred_std", "mean"),
            mean_train_seconds=("train_seconds", "mean"),
        )
        .reset_index()
        .sort_values(["mean_brier", "std_brier"])
    )
    summary.to_csv(os.path.join(args.output_dir, "feature_set_metrics.csv"), index=False, encoding="utf-8")
    summary.to_csv(os.path.join(args.output_dir, "smoothing_grid.csv"), index=False, encoding="utf-8")

    print("\nTop feature sets")
    print(summary.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
