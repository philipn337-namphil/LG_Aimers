import os

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


OUT_DIR = "output/leaderboard_zero_diagnosis"
OOF_PATH = "output/model_comparison/oof_predictions.csv"
METRICS_PATH = "output/model_comparison/model_fold_metrics.csv"
TRAIN_PATH = "data/train.csv"


def clip_prob(x):
    return np.clip(np.asarray(x, dtype=np.float64), 1e-6, 1.0 - 1e-6)


def linear_forecast(year_rates: pd.Series) -> float:
    s = year_rates.dropna().sort_index()
    if len(s) == 1:
        return float(s.iloc[-1])
    x = s.index.to_numpy(dtype=np.float64)
    y = s.to_numpy(dtype=np.float64)
    slope, intercept = np.polyfit(x, y, 1)
    return float(np.clip(intercept + slope * (x[-1] + 1), 0.42, 0.62))


def score_from_brier(brier, r):
    denom = r * (1.0 - r)
    return max(0.0, 100000.0 * (1.0 - brier / denom))


def metric_row(y, pred):
    pred = clip_prob(pred)
    actual = float(np.mean(y))
    brier = float(brier_score_loss(y, pred))
    return {
        "brier": brier,
        "auc": float(roc_auc_score(y, pred)),
        "logloss": float(log_loss(y, pred)),
        "pred_mean": float(np.mean(pred)),
        "pred_std": float(np.std(pred)),
        "actual_rate": actual,
        "pred_minus_actual": float(np.mean(pred) - actual),
        "avg_rate_brier": float(actual * (1.0 - actual)),
        "pseudo_score": score_from_brier(brier, actual),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    train = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig", usecols=["season", "control_success"])
    yearly = (
        train.groupby("season")["control_success"]
        .agg(sample_count="count", control_success_mean="mean")
        .reset_index()
        .sort_values("season")
    )
    yearly["avg_rate_brier"] = yearly["control_success_mean"] * (1.0 - yearly["control_success_mean"])
    yearly.to_csv(os.path.join(OUT_DIR, "yearly_global_rates.csv"), index=False, encoding="utf-8")

    rates = yearly.set_index("season")["control_success_mean"]
    fold_priors = {}
    for valid_year in [2022, 2023, 2024]:
        cal_year = valid_year - 1
        hist = rates.loc[rates.index <= cal_year]
        fold_priors[valid_year] = {
            "linear_trend_prior": linear_forecast(hist),
            "previous_year_rate": float(rates.loc[cal_year]),
            "history_mean_rate": float(train.loc[train["season"] <= cal_year, "control_success"].mean()),
        }

    oof = pd.read_csv(OOF_PATH, usecols=["valid_year", "target", "catboost_raw", "catboost_platt"])
    prior_df = pd.DataFrame.from_dict(fold_priors, orient="index")
    prior_df.index.name = "valid_year"
    prior_df = prior_df.reset_index()
    oof = oof.merge(prior_df, on="valid_year", how="left")

    rows = []
    for valid_year, fold in oof.groupby("valid_year", sort=True):
        y = fold["target"].to_numpy(dtype=np.int8)
        raw = fold["catboost_raw"].to_numpy(dtype=np.float64)
        platt = fold["catboost_platt"].to_numpy(dtype=np.float64)
        trend_prior = float(fold["linear_trend_prior"].iloc[0])
        prev_prior = float(fold["previous_year_rate"].iloc[0])
        hist_prior = float(fold["history_mean_rate"].iloc[0])

        variants = {
            "A_catboost_raw": raw,
            "B_catboost_platt": platt,
            "C_platt_trend_w025": 0.75 * platt + 0.25 * trend_prior,
            "D_platt_trend_w050": 0.50 * platt + 0.50 * trend_prior,
            "E_platt_trend_w075": 0.25 * platt + 0.75 * trend_prior,
            "F_trend_prior_w100": np.full(len(fold), trend_prior, dtype=np.float64),
            "G_prev_year_prior_w075": 0.25 * platt + 0.75 * prev_prior,
            "H_history_mean_prior_w075": 0.25 * platt + 0.75 * hist_prior,
            "I_platt_weak_trend_w010": 0.90 * platt + 0.10 * trend_prior,
            "J_platt_weak_trend_w020": 0.80 * platt + 0.20 * trend_prior,
            "K_platt_weak_trend_w030": 0.70 * platt + 0.30 * trend_prior,
        }
        for name, pred in variants.items():
            prior_type = "none"
            prior_value = np.nan
            trend_weight = 0.0
            if "trend" in name:
                prior_type = "linear_trend"
                prior_value = trend_prior
                if "w010" in name:
                    trend_weight = 0.10
                elif "w020" in name:
                    trend_weight = 0.20
                elif "w025" in name:
                    trend_weight = 0.25
                elif "w030" in name:
                    trend_weight = 0.30
                elif "w050" in name:
                    trend_weight = 0.50
                elif "w075" in name:
                    trend_weight = 0.75
                elif "w100" in name:
                    trend_weight = 1.00
            elif "prev_year" in name:
                prior_type = "previous_year_rate"
                prior_value = prev_prior
                trend_weight = 0.75
            elif "history_mean" in name:
                prior_type = "history_mean_rate"
                prior_value = hist_prior
                trend_weight = 0.75
            rows.append(
                {
                    "valid_year": int(valid_year),
                    "variant": name,
                    "prior_type": prior_type,
                    "prior_value": prior_value,
                    "prior_weight": trend_weight,
                    **metric_row(y, pred),
                }
            )

    fold_metrics = pd.DataFrame(rows)
    fold_metrics.to_csv(os.path.join(OUT_DIR, "variant_fold_metrics.csv"), index=False, encoding="utf-8")
    fold_metrics[
        [
            "valid_year",
            "variant",
            "brier",
            "actual_rate",
            "avg_rate_brier",
            "pseudo_score",
            "pred_mean",
            "pred_std",
            "pred_minus_actual",
        ]
    ].to_csv(os.path.join(OUT_DIR, "pseudo_scores.csv"), index=False, encoding="utf-8")

    summary = (
        fold_metrics.groupby("variant")
        .agg(
            mean_brier=("brier", "mean"),
            std_brier=("brier", "std"),
            worst_brier=("brier", "max"),
            mean_auc=("auc", "mean"),
            min_auc=("auc", "min"),
            mean_logloss=("logloss", "mean"),
            mean_pred_std=("pred_std", "mean"),
            mean_abs_pred_bias=("pred_minus_actual", lambda s: float(np.mean(np.abs(s)))),
            mean_pseudo_score=("pseudo_score", "mean"),
            min_pseudo_score=("pseudo_score", "min"),
            zero_score_folds=("pseudo_score", lambda s: int((s <= 0).sum())),
        )
        .reset_index()
        .sort_values(["mean_brier", "worst_brier"])
    )
    summary.to_csv(os.path.join(OUT_DIR, "variant_summary.csv"), index=False, encoding="utf-8")

    print("Yearly rates")
    print(yearly.to_string(index=False))
    print("\nFold priors")
    print(prior_df.to_string(index=False))
    print("\nTop variants")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
