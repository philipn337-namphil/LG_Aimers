import os

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


OUT_DIR = "output/holdout_aware_calibration"
OOF_PATH = "output/model_comparison/oof_predictions.csv"
TRAIN_PATH = "data/train.csv"
EPS = 1e-6


def clip_prob(x):
    return np.clip(np.asarray(x, dtype=np.float64), EPS, 1.0 - EPS)


def logit(p):
    p = clip_prob(p)
    return np.log(p / (1.0 - p))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def linear_forecast(year_rates: pd.Series, tail: int | None = None) -> float:
    s = year_rates.dropna().sort_index()
    if tail is not None and len(s) > tail:
        s = s.tail(tail)
    if len(s) == 1:
        return float(s.iloc[-1])
    x = s.index.to_numpy(dtype=np.float64)
    y = s.to_numpy(dtype=np.float64)
    slope, intercept = np.polyfit(x, y, 1)
    return float(np.clip(intercept + slope * (x[-1] + 1), 0.42, 0.62))


def logit_mean_match(pred, target_mean):
    target_mean = float(np.clip(target_mean, EPS, 1.0 - EPS))
    base = logit(pred)
    lo, hi = -20.0, 20.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        mean = sigmoid(base + mid).mean()
        if mean < target_mean:
            lo = mid
        else:
            hi = mid
    return clip_prob(sigmoid(base + (lo + hi) / 2.0))


def probability_mean_match(pred, target_mean):
    pred = clip_prob(pred)
    target_mean = float(np.clip(target_mean, EPS, 1.0 - EPS))
    lo, hi = -1.0, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        mean = np.clip(pred + mid, EPS, 1.0 - EPS).mean()
        if mean < target_mean:
            lo = mid
        else:
            hi = mid
    return clip_prob(pred + (lo + hi) / 2.0)


def pseudo_score(brier, actual_rate):
    denom = actual_rate * (1.0 - actual_rate)
    return max(0.0, 100000.0 * (1.0 - brier / denom))


def metrics(y, pred):
    pred = clip_prob(pred)
    actual_rate = float(np.mean(y))
    brier = float(brier_score_loss(y, pred))
    oracle = float(actual_rate * (1.0 - actual_rate))
    return {
        "brier": brier,
        "constant_brier": oracle,
        "skill_margin": oracle - brier,
        "pseudo_score": pseudo_score(brier, actual_rate),
        "auc": float(roc_auc_score(y, pred)),
        "logloss": float(log_loss(y, pred)),
        "pred_mean": float(pred.mean()),
        "pred_std": float(pred.std()),
        "actual_rate": actual_rate,
        "pred_minus_actual": float(pred.mean() - actual_rate),
    }


def weighted_recent(hist: pd.Series, weights):
    s = hist.dropna().sort_index().tail(len(weights))
    w = np.asarray(weights[-len(s):], dtype=np.float64)
    w = w / w.sum()
    return float(np.dot(s.to_numpy(dtype=np.float64), w))


def rate_estimators_for_year(year_rates: pd.Series, valid_year: int):
    hist = year_rates[year_rates.index < valid_year].dropna().sort_index()
    last = float(hist.iloc[-1])
    trend_all = linear_forecast(hist)
    trend_2 = linear_forecast(hist, tail=2)
    trend_3 = linear_forecast(hist, tail=3)
    trend_delta_all = trend_all - last
    out = {
        "A_no_adjustment": np.nan,
        "B_previous_year_rate": last,
        "C_recent_2yr_mean": float(hist.tail(2).mean()),
        "D_recent_3yr_mean": float(hist.tail(3).mean()),
        "E_weighted_060_030_010": weighted_recent(hist, [0.1, 0.3, 0.6]),
        "E_weighted_050_030_020": weighted_recent(hist, [0.2, 0.3, 0.5]),
        "F_linear_trend_all": trend_all,
        "G_linear_trend_recent2": trend_2,
        "G_linear_trend_recent3": trend_3,
    }
    for gamma in [0.0, 0.25, 0.5, 0.75, 1.0]:
        out[f"H_damped_trend_gamma_{gamma:.2f}"] = float(np.clip(last + gamma * trend_delta_all, 0.42, 0.62))
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        out[f"I_trend_last_blend_alpha_{alpha:.2f}"] = float(np.clip(alpha * trend_all + (1.0 - alpha) * last, 0.42, 0.62))
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    train = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig", usecols=["season", "control_success"])
    yearly = (
        train.groupby("season")["control_success"]
        .agg(sample_count="count", actual_rate="mean")
        .reset_index()
        .sort_values("season")
    )
    yearly["constant_brier"] = yearly["actual_rate"] * (1.0 - yearly["actual_rate"])
    yearly.to_csv(os.path.join(OUT_DIR, "yearly_global_rates.csv"), index=False, encoding="utf-8")
    year_rates = yearly.set_index("season")["actual_rate"]

    oof = pd.read_csv(OOF_PATH, usecols=["valid_year", "target", "catboost_platt"])
    rows = []
    rate_rows = []
    for valid_year, fold in oof.groupby("valid_year", sort=True):
        y = fold["target"].to_numpy(dtype=np.int8)
        platt = clip_prob(fold["catboost_platt"].to_numpy(dtype=np.float64))
        actual = float(np.mean(y))
        estimators = rate_estimators_for_year(year_rates, int(valid_year))
        platt_metrics = metrics(y, platt)

        for estimator, target_rate in estimators.items():
            if estimator == "A_no_adjustment":
                adjusted = {"none": platt}
                estimated_rate = float(platt.mean())
            else:
                estimated_rate = float(target_rate)
                adjusted = {
                    "probability_shift": probability_mean_match(platt, estimated_rate),
                    "logit_shift": logit_mean_match(platt, estimated_rate),
                }
            rate_rows.append(
                {
                    "valid_year": int(valid_year),
                    "estimator": estimator,
                    "actual_rate": actual,
                    "platt_pred_mean": float(platt.mean()),
                    "estimated_target_rate": estimated_rate,
                    "rate_estimation_error": estimated_rate - actual,
                }
            )
            for method, pred in adjusted.items():
                row = {
                    "valid_year": int(valid_year),
                    "estimator": estimator,
                    "mean_shift_method": method,
                    "estimated_target_rate": estimated_rate,
                    "rate_estimation_error": estimated_rate - actual,
                    "platt_pred_mean": float(platt.mean()),
                    "adjusted_pred_mean": float(pred.mean()),
                    "raw_platt_brier": platt_metrics["brier"],
                    **metrics(y, pred),
                }
                rows.append(row)

    fold_metrics = pd.DataFrame(rows)
    rate_metrics = pd.DataFrame(rate_rows).drop_duplicates(["valid_year", "estimator"])
    fold_metrics.to_csv(os.path.join(OUT_DIR, "mean_shift_metrics.csv"), index=False, encoding="utf-8")
    rate_metrics.to_csv(os.path.join(OUT_DIR, "rate_estimator_fold_metrics.csv"), index=False, encoding="utf-8")

    summary = (
        fold_metrics.groupby(["estimator", "mean_shift_method"])
        .agg(
            mean_brier=("brier", "mean"),
            std_brier=("brier", "std"),
            worst_brier=("brier", "max"),
            mean_skill_margin=("skill_margin", "mean"),
            worst_skill_margin=("skill_margin", "min"),
            mean_pseudo_score=("pseudo_score", "mean"),
            positive_score_folds=("skill_margin", lambda s: int((s > 0).sum())),
            mean_auc=("auc", "mean"),
            mean_logloss=("logloss", "mean"),
            mean_pred_std=("pred_std", "mean"),
            mean_abs_rate_error=("rate_estimation_error", lambda s: float(np.mean(np.abs(s)))),
            brier_2024=("brier", lambda s: float(s[fold_metrics.loc[s.index, "valid_year"] == 2024].iloc[0])),
            skill_margin_2024=("skill_margin", lambda s: float(s[fold_metrics.loc[s.index, "valid_year"] == 2024].iloc[0])),
            pseudo_score_2024=("pseudo_score", lambda s: float(s[fold_metrics.loc[s.index, "valid_year"] == 2024].iloc[0])),
            rate_error_2024=("rate_estimation_error", lambda s: float(s[fold_metrics.loc[s.index, "valid_year"] == 2024].iloc[0])),
        )
        .reset_index()
        .sort_values(["positive_score_folds", "skill_margin_2024", "mean_pseudo_score"], ascending=[False, False, False])
    )
    summary.to_csv(os.path.join(OUT_DIR, "rate_estimator_summary.csv"), index=False, encoding="utf-8")
    damped = fold_metrics[
        fold_metrics["estimator"].str.startswith("H_damped_trend")
        | fold_metrics["estimator"].str.startswith("I_trend_last_blend")
    ].copy()
    damped.to_csv(os.path.join(OUT_DIR, "damped_trend_grid.csv"), index=False, encoding="utf-8")

    print("Yearly rates")
    print(yearly.to_string(index=False))
    print("\nTop summary")
    print(summary.head(40).to_string(index=False))


if __name__ == "__main__":
    main()
