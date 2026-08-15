import os

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


OUT_DIR = "output/final_strength_selection"
OOF_PATH = "output/model_comparison/oof_predictions.csv"
TRAIN_PATH = "data/train.csv"
EPS = 1e-6

NARROW_LAMBDAS = [0.25, 0.30, 0.325, 0.35, 0.375, 0.40, 0.425, 0.45, 0.475, 0.50, 0.525, 0.55, 0.60]
NARROW_TEMPERATURES = [1.75, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.75, 3.0, 3.5]
CAPS = [0.005, 0.0075, 0.010, 0.0125, 0.015, 0.020, 0.025, 0.030, None]


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


def target_rate_for_year(year_rates: pd.Series, valid_year: int) -> float:
    hist = year_rates[year_rates.index < valid_year].dropna().sort_index()
    return linear_forecast(hist, tail=3)


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


def metrics(y, pred):
    pred = clip_prob(pred)
    actual_rate = float(np.mean(y))
    constant_brier = float(actual_rate * (1.0 - actual_rate))
    model_brier = float(brier_score_loss(y, pred))
    skill_margin = constant_brier - model_brier
    return {
        "actual_rate": actual_rate,
        "constant_brier": constant_brier,
        "model_brier": model_brier,
        "skill_margin": float(skill_margin),
        "pseudo_score": float(max(0.0, 100000.0 * skill_margin / constant_brier)),
        "auc": float(roc_auc_score(y, pred)),
        "logloss": float(log_loss(y, pred)),
        "pred_mean": float(pred.mean()),
        "pred_std": float(pred.std()),
    }


def temperature_scale(adjusted, target_rate, temperature):
    centered = logit(adjusted) - float(logit(target_rate))
    pred = sigmoid(float(logit(target_rate)) + centered / float(temperature))
    return logit_mean_match(pred, target_rate)


def apply_cap(pred, target_rate, cap_type, cap_value):
    if cap_type == "no_cap":
        return clip_prob(pred)
    d = clip_prob(pred) - target_rate
    if cap_type == "hard_cap":
        d_final = np.clip(d, -cap_value, cap_value)
    elif cap_type == "soft_cap":
        d_final = cap_value * np.tanh(d / cap_value)
    else:
        raise ValueError(f"Unknown cap_type={cap_type}")
    return clip_prob(target_rate + d_final)


def strategy_pred(adjusted, target_rate, strategy, value):
    if strategy == "shrink":
        return clip_prob(target_rate + float(value) * (adjusted - target_rate))
    if strategy == "temperature":
        return temperature_scale(adjusted, target_rate, float(value))
    raise ValueError(f"Unknown strategy={strategy}")


def add_eval_row(rows, y, pred, valid_year, target_rate, strategy, value, cap_type="no_cap", cap_value=None):
    rows.append(
        {
            "valid_year": int(valid_year),
            "base_model": "catboost",
            "calibration": "platt",
            "target_rate_estimator": "linear_trend_recent3",
            "mean_matching": "logit_intercept_shift",
            "strategy": strategy,
            "parameter": float(value),
            "cap_type": cap_type,
            "cap_value": np.nan if cap_value is None else float(cap_value),
            "target_rate": float(target_rate),
            **metrics(y, pred),
        }
    )


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    summary = (
        rows.groupby(["strategy", "parameter", "cap_type", "cap_value"], dropna=False)
        .agg(
            mean_brier=("model_brier", "mean"),
            worst_brier=("model_brier", "max"),
            mean_pseudo_score=("pseudo_score", "mean"),
            positive_fold_count=("skill_margin", lambda s: int((s > 0).sum())),
            mean_skill_margin=("skill_margin", "mean"),
            worst_skill_margin=("skill_margin", "min"),
            mean_pred_std=("pred_std", "mean"),
            mean_auc=("auc", "mean"),
            brier_2022=("model_brier", lambda s: float(s[rows.loc[s.index, "valid_year"] == 2022].iloc[0])),
            brier_2023=("model_brier", lambda s: float(s[rows.loc[s.index, "valid_year"] == 2023].iloc[0])),
            brier_2024=("model_brier", lambda s: float(s[rows.loc[s.index, "valid_year"] == 2024].iloc[0])),
            pseudo_2022=("pseudo_score", lambda s: float(s[rows.loc[s.index, "valid_year"] == 2022].iloc[0])),
            pseudo_2023=("pseudo_score", lambda s: float(s[rows.loc[s.index, "valid_year"] == 2023].iloc[0])),
            pseudo_2024=("pseudo_score", lambda s: float(s[rows.loc[s.index, "valid_year"] == 2024].iloc[0])),
            skill_2022=("skill_margin", lambda s: float(s[rows.loc[s.index, "valid_year"] == 2022].iloc[0])),
            skill_2023=("skill_margin", lambda s: float(s[rows.loc[s.index, "valid_year"] == 2023].iloc[0])),
            skill_2024=("skill_margin", lambda s: float(s[rows.loc[s.index, "valid_year"] == 2024].iloc[0])),
            pred_std_2024=("pred_std", lambda s: float(s[rows.loc[s.index, "valid_year"] == 2024].iloc[0])),
        )
        .reset_index()
    )
    summary["ready_like"] = (
        (summary["pseudo_2022"] > 0)
        & (summary["pseudo_2024"] > 0)
        & (summary["positive_fold_count"] >= 2)
        & (summary["skill_2023"] > -0.0010)
    )
    return summary.sort_values(
        ["ready_like", "positive_fold_count", "skill_2024", "worst_skill_margin", "mean_pseudo_score"],
        ascending=[False, False, False, False, False],
    )


def direction_and_bucket_analysis(fold_data):
    rows = []
    bins = [0.0, 0.005, 0.01, 0.02, 0.04, np.inf]
    labels = ["0-0.005", "0.005-0.01", "0.01-0.02", "0.02-0.04", "0.04+"]
    for valid_year, (y, adjusted, target_rate) in fold_data.items():
        pred = clip_prob(adjusted)
        for name, mask in [("low_half", pred < np.median(pred)), ("high_half", pred >= np.median(pred))]:
            rows.append(
                {
                    "valid_year": int(valid_year),
                    "analysis": "median_direction",
                    "group": name,
                    "n": int(mask.sum()),
                    "actual_rate": float(np.mean(y[mask])),
                    "prediction_mean": float(np.mean(pred[mask])),
                    "pred_std": float(np.std(pred[mask])),
                    "model_brier": float(np.mean((y[mask] - pred[mask]) ** 2)),
                    "constant_brier": float(np.mean((y[mask] - target_rate) ** 2)),
                    "brier_gain_loss": float(np.mean((y[mask] - target_rate) ** 2) - np.mean((y[mask] - pred[mask]) ** 2)),
                }
            )
        rows.append(
            {
                "valid_year": int(valid_year),
                "analysis": "median_direction",
                "group": "high_minus_low",
                "n": int(len(y)),
                "actual_rate": float(np.mean(y[pred >= np.median(pred)]) - np.mean(y[pred < np.median(pred)])),
                "prediction_mean": float(np.mean(pred[pred >= np.median(pred)]) - np.mean(pred[pred < np.median(pred)])),
                "pred_std": float(np.std(pred)),
                "model_brier": np.nan,
                "constant_brier": np.nan,
                "brier_gain_loss": np.nan,
            }
        )
        work = pd.DataFrame({"target": y, "pred": pred, "abs_dev": np.abs(pred - target_rate)})
        work["bucket"] = pd.cut(work["abs_dev"], bins=bins, labels=labels, right=False, include_lowest=True)
        for bucket, part in work.groupby("bucket", observed=False):
            if len(part) == 0:
                continue
            model_brier = float(np.mean((part["target"].to_numpy() - part["pred"].to_numpy()) ** 2))
            const_brier = float(np.mean((part["target"].to_numpy() - target_rate) ** 2))
            rows.append(
                {
                    "valid_year": int(valid_year),
                    "analysis": "deviation_bucket",
                    "group": str(bucket),
                    "n": int(len(part)),
                    "actual_rate": float(part["target"].mean()),
                    "prediction_mean": float(part["pred"].mean()),
                    "pred_std": float(part["pred"].std()),
                    "model_brier": model_brier,
                    "constant_brier": const_brier,
                    "brier_gain_loss": const_brier - model_brier,
                }
            )
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    train = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig", usecols=["season", "control_success"])
    year_rates = train.groupby("season")["control_success"].mean().sort_index()
    oof = pd.read_csv(OOF_PATH, usecols=["valid_year", "target", "catboost_platt"])

    fold_data = {}
    for valid_year, fold in oof.groupby("valid_year", sort=True):
        valid_year = int(valid_year)
        y = fold["target"].to_numpy(dtype=np.int8)
        platt = clip_prob(fold["catboost_platt"].to_numpy(dtype=np.float64))
        target_rate = target_rate_for_year(year_rates, valid_year)
        fold_data[valid_year] = (y, logit_mean_match(platt, target_rate), target_rate)

    shrink_rows = []
    temp_rows = []
    for valid_year, (y, adjusted, target_rate) in fold_data.items():
        for lam in NARROW_LAMBDAS:
            add_eval_row(shrink_rows, y, strategy_pred(adjusted, target_rate, "shrink", lam), valid_year, target_rate, "shrink", lam)
        for temp in NARROW_TEMPERATURES:
            add_eval_row(temp_rows, y, strategy_pred(adjusted, target_rate, "temperature", temp), valid_year, target_rate, "temperature", temp)

    shrink_grid = pd.DataFrame(shrink_rows)
    temp_grid = pd.DataFrame(temp_rows)
    shrink_grid.to_csv(os.path.join(OUT_DIR, "narrow_shrink_grid.csv"), index=False, encoding="utf-8")
    temp_grid.to_csv(os.path.join(OUT_DIR, "narrow_temperature_grid.csv"), index=False, encoding="utf-8")

    no_cap_summary = pd.concat([summarize(shrink_grid), summarize(temp_grid)], ignore_index=True)
    top = no_cap_summary[
        (no_cap_summary["positive_fold_count"] >= 2)
        & (no_cap_summary["pseudo_2024"] > 0)
        & (no_cap_summary["skill_2023"] > -0.0010)
    ].copy()
    top = top.sort_values(["skill_2024", "worst_skill_margin", "mean_pseudo_score"], ascending=[False, False, False]).head(5)

    cap_rows = []
    for _, candidate in top.iterrows():
        strategy = candidate["strategy"]
        parameter = float(candidate["parameter"])
        for cap in CAPS:
            cap_variants = [("no_cap", None)] if cap is None else [("hard_cap", cap), ("soft_cap", cap)]
            for cap_type, cap_value in cap_variants:
                for valid_year, (y, adjusted, target_rate) in fold_data.items():
                    base_pred = strategy_pred(adjusted, target_rate, strategy, parameter)
                    capped = apply_cap(base_pred, target_rate, cap_type, cap_value)
                    add_eval_row(cap_rows, y, capped, valid_year, target_rate, strategy, parameter, cap_type, cap_value)

    cap_grid = pd.DataFrame(cap_rows).drop_duplicates(
        ["valid_year", "strategy", "parameter", "cap_type", "cap_value"], keep="first"
    )
    cap_grid.to_csv(os.path.join(OUT_DIR, "cap_grid.csv"), index=False, encoding="utf-8")

    all_summary = pd.concat([no_cap_summary, summarize(cap_grid)], ignore_index=True).drop_duplicates(
        ["strategy", "parameter", "cap_type", "cap_value"], keep="first"
    )
    all_summary = all_summary.sort_values(
        ["ready_like", "positive_fold_count", "skill_2024", "worst_skill_margin", "mean_pseudo_score"],
        ascending=[False, False, False, False, False],
    )

    all_summary["selection_note"] = ""
    conservative_mask = (
        (all_summary["ready_like"])
        & (all_summary["pred_std_2024"] >= 0.004)
        & (all_summary["skill_2023"] > -0.0008)
    )
    conservative = all_summary[conservative_mask].sort_values(
        ["worst_skill_margin", "skill_2024", "mean_pseudo_score"], ascending=[False, False, False]
    ).head(1)
    aggressive = all_summary[all_summary["ready_like"]].sort_values(
        ["skill_2024", "mean_pseudo_score", "worst_skill_margin"], ascending=[False, False, False]
    ).head(1)
    if not conservative.empty:
        all_summary.loc[conservative.index, "selection_note"] = "Candidate A - conservative fold-stable"
    if not aggressive.empty:
        note = "Candidate B - higher expected 2024 score"
        if all_summary.loc[aggressive.index[0], "selection_note"]:
            note = all_summary.loc[aggressive.index[0], "selection_note"] + "; " + note
        all_summary.loc[aggressive.index, "selection_note"] = note

    all_summary.to_csv(os.path.join(OUT_DIR, "robust_strategy_summary.csv"), index=False, encoding="utf-8")
    direction_and_bucket_analysis(fold_data).to_csv(
        os.path.join(OUT_DIR, "fold_direction_analysis.csv"), index=False, encoding="utf-8"
    )

    print("Top robust strategies")
    print(
        all_summary.head(20)[
            [
                "strategy",
                "parameter",
                "cap_type",
                "cap_value",
                "pseudo_2022",
                "pseudo_2023",
                "pseudo_2024",
                "mean_brier",
                "worst_skill_margin",
                "positive_fold_count",
                "pred_std_2024",
                "mean_auc",
                "selection_note",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
