import os

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


OUT_DIR = "output/probability_strength_calibration"
OOF_PATH = "output/model_comparison/oof_predictions.csv"
TRAIN_PATH = "data/train.csv"
EPS = 1e-6

LAMBDAS = [0.00, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.75, 1.00]
TEMPERATURES = [1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0, 15.0, 20.0]


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
    model_brier = float(brier_score_loss(y, pred))
    constant_brier = float(actual_rate * (1.0 - actual_rate))
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


def summarize_strategy(df: pd.DataFrame, strategy_col: str, value_col: str, strategy_name: str) -> pd.DataFrame:
    summary = (
        df.groupby(value_col)
        .agg(
            strategy=(strategy_col, "first"),
            mean_model_brier=("model_brier", "mean"),
            worst_model_brier=("model_brier", "max"),
            mean_skill_margin=("skill_margin", "mean"),
            worst_skill_margin=("skill_margin", "min"),
            mean_pseudo_score=("pseudo_score", "mean"),
            positive_score_folds=("skill_margin", lambda s: int((s > 0).sum())),
            mean_auc=("auc", "mean"),
            mean_logloss=("logloss", "mean"),
            mean_pred_std=("pred_std", "mean"),
            skill_margin_2024=("skill_margin", lambda s: float(s[df.loc[s.index, "valid_year"] == 2024].iloc[0])),
            pseudo_score_2024=("pseudo_score", lambda s: float(s[df.loc[s.index, "valid_year"] == 2024].iloc[0])),
            model_brier_2024=("model_brier", lambda s: float(s[df.loc[s.index, "valid_year"] == 2024].iloc[0])),
            pred_std_2024=("pred_std", lambda s: float(s[df.loc[s.index, "valid_year"] == 2024].iloc[0])),
        )
        .reset_index()
    )
    summary["strategy"] = strategy_name
    return summary.sort_values(
        ["positive_score_folds", "skill_margin_2024", "worst_skill_margin", "mean_pseudo_score"],
        ascending=[False, False, False, False],
    )


def deviation_bucket_analysis(y, adjusted, target_rate):
    bins = [0.0, 0.005, 0.01, 0.02, 0.04, np.inf]
    labels = ["0-0.005", "0.005-0.01", "0.01-0.02", "0.02-0.04", "0.04+"]
    work = pd.DataFrame(
        {
            "target": y,
            "pred": clip_prob(adjusted),
            "deviation_abs": np.abs(clip_prob(adjusted) - target_rate),
        }
    )
    work["bucket"] = pd.cut(work["deviation_abs"], bins=bins, labels=labels, right=False, include_lowest=True)
    rows = []
    for bucket, part in work.groupby("bucket", observed=False):
        if len(part) == 0:
            continue
        brier = float(np.mean((part["target"].to_numpy() - part["pred"].to_numpy()) ** 2))
        constant_brier = float(np.mean((part["target"].to_numpy() - target_rate) ** 2))
        rows.append(
            {
                "valid_year": 2024,
                "bucket": str(bucket),
                "n": int(len(part)),
                "actual_rate": float(part["target"].mean()),
                "prediction_mean": float(part["pred"].mean()),
                "brier": brier,
                "constant_prediction_brier": constant_brier,
                "brier_gain_loss": constant_brier - brier,
            }
        )
    return pd.DataFrame(rows)


def direction_signal_analysis(y, adjusted, target_rate):
    pred = clip_prob(adjusted)
    rows = []
    for name, mask in [("low", pred < target_rate), ("high", pred >= target_rate)]:
        yy = y[mask]
        pp = pred[mask]
        rows.append(
            {
                "valid_year": 2024,
                "analysis": "direction",
                "group": name,
                "n": int(len(yy)),
                "actual_rate": float(np.mean(yy)),
                "prediction_mean": float(np.mean(pp)),
                "deviation_abs_mean": float(np.mean(np.abs(pp - target_rate))),
            }
        )
    rows.append(
        {
            "valid_year": 2024,
            "analysis": "direction",
            "group": "high_minus_low",
            "n": int(len(y)),
            "actual_rate": float(np.mean(y[pred >= target_rate]) - np.mean(y[pred < target_rate])),
            "prediction_mean": float(np.mean(pred[pred >= target_rate]) - np.mean(pred[pred < target_rate])),
            "deviation_abs_mean": float("nan"),
        }
    )

    q = pd.qcut(np.abs(pred - target_rate), q=5, labels=False, duplicates="drop")
    for qid in sorted(pd.Series(q).dropna().unique()):
        mask = q == qid
        rows.append(
            {
                "valid_year": 2024,
                "analysis": "deviation_abs_quantile",
                "group": f"q{int(qid) + 1}",
                "n": int(mask.sum()),
                "actual_rate": float(np.mean(y[mask])),
                "prediction_mean": float(np.mean(pred[mask])),
                "deviation_abs_mean": float(np.mean(np.abs(pred[mask] - target_rate))),
            }
        )

    signed_q = pd.qcut(pred - target_rate, q=10, labels=False, duplicates="drop")
    for qid in sorted(pd.Series(signed_q).dropna().unique()):
        mask = signed_q == qid
        rows.append(
            {
                "valid_year": 2024,
                "analysis": "signed_prediction_quantile",
                "group": f"q{int(qid) + 1}",
                "n": int(mask.sum()),
                "actual_rate": float(np.mean(y[mask])),
                "prediction_mean": float(np.mean(pred[mask])),
                "deviation_abs_mean": float(np.mean(np.abs(pred[mask] - target_rate))),
            }
        )
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    train = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig", usecols=["season", "control_success"])
    year_rates = train.groupby("season")["control_success"].mean().sort_index()
    oof = pd.read_csv(OOF_PATH, usecols=["valid_year", "target", "catboost_platt"])

    shrink_rows = []
    temp_rows = []
    adjusted_by_year = {}
    for valid_year, fold in oof.groupby("valid_year", sort=True):
        valid_year = int(valid_year)
        y = fold["target"].to_numpy(dtype=np.int8)
        platt = clip_prob(fold["catboost_platt"].to_numpy(dtype=np.float64))
        target_rate = target_rate_for_year(year_rates, valid_year)
        adjusted = logit_mean_match(platt, target_rate)
        adjusted_by_year[valid_year] = (y, adjusted, target_rate)

        base = {
            "valid_year": valid_year,
            "target_rate_estimator": "linear_trend_recent3",
            "target_rate": target_rate,
            "platt_pred_mean": float(platt.mean()),
            "mean_matched_pred_mean": float(adjusted.mean()),
            "mean_matched_pred_std": float(adjusted.std()),
        }
        for lam in LAMBDAS:
            pred = clip_prob(target_rate + lam * (adjusted - target_rate))
            shrink_rows.append({"strategy": "shrink_to_mean", "lambda": lam, **base, **metrics(y, pred)})

        for temp in TEMPERATURES:
            pred = temperature_scale(adjusted, target_rate, temp)
            temp_rows.append({"strategy": "temperature", "temperature": temp, **base, **metrics(y, pred)})

    shrink_grid = pd.DataFrame(shrink_rows)
    temperature_grid = pd.DataFrame(temp_rows)
    shrink_grid.to_csv(os.path.join(OUT_DIR, "shrink_grid.csv"), index=False, encoding="utf-8")
    temperature_grid.to_csv(os.path.join(OUT_DIR, "temperature_grid.csv"), index=False, encoding="utf-8")

    shrink_summary = summarize_strategy(shrink_grid, "strategy", "lambda", "shrink_to_mean")
    temp_summary = summarize_strategy(temperature_grid, "strategy", "temperature", "temperature")
    strategy_summary = pd.concat([shrink_summary, temp_summary], ignore_index=True)
    strategy_summary.to_csv(os.path.join(OUT_DIR, "strategy_summary.csv"), index=False, encoding="utf-8")

    y2024, adjusted2024, target2024 = adjusted_by_year[2024]
    deviation_bucket_analysis(y2024, adjusted2024, target2024).to_csv(
        os.path.join(OUT_DIR, "deviation_bucket_analysis.csv"), index=False, encoding="utf-8"
    )
    direction_signal_analysis(y2024, adjusted2024, target2024).to_csv(
        os.path.join(OUT_DIR, "direction_signal_analysis.csv"), index=False, encoding="utf-8"
    )

    print("Top shrink summary")
    print(shrink_summary.head(10).to_string(index=False))
    print("\nTop temperature summary")
    print(temp_summary.head(10).to_string(index=False))
    print("\nFold-optimal shrink")
    print(
        shrink_grid.sort_values(["valid_year", "skill_margin"], ascending=[True, False])
        .groupby("valid_year")
        .head(1)[["valid_year", "lambda", "model_brier", "constant_brier", "skill_margin", "pseudo_score", "pred_std"]]
        .to_string(index=False)
    )
    print("\nFold-optimal temperature")
    print(
        temperature_grid.sort_values(["valid_year", "skill_margin"], ascending=[True, False])
        .groupby("valid_year")
        .head(1)[["valid_year", "temperature", "model_brier", "constant_brier", "skill_margin", "pseudo_score", "pred_std"]]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
