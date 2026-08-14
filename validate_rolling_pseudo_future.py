import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from calibration_utils import apply_calibration, fit_calibrators
from model_utils import FeatureBuilder, TARGET_COL, make_model


FOLDS = [
    (2019, 2020, 2021, 2022),
    (2019, 2021, 2022, 2023),
    (2019, 2022, 2023, 2024),
]

COARSE_WEIGHTS = np.round(
    np.unique(np.concatenate([np.arange(0.0, 1.0001, 0.1), np.arange(0.65, 0.8501, 0.05), [0.75]])),
    2,
)


def clip_prob(pred):
    return np.clip(np.asarray(pred, dtype=np.float64), 1e-6, 1.0 - 1e-6)


def evaluate(y, pred):
    pred = clip_prob(pred)
    return {
        "brier": float(brier_score_loss(y, pred)),
        "logloss": float(log_loss(y, pred)),
        "pred_mean": float(pred.mean()),
        "target_mean": float(y.mean()),
        "bias": float(pred.mean() - y.mean()),
    }


def linear_forecast(year_rates: pd.Series, years_back: int | None = None) -> float:
    s = year_rates.dropna().sort_index()
    if years_back is not None:
        s = s.tail(years_back)
    if len(s) == 0:
        raise ValueError("No history for prior forecast.")
    if len(s) == 1:
        return float(s.iloc[-1])
    x = s.index.to_numpy(dtype=np.float64)
    y = s.to_numpy(dtype=np.float64)
    slope, intercept = np.polyfit(x, y, 1)
    return float(intercept + slope * (x[-1] + 1))


def recency_weighted_linear_forecast(year_rates: pd.Series, half_life: float = 1.5) -> float:
    s = year_rates.dropna().sort_index()
    if len(s) == 0:
        raise ValueError("No history for prior forecast.")
    if len(s) == 1:
        return float(s.iloc[-1])
    x = s.index.to_numpy(dtype=np.float64)
    y = s.to_numpy(dtype=np.float64)
    age = x.max() - x
    weights = np.power(0.5, age / half_life)
    slope, intercept = np.polyfit(x, y, 1, w=weights)
    return float(intercept + slope * (x[-1] + 1))


def exp_smoothing_forecast(year_rates: pd.Series, alpha: float = 0.65) -> float:
    s = year_rates.dropna().sort_index()
    if len(s) == 0:
        raise ValueError("No history for prior forecast.")
    level = float(s.iloc[0])
    for value in s.iloc[1:]:
        level = alpha * float(value) + (1.0 - alpha) * level
    return level


def prior_rates(history_df: pd.DataFrame) -> dict:
    rates = history_df.groupby("season")[TARGET_COL].mean()
    priors = {
        "prev_year_rate": float(rates.iloc[-1]),
        "linear_all_history": linear_forecast(rates),
        "linear_recent_2y": linear_forecast(rates, years_back=2),
        "linear_recent_3y": linear_forecast(rates, years_back=3),
        "recency_weighted_linear": recency_weighted_linear_forecast(rates),
        "exp_smoothing_0.65": exp_smoothing_forecast(rates, alpha=0.65),
    }
    return {k: float(np.clip(v, 0.42, 0.62)) for k, v in priors.items()}


def fine_weights(best_weight: float) -> np.ndarray:
    lo = max(0.0, best_weight - 0.1)
    hi = min(1.0, best_weight + 0.1)
    return np.round(np.arange(lo, hi + 0.0001, 0.02), 2)


def run_fold(df: pd.DataFrame, fold: tuple[int, int, int, int]):
    train_start, train_end, cal_year, valid_year = fold
    train_df = df[(df["season"] >= train_start) & (df["season"] <= train_end)].copy()
    cal_df = df[df["season"] == cal_year].copy()
    valid_df = df[df["season"] == valid_year].copy()
    history_for_prior = df[df["season"] <= cal_year].copy()

    print(
        f"Fold train={train_start}-{train_end} cal={cal_year} valid={valid_year} "
        f"rows train={len(train_df)} cal={len(cal_df)} valid={len(valid_df)}"
    )

    y_train = train_df[TARGET_COL].astype("int8")
    builder = FeatureBuilder(alpha=80.0)
    X_train = builder.fit_transform(train_df.drop(columns=[TARGET_COL]), y_train)
    model = make_model()
    model.fit(X_train, y_train)

    X_cal = builder.transform(cal_df.drop(columns=[TARGET_COL]))
    y_cal = cal_df[TARGET_COL].astype("int8")
    cal_raw = model.predict_proba(X_cal)[:, 1]
    calibration = fit_calibrators(cal_raw, y_cal, cal_df[["game_type"]])

    X_valid = builder.transform(valid_df.drop(columns=[TARGET_COL]))
    y_valid = valid_df[TARGET_COL].astype("int8")
    raw = model.predict_proba(X_valid)[:, 1]
    platt = apply_calibration(raw, valid_df[["game_type"]], calibration, "platt")

    metric_rows = []
    blend_rows = []

    raw_metrics = evaluate(y_valid, raw)
    platt_metrics = evaluate(y_valid, platt)
    metric_rows.append({"valid_year": valid_year, "method": "raw_model", "prior_name": "", "weight": np.nan, **raw_metrics})
    metric_rows.append({"valid_year": valid_year, "method": "platt", "prior_name": "", "weight": 0.0, **platt_metrics})

    priors = prior_rates(history_for_prior)
    actual = float(y_valid.mean())
    for prior_name, prior_rate in priors.items():
        prior_pred = np.full(len(valid_df), prior_rate, dtype=np.float64)
        prior_metrics = evaluate(y_valid, prior_pred)
        metric_rows.append(
            {
                "valid_year": valid_year,
                "method": "prior",
                "prior_name": prior_name,
                "weight": 1.0,
                "prior_rate": prior_rate,
                "actual_valid_rate": actual,
                "prior_error": prior_rate - actual,
                **prior_metrics,
            }
        )

        coarse = []
        for w in COARSE_WEIGHTS:
            pred = (1.0 - w) * platt + w * prior_pred
            row = {
                "valid_year": valid_year,
                "prior_name": prior_name,
                "weight": float(w),
                "prior_rate": prior_rate,
                "actual_valid_rate": actual,
                "prior_error": prior_rate - actual,
                **evaluate(y_valid, pred),
            }
            coarse.append(row)
        best_coarse = min(coarse, key=lambda r: r["brier"])
        blend_rows.extend(coarse)

        existing_weights = {r["weight"] for r in coarse}
        for w in fine_weights(best_coarse["weight"]):
            if float(w) in existing_weights:
                continue
            pred = (1.0 - w) * platt + w * prior_pred
            blend_rows.append(
                {
                    "valid_year": valid_year,
                    "prior_name": prior_name,
                    "weight": float(w),
                    "prior_rate": prior_rate,
                    "actual_valid_rate": actual,
                    "prior_error": prior_rate - actual,
                    **evaluate(y_valid, pred),
                }
            )

    return metric_rows, blend_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", default="data/train.csv")
    parser.add_argument("--output-dir", default="output/rolling_pseudo_future")
    args = parser.parse_args()

    df = pd.read_csv(args.train_path, encoding="utf-8-sig")
    os.makedirs(args.output_dir, exist_ok=True)

    all_metrics = []
    all_blends = []
    for fold in FOLDS:
        metric_rows, blend_rows = run_fold(df, fold)
        all_metrics.extend(metric_rows)
        all_blends.extend(blend_rows)

    metrics = pd.DataFrame(all_metrics)
    blends = pd.DataFrame(all_blends)

    metrics_path = os.path.join(args.output_dir, "fold_metrics.csv")
    blends_path = os.path.join(args.output_dir, "blend_grid.csv")
    metrics.to_csv(metrics_path, index=False, encoding="utf-8")
    blends.to_csv(blends_path, index=False, encoding="utf-8")

    method_summary = (
        pd.concat(
            [
                metrics[metrics["method"].isin(["raw_model", "platt", "prior"])][
                    ["valid_year", "method", "prior_name", "weight", "brier", "logloss", "bias"]
                ],
                blends.assign(method="blend")[["valid_year", "method", "prior_name", "weight", "brier", "logloss", "bias"]],
            ],
            ignore_index=True,
        )
        .groupby(["method", "prior_name", "weight"], dropna=False)
        .agg(
            mean_brier=("brier", "mean"),
            std_brier=("brier", "std"),
            mean_logloss=("logloss", "mean"),
            mean_bias=("bias", "mean"),
            folds=("valid_year", "nunique"),
        )
        .reset_index()
        .sort_values(["mean_brier", "std_brier"])
    )
    summary_path = os.path.join(args.output_dir, "method_summary.csv")
    method_summary.to_csv(summary_path, index=False, encoding="utf-8")

    fixed_weight_summary = (
        blends.groupby(["prior_name", "weight"])
        .agg(
            mean_brier=("brier", "mean"),
            std_brier=("brier", "std"),
            max_brier=("brier", "max"),
            mean_prior_error=("prior_error", "mean"),
            folds=("valid_year", "nunique"),
        )
        .reset_index()
    )
    fixed_weight_summary = fixed_weight_summary[fixed_weight_summary["folds"] == len(FOLDS)].sort_values(
        ["mean_brier", "std_brier"]
    )
    fixed_path = os.path.join(args.output_dir, "fixed_weight_summary.csv")
    fixed_weight_summary.to_csv(fixed_path, index=False, encoding="utf-8")

    best_per_fold = (
        blends.sort_values("brier")
        .groupby("valid_year", as_index=False)
        .first()
        .sort_values("valid_year")
    )
    best_path = os.path.join(args.output_dir, "best_blend_per_fold.csv")
    best_per_fold.to_csv(best_path, index=False, encoding="utf-8")

    print("\nBest blend per fold")
    print(best_per_fold[["valid_year", "prior_name", "weight", "brier", "prior_rate", "actual_valid_rate", "prior_error"]].to_string(index=False))
    print("\nBest fixed prior+weight")
    print(fixed_weight_summary.head(10).to_string(index=False))
    print(f"\nSaved: {metrics_path}")
    print(f"Saved: {blends_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {fixed_path}")
    print(f"Saved: {best_path}")


if __name__ == "__main__":
    main()
