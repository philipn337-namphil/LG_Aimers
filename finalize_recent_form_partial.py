import os

import pandas as pd

from validate_recent_form_features import (
    FOLDS,
    TARGET_COL,
    add_basic,
    add_static_recent_from_history,
    add_train_recent_features,
    rolling_feature_stats,
)


def main():
    out_dir = "output/recent_form"
    partial_path = os.path.join(out_dir, "fold_metrics_partial.csv")
    fold_metrics = pd.read_csv(partial_path)

    trend = fold_metrics[fold_metrics["stage"] == "platt_plus_trend_w075"].copy()
    complete_keys = (
        trend.groupby(["alpha", "feature_set"])["valid_year"]
        .agg(lambda s: set(s.astype(int)))
        .reset_index(name="years")
    )
    complete_keys = complete_keys[complete_keys["years"].map(lambda s: s == {2022, 2023, 2024})][["alpha", "feature_set"]]
    fold_metrics = fold_metrics.merge(complete_keys, on=["alpha", "feature_set"], how="inner")

    trend = fold_metrics[fold_metrics["stage"] == "platt_plus_trend_w075"].copy()
    baseline = trend[trend["feature_set"] == "baseline"][["alpha", "valid_year", "brier", "auc"]].rename(
        columns={"brier": "baseline_brier", "auc": "baseline_auc"}
    )
    fold_metrics = fold_metrics.merge(baseline, on=["alpha", "valid_year"], how="left")
    fold_metrics["improvement_vs_recent_baseline"] = fold_metrics["baseline_brier"] - fold_metrics["brier"]
    fold_metrics["auc_delta_vs_recent_baseline"] = fold_metrics["auc"] - fold_metrics["baseline_auc"]
    fold_metrics.to_csv(os.path.join(out_dir, "fold_metrics.csv"), index=False, encoding="utf-8")

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
    summary.to_csv(os.path.join(out_dir, "feature_set_metrics.csv"), index=False, encoding="utf-8")
    summary.to_csv(os.path.join(out_dir, "smoothing_grid.csv"), index=False, encoding="utf-8")

    df = add_basic(pd.read_csv("data/train.csv", encoding="utf-8-sig"))
    stats_rows = []
    for alpha in sorted(fold_metrics["alpha"].unique()):
        for train_start, train_end, cal_year, valid_year in FOLDS:
            raw_train = df[(df["season"] >= train_start) & (df["season"] <= train_end)].copy()
            raw_cal = df[df["season"] == cal_year].copy()
            raw_valid = df[df["season"] == valid_year].copy()
            rf_train = add_train_recent_features(raw_train, int(alpha))
            rf_valid = add_static_recent_from_history(
                raw_valid, pd.concat([raw_train, raw_cal], ignore_index=True), int(alpha)
            )
            rf_cols = [c for c in rf_train.columns if c.startswith("rf_") and not c.endswith("_fallback")]
            st = rolling_feature_stats(rf_valid, rf_cols)
            st["alpha"] = alpha
            st["valid_year"] = valid_year
            stats_rows.append(st)
    pd.concat(stats_rows, ignore_index=True).to_csv(
        os.path.join(out_dir, "rolling_feature_stats.csv"), index=False, encoding="utf-8"
    )

    stage1_fi = "output/recent_form_stage1/feature_importance.csv"
    if os.path.exists(stage1_fi):
        pd.read_csv(stage1_fi).to_csv(os.path.join(out_dir, "feature_importance.csv"), index=False, encoding="utf-8")
    else:
        pd.DataFrame(columns=["feature", "importance", "alpha", "feature_set", "valid_year"]).to_csv(
            os.path.join(out_dir, "feature_importance.csv"), index=False, encoding="utf-8"
        )

    print(summary.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
