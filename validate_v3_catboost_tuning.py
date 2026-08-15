import os
import time
from itertools import product

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from calibration_utils import apply_calibration, fit_calibrators
from model_utils import FeatureBuilder, TARGET_COL


OUT_DIR = "output/v3_catboost_tuning"
TRAIN_PATH = "data/train.csv"
EPS = 1e-6
TEMPERATURE = 2.3
HARD_CAP = 0.020
FOLDS = [
    (2019, 2020, 2021, 2022),
    (2019, 2021, 2022, 2023),
    (2019, 2022, 2023, 2024),
]
BASELINE_PARAMS = {
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


def clip_prob(pred):
    return np.clip(np.asarray(pred, dtype=np.float64), EPS, 1.0 - EPS)


def logit(pred):
    pred = clip_prob(pred)
    return np.log(pred / (1.0 - pred))


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
    return linear_forecast(year_rates[year_rates.index < valid_year], tail=3)


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


def apply_v2_strength(platt_pred, target_rate):
    adjusted = logit_mean_match(platt_pred, target_rate)
    centered = logit(adjusted) - float(logit(target_rate))
    temp_pred = sigmoid(float(logit(target_rate)) + centered / TEMPERATURE)
    temp_pred = logit_mean_match(temp_pred, target_rate)
    deviation = clip_prob(temp_pred) - target_rate
    return clip_prob(target_rate + np.clip(deviation, -HARD_CAP, HARD_CAP))


def metric_dict(y, pred, prefix):
    pred = clip_prob(pred)
    actual_rate = float(np.mean(y))
    constant_brier = float(actual_rate * (1.0 - actual_rate))
    model_brier = float(brier_score_loss(y, pred))
    skill_margin = constant_brier - model_brier
    return {
        f"{prefix}_actual_rate": actual_rate,
        f"{prefix}_constant_brier": constant_brier,
        f"{prefix}_brier": model_brier,
        f"{prefix}_skill_margin": float(skill_margin),
        f"{prefix}_pseudo_score": float(max(0.0, 100000.0 * skill_margin / constant_brier)),
        f"{prefix}_auc": float(roc_auc_score(y, pred)),
        f"{prefix}_logloss": float(log_loss(y, pred)),
        f"{prefix}_pred_mean": float(pred.mean()),
        f"{prefix}_pred_std": float(pred.std()),
    }


def clean_params(params):
    out = dict(BASELINE_PARAMS)
    out.update(params)
    return out


def config_key(params):
    keys = ["iterations", "learning_rate", "depth", "l2_leaf_reg", "random_strength", "bagging_temperature", "rsm"]
    return "|".join(f"{k}={params.get(k, BASELINE_PARAMS.get(k, ''))}" for k in keys)


def make_stage1_configs():
    configs = [
        ("baseline_current", {}),
        ("shallow_more_trees_d4_i450_lr003_l2_5", {"depth": 4, "iterations": 450, "learning_rate": 0.03, "l2_leaf_reg": 5.0}),
        ("shallow_more_trees_d5_i450_lr003_l2_5", {"depth": 5, "iterations": 450, "learning_rate": 0.03, "l2_leaf_reg": 5.0}),
        ("shallow_slow_d5_i600_lr002_l2_5", {"depth": 5, "iterations": 600, "learning_rate": 0.02, "l2_leaf_reg": 5.0}),
        ("baseline_more_trees_i300_lr003", {"iterations": 300, "learning_rate": 0.03}),
        ("baseline_more_trees_i450_lr002", {"iterations": 450, "learning_rate": 0.02}),
        ("baseline_more_trees_i450_lr003", {"iterations": 450, "learning_rate": 0.03}),
        ("baseline_fast_i200_lr006", {"iterations": 200, "learning_rate": 0.06}),
        ("baseline_l2_3", {"l2_leaf_reg": 3.0}),
        ("baseline_l2_10", {"l2_leaf_reg": 10.0}),
        ("baseline_l2_20", {"l2_leaf_reg": 20.0}),
        ("deeper_d7_l2_10", {"depth": 7, "l2_leaf_reg": 10.0}),
        ("deeper_d7_i300_lr003_l2_10", {"depth": 7, "iterations": 300, "learning_rate": 0.03, "l2_leaf_reg": 10.0}),
        ("deeper_d8_l2_20", {"depth": 8, "l2_leaf_reg": 20.0}),
        ("deeper_d8_i300_lr003_l2_20", {"depth": 8, "iterations": 300, "learning_rate": 0.03, "l2_leaf_reg": 20.0}),
        ("random_strength_05", {"random_strength": 0.5}),
        ("random_strength_1", {"random_strength": 1.0}),
        ("random_strength_2", {"random_strength": 2.0}),
        ("bagging_temp_05", {"bagging_temperature": 0.5}),
        ("bagging_temp_1", {"bagging_temperature": 1.0}),
        ("bagging_temp_2", {"bagging_temperature": 2.0}),
        ("rsm_085", {"rsm": 0.85}),
        ("rsm_07_l2_10", {"rsm": 0.70, "l2_leaf_reg": 10.0}),
        (
            "conservative_d5_i300_lr003_l2_10_rs1_bag1_rsm085",
            {"depth": 5, "iterations": 300, "learning_rate": 0.03, "l2_leaf_reg": 10.0, "random_strength": 1.0, "bagging_temperature": 1.0, "rsm": 0.85},
        ),
        (
            "regularized_d6_i300_lr003_l2_20_rs1_bag1",
            {"depth": 6, "iterations": 300, "learning_rate": 0.03, "l2_leaf_reg": 20.0, "random_strength": 1.0, "bagging_temperature": 1.0},
        ),
    ]
    return [(name, clean_params(params)) for name, params in configs]


def make_stage2_configs(stage1_summary):
    top = stage1_summary[
        (stage1_summary["final_pseudo_2022"] > 0)
        & (stage1_summary["final_pseudo_2024"] > 0)
        & (stage1_summary["final_skill_2023"] > -0.0012)
    ].copy()
    if top.empty:
        top = stage1_summary.copy()
    top = top.sort_values(["final_mean_auc", "final_auc_2024", "final_worst_skill_margin"], ascending=[False, False, False]).head(5)
    out = []
    seen = set()
    for _, row in top.iterrows():
        base = {k: row[k] for k in ["iterations", "learning_rate", "depth", "l2_leaf_reg", "random_strength", "bagging_temperature", "rsm"] if pd.notna(row.get(k))}
        for updates in [
            {"iterations": int(min(600, base.get("iterations", 220) + 100))},
            {"learning_rate": max(0.02, round(base.get("learning_rate", 0.045) * 0.85, 5)), "iterations": int(min(600, base.get("iterations", 220) + 100))},
            {"l2_leaf_reg": min(20.0, float(base.get("l2_leaf_reg", 5.0)) * 1.5)},
            {"depth": int(max(4, min(8, base.get("depth", 6) - 1))), "iterations": int(min(600, base.get("iterations", 220) + 150))},
        ]:
            params = dict(base)
            params.update(updates)
            params = clean_params(params)
            key = config_key(params)
            if key in seen:
                continue
            seen.add(key)
            out.append((f"stage2_from_{row['config_name']}_{len(out)+1}", params))
            if len(out) >= 10:
                return out
    return out


def prepare_folds(df):
    fold_parts = []
    year_rates = df.groupby("season")[TARGET_COL].mean().sort_index()
    for train_start, train_end, cal_year, valid_year in FOLDS:
        train_df = df[(df["season"] >= train_start) & (df["season"] <= train_end)].copy()
        cal_df = df[df["season"] == cal_year].copy()
        valid_df = df[df["season"] == valid_year].copy()
        y_train = train_df[TARGET_COL].astype("int8")
        y_cal = cal_df[TARGET_COL].astype("int8").to_numpy()
        y_valid = valid_df[TARGET_COL].astype("int8").to_numpy()
        builder = FeatureBuilder(alpha=80.0)
        t0 = time.time()
        X_train = builder.fit_transform(train_df.drop(columns=[TARGET_COL]), y_train)
        X_cal = builder.transform(cal_df.drop(columns=[TARGET_COL]))
        X_valid = builder.transform(valid_df.drop(columns=[TARGET_COL]))
        feature_seconds = time.time() - t0
        fold_parts.append(
            {
                "fold": f"{train_start}-{train_end}_cal{cal_year}_valid{valid_year}",
                "train_start": train_start,
                "train_end": train_end,
                "cal_year": cal_year,
                "valid_year": valid_year,
                "X_train": X_train,
                "X_cal": X_cal,
                "X_valid": X_valid,
                "y_train": y_train,
                "y_cal": y_cal,
                "y_valid": y_valid,
                "cal_meta": cal_df[["game_type"]],
                "valid_meta": valid_df[["game_type"]],
                "target_rate": target_rate_for_year(year_rates, valid_year),
                "feature_names": list(X_train.columns),
                "feature_seconds": feature_seconds,
            }
        )
        print(
            f"Prepared fold {valid_year}: train={len(train_df)} cal={len(cal_df)} valid={len(valid_df)} "
            f"features={X_train.shape[1]} seconds={feature_seconds:.2f}"
        )
    return fold_parts


def evaluate_configs(configs, folds, stage_name):
    fold_rows = []
    runtime_rows = []
    pred_store = {}
    feature_rows = []
    for config_name, params in configs:
        print(f"\n[{stage_name}] {config_name} {params}")
        final_oof = []
        for fold in folds:
            model = CatBoostClassifier(**params)
            t0 = time.time()
            model.fit(fold["X_train"], fold["y_train"])
            train_seconds = time.time() - t0
            t0 = time.time()
            raw_cal = clip_prob(model.predict_proba(fold["X_cal"])[:, 1])
            raw_valid = clip_prob(model.predict_proba(fold["X_valid"])[:, 1])
            inference_seconds = time.time() - t0
            calibration = fit_calibrators(raw_cal, fold["y_cal"], fold["cal_meta"])
            platt_valid = apply_calibration(raw_valid, fold["valid_meta"], calibration, "platt")
            final_valid = apply_v2_strength(platt_valid, fold["target_rate"])
            final_oof.append(final_valid)

            row = {
                "stage": stage_name,
                "config_name": config_name,
                "fold": fold["fold"],
                "valid_year": fold["valid_year"],
                "target_rate": fold["target_rate"],
                **{k: params.get(k, np.nan) for k in ["iterations", "learning_rate", "depth", "l2_leaf_reg", "random_strength", "bagging_temperature", "rsm"]},
                **metric_dict(fold["y_valid"], raw_valid, "raw"),
                **metric_dict(fold["y_valid"], final_valid, "final"),
            }
            fold_rows.append(row)
            runtime_rows.append(
                {
                    "stage": stage_name,
                    "config_name": config_name,
                    "valid_year": fold["valid_year"],
                    "feature_seconds": fold["feature_seconds"],
                    "train_seconds": train_seconds,
                    "inference_seconds": inference_seconds,
                    "rows_train": len(fold["y_train"]),
                    "rows_cal": len(fold["y_cal"]),
                    "rows_valid": len(fold["y_valid"]),
                }
            )
            try:
                fi = model.get_feature_importance()
                top_idx = np.argsort(fi)[::-1][:30]
                for rank, idx in enumerate(top_idx, start=1):
                    feature_rows.append(
                        {
                            "stage": stage_name,
                            "config_name": config_name,
                            "valid_year": fold["valid_year"],
                            "rank": rank,
                            "feature": fold["feature_names"][idx],
                            "importance": float(fi[idx]),
                        }
                    )
            except Exception as exc:
                feature_rows.append(
                    {
                        "stage": stage_name,
                        "config_name": config_name,
                        "valid_year": fold["valid_year"],
                        "rank": 0,
                        "feature": f"IMPORTANCE_ERROR:{repr(exc)}",
                        "importance": np.nan,
                    }
                )
            print(
                f"  valid={fold['valid_year']} raw_auc={row['raw_auc']:.6f} final_auc={row['final_auc']:.6f} "
                f"pseudo={row['final_pseudo_score']:.3f} train_s={train_seconds:.2f}"
            )
        pred_store[config_name] = np.concatenate(final_oof)
    return pd.DataFrame(fold_rows), pd.DataFrame(runtime_rows), pred_store, pd.DataFrame(feature_rows)


def summarize_fold_metrics(fold_metrics):
    rows = []
    param_cols = ["iterations", "learning_rate", "depth", "l2_leaf_reg", "random_strength", "bagging_temperature", "rsm"]
    for (stage, config_name), g in fold_metrics.groupby(["stage", "config_name"], sort=False):
        row = {"stage": stage, "config_name": config_name}
        for col in param_cols:
            row[col] = g[col].dropna().iloc[0] if g[col].notna().any() else np.nan
        for prefix in ["raw", "final"]:
            row[f"{prefix}_mean_auc"] = float(g[f"{prefix}_auc"].mean())
            row[f"{prefix}_worst_auc"] = float(g[f"{prefix}_auc"].min())
            row[f"{prefix}_mean_brier"] = float(g[f"{prefix}_brier"].mean())
            row[f"{prefix}_worst_brier"] = float(g[f"{prefix}_brier"].max())
            row[f"{prefix}_mean_skill_margin"] = float(g[f"{prefix}_skill_margin"].mean())
            row[f"{prefix}_worst_skill_margin"] = float(g[f"{prefix}_skill_margin"].min())
            row[f"{prefix}_mean_pseudo_score"] = float(g[f"{prefix}_pseudo_score"].mean())
            row[f"{prefix}_positive_fold_count"] = int((g[f"{prefix}_skill_margin"] > 0).sum())
            row[f"{prefix}_mean_pred_std"] = float(g[f"{prefix}_pred_std"].mean())
            for year in [2022, 2023, 2024]:
                y = g[g["valid_year"] == year].iloc[0]
                row[f"{prefix}_auc_{year}"] = float(y[f"{prefix}_auc"])
                row[f"{prefix}_brier_{year}"] = float(y[f"{prefix}_brier"])
                row[f"{prefix}_pseudo_{year}"] = float(y[f"{prefix}_pseudo_score"])
                row[f"{prefix}_skill_{year}"] = float(y[f"{prefix}_skill_margin"])
        rows.append(row)
    summary = pd.DataFrame(rows)
    baseline = summary[summary["config_name"] == "baseline_current"].iloc[0]
    for col in [
        "raw_mean_auc",
        "raw_auc_2022",
        "raw_auc_2023",
        "raw_auc_2024",
        "final_mean_auc",
        "final_auc_2022",
        "final_auc_2023",
        "final_auc_2024",
        "final_mean_brier",
        "final_brier_2022",
        "final_brier_2023",
        "final_brier_2024",
        "final_pseudo_2022",
        "final_pseudo_2023",
        "final_pseudo_2024",
    ]:
        summary[f"delta_vs_v2_{col}"] = summary[col] - float(baseline[col])
    return summary.sort_values(
        ["final_mean_auc", "final_auc_2024", "final_worst_skill_margin", "final_mean_pseudo_score"],
        ascending=[False, False, False, False],
    )


def candidate_labels(summary, pred_corr):
    merged = summary.merge(pred_corr[["config_name", "corr_vs_v2"]], on="config_name", how="left")
    labels = {}
    stable = merged[
        (merged["final_pseudo_2022"] > 0)
        & (merged["final_pseudo_2024"] > 0)
        & (merged["final_skill_2023"] > -0.0010)
    ].sort_values(["final_worst_skill_margin", "final_mean_auc", "final_auc_2024"], ascending=[False, False, False])
    if not stable.empty:
        labels[stable.iloc[0]["config_name"]] = "Candidate 1 - fold-stable"
    discr = merged.sort_values(["final_mean_auc", "final_auc_2024"], ascending=[False, False])
    if not discr.empty:
        labels[discr.iloc[0]["config_name"]] = labels.get(discr.iloc[0]["config_name"], "") + "; Candidate 2 - highest discrimination"
    low_corr = merged[
        (~merged["config_name"].isin(labels))
        & (merged["final_mean_auc"] >= merged["final_mean_auc"].quantile(0.75))
    ].sort_values(["corr_vs_v2", "final_mean_auc"], ascending=[True, False])
    if not low_corr.empty:
        labels[low_corr.iloc[0]["config_name"]] = "Candidate 3 - lower correlation ensemble probe"
    return labels


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig")
    folds = prepare_folds(df)

    stage1_configs = make_stage1_configs()
    stage1_fold, stage1_runtime, stage1_preds, stage1_fi = evaluate_configs(stage1_configs, folds, "stage1")
    stage1_summary = summarize_fold_metrics(stage1_fold)
    stage1_fold.to_csv(os.path.join(OUT_DIR, "stage1_results.csv"), index=False, encoding="utf-8")

    stage2_configs = make_stage2_configs(stage1_summary)
    existing = {config_key(params) for _, params in stage1_configs}
    stage2_configs = [(n, p) for n, p in stage2_configs if config_key(p) not in existing]
    if stage2_configs:
        stage2_fold, stage2_runtime, stage2_preds, stage2_fi = evaluate_configs(stage2_configs, folds, "stage2")
    else:
        stage2_fold, stage2_runtime, stage2_fi = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        stage2_preds = {}
    stage2_fold.to_csv(os.path.join(OUT_DIR, "stage2_results.csv"), index=False, encoding="utf-8")

    fold_metrics = pd.concat([stage1_fold, stage2_fold], ignore_index=True)
    runtime = pd.concat([stage1_runtime, stage2_runtime], ignore_index=True)
    feature_importance = pd.concat([stage1_fi, stage2_fi], ignore_index=True)
    preds = {**stage1_preds, **stage2_preds}
    summary = summarize_fold_metrics(fold_metrics)

    v2_pred = preds["baseline_current"]
    corr_rows = []
    for config_name, pred in preds.items():
        corr_rows.append(
            {
                "config_name": config_name,
                "corr_vs_v2": float(np.corrcoef(v2_pred, pred)[0, 1]),
                "mean_abs_diff_vs_v2": float(np.mean(np.abs(pred - v2_pred))),
                "max_abs_diff_vs_v2": float(np.max(np.abs(pred - v2_pred))),
            }
        )
    pred_corr = pd.DataFrame(corr_rows)
    labels = candidate_labels(summary, pred_corr)
    summary["selection_note"] = summary["config_name"].map(labels).fillna("")
    summary = summary.merge(pred_corr, on="config_name", how="left")

    selected_names = [name for name in labels if name in set(summary["config_name"])]
    feature_top = feature_importance[feature_importance["config_name"].isin(selected_names)].copy()
    if "baseline_current" in set(feature_importance["config_name"]):
        feature_top = pd.concat(
            [feature_importance[feature_importance["config_name"] == "baseline_current"], feature_top],
            ignore_index=True,
        ).drop_duplicates(["stage", "config_name", "valid_year", "rank", "feature"])
    feature_top = feature_top[feature_top["rank"].between(1, 30)]

    fold_metrics.to_csv(os.path.join(OUT_DIR, "fold_metrics.csv"), index=False, encoding="utf-8")
    summary.to_csv(os.path.join(OUT_DIR, "candidate_summary.csv"), index=False, encoding="utf-8")
    pred_corr.to_csv(os.path.join(OUT_DIR, "prediction_correlation.csv"), index=False, encoding="utf-8")
    feature_top.to_csv(os.path.join(OUT_DIR, "feature_importance_top.csv"), index=False, encoding="utf-8")
    runtime.to_csv(os.path.join(OUT_DIR, "runtime_summary.csv"), index=False, encoding="utf-8")

    print("\nTop summary")
    print(
        summary[
            [
                "stage",
                "config_name",
                "final_mean_auc",
                "final_auc_2024",
                "final_pseudo_2022",
                "final_pseudo_2023",
                "final_pseudo_2024",
                "final_mean_brier",
                "final_worst_skill_margin",
                "corr_vs_v2",
                "selection_note",
            ]
        ].head(20).to_string(index=False)
    )
    print(f"\nconfig_count={summary['config_name'].nunique()}")


if __name__ == "__main__":
    main()
