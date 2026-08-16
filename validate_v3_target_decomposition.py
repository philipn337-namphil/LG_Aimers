import os
from itertools import combinations

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from scipy.stats import spearmanr
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from calibration_utils import apply_calibration, fit_calibrators, logit, sigmoid
from model_utils import FeatureBuilder, TARGET_COL
from validate_v3_asof_signal_reconstruction import V2_CATBOOST_PARAMS, apply_v2_strength, clip_prob, target_rate_for_year


OUT_DIR = "output/v3_target_decomposition"
TRAIN_PATH = "data/train.csv"
FOLDS = [
    (2019, 2020, 2021, 2022),
    (2019, 2021, 2022, 2023),
    (2019, 2022, 2023, 2024),
]
REG_PARAMS = {
    "loss_function": "RMSE",
    "iterations": 220,
    "learning_rate": 0.045,
    "depth": 6,
    "l2_leaf_reg": 5.0,
    "random_seed": 42,
    "verbose": False,
    "allow_writing_files": False,
    "thread_count": -1,
}


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


def safe_year_rates(df, years):
    return df[df["season"].isin(years)].groupby("season")[TARGET_COL].mean().sort_index()


def deterministic_future_baseline(df, year):
    year_rates = df[df["season"] < year].groupby("season")[TARGET_COL].mean().sort_index()
    return target_rate_for_year(year_rates, year)


def loo_season_baseline(y, season):
    tmp = pd.DataFrame({"y": np.asarray(y, dtype=np.float64), "season": np.asarray(season)})
    sums = tmp.groupby("season")["y"].transform("sum")
    counts = tmp.groupby("season")["y"].transform("count")
    global_mean = float(tmp["y"].mean())
    return ((sums - tmp["y"]) / np.maximum(counts - 1, 1)).fillna(global_mean).to_numpy(dtype=np.float64)


def season_baseline_for_rows(history_df, rows):
    rates = history_df.groupby("season")[TARGET_COL].mean().sort_index()
    out = []
    for year in rows["season"].astype(int):
        out.append(target_rate_for_year(rates[rates.index < year], int(year)))
    return np.asarray(out, dtype=np.float64)


def target_distribution_audit(df):
    rows = []
    dist_rows = []
    y = df[TARGET_COL].astype(float).to_numpy()
    q_loo = loo_season_baseline(y, df["season"])
    targets = {
        "A_ORIGINAL": y,
        "B_SEASON_CENTERED": y - q_loo,
        "C_LOGIT_RESIDUAL": np.clip((y - q_loo) / np.maximum(q_loo * (1.0 - q_loo), 1e-6), -4.0, 4.0),
    }
    for name, vals in targets.items():
        s = pd.Series(vals)
        q = s.quantile([0.01, 0.05, 0.5, 0.95, 0.99])
        rows.append({
            "target_definition": name,
            "mean": float(s.mean()),
            "std": float(s.std()),
            "min": float(s.min()),
            "max": float(s.max()),
            "p01": float(q.loc[0.01]),
            "p05": float(q.loc[0.05]),
            "median": float(q.loc[0.5]),
            "p95": float(q.loc[0.95]),
            "p99": float(q.loc[0.99]),
            "corr_with_original": float(np.corrcoef(y, vals)[0, 1]),
        })
        tmp = pd.DataFrame({"season": df["season"].to_numpy(), "target": vals})
        by = tmp.groupby("season")["target"].agg(["mean", "std"]).reset_index()
        by["target_definition"] = name
        by["year_to_year_mean_drift"] = by["mean"].diff()
        dist_rows.append(by)
    return pd.DataFrame(rows), pd.concat(dist_rows, ignore_index=True)


def prepare_features(train_df, cal_df, valid_df):
    y_train = train_df[TARGET_COL].astype("int8")
    y_cal = cal_df[TARGET_COL].astype("int8")
    y_valid = valid_df[TARGET_COL].astype("int8")
    builder = FeatureBuilder()
    x_train = builder.fit_transform(train_df.drop(columns=[TARGET_COL]), y_train)
    x_cal = builder.transform(cal_df.drop(columns=[TARGET_COL]))
    x_valid = builder.transform(valid_df.drop(columns=[TARGET_COL]))
    return x_train, x_cal, x_valid, y_train, y_cal, y_valid


def candidate_specs():
    return [
        ("A_ORIGINAL", "binary classifier", "control_success", "CatBoostClassifier Logloss"),
        ("B_SEASON_CENTERED", "season-centered residual", "control_success - leave-one-out season baseline", "CatBoostRegressor RMSE"),
        ("C_LOGIT_RESIDUAL", "pseudo-logit residual", "clip((y - q_loo)/(q_loo*(1-q_loo)), -4, 4)", "CatBoostRegressor RMSE"),
        ("D_BASELINE_PLUS_MODEL_DEVIATION", "classifier deviation from model mean", "control_success classifier; final p=q_future + raw - mean(raw_cal)", "CatBoostClassifier Logloss"),
    ]


def native_predictions(name, x_train, x_cal, x_valid, y_train, train_df, cal_df, valid_df, full_df):
    if name == "A_ORIGINAL":
        model = CatBoostClassifier(**V2_CATBOOST_PARAMS)
        model.fit(x_train, y_train)
        return model, clip_prob(model.predict_proba(x_cal)[:, 1]), clip_prob(model.predict_proba(x_valid)[:, 1])
    if name == "D_BASELINE_PLUS_MODEL_DEVIATION":
        model = CatBoostClassifier(**V2_CATBOOST_PARAMS)
        model.fit(x_train, y_train)
        raw_cal = clip_prob(model.predict_proba(x_cal)[:, 1])
        raw_valid = clip_prob(model.predict_proba(x_valid)[:, 1])
        q_cal = season_baseline_for_rows(full_df[full_df["season"] < int(cal_df["season"].iloc[0])], cal_df)
        q_valid = np.full(len(valid_df), deterministic_future_baseline(full_df, int(valid_df["season"].iloc[0])))
        ref = float(raw_cal.mean())
        return model, clip_prob(q_cal + raw_cal - ref), clip_prob(q_valid + raw_valid - ref)
    q_loo = loo_season_baseline(y_train, train_df["season"])
    if name == "B_SEASON_CENTERED":
        target = y_train.to_numpy(dtype=np.float64) - q_loo
        model = CatBoostRegressor(**REG_PARAMS)
        model.fit(x_train, target)
        q_cal = season_baseline_for_rows(full_df[full_df["season"] < int(cal_df["season"].iloc[0])], cal_df)
        q_valid = np.full(len(valid_df), deterministic_future_baseline(full_df, int(valid_df["season"].iloc[0])))
        return model, clip_prob(q_cal + model.predict(x_cal)), clip_prob(q_valid + model.predict(x_valid))
    if name == "C_LOGIT_RESIDUAL":
        target = np.clip((y_train.to_numpy(dtype=np.float64) - q_loo) / np.maximum(q_loo * (1.0 - q_loo), 1e-6), -4.0, 4.0)
        model = CatBoostRegressor(**REG_PARAMS)
        model.fit(x_train, target)
        q_cal = season_baseline_for_rows(full_df[full_df["season"] < int(cal_df["season"].iloc[0])], cal_df)
        q_valid = np.full(len(valid_df), deterministic_future_baseline(full_df, int(valid_df["season"].iloc[0])))
        return model, clip_prob(sigmoid(logit(q_cal) + model.predict(x_cal))), clip_prob(sigmoid(logit(q_valid) + model.predict(x_valid)))
    raise ValueError(name)


def evaluate(df):
    fold_rows = []
    y2023_rows = []
    pred_parts = []
    fi_rows = []
    for name, target_def, target_formula, model_type in candidate_specs():
        for train_start, train_end, cal_year, valid_year in FOLDS:
            train_df = df[(df["season"] >= train_start) & (df["season"] <= train_end)].copy()
            cal_df = df[df["season"] == cal_year].copy()
            valid_df = df[df["season"] == valid_year].copy()
            x_train, x_cal, x_valid, y_train, y_cal, y_valid = prepare_features(train_df, cal_df, valid_df)
            model, native_cal, native_valid = native_predictions(name, x_train, x_cal, x_valid, y_train, train_df, cal_df, valid_df, df)
            platt = fit_calibrators(native_cal, y_cal, cal_df[["game_type"]])
            platt_valid = apply_calibration(native_valid, valid_df[["game_type"]], platt, "platt")
            strength_valid = apply_v2_strength(platt_valid, deterministic_future_baseline(df, valid_year))
            for mode, pred in [("native", native_valid), ("platt", platt_valid), ("v2_strength", strength_valid)]:
                fold_rows.append({
                    "candidate_name": name,
                    "target_definition": target_def,
                    "target_formula": target_formula,
                    "model_type": model_type,
                    "calibration_mode": mode,
                    "valid_year": valid_year,
                    "estimated_global_baseline": deterministic_future_baseline(df, valid_year),
                    **metric_dict(y_valid, pred, "metric"),
                })
            if valid_year == 2023:
                for mode, pred in [("native", native_valid), ("platt", platt_valid), ("v2_strength", strength_valid)]:
                    median = np.median(pred)
                    low = pred < median
                    high = ~low
                    y = y_valid.to_numpy()
                    y2023_rows.append({
                        "candidate_name": name,
                        "calibration_mode": mode,
                        "auc": float(roc_auc_score(y, pred)),
                        "brier": float(brier_score_loss(y, pred)),
                        "constant_brier": float(y.mean() * (1 - y.mean())),
                        "skill_margin": float(y.mean() * (1 - y.mean()) - brier_score_loss(y, pred)),
                        "pseudo_score": float(max(0.0, 100000.0 * (y.mean() * (1 - y.mean()) - brier_score_loss(y, pred)) / (y.mean() * (1 - y.mean())))),
                        "prediction_mean": float(pred.mean()),
                        "prediction_std": float(pred.std()),
                        "low_half_actual_rate": float(y[low].mean()),
                        "high_half_actual_rate": float(y[high].mean()),
                        "low_half_prediction_mean": float(pred[low].mean()),
                        "high_half_prediction_mean": float(pred[high].mean()),
                        "high_minus_low_actual_rate": float(y[high].mean() - y[low].mean()),
                    })
            pred_parts.append(pd.DataFrame({
                "row_id": valid_df["row_id"].to_numpy(),
                "valid_year": valid_year,
                "target": y_valid.to_numpy(),
                "candidate_name": name,
                "native_pred": native_valid,
                "platt_pred": platt_valid,
                "v2_strength_pred": strength_valid,
                "estimated_global_baseline": deterministic_future_baseline(df, valid_year),
            }))
            if name != "A_ORIGINAL":
                imp = model.get_feature_importance()
                for rank, idx in enumerate(np.argsort(imp)[::-1][:40], start=1):
                    fi_rows.append({
                        "candidate_name": name,
                        "valid_year": valid_year,
                        "rank": rank,
                        "feature": x_train.columns[idx],
                        "importance": float(imp[idx]),
                    })
            print(f"{name} valid={valid_year} native_auc={roc_auc_score(y_valid, native_valid):.6f}")
    return pd.DataFrame(fold_rows), pd.DataFrame(y2023_rows), pd.concat(pred_parts, ignore_index=True), pd.DataFrame(fi_rows)


def summarize(fold_metrics):
    rows = []
    base = fold_metrics[(fold_metrics["candidate_name"] == "A_ORIGINAL") & (fold_metrics["calibration_mode"] == "v2_strength")].set_index("valid_year")
    for (cand, mode), g in fold_metrics.groupby(["candidate_name", "calibration_mode"], sort=False):
        by = g.set_index("valid_year")
        row = {
            "candidate_name": cand,
            "calibration_mode": mode,
            "target_definition": g["target_definition"].iloc[0],
            "baseline_estimator": "linear_trend_recent3 with deterministic fallback",
            "model_type": g["model_type"].iloc[0],
            "mean_auc": float(g["metric_auc"].mean()),
            "mean_brier": float(g["metric_brier"].mean()),
            "worst_skill_margin": float(g["metric_skill_margin"].min()),
            "positive_fold_count": int((g["metric_skill_margin"] > 0).sum()),
        }
        for year in [2022, 2023, 2024]:
            row[f"auc_{year}"] = float(by.loc[year, "metric_auc"])
            row[f"pseudo_{year}"] = float(by.loc[year, "metric_pseudo_score"])
            row[f"skill_{year}"] = float(by.loc[year, "metric_skill_margin"])
            row[f"brier_{year}"] = float(by.loc[year, "metric_brier"])
            row[f"delta_auc_{year}_vs_v2"] = float(by.loc[year, "metric_auc"] - base.loc[year, "metric_auc"])
            row[f"delta_skill_{year}_vs_v2"] = float(by.loc[year, "metric_skill_margin"] - base.loc[year, "metric_skill_margin"])
        row["delta_mean_auc_vs_v2"] = row["mean_auc"] - float(base["metric_auc"].mean())
        row["success_a"] = row["mean_auc"] >= 0.523
        row["success_b"] = row["pseudo_2024"] >= 25 and row["pseudo_2022"] > 0 and row["skill_2023"] >= float(base.loc[2023, "metric_skill_margin"])
        row["success_c"] = row["skill_2023"] >= float(base.loc[2023, "metric_skill_margin"]) + 0.0001 and row["pseudo_2022"] > 0 and row["pseudo_2024"] > 0
        row["success_d"] = all(row[f"delta_skill_{y}_vs_v2"] > 0 for y in [2022, 2023, 2024])
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["mean_auc", "pseudo_2024", "skill_2023"], ascending=[False, False, False])


def deviation_buckets(preds, top_candidates):
    rows = []
    bins = [0, 0.005, 0.010, 0.020, 0.040, np.inf]
    labels = ["0_005", "005_010", "010_020", "020_040", "040_plus"]
    for cand in top_candidates:
        p = preds[preds["candidate_name"] == cand].copy()
        for mode_col in ["native_pred", "platt_pred", "v2_strength_pred"]:
            p["dev_abs"] = (p[mode_col] - p["estimated_global_baseline"]).abs()
            p["bucket"] = pd.cut(p["dev_abs"], bins=bins, labels=labels, include_lowest=True)
            for (year, bucket), g in p.groupby(["valid_year", "bucket"], observed=True):
                y = g["target"].to_numpy()
                pred = g[mode_col].to_numpy()
                baseline = g["estimated_global_baseline"].to_numpy()
                rows.append({
                    "candidate_name": cand,
                    "prediction_mode": mode_col.replace("_pred", ""),
                    "valid_year": year,
                    "deviation_bucket": bucket,
                    "n": len(g),
                    "actual_rate": float(y.mean()),
                    "prediction_mean": float(pred.mean()),
                    "brier": float(brier_score_loss(y, pred)),
                    "baseline_brier": float(np.mean((baseline - y) ** 2)),
                    "skill_gain_vs_baseline": float(np.mean((baseline - y) ** 2) - brier_score_loss(y, pred)),
                })
    return pd.DataFrame(rows)


def prediction_correlation(preds, top_candidates):
    base = preds[preds["candidate_name"] == "A_ORIGINAL"][["row_id", "valid_year", "target", "v2_strength_pred"]].rename(columns={"v2_strength_pred": "v2_pred"})
    rows = []
    for cand in top_candidates:
        if cand == "A_ORIGINAL":
            continue
        c = preds[preds["candidate_name"] == cand][["row_id", "valid_year", "target", "v2_strength_pred"]].rename(columns={"v2_strength_pred": "cand_pred"})
        m = base.merge(c, on=["row_id", "valid_year", "target"])
        for scope, g in [("overall", m), *[(str(y), gy) for y, gy in m.groupby("valid_year")]]:
            ev = (g["v2_pred"] - g["target"]) ** 2
            ec = (g["cand_pred"] - g["target"]) ** 2
            rows.append({
                "candidate_name": cand,
                "scope": scope,
                "pearson_corr": float(np.corrcoef(g["v2_pred"], g["cand_pred"])[0, 1]),
                "spearman_corr": float(spearmanr(g["v2_pred"], g["cand_pred"]).correlation),
                "mean_abs_prediction_diff": float(np.mean(np.abs(g["v2_pred"] - g["cand_pred"]))),
                "squared_error_corr": float(np.corrcoef(ev, ec)[0, 1]),
            })
    return pd.DataFrame(rows)


def error_decomposition(preds, top_candidates):
    base = preds[preds["candidate_name"] == "A_ORIGINAL"][["row_id", "valid_year", "target", "v2_strength_pred"]].rename(columns={"v2_strength_pred": "v2_pred"})
    rows = []
    for cand in top_candidates:
        if cand == "A_ORIGINAL":
            continue
        c = preds[preds["candidate_name"] == cand][["row_id", "valid_year", "target", "v2_strength_pred"]].rename(columns={"v2_strength_pred": "cand_pred"})
        m = base.merge(c, on=["row_id", "valid_year", "target"])
        for scope, g in [("overall", m), *[(str(y), gy) for y, gy in m.groupby("valid_year")]]:
            ev = (g["v2_pred"] - g["target"]) ** 2
            ec = (g["cand_pred"] - g["target"]) ** 2
            rows.append({
                "candidate_name": cand,
                "scope": scope,
                "n": len(g),
                "candidate_better_rate": float((ec < ev).mean()),
                "v2_better_rate": float((ev < ec).mean()),
                "tie_rate": float(np.isclose(ec, ev).mean()),
                "both_poor_top_decile_rate": float(((ec >= np.quantile(ec, 0.9)) & (ev >= np.quantile(ev, 0.9))).mean()),
            })
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig")
    target_summary, target_by_year = target_distribution_audit(df)
    fold_metrics, y2023, preds, fi = evaluate(df)
    summary = summarize(fold_metrics)
    top = summary["candidate_name"].drop_duplicates().head(3).tolist()
    dev = deviation_buckets(preds, top)
    corr = prediction_correlation(preds, top)
    err = error_decomposition(preds, top)
    verdict = "TARGET DECOMPOSITION SIGNAL FOUND" if summary[["success_a", "success_b", "success_c", "success_d"]].any(axis=None) else "TARGET DECOMPOSITION NOT USEFUL"
    target_summary.to_csv(os.path.join(OUT_DIR, "target_definition_summary.csv"), index=False, encoding="utf-8")
    target_by_year.to_csv(os.path.join(OUT_DIR, "target_distribution_by_year.csv"), index=False, encoding="utf-8")
    fold_metrics.to_csv(os.path.join(OUT_DIR, "fold_metrics.csv"), index=False, encoding="utf-8")
    summary.to_csv(os.path.join(OUT_DIR, "candidate_summary.csv"), index=False, encoding="utf-8")
    y2023.to_csv(os.path.join(OUT_DIR, "year2023_analysis.csv"), index=False, encoding="utf-8")
    dev.to_csv(os.path.join(OUT_DIR, "deviation_bucket_analysis.csv"), index=False, encoding="utf-8")
    corr.to_csv(os.path.join(OUT_DIR, "prediction_correlation.csv"), index=False, encoding="utf-8")
    err.to_csv(os.path.join(OUT_DIR, "error_decomposition.csv"), index=False, encoding="utf-8")
    pd.DataFrame([{"verdict": verdict}]).to_csv(os.path.join(OUT_DIR, "verdict.csv"), index=False, encoding="utf-8")
    fi.to_csv(os.path.join(OUT_DIR, "feature_importance.csv"), index=False, encoding="utf-8")
    preds.to_csv(os.path.join(OUT_DIR, "oof_predictions.csv"), index=False, encoding="utf-8")
    print(summary.head(12).to_string(index=False))
    print(verdict)


if __name__ == "__main__":
    main()
