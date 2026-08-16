import os
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.stats import spearmanr
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from calibration_utils import apply_calibration, fit_calibrators, logit, sigmoid
from model_utils import FeatureBuilder, TARGET_COL
from validate_v3_asof_signal_reconstruction import V2_CATBOOST_PARAMS, apply_v2_strength, clip_prob, target_rate_for_year


OUT_DIR = "output/v3_hierarchical"
TRAIN_PATH = "data/train.csv"
FOLDS = [
    (2019, 2020, 2021, 2022),
    (2019, 2021, 2022, 2023),
    (2019, 2022, 2023, 2024),
]
EPS = 1e-6
ALPHA_GRID = [20, 50, 100, 300, 1000]
ALPHA_PITCHER = 100
ALPHA_CONTEXT = 300


def add_context_columns(df):
    out = df.copy()
    out["count_state"] = out["balls_before"].astype(str) + "-" + out["strikes_before"].astype(str)
    return out


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


def future_global_rate(df, year):
    rates = df[df["season"] < year].groupby("season")[TARGET_COL].mean().sort_index()
    return target_rate_for_year(rates, year)


def group_key(df, definition):
    if definition == "pitcher":
        return df["pitcher_id"].astype(str)
    if definition == "pitcher_count":
        return df["pitcher_id"].astype(str) + "|" + df["count_state"].astype(str)
    if definition == "pitcher_game":
        return df["pitcher_id"].astype(str) + "|" + df["game_type"].astype(str)
    if definition == "pitcher_batter_hand":
        return df["pitcher_id"].astype(str) + "|" + df["batter_hand"].astype(str)
    if definition == "pitcher_base":
        return df["pitcher_id"].astype(str) + "|" + df["base_state"].astype(str)
    if definition == "pitcher_game_count":
        return df["pitcher_id"].astype(str) + "|" + df["game_type"].astype(str) + "|" + df["count_state"].astype(str)
    raise ValueError(definition)


def hierarchy_feasibility(df):
    rows = []
    definitions = ["pitcher", "pitcher_game", "pitcher_count", "pitcher_batter_hand", "pitcher_base", "pitcher_game_count"]
    for _, train_end, _, valid_year in FOLDS:
        hist = df[df["season"] <= train_end].copy()
        valid = df[df["season"] == valid_year].copy()
        for definition in definitions:
            hist_key = group_key(hist, definition)
            valid_key = group_key(valid, definition)
            grouped = pd.DataFrame({"group": hist_key, "y": hist[TARGET_COL].astype(int)}).groupby("group")["y"].agg(["count", "sum"])
            grouped["fail"] = grouped["count"] - grouped["sum"]
            valid_known = valid_key.isin(grouped.index)
            rows.append(
                {
                    "valid_year": valid_year,
                    "hierarchy_level": definition,
                    "group_count": int(len(grouped)),
                    "row_count": int(len(hist)),
                    "median_sample_size": float(grouped["count"].median()),
                    "p10_sample_size": float(grouped["count"].quantile(0.10)),
                    "p90_sample_size": float(grouped["count"].quantile(0.90)),
                    "low_sample_group_rate_lt20": float((grouped["count"] < 20).mean()),
                    "low_sample_group_rate_lt50": float((grouped["count"] < 50).mean()),
                    "both_class_group_rate": float(((grouped["sum"] > 0) & (grouped["fail"] > 0)).mean()),
                    "validation_row_coverage": float(valid_known.mean()),
                    "validation_unseen_row_rate": float((~valid_known).mean()),
                }
            )
    return pd.DataFrame(rows)


def posterior_map(history, key, parent_rate, alpha):
    grouped = pd.DataFrame({"key": key, "y": history[TARGET_COL].astype(float)}).groupby("key")["y"].agg(["sum", "count"])
    grouped["posterior"] = (grouped["sum"] + alpha * parent_rate) / (grouped["count"] + alpha)
    return grouped["posterior"].to_dict(), grouped["count"].to_dict()


def pitcher_posteriors(history, q, alpha=ALPHA_PITCHER):
    return posterior_map(history, group_key(history, "pitcher"), q, alpha)


def predict_global_pitcher(history, rows, q_future, alpha_pitcher=ALPHA_PITCHER):
    pmap, cmap = pitcher_posteriors(history, q_future, alpha_pitcher)
    keys = group_key(rows, "pitcher")
    pred = keys.map(pmap).fillna(q_future).to_numpy(dtype=np.float64)
    count = keys.map(cmap).fillna(0).to_numpy(dtype=np.float64)
    return clip_prob(pred), count


def predict_pitcher_context(history, rows, q_future, context_definition, alpha_pitcher=ALPHA_PITCHER, alpha_context=ALPHA_CONTEXT):
    p_pitcher, p_count = predict_global_pitcher(history, rows, q_future, alpha_pitcher)
    hist_pitcher_map, _ = pitcher_posteriors(history, q_future, alpha_pitcher)
    hist_parent = group_key(history, "pitcher").map(hist_pitcher_map).fillna(q_future).to_numpy(dtype=np.float64)
    ckey_hist = group_key(history, context_definition)
    tmp = pd.DataFrame({"key": ckey_hist, "y": history[TARGET_COL].astype(float), "parent": hist_parent})
    grouped = tmp.groupby("key").agg(success=("y", "sum"), count=("y", "count"), parent=("parent", "mean"))
    grouped["posterior"] = (grouped["success"] + alpha_context * grouped["parent"]) / (grouped["count"] + alpha_context)
    pmap = grouped["posterior"].to_dict()
    cmap = grouped["count"].to_dict()
    rkey = group_key(rows, context_definition)
    pred = rkey.map(pmap).fillna(pd.Series(p_pitcher, index=rows.index)).to_numpy(dtype=np.float64)
    ccount = rkey.map(cmap).fillna(0).to_numpy(dtype=np.float64)
    return clip_prob(pred), p_count, ccount


def predict_multi_context_logit(history, rows, q_future):
    p_pitcher_rows, p_count = predict_global_pitcher(history, rows, q_future)
    hist_p_rows, _ = predict_global_pitcher(history, history, q_future)
    effects = logit(p_pitcher_rows) - logit(q_future)
    for definition in ["pitcher_count", "pitcher_game"]:
        hist_pred, _, _ = predict_pitcher_context(history, history, q_future, definition)
        hist_parent = hist_p_rows
        ckey = group_key(history, definition)
        grouped = pd.DataFrame(
            {
                "key": ckey,
                "effect": logit(hist_pred) - logit(hist_parent),
                "n": 1,
            }
        ).groupby("key").agg(effect=("effect", "mean"), count=("n", "sum"))
        row_key = group_key(rows, definition)
        row_effect = row_key.map(grouped["effect"].to_dict()).fillna(0.0).to_numpy(dtype=np.float64)
        row_count = row_key.map(grouped["count"].to_dict()).fillna(0.0).to_numpy(dtype=np.float64)
        shrink = row_count / (row_count + ALPHA_CONTEXT)
        effects += row_effect * shrink
    return clip_prob(sigmoid(logit(q_future) + np.clip(effects, -0.20, 0.20))), p_count


def prepare_features(train_df, cal_df, valid_df):
    y_train = train_df[TARGET_COL].astype("int8")
    y_cal = cal_df[TARGET_COL].astype("int8")
    y_valid = valid_df[TARGET_COL].astype("int8")
    builder = FeatureBuilder()
    x_train = builder.fit_transform(train_df.drop(columns=[TARGET_COL]), y_train)
    x_cal = builder.transform(cal_df.drop(columns=[TARGET_COL]))
    x_valid = builder.transform(valid_df.drop(columns=[TARGET_COL]))
    return x_train, x_cal, x_valid, y_train, y_cal, y_valid


def v2_predictions(train_df, cal_df, valid_df):
    x_train, x_cal, x_valid, y_train, y_cal, y_valid = prepare_features(train_df, cal_df, valid_df)
    model = CatBoostClassifier(**V2_CATBOOST_PARAMS)
    model.fit(x_train, y_train)
    raw_cal = clip_prob(model.predict_proba(x_cal)[:, 1])
    raw_valid = clip_prob(model.predict_proba(x_valid)[:, 1])
    calibrator = fit_calibrators(raw_cal, y_cal, cal_df[["game_type"]])
    platt_valid = apply_calibration(raw_valid, valid_df[["game_type"]], calibrator, "platt")
    return raw_cal, raw_valid, platt_valid, y_cal, y_valid


def candidate_specs():
    return [
        {"candidate_name": "A_GLOBAL_PITCHER", "structure": "global -> pitcher", "kind": "pure", "context": None},
        {"candidate_name": "B_PITCHER_COUNT", "structure": "global -> pitcher -> count_state", "kind": "pure", "context": "pitcher_count"},
        {"candidate_name": "C_PITCHER_GAME", "structure": "global -> pitcher -> game_type", "kind": "pure", "context": "pitcher_game"},
        {"candidate_name": "D_PITCHER_BATTER_HAND", "structure": "global -> pitcher -> batter_hand", "kind": "pure", "context": "pitcher_batter_hand"},
        {"candidate_name": "E_MULTI_CONTEXT", "structure": "logit global + pitcher + count + game offsets", "kind": "multi", "context": None},
        {"candidate_name": "F_PITCHER_COUNT_PLUS_CATBOOST", "structure": "pitcher-count hierarchy + beta 0.5 CatBoost deviation", "kind": "hybrid", "context": "pitcher_count"},
    ]


def hierarchy_predictions(df, train_end, cal_year, valid_year, spec, v2_platt_valid=None):
    hist_for_cal = df[df["season"] <= train_end].copy()
    hist_for_valid = df[df["season"] < valid_year].copy()
    cal = df[df["season"] == cal_year].copy()
    valid = df[df["season"] == valid_year].copy()
    q_cal = future_global_rate(df, cal_year)
    q_valid = future_global_rate(df, valid_year)
    if spec["kind"] == "multi":
        cal_pred, cal_pitcher_count = predict_multi_context_logit(hist_for_cal, cal, q_cal)
        valid_pred, valid_pitcher_count = predict_multi_context_logit(hist_for_valid, valid, q_valid)
        return cal_pred, valid_pred, valid_pitcher_count
    if spec["context"] is None:
        cal_pred, cal_pitcher_count = predict_global_pitcher(hist_for_cal, cal, q_cal)
        valid_pred, valid_pitcher_count = predict_global_pitcher(hist_for_valid, valid, q_valid)
    else:
        cal_pred, cal_pitcher_count, _ = predict_pitcher_context(hist_for_cal, cal, q_cal, spec["context"])
        valid_pred, valid_pitcher_count, _ = predict_pitcher_context(hist_for_valid, valid, q_valid, spec["context"])
    if spec["kind"] == "hybrid":
        if v2_platt_valid is None:
            raise ValueError("hybrid candidate requires CatBoost validation prediction")
        residual = logit(v2_platt_valid) - logit(q_valid)
        valid_pred = clip_prob(sigmoid(logit(valid_pred) + 0.5 * residual))
    return cal_pred, valid_pred, valid_pitcher_count


def smoothing_policy_analysis(df):
    rows = []
    for alpha in ALPHA_GRID:
        for _, train_end, cal_year, valid_year in FOLDS:
            hist_for_valid = df[df["season"] < valid_year].copy()
            valid = df[df["season"] == valid_year].copy()
            q = future_global_rate(df, valid_year)
            pred, _ = predict_global_pitcher(hist_for_valid, valid, q, alpha_pitcher=alpha)
            y = valid[TARGET_COL].astype(int).to_numpy()
            rows.append({"alpha_pitcher": alpha, "valid_year": valid_year, **metric_dict(y, pred, "metric")})
    return pd.DataFrame(rows)


def effect_persistence(df):
    rows = []
    for threshold in [0, 50, 100, 300]:
        for _, train_end, _, valid_year in FOLDS:
            hist = df[df["season"] <= train_end].copy()
            valid = df[df["season"] == valid_year].copy()
            qh = float(hist[TARGET_COL].mean())
            qv = float(valid[TARGET_COL].mean())
            h = pd.DataFrame({"group": group_key(hist, "pitcher"), "y": hist[TARGET_COL].astype(float)}).groupby("group")["y"].agg(["sum", "count"])
            v = pd.DataFrame({"group": group_key(valid, "pitcher"), "y": valid[TARGET_COL].astype(float)}).groupby("group")["y"].agg(["sum", "count"])
            h["effect"] = logit((h["sum"] + ALPHA_PITCHER * qh) / (h["count"] + ALPHA_PITCHER)) - logit(qh)
            v["effect"] = logit((v["sum"] + ALPHA_PITCHER * qv) / (v["count"] + ALPHA_PITCHER)) - logit(qv)
            m = h[["effect", "count"]].join(v[["effect", "count"]], how="inner", lsuffix="_hist", rsuffix="_valid")
            m = m[m["count_hist"] >= threshold]
            rows.append(correlation_row("pitcher", threshold, valid_year, m))
    return pd.DataFrame(rows)


def context_persistence(df):
    rows = []
    for definition in ["pitcher_count", "pitcher_game", "pitcher_batter_hand", "pitcher_base", "pitcher_game_count"]:
        for threshold in [0, 50, 100, 300]:
            for _, train_end, _, valid_year in FOLDS:
                hist = df[df["season"] <= train_end].copy()
                valid = df[df["season"] == valid_year].copy()
                qh = float(hist[TARGET_COL].mean())
                qv = float(valid[TARGET_COL].mean())
                hk = group_key(hist, definition)
                vk = group_key(valid, definition)
                h = pd.DataFrame({"group": hk, "y": hist[TARGET_COL].astype(float)}).groupby("group")["y"].agg(["sum", "count"])
                v = pd.DataFrame({"group": vk, "y": valid[TARGET_COL].astype(float)}).groupby("group")["y"].agg(["sum", "count"])
                h["effect"] = logit((h["sum"] + ALPHA_CONTEXT * qh) / (h["count"] + ALPHA_CONTEXT)) - logit(qh)
                v["effect"] = logit((v["sum"] + ALPHA_CONTEXT * qv) / (v["count"] + ALPHA_CONTEXT)) - logit(qv)
                m = h[["effect", "count"]].join(v[["effect", "count"]], how="inner", lsuffix="_hist", rsuffix="_valid")
                m = m[m["count_hist"] >= threshold]
                rows.append(correlation_row(definition, threshold, valid_year, m))
    return pd.DataFrame(rows)


def correlation_row(level, threshold, valid_year, m):
    if len(m) < 3 or m["effect_hist"].nunique() < 2 or m["effect_valid"].nunique() < 2:
        pearson = np.nan
        spear = np.nan
        sign = np.nan
    else:
        pearson = float(np.corrcoef(m["effect_hist"], m["effect_valid"])[0, 1])
        spear = float(spearmanr(m["effect_hist"], m["effect_valid"]).correlation)
        sign = float((np.sign(m["effect_hist"]) == np.sign(m["effect_valid"])).mean())
    return {
        "level": level,
        "history_count_threshold": threshold,
        "valid_year": valid_year,
        "matched_group_count": int(len(m)),
        "pearson": pearson,
        "spearman": spear,
        "sign_agreement": sign,
        "median_history_count": float(m["count_hist"].median()) if len(m) else np.nan,
        "median_validation_count": float(m["count_valid"].median()) if len(m) else np.nan,
    }


def evaluate(df):
    fold_rows = []
    y2023_rows = []
    bucket_rows = []
    pred_parts = []
    runtime_rows = []
    for train_start, train_end, cal_year, valid_year in FOLDS:
        train = df[(df["season"] >= train_start) & (df["season"] <= train_end)].copy()
        cal = df[df["season"] == cal_year].copy()
        valid = df[df["season"] == valid_year].copy()
        t0 = time.time()
        _, _, v2_platt, _, y_valid = v2_predictions(train, cal, valid)
        v2_final = apply_v2_strength(v2_platt, future_global_rate(df, valid_year))
        runtime_rows.append({"candidate_name": "V2_BASELINE", "valid_year": valid_year, "seconds": time.time() - t0})
        for mode, pred in [("platt", v2_platt), ("v2_strength", v2_final)]:
            fold_rows.append(
                {
                    "candidate_name": "V2_BASELINE",
                    "hierarchy_structure": "flat CatBoost V2",
                    "smoothing_policy": "none",
                    "combination": "CatBoost",
                    "calibration_mode": mode,
                    "valid_year": valid_year,
                    **metric_dict(y_valid, pred, "metric"),
                }
            )
        pred_parts.append(
            pd.DataFrame(
                {
                    "row_id": valid["row_id"].to_numpy(),
                    "valid_year": valid_year,
                    "target": y_valid.to_numpy(),
                    "candidate_name": "V2_BASELINE",
                    "native_pred": v2_platt,
                    "platt_pred": v2_platt,
                    "v2_strength_pred": v2_final,
                    "hist_pitcher_count": valid["asof_pitcher_n"].fillna(0).to_numpy(),
                }
            )
        )
        v2_buckets = pd.cut(valid["asof_pitcher_n"].fillna(0).to_numpy(), bins=[-1, 50, 300, 1000, np.inf], labels=["lt50", "50_300", "300_1000", "1000_plus"])
        for mode_name, pred in [("platt", v2_platt), ("v2_strength", v2_final)]:
            tmp = pd.DataFrame({"bucket": v2_buckets, "y": y_valid.to_numpy(), "pred": pred})
            for b, g in tmp.groupby("bucket", observed=True):
                if len(g) < 1000:
                    continue
                bucket_rows.append(
                    {
                        "candidate_name": "V2_BASELINE",
                        "calibration_mode": mode_name,
                        "valid_year": valid_year,
                        "pitcher_sample_bucket": str(b),
                        "n": len(g),
                        "actual_rate": float(g["y"].mean()),
                        "brier": float(brier_score_loss(g["y"], g["pred"])),
                        "auc": float(roc_auc_score(g["y"], g["pred"])) if g["y"].nunique() == 2 else np.nan,
                        "prediction_std": float(g["pred"].std()),
                    }
                )
        for spec in candidate_specs():
            t1 = time.time()
            cal_pred, native_valid, hist_pitcher_count = hierarchy_predictions(df, train_end, cal_year, valid_year, spec, v2_platt_valid=v2_platt)
            calibrator = fit_calibrators(cal_pred, cal[TARGET_COL].astype(int), cal[["game_type"]])
            platt_valid = apply_calibration(native_valid, valid[["game_type"]], calibrator, "platt")
            strength_valid = apply_v2_strength(platt_valid, future_global_rate(df, valid_year))
            runtime_rows.append({"candidate_name": spec["candidate_name"], "valid_year": valid_year, "seconds": time.time() - t1})
            for mode, pred in [("native", native_valid), ("platt", platt_valid), ("v2_strength", strength_valid)]:
                fold_rows.append(
                    {
                        "candidate_name": spec["candidate_name"],
                        "hierarchy_structure": spec["structure"],
                        "smoothing_policy": f"alpha_pitcher={ALPHA_PITCHER}; alpha_context={ALPHA_CONTEXT}",
                        "combination": spec["kind"],
                        "calibration_mode": mode,
                        "valid_year": valid_year,
                        **metric_dict(y_valid, pred, "metric"),
                    }
                )
            if valid_year == 2023:
                for mode, pred in [("native", native_valid), ("platt", platt_valid), ("v2_strength", strength_valid)]:
                    med = np.median(pred)
                    low = pred < med
                    high = ~low
                    y = y_valid.to_numpy()
                    y2023_rows.append(
                        {
                            "candidate_name": spec["candidate_name"],
                            "calibration_mode": mode,
                            "auc": float(roc_auc_score(y, pred)),
                            "brier": float(brier_score_loss(y, pred)),
                            "skill_margin": float(y.mean() * (1 - y.mean()) - brier_score_loss(y, pred)),
                            "pseudo_score": float(max(0.0, 100000.0 * (y.mean() * (1 - y.mean()) - brier_score_loss(y, pred)) / (y.mean() * (1 - y.mean())))),
                            "prediction_std": float(pred.std()),
                            "low_half_actual_rate": float(y[low].mean()),
                            "high_half_actual_rate": float(y[high].mean()),
                            "high_minus_low_actual_rate": float(y[high].mean() - y[low].mean()),
                        }
                    )
            buckets = pd.cut(hist_pitcher_count, bins=[-1, 50, 300, 1000, np.inf], labels=["lt50", "50_300", "300_1000", "1000_plus"])
            for mode_name, pred in [("native", native_valid), ("platt", platt_valid), ("v2_strength", strength_valid)]:
                tmp = pd.DataFrame({"bucket": buckets, "y": y_valid.to_numpy(), "pred": pred})
                for b, g in tmp.groupby("bucket", observed=True):
                    if len(g) < 1000:
                        continue
                    bucket_rows.append(
                        {
                            "candidate_name": spec["candidate_name"],
                            "calibration_mode": mode_name,
                            "valid_year": valid_year,
                            "pitcher_sample_bucket": str(b),
                            "n": len(g),
                            "actual_rate": float(g["y"].mean()),
                            "brier": float(brier_score_loss(g["y"], g["pred"])),
                            "auc": float(roc_auc_score(g["y"], g["pred"])) if g["y"].nunique() == 2 else np.nan,
                            "prediction_std": float(g["pred"].std()),
                        }
                    )
            pred_parts.append(
                pd.DataFrame(
                    {
                        "row_id": valid["row_id"].to_numpy(),
                        "valid_year": valid_year,
                        "target": y_valid.to_numpy(),
                        "candidate_name": spec["candidate_name"],
                        "native_pred": native_valid,
                        "platt_pred": platt_valid,
                        "v2_strength_pred": strength_valid,
                        "hist_pitcher_count": hist_pitcher_count,
                    }
                )
            )
            print(f"{spec['candidate_name']} valid={valid_year} auc={roc_auc_score(y_valid, strength_valid):.6f}")
    return pd.DataFrame(fold_rows), pd.DataFrame(y2023_rows), pd.DataFrame(bucket_rows), pd.concat(pred_parts, ignore_index=True), pd.DataFrame(runtime_rows)


def summarize(fold_metrics, bucket_analysis):
    rows = []
    base = fold_metrics[(fold_metrics["candidate_name"] == "V2_BASELINE") & (fold_metrics["calibration_mode"] == "v2_strength")].set_index("valid_year")
    base_buckets = bucket_analysis[bucket_analysis["calibration_mode"] == "v2_strength"]
    for (candidate, mode), g in fold_metrics.groupby(["candidate_name", "calibration_mode"], sort=False):
        by = g.set_index("valid_year")
        row = {
            "candidate_name": candidate,
            "calibration_mode": mode,
            "hierarchy_structure": g["hierarchy_structure"].iloc[0],
            "smoothing_policy": g["smoothing_policy"].iloc[0],
            "model_combination": g["combination"].iloc[0],
            "mean_auc": float(g["metric_auc"].mean()),
            "mean_brier": float(g["metric_brier"].mean()),
            "worst_skill_margin": float(g["metric_skill_margin"].min()),
            "positive_fold_count": int((g["metric_skill_margin"] > 0).sum()),
        }
        for year in [2022, 2023, 2024]:
            row[f"auc_{year}"] = float(by.loc[year, "metric_auc"])
            row[f"pseudo_{year}"] = float(by.loc[year, "metric_pseudo_score"])
            row[f"skill_{year}"] = float(by.loc[year, "metric_skill_margin"])
            row[f"delta_auc_{year}_vs_v2"] = float(by.loc[year, "metric_auc"] - base.loc[year, "metric_auc"])
            row[f"delta_skill_{year}_vs_v2"] = float(by.loc[year, "metric_skill_margin"] - base.loc[year, "metric_skill_margin"])
        row["delta_mean_auc_vs_v2"] = row["mean_auc"] - float(base["metric_auc"].mean())
        cand_b = bucket_analysis[(bucket_analysis["candidate_name"] == candidate) & (bucket_analysis["calibration_mode"] == mode)]
        improve_buckets = []
        for _, r in cand_b.iterrows():
            b0 = base_buckets[
                (base_buckets["candidate_name"] == "V2_BASELINE")
                & (base_buckets["valid_year"] == r["valid_year"])
                & (base_buckets["pitcher_sample_bucket"] == r["pitcher_sample_bucket"])
            ]
            if not b0.empty:
                improve_buckets.append(float(r["brier"] < b0["brier"].iloc[0]))
        row["bucket_improvement_rate_vs_v2"] = float(np.mean(improve_buckets)) if improve_buckets else np.nan
        row["success_a"] = row["mean_auc"] >= 0.523
        row["success_b"] = row["skill_2023"] >= float(base.loc[2023, "metric_skill_margin"]) + 0.0001 and row["pseudo_2022"] > 0 and row["pseudo_2024"] > 0
        row["success_c"] = row["pseudo_2024"] >= 25 and row["pseudo_2022"] > 0 and row["skill_2023"] >= float(base.loc[2023, "metric_skill_margin"])
        row["success_d"] = all(row[f"delta_skill_{y}_vs_v2"] > 0 for y in [2022, 2023, 2024])
        row["success_e"] = row["bucket_improvement_rate_vs_v2"] >= 0.75 and row["positive_fold_count"] >= 2
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["mean_auc", "pseudo_2024", "skill_2023"], ascending=[False, False, False])


def prediction_correlation(preds, top_candidates):
    base = preds[preds["candidate_name"] == "V2_BASELINE"][["row_id", "valid_year", "target", "v2_strength_pred"]].rename(columns={"v2_strength_pred": "v2_pred"})
    rows = []
    for candidate in top_candidates:
        if candidate == "V2_BASELINE":
            continue
        cand = preds[preds["candidate_name"] == candidate][["row_id", "valid_year", "target", "v2_strength_pred"]].rename(columns={"v2_strength_pred": "cand_pred"})
        m = base.merge(cand, on=["row_id", "valid_year", "target"])
        for scope, g in [("overall", m), *[(str(y), gy) for y, gy in m.groupby("valid_year")]]:
            ev = (g["v2_pred"] - g["target"]) ** 2
            ec = (g["cand_pred"] - g["target"]) ** 2
            rows.append(
                {
                    "candidate_name": candidate,
                    "scope": scope,
                    "pearson_corr": float(np.corrcoef(g["v2_pred"], g["cand_pred"])[0, 1]),
                    "spearman_corr": float(spearmanr(g["v2_pred"], g["cand_pred"]).correlation),
                    "mean_abs_prediction_diff": float(np.mean(np.abs(g["v2_pred"] - g["cand_pred"]))),
                    "squared_error_corr": float(np.corrcoef(ev, ec)[0, 1]),
                }
            )
    return pd.DataFrame(rows)


def error_complementarity(preds, top_candidates):
    base = preds[preds["candidate_name"] == "V2_BASELINE"][["row_id", "valid_year", "target", "v2_strength_pred"]].rename(columns={"v2_strength_pred": "v2_pred"})
    rows = []
    for candidate in top_candidates:
        if candidate == "V2_BASELINE":
            continue
        cand = preds[preds["candidate_name"] == candidate][["row_id", "valid_year", "target", "v2_strength_pred", "hist_pitcher_count"]].rename(columns={"v2_strength_pred": "cand_pred"})
        m = base.merge(cand, on=["row_id", "valid_year", "target"])
        m["sample_bucket"] = pd.cut(m["hist_pitcher_count"], bins=[-1, 50, 300, 1000, np.inf], labels=["lt50", "50_300", "300_1000", "1000_plus"])
        scopes = [("overall", m)] + [(str(y), gy) for y, gy in m.groupby("valid_year")] + [(f"bucket_{b}", gb) for b, gb in m.groupby("sample_bucket", observed=True)]
        for scope, g in scopes:
            if len(g) == 0:
                continue
            ev = (g["v2_pred"] - g["target"]) ** 2
            ec = (g["cand_pred"] - g["target"]) ** 2
            rows.append(
                {
                    "candidate_name": candidate,
                    "scope": scope,
                    "n": int(len(g)),
                    "hierarchy_better_rate": float((ec < ev).mean()),
                    "v2_better_rate": float((ev < ec).mean()),
                    "tie_rate": float(np.isclose(ec, ev).mean()),
                    "both_poor_top_decile_rate": float(((ec >= np.quantile(ec, 0.9)) & (ev >= np.quantile(ev, 0.9))).mean()),
                }
            )
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = add_context_columns(pd.read_csv(TRAIN_PATH, encoding="utf-8-sig"))
    feasibility = hierarchy_feasibility(df)
    smoothing = smoothing_policy_analysis(df)
    pitcher_persistence = effect_persistence(df)
    context_persist = context_persistence(df)
    fold_metrics, y2023, buckets, preds, runtime = evaluate(df)
    summary = summarize(fold_metrics, buckets)
    top_candidates = summary["candidate_name"].drop_duplicates().head(3).tolist()
    corr = prediction_correlation(preds, top_candidates)
    err = error_complementarity(preds, top_candidates)
    verdict = "HIERARCHICAL SIGNAL FOUND" if summary[["success_a", "success_b", "success_c", "success_d", "success_e"]].any(axis=None) else "HIERARCHICAL FORMULATION NOT USEFUL"
    feasibility.to_csv(os.path.join(OUT_DIR, "hierarchy_feasibility.csv"), index=False, encoding="utf-8")
    pitcher_persistence.to_csv(os.path.join(OUT_DIR, "effect_persistence.csv"), index=False, encoding="utf-8")
    context_persist.to_csv(os.path.join(OUT_DIR, "context_persistence.csv"), index=False, encoding="utf-8")
    fold_metrics.to_csv(os.path.join(OUT_DIR, "fold_metrics.csv"), index=False, encoding="utf-8")
    summary.to_csv(os.path.join(OUT_DIR, "candidate_summary.csv"), index=False, encoding="utf-8")
    buckets.to_csv(os.path.join(OUT_DIR, "sample_size_bucket_analysis.csv"), index=False, encoding="utf-8")
    y2023.to_csv(os.path.join(OUT_DIR, "year2023_analysis.csv"), index=False, encoding="utf-8")
    corr.to_csv(os.path.join(OUT_DIR, "prediction_correlation.csv"), index=False, encoding="utf-8")
    err.to_csv(os.path.join(OUT_DIR, "error_complementarity.csv"), index=False, encoding="utf-8")
    smoothing.to_csv(os.path.join(OUT_DIR, "smoothing_policy_analysis.csv"), index=False, encoding="utf-8")
    runtime.to_csv(os.path.join(OUT_DIR, "runtime_summary.csv"), index=False, encoding="utf-8")
    pd.DataFrame([{"verdict": verdict}]).to_csv(os.path.join(OUT_DIR, "verdict.csv"), index=False, encoding="utf-8")
    preds.to_csv(os.path.join(OUT_DIR, "oof_predictions.csv"), index=False, encoding="utf-8")
    print(summary.head(12).to_string(index=False))
    print(verdict)


if __name__ == "__main__":
    main()
