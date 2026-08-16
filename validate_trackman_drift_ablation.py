import os
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

import build_trackman_mechanical_drift_foundation as drift
from calibration_utils import apply_calibration, fit_calibrators
from model_utils import FeatureBuilder, TARGET_COL
from validate_v3_asof_signal_reconstruction import (
    V2_CATBOOST_PARAMS,
    apply_v2_strength,
    clip_prob,
    target_rate_for_year,
)


OUT_DIR = "output/v3_trackman_drift_ablation"
TRAIN_PATH = "data/train.csv"
FOLDS = [
    (2019, 2020, 2021, 2022),
    (2019, 2021, 2022, 2023),
    (2019, 2022, 2023, 2024),
]
PHYSICAL = ["rel_height", "rel_side", "extension", "rel_speed", "spin_rate", "induced_vert_break", "horz_break"]
RELEASE = ["rel_height", "rel_side", "extension"]
MOVEMENT = ["rel_speed", "spin_rate", "induced_vert_break", "horz_break"]


def metric_dict(y, pred, prefix):
    pred = clip_prob(pred)
    actual_rate = float(np.mean(y))
    constant_brier = float(actual_rate * (1.0 - actual_rate))
    model_brier = float(brier_score_loss(y, pred))
    skill_margin = constant_brier - model_brier
    out = {
        f"{prefix}_actual_rate": actual_rate,
        f"{prefix}_constant_brier": constant_brier,
        f"{prefix}_brier": model_brier,
        f"{prefix}_skill_margin": float(skill_margin),
        f"{prefix}_pseudo_score": float(max(0.0, 100000.0 * skill_margin / constant_brier)),
        f"{prefix}_auc": float(roc_auc_score(y, pred)) if len(np.unique(y)) == 2 else np.nan,
        f"{prefix}_logloss": float(log_loss(y, pred)),
        f"{prefix}_pred_std": float(pred.std()),
        f"{prefix}_pred_mean": float(pred.mean()),
    }
    return out


def feature_families():
    level = [f"tm_{c}_longterm_mean" for c in PHYSICAL]
    release_level = [f"tm_{c}_longterm_mean" for c in RELEASE]
    recent_delta = [f"tm_{c}_recent2_minus_longterm" for c in PHYSICAL] + [
        f"tm_{c}_recent1_minus_longterm" for c in RELEASE
    ]
    slope = [f"tm_{c}_slope" for c in PHYSICAL]
    league = [f"tm_{c}_drift_relative_league" for c in PHYSICAL]
    adjusted = [f"tm_{c}_pitch_type_adjusted_drift" for c in PHYSICAL]
    reliability = [
        "tm_has_match",
        "tm_history_pitch_count",
        "tm_history_game_count",
        "tm_history_season_count",
        "tm_recent1_pitch_count",
        "tm_recent2_pitch_count",
        "tm_recent1_game_count",
        "tm_drift_timepoints",
        "tm_drift_reliability",
        "tm_mapping_score",
    ]
    release_core = []
    for c in RELEASE:
        release_core += [
            f"tm_{c}_longterm_mean",
            f"tm_{c}_recent2_minus_longterm",
            f"tm_{c}_slope",
            f"tm_{c}_drift_relative_league",
        ]
    movement_core = []
    for c in MOVEMENT:
        movement_core += [
            f"tm_{c}_longterm_mean",
            f"tm_{c}_recent2_minus_longterm",
            f"tm_{c}_slope",
            f"tm_{c}_drift_relative_league",
        ]
    return [
        ("BASE", []),
        ("A_LEVEL", level),
        ("B_RECENT_DELTA", recent_delta),
        ("C_SLOPE", slope),
        ("D_LEAGUE_RELATIVE", league),
        ("E_PITCHTYPE_ADJUSTED", adjusted),
        ("F_RELIABILITY", reliability),
        ("G_RELEASE_CORE", release_core),
        ("H_MOVEMENT_CORE", movement_core),
        ("I_RELEASE_PLUS_RELIABILITY", release_core + reliability),
        ("K_STABLE_TOP10", [
            "tm_rel_height_longterm_mean",
            "tm_extension_longterm_mean",
            "tm_rel_height_recent2_minus_longterm",
            "tm_rel_side_recent2_minus_longterm",
            "tm_extension_recent2_minus_longterm",
            "tm_rel_height_slope",
            "tm_rel_height_recent2_mean",
            "tm_extension_slope",
            "tm_extension_recent2_mean",
            "tm_rel_side_slope",
            *reliability,
        ]),
    ]


def load_foundation_all_cutoffs():
    tm, train_meta = drift.load_data()
    old = drift.CUTOFFS
    try:
        drift.CUTOFFS = [2020, 2021, 2022, 2023, 2024, 2025]
        _, maps = drift.mapping_quality_by_cutoff(train_meta, tm)
        foundation, _ = drift.drift_foundation_tables(tm, maps)
    finally:
        drift.CUTOFFS = old
    return foundation


def attach_trackman_by_row_year(df, foundation, family_cols, train_fill_values=None):
    out = df.copy()
    if not family_cols:
        out["tm_has_match"] = 0.0
        return out, {}
    need_cols = sorted(set(["cutoff_year", "pitcher_id", "pitcher_trackman_id", "source_max_season", "source_max_date", "tm_mapping_score", *family_cols]))
    source = foundation[[c for c in need_cols if c in foundation.columns]].copy()
    out["cutoff_year"] = out["season"].astype(int)
    out = out.merge(source, on=["cutoff_year", "pitcher_id"], how="left")
    out["tm_has_match"] = out["pitcher_trackman_id"].notna().astype("float32")
    if "tm_has_match" in family_cols and "tm_has_match" not in out.columns:
        out["tm_has_match"] = out["pitcher_trackman_id"].notna().astype("float32")
    numeric_cols = [c for c in family_cols if c in out.columns]
    fill_values = {} if train_fill_values is None else dict(train_fill_values)
    if train_fill_values is None:
        for c in numeric_cols:
            if c == "tm_has_match":
                fill_values[c] = 0.0
            elif any(token in c for token in ["minus", "slope", "drift", "relative", "abs_drift"]):
                fill_values[c] = 0.0
            elif any(token in c for token in ["count", "timepoints", "reliability", "score"]):
                fill_values[c] = 0.0
            else:
                fill_values[c] = float(out.loc[out["tm_has_match"] == 1, c].median()) if out[c].notna().any() else 0.0
    for c in numeric_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(fill_values.get(c, 0.0))
    return out, fill_values


def prepare_split(df, foundation, train_start, train_end, cal_year, valid_year, family_cols):
    train = df[(df["season"] >= train_start) & (df["season"] <= train_end)].copy()
    cal = df[df["season"] == cal_year].copy()
    valid = df[df["season"] == valid_year].copy()
    train, fills = attach_trackman_by_row_year(train, foundation, family_cols)
    cal, _ = attach_trackman_by_row_year(cal, foundation, family_cols, fills)
    valid, _ = attach_trackman_by_row_year(valid, foundation, family_cols, fills)
    return train, cal, valid


def evaluate_candidate(df, foundation, candidate_name, family_cols):
    fold_rows = []
    matched_rows = []
    reliability_rows = []
    fi_rows = []
    pred_parts = []
    year_rates = df.groupby("season")[TARGET_COL].mean().sort_index()
    for train_start, train_end, cal_year, valid_year in FOLDS:
        train, cal, valid = prepare_split(df, foundation, train_start, train_end, cal_year, valid_year, family_cols)
        valid_meta, _ = attach_trackman_by_row_year(
            df[df["season"] == valid_year].copy(),
            foundation,
            ["tm_drift_reliability", "tm_history_pitch_count", "tm_history_game_count", "tm_history_season_count"],
        )
        y_train = train[TARGET_COL].astype("int8")
        y_cal = cal[TARGET_COL].astype("int8")
        y_valid = valid[TARGET_COL].astype("int8")
        builder = FeatureBuilder()
        X_train = builder.fit_transform(train.drop(columns=[TARGET_COL]), y_train)
        X_cal = builder.transform(cal.drop(columns=[TARGET_COL]))
        X_valid = builder.transform(valid.drop(columns=[TARGET_COL]))
        model = CatBoostClassifier(**V2_CATBOOST_PARAMS)
        t0 = time.time()
        model.fit(X_train, y_train)
        train_seconds = time.time() - t0
        raw_cal = clip_prob(model.predict_proba(X_cal)[:, 1])
        raw_valid = clip_prob(model.predict_proba(X_valid)[:, 1])
        calibration = fit_calibrators(raw_cal, y_cal, cal[["game_type"]])
        platt_valid = apply_calibration(raw_valid, valid[["game_type"]], calibration, "platt")
        final_valid = apply_v2_strength(platt_valid, target_rate_for_year(year_rates, valid_year))
        row = {
            "candidate_name": candidate_name,
            "valid_year": valid_year,
            "feature_count": len(family_cols),
            "tm_match_row_rate": float(valid.get("tm_has_match", pd.Series(0, index=valid.index)).mean()),
            "train_seconds": train_seconds,
            **metric_dict(y_valid, raw_valid, "raw"),
            **metric_dict(y_valid, final_valid, "final"),
        }
        fold_rows.append(row)
        tmp_pred = pd.DataFrame({
            "row_id": valid["row_id"].to_numpy(),
            "valid_year": valid_year,
            "target": y_valid.to_numpy(),
            "tm_has_match": valid_meta.get("tm_has_match", pd.Series(0, index=valid_meta.index)).to_numpy(),
            "tm_drift_reliability": valid_meta.get("tm_drift_reliability", pd.Series(0, index=valid_meta.index)).to_numpy(),
            "pred_raw": raw_valid,
            "pred_final": final_valid,
        })
        tmp_pred["candidate_name"] = candidate_name
        pred_parts.append(tmp_pred)
        for has_match, gidx in tmp_pred.groupby("tm_has_match").groups.items():
            idx = np.asarray(list(gidx), dtype=np.int64)
            if len(idx) < 2:
                continue
            y = tmp_pred.iloc[idx]["target"].to_numpy()
            p = tmp_pred.iloc[idx]["pred_final"].to_numpy()
            matched_rows.append({
                "candidate_name": candidate_name,
                "valid_year": valid_year,
                "tm_has_match": int(has_match),
                "n": len(idx),
                "actual_rate": float(y.mean()),
                "brier": float(brier_score_loss(y, p)),
                "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan,
                "prediction_std": float(p.std()),
            })
        matched_valid = tmp_pred[tmp_pred["tm_has_match"] == 1].copy()
        if len(matched_valid) >= 100 and "tm_drift_reliability" in matched_valid:
            med = matched_valid["tm_drift_reliability"].median()
            for bucket, bg in matched_valid.groupby(np.where(matched_valid["tm_drift_reliability"] >= med, "high", "low")):
                y = bg["target"].to_numpy()
                p = bg["pred_final"].to_numpy()
                reliability_rows.append({
                    "candidate_name": candidate_name,
                    "valid_year": valid_year,
                    "bucket": bucket,
                    "n": len(bg),
                    "median_cut": float(med),
                    "actual_rate": float(y.mean()),
                    "brier": float(brier_score_loss(y, p)),
                    "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan,
                    "prediction_std": float(p.std()),
                })
        if candidate_name != "BASE":
            importances = model.get_feature_importance()
            for rank, idx in enumerate(np.argsort(importances)[::-1][:80], start=1):
                feature = X_train.columns[idx]
                if feature.startswith("tm_") or rank <= 30:
                    fi_rows.append({
                        "candidate_name": candidate_name,
                        "valid_year": valid_year,
                        "rank": rank,
                        "feature": feature,
                        "importance": float(importances[idx]),
                        "is_trackman": feature.startswith("tm_"),
                    })
    return pd.DataFrame(fold_rows), pd.DataFrame(matched_rows), pd.DataFrame(reliability_rows), pd.DataFrame(fi_rows), pd.concat(pred_parts, ignore_index=True)


def summarize(fold_metrics):
    base = fold_metrics[fold_metrics["candidate_name"] == "BASE"].set_index("valid_year")
    rows = []
    for name, g in fold_metrics.groupby("candidate_name", sort=False):
        by = g.set_index("valid_year")
        row = {
            "candidate_name": name,
            "feature_count": int(g["feature_count"].iloc[0]),
            "mean_auc": float(g["raw_auc"].mean()),
            "mean_brier": float(g["final_brier"].mean()),
            "mean_skill_margin": float(g["final_skill_margin"].mean()),
            "worst_skill_margin": float(g["final_skill_margin"].min()),
            "mean_tm_match_row_rate": float(g["tm_match_row_rate"].mean()),
        }
        for year in [2022, 2023, 2024]:
            row[f"auc_{year}"] = float(by.loc[year, "raw_auc"])
            row[f"pseudo_{year}"] = float(by.loc[year, "final_pseudo_score"])
            row[f"skill_{year}"] = float(by.loc[year, "final_skill_margin"])
            row[f"brier_{year}"] = float(by.loc[year, "final_brier"])
            row[f"delta_auc_{year}"] = float(by.loc[year, "raw_auc"] - base.loc[year, "raw_auc"])
            row[f"delta_skill_{year}"] = float(by.loc[year, "final_skill_margin"] - base.loc[year, "final_skill_margin"])
            row[f"delta_pseudo_{year}"] = float(by.loc[year, "final_pseudo_score"] - base.loc[year, "final_pseudo_score"])
        row["delta_mean_auc"] = row["mean_auc"] - float(base["raw_auc"].mean())
        row["delta_mean_brier"] = row["mean_brier"] - float(base["final_brier"].mean())
        row["success_a"] = row["mean_auc"] >= 0.523
        row["success_b"] = row["pseudo_2024"] >= 25 and row["pseudo_2022"] > 0 and row["skill_2023"] >= float(base.loc[2023, "final_skill_margin"])
        row["success_c"] = row["skill_2023"] >= float(base.loc[2023, "final_skill_margin"]) + 0.00005 and row["pseudo_2022"] > 0 and row["pseudo_2024"] > 0
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["mean_auc", "pseudo_2024", "skill_2023"], ascending=[False, False, False])


def prediction_correlation(preds, top_names):
    base = preds[preds["candidate_name"] == "BASE"].copy()
    rows = []
    for name in top_names:
        if name == "BASE":
            continue
        cand = preds[preds["candidate_name"] == name].copy()
        merged = base.merge(cand, on=["row_id", "valid_year", "target"], suffixes=("_base", "_cand"))
        for scope, g in [("overall", merged), *[(str(y), gy) for y, gy in merged.groupby("valid_year")]]:
            err_b = (g["pred_final_base"] - g["target"]) ** 2
            err_c = (g["pred_final_cand"] - g["target"]) ** 2
            rows.append({
                "candidate_name": name,
                "scope": scope,
                "pearson_corr": float(np.corrcoef(g["pred_raw_base"], g["pred_raw_cand"])[0, 1]),
                "final_pearson_corr": float(np.corrcoef(g["pred_final_base"], g["pred_final_cand"])[0, 1]),
                "squared_error_corr": float(np.corrcoef(err_b, err_c)[0, 1]),
                "mean_abs_prediction_diff": float(np.mean(np.abs(g["pred_final_base"] - g["pred_final_cand"]))),
            })
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig")
    foundation = load_foundation_all_cutoffs()
    leakage_rows = []
    for year in [2022, 2023, 2024]:
        f = foundation[foundation["cutoff_year"] == year]
        leakage_rows.append({
            "valid_year": year,
            "status": "PASS" if int(f["source_max_season"].max()) <= year - 1 else "FAIL",
            "max_source_season": int(f["source_max_season"].max()),
            "allowed_max_source_season": year - 1,
        })
    pd.DataFrame(leakage_rows).to_csv(os.path.join(OUT_DIR, "leakage_assertion_results.csv"), index=False, encoding="utf-8")

    all_fold = []
    all_matched = []
    all_rel = []
    all_fi = []
    all_preds = []
    for name, cols in feature_families():
        fm, ms, rel, fi, preds = evaluate_candidate(df, foundation, name, cols)
        all_fold.append(fm)
        all_matched.append(ms)
        all_rel.append(rel)
        all_fi.append(fi)
        all_preds.append(preds)
        print(f"{name}: mean_auc={fm['raw_auc'].mean():.6f} pseudo2024={fm[fm.valid_year==2024]['final_pseudo_score'].iloc[0]:.3f}")
    fold_metrics = pd.concat(all_fold, ignore_index=True)
    summary = summarize(fold_metrics)

    # Limited best combination only if any non-base family shows positive mean AUC delta.
    improving = summary[(summary["candidate_name"] != "BASE") & (summary["delta_mean_auc"] > 0)].head(3)
    if not improving.empty:
        best_cols = []
        fam_map = dict(feature_families())
        for name in improving["candidate_name"]:
            best_cols.extend(fam_map[name])
        fm, ms, rel, fi, preds = evaluate_candidate(df, foundation, "J_BEST_COMBINATION", sorted(set(best_cols)))
        fold_metrics = pd.concat([fold_metrics, fm], ignore_index=True)
        all_matched.append(ms)
        all_rel.append(rel)
        all_fi.append(fi)
        all_preds.append(preds)
        summary = summarize(fold_metrics)

    top_names = summary.head(4)["candidate_name"].tolist()
    preds_all = pd.concat(all_preds, ignore_index=True)
    corr = prediction_correlation(preds_all, top_names)
    fi_all = pd.concat([x for x in all_fi if not x.empty], ignore_index=True)
    top3 = [n for n in summary[summary["candidate_name"] != "BASE"].head(3)["candidate_name"]]
    fi_top = fi_all[fi_all["candidate_name"].isin(top3)].copy()
    family_rows = [{"candidate_name": name, "features": ";".join(cols), "feature_count": len(cols)} for name, cols in feature_families()]
    if "J_BEST_COMBINATION" in set(summary["candidate_name"]):
        family_rows.append({"candidate_name": "J_BEST_COMBINATION", "features": "union of improving families", "feature_count": int(summary[summary.candidate_name == "J_BEST_COMBINATION"]["feature_count"].iloc[0])})

    matched = pd.concat([x for x in all_matched if not x.empty], ignore_index=True)
    rel = pd.concat([x for x in all_rel if not x.empty], ignore_index=True)
    summary["success_d_subset_signal"] = False
    if not matched.empty and "BASE" in set(matched["candidate_name"]):
        base_matched = matched[matched["candidate_name"] == "BASE"][["valid_year", "tm_has_match", "auc", "brier"]].rename(
            columns={"auc": "base_auc", "brier": "base_brier"}
        )
        matched_delta = matched.merge(base_matched, on=["valid_year", "tm_has_match"], how="left")
        matched_delta["delta_auc_vs_base"] = matched_delta["auc"] - matched_delta["base_auc"]
        matched_delta["delta_brier_vs_base"] = matched_delta["brier"] - matched_delta["base_brier"]
        matched_delta.to_csv(os.path.join(OUT_DIR, "matched_subset_delta.csv"), index=False, encoding="utf-8")
        for idx, row in summary.iterrows():
            if row["candidate_name"] == "BASE":
                continue
            mg = matched_delta[(matched_delta["candidate_name"] == row["candidate_name"]) & (matched_delta["tm_has_match"] == 1)]
            if len(mg) == 3 and (mg["delta_auc_vs_base"] > 0).all() and (mg["delta_brier_vs_base"] < 0).all():
                summary.loc[idx, "success_d_subset_signal"] = True
    verdict = "TRACKMAN MECHANICAL DRIFT NOT USEFUL"
    if summary[["success_a", "success_b", "success_c"]].any(axis=None):
        verdict = "TRACKMAN DRIFT SIGNAL FOUND"
    elif bool(summary["success_d_subset_signal"].any()):
        verdict = "TRACKMAN SIGNAL LIMITED TO MATCHED SUBSET"
    pd.DataFrame([{"verdict": verdict}]).to_csv(os.path.join(OUT_DIR, "verdict.csv"), index=False, encoding="utf-8")
    pd.DataFrame(family_rows).to_csv(os.path.join(OUT_DIR, "feature_family_metrics.csv"), index=False, encoding="utf-8")
    fold_metrics.to_csv(os.path.join(OUT_DIR, "fold_metrics.csv"), index=False, encoding="utf-8")
    summary.to_csv(os.path.join(OUT_DIR, "candidate_summary.csv"), index=False, encoding="utf-8")
    matched.to_csv(os.path.join(OUT_DIR, "matched_subset_analysis.csv"), index=False, encoding="utf-8")
    rel.to_csv(os.path.join(OUT_DIR, "reliability_bucket_analysis.csv"), index=False, encoding="utf-8")
    fi_top.to_csv(os.path.join(OUT_DIR, "feature_importance.csv"), index=False, encoding="utf-8")
    corr.to_csv(os.path.join(OUT_DIR, "prediction_correlation.csv"), index=False, encoding="utf-8")
    preds_all.to_csv(os.path.join(OUT_DIR, "oof_predictions.csv"), index=False, encoding="utf-8")
    print(summary.head(12).to_string(index=False))
    print(verdict)


if __name__ == "__main__":
    main()
