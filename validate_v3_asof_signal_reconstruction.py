import os
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from calibration_utils import apply_calibration, fit_calibrators
from model_utils import FeatureBuilder, TARGET_COL


OUT_DIR = "output/v3_asof_signal_reconstruction"
TRAIN_PATH = "data/train.csv"
EPS = 1e-6
TEMPERATURE = 2.3
HARD_CAP = 0.020
FOLDS = [
    (2019, 2020, 2021, 2022),
    (2019, 2021, 2022, 2023),
    (2019, 2022, 2023, 2024),
]
V2_CATBOOST_PARAMS = {
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
TUNED_CATBOOST_PARAMS = {
    "loss_function": "Logloss",
    "iterations": 450,
    "learning_rate": 0.03,
    "depth": 4,
    "l2_leaf_reg": 10.0,
    "random_seed": 42,
    "verbose": False,
    "allow_writing_files": False,
    "thread_count": -1,
    "random_strength": 1.0,
    "bagging_temperature": 1.0,
    "rsm": 0.85,
}
PITCHER_RECENT_COLS = [
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
]
ASOF_COLS = [
    "asof_pitcher_n",
    "asof_pitcher_success_rate",
    "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
    *PITCHER_RECENT_COLS,
    "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate",
    "asof_batter_n",
    "asof_batter_success_rate",
    "asof_batter_middle_rate",
    "asof_pitcher_pitchmix_n",
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
]


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


def add_asof_reconstruction(df, families):
    out = df.copy()
    has = set(out.columns)
    def col(name):
        return pd.to_numeric(out[name], errors="coerce") if name in has else pd.Series(np.nan, index=out.index)

    career = col("asof_pitcher_success_rate")
    prev1 = col("asof_pitcher_prev1_game_success_rate")
    prev3 = col("asof_pitcher_prev3_game_success_rate")
    prev5 = col("asof_pitcher_prev5_game_success_rate")
    batter = col("asof_batter_success_rate")
    reverse = col("asof_pitcher_reverse_rate")
    middle = col("asof_pitcher_middle_rate")
    fastball = col("asof_pitcher_fastball_rate")
    breaking = col("asof_pitcher_breaking_rate")
    offspeed = col("asof_pitcher_offspeed_rate")
    balls = col("balls_before")
    strikes = col("strikes_before")
    inning = col("inning")
    li = col("li")
    recent = pd.concat([prev1, prev3, prev5], axis=1)

    if "A" in families:
        out["rf_prev1_minus_career"] = prev1 - career
        out["rf_prev3_minus_career"] = prev3 - career
        out["rf_prev5_minus_career"] = prev5 - career
        out["rf_prev1_minus_prev5"] = prev1 - prev5
        out["rf_prev3_minus_prev5"] = prev3 - prev5
        out["rf_recent_mean_minus_career"] = recent.mean(axis=1) - career
    if "B" in families:
        out["rf_recent_range"] = recent.max(axis=1) - recent.min(axis=1)
        out["rf_recent_std"] = recent.std(axis=1)
        out["rf_short_vs_medium"] = prev1 - prev3
        out["rf_medium_vs_long"] = prev3 - prev5
        out["rf_recent_trend"] = 0.5 * prev1 + 0.3 * prev3 + 0.2 * prev5 - career
    if "C" in families:
        out["rf_success_minus_reverse"] = career - reverse
        out["rf_success_plus_reverse"] = career + reverse
        out["rf_success_minus_middle"] = career - middle
        out["rf_non_success_command_gap"] = reverse - middle
    if "D" in families:
        pc = career - career.mean()
        bc = batter - batter.mean()
        out["rf_pitcher_minus_batter"] = career - batter
        out["rf_pitcher_plus_batter"] = career + batter
        out["rf_pitcher_centered_x_batter_centered"] = pc * bc
        out["rf_recent_mean_minus_batter"] = recent.mean(axis=1) - batter
    if "E" in families:
        recent_dev = prev3 - career
        out["rf_prev3_dev_x_balls"] = recent_dev * balls
        out["rf_prev3_dev_x_strikes"] = recent_dev * strikes
        out["rf_prev3_dev_x_two_strike"] = recent_dev * (strikes == 2).astype("float32")
        out["rf_prev3_dev_x_hitter_count"] = recent_dev * (balls > strikes).astype("float32")
        out["rf_prev3_dev_x_late"] = recent_dev * (inning >= 7).astype("float32")
        out["rf_prev3_dev_x_li"] = recent_dev * li
    if "P" in families:
        out["rf_fastball_minus_breaking"] = fastball - breaking
        out["rf_fastball_minus_offspeed"] = fastball - offspeed
        out["rf_breaking_plus_offspeed"] = breaking + offspeed
        out["rf_pitchmix_entropy_proxy"] = -(fastball**2 + breaking**2 + offspeed**2)
    return out


def candidate_families():
    return [
        ("BASE", ""),
        ("A_recent_vs_longterm", "A"),
        ("B_recent_trend", "B"),
        ("C_skill_decomposition", "C"),
        ("D_pitcher_batter", "D"),
        ("E_recent_context", "E"),
        ("P_pitchmix", "P"),
        ("A_B_recent_combo", "AB"),
        ("A_C_skill_combo", "AC"),
        ("A_D_env_combo", "AD"),
        ("A_E_context_combo", "AE"),
        ("A_B_C_core", "ABC"),
        ("A_B_D_core_env", "ABD"),
        ("A_B_C_D_all_signal", "ABCD"),
    ]


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


def asof_inventory(df):
    rows = []
    y = df[TARGET_COL].astype("float64")
    for name in [c for c in ASOF_COLS if c in df.columns]:
        s = pd.to_numeric(df[name], errors="coerce")
        valid = s.notna()
        q25, q75 = s.quantile([0.25, 0.75])
        low = y[s <= q25].mean()
        high = y[s >= q75].mean()
        row = {
            "feature": name,
            "missing_rate": float(s.isna().mean()),
            "mean": float(s.mean()),
            "std": float(s.std()),
            "target_separation_q75_minus_q25": float(high - low),
            "pearson_corr": float(s[valid].corr(y[valid])) if valid.sum() else np.nan,
        }
        for year in [2022, 2023, 2024]:
            mask = (df["season"] == year) & valid
            row[f"corr_{year}"] = float(s[mask].corr(y[mask])) if mask.sum() else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("target_separation_q75_minus_q25", ascending=False)


def feature_relation(df, new_features):
    rows = []
    y = df[TARGET_COL].astype("float64")
    for feat in new_features:
        if feat not in df.columns:
            continue
        s = pd.to_numeric(df[feat], errors="coerce")
        valid = s.notna()
        q25, q75 = s.quantile([0.25, 0.75])
        rows.append(
            {
                "feature": feat,
                "missing_rate": float(s.isna().mean()),
                "mean": float(s.mean()),
                "std": float(s.std()),
                "target_separation_q75_minus_q25": float(y[s >= q75].mean() - y[s <= q25].mean()),
                "pearson_corr": float(s[valid].corr(y[valid])) if valid.sum() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def prepare_fold(df, train_start, train_end, cal_year, valid_year, families):
    train_df = add_asof_reconstruction(df[(df["season"] >= train_start) & (df["season"] <= train_end)].copy(), families)
    cal_df = add_asof_reconstruction(df[df["season"] == cal_year].copy(), families)
    valid_df = add_asof_reconstruction(df[df["season"] == valid_year].copy(), families)
    y_train = train_df[TARGET_COL].astype("int8")
    y_cal = cal_df[TARGET_COL].astype("int8").to_numpy()
    y_valid = valid_df[TARGET_COL].astype("int8").to_numpy()
    builder = FeatureBuilder(alpha=80.0)
    X_train = builder.fit_transform(train_df.drop(columns=[TARGET_COL]), y_train)
    X_cal = builder.transform(cal_df.drop(columns=[TARGET_COL]))
    X_valid = builder.transform(valid_df.drop(columns=[TARGET_COL]))
    return train_df, cal_df, valid_df, X_train, X_cal, X_valid, y_train, y_cal, y_valid, builder


def evaluate_candidate(df, year_rates, candidate_name, families, params, model_label):
    rows = []
    runtime_rows = []
    fi_rows = []
    y_oof = []
    pred_oof = []
    pred_oof_by_year = {}
    feature_rel_parts = []
    for train_start, train_end, cal_year, valid_year in FOLDS:
        t0 = time.time()
        train_df, cal_df, valid_df, X_train, X_cal, X_valid, y_train, y_cal, y_valid, builder = prepare_fold(
            df, train_start, train_end, cal_year, valid_year, families
        )
        feature_seconds = time.time() - t0
        model = CatBoostClassifier(**params)
        t0 = time.time()
        model.fit(X_train, y_train)
        train_seconds = time.time() - t0
        t0 = time.time()
        raw_cal = clip_prob(model.predict_proba(X_cal)[:, 1])
        raw_valid = clip_prob(model.predict_proba(X_valid)[:, 1])
        inference_seconds = time.time() - t0
        calibration = fit_calibrators(raw_cal, y_cal, cal_df[["game_type"]])
        platt_valid = apply_calibration(raw_valid, valid_df[["game_type"]], calibration, "platt")
        final_valid = apply_v2_strength(platt_valid, target_rate_for_year(year_rates, valid_year))
        y_oof.append(y_valid)
        pred_oof.append(final_valid)
        pred_oof_by_year[valid_year] = final_valid
        high_mask = final_valid >= np.median(final_valid)
        low_mask = ~high_mask
        row = {
            "candidate_name": candidate_name,
            "model_label": model_label,
            "families": families,
            "fold": f"{train_start}-{train_end}_cal{cal_year}_valid{valid_year}",
            "valid_year": valid_year,
            "high_half_actual_rate": float(np.mean(y_valid[high_mask])),
            "low_half_actual_rate": float(np.mean(y_valid[low_mask])),
            "high_minus_low_actual_rate": float(np.mean(y_valid[high_mask]) - np.mean(y_valid[low_mask])),
            **{k: params.get(k, np.nan) for k in ["iterations", "learning_rate", "depth", "l2_leaf_reg", "random_strength", "bagging_temperature", "rsm"]},
            **metric_dict(y_valid, raw_valid, "raw"),
            **metric_dict(y_valid, final_valid, "final"),
        }
        rows.append(row)
        runtime_rows.append(
            {
                "candidate_name": candidate_name,
                "model_label": model_label,
                "valid_year": valid_year,
                "feature_seconds": feature_seconds,
                "train_seconds": train_seconds,
                "inference_seconds": inference_seconds,
            }
        )
        fi = model.get_feature_importance()
        for rank, idx in enumerate(np.argsort(fi)[::-1][:40], start=1):
            feat = X_train.columns[idx]
            fi_rows.append(
                {
                    "candidate_name": candidate_name,
                    "model_label": model_label,
                    "valid_year": valid_year,
                    "rank": rank,
                    "feature": feat,
                    "importance": float(fi[idx]),
                    "is_reconstructed": feat.startswith("rf_"),
                }
            )
        new_features = [c for c in valid_df.columns if c.startswith("rf_")]
        rel = feature_relation(valid_df, new_features)
        if not rel.empty:
            rel["candidate_name"] = candidate_name
            rel["valid_year"] = valid_year
            feature_rel_parts.append(rel)
        print(
            f"{candidate_name}/{model_label} valid={valid_year} auc={row['final_auc']:.6f} "
            f"pseudo={row['final_pseudo_score']:.3f} skill={row['final_skill_margin']:.7f} "
            f"train_s={train_seconds:.2f}"
        )
    return (
        pd.DataFrame(rows),
        pd.DataFrame(runtime_rows),
        pd.DataFrame(fi_rows),
        np.concatenate(y_oof),
        np.concatenate(pred_oof),
        pred_oof_by_year,
        pd.concat(feature_rel_parts, ignore_index=True) if feature_rel_parts else pd.DataFrame(),
    )


def summarize(fold_metrics):
    rows = []
    for (candidate_name, model_label), g in fold_metrics.groupby(["candidate_name", "model_label"], sort=False):
        row = {
            "candidate_name": candidate_name,
            "model_label": model_label,
            "families": g["families"].iloc[0],
            "mean_auc": float(g["final_auc"].mean()),
            "worst_auc": float(g["final_auc"].min()),
            "mean_brier": float(g["final_brier"].mean()),
            "worst_brier": float(g["final_brier"].max()),
            "mean_skill_margin": float(g["final_skill_margin"].mean()),
            "worst_skill_margin": float(g["final_skill_margin"].min()),
            "mean_pseudo_score": float(g["final_pseudo_score"].mean()),
            "positive_fold_count": int((g["final_skill_margin"] > 0).sum()),
            "mean_pred_std": float(g["final_pred_std"].mean()),
        }
        for year in [2022, 2023, 2024]:
            y = g[g["valid_year"] == year].iloc[0]
            row[f"auc_{year}"] = float(y["final_auc"])
            row[f"brier_{year}"] = float(y["final_brier"])
            row[f"pseudo_{year}"] = float(y["final_pseudo_score"])
            row[f"skill_{year}"] = float(y["final_skill_margin"])
            row[f"pred_std_{year}"] = float(y["final_pred_std"])
            row[f"high_minus_low_actual_rate_{year}"] = float(y["high_minus_low_actual_rate"])
        rows.append(row)
    out = pd.DataFrame(rows)
    base = out[(out["candidate_name"] == "BASE") & (out["model_label"] == "v2_params")].iloc[0]
    for col in ["mean_auc", "auc_2022", "auc_2023", "auc_2024", "mean_brier", "brier_2022", "brier_2023", "brier_2024", "pseudo_2022", "pseudo_2023", "pseudo_2024", "skill_2023"]:
        out[f"delta_vs_v2_{col}"] = out[col] - float(base[col])
    out["success_flag"] = (
        (out["mean_auc"] >= 0.523)
        | ((out["pseudo_2024"] >= 25.0) & (out["pseudo_2022"] > 0) & (out["skill_2023"] >= base["skill_2023"]))
        | ((out["skill_2023"] >= base["skill_2023"] + 0.00005) & (out["pseudo_2022"] > 0) & (out["pseudo_2024"] > 0))
    )
    return out.sort_values(["mean_auc", "auc_2024", "skill_2023"], ascending=[False, False, False])


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig")
    year_rates = df.groupby("season")[TARGET_COL].mean().sort_index()
    asof_inventory(df).to_csv(os.path.join(OUT_DIR, "feature_inventory.csv"), index=False, encoding="utf-8")

    fold_parts = []
    runtime_parts = []
    fi_parts = []
    rel_parts = []
    pred_by_candidate = {}
    pred_by_year = {}
    for name, families in candidate_families():
        fm, rt, fi, y_oof, pred_oof, pred_year, rel = evaluate_candidate(
            df, year_rates, name, families, V2_CATBOOST_PARAMS, "v2_params"
        )
        fold_parts.append(fm)
        runtime_parts.append(rt)
        fi_parts.append(fi)
        rel_parts.append(rel)
        pred_by_candidate[(name, "v2_params")] = pred_oof
        pred_by_year[(name, "v2_params")] = pred_year

    fold_metrics = pd.concat(fold_parts, ignore_index=True)
    summary = summarize(fold_metrics)
    base_pred = pred_by_candidate[("BASE", "v2_params")]
    best = summary[(summary["candidate_name"] != "BASE") & (summary["model_label"] == "v2_params")].iloc[0]
    run_tuned = bool(
        (best["mean_auc"] >= summary[(summary["candidate_name"] == "BASE") & (summary["model_label"] == "v2_params")]["mean_auc"].iloc[0] + 0.0005)
        or (best["pseudo_2024"] >= 25.0)
        or best["success_flag"]
    )
    if run_tuned:
        fm, rt, fi, y_oof, pred_oof, pred_year, rel = evaluate_candidate(
            df, year_rates, f"{best['candidate_name']}_tuned_catboost", best["families"], TUNED_CATBOOST_PARAMS, "tuned_params"
        )
        fold_metrics = pd.concat([fold_metrics, fm], ignore_index=True)
        runtime_parts.append(rt)
        fi_parts.append(fi)
        rel_parts.append(rel)
        pred_by_candidate[(f"{best['candidate_name']}_tuned_catboost", "tuned_params")] = pred_oof
        pred_by_year[(f"{best['candidate_name']}_tuned_catboost", "tuned_params")] = pred_year
        summary = summarize(fold_metrics)

    corr_rows = []
    for (name, model_label), pred in pred_by_candidate.items():
        row = {
            "candidate_name": name,
            "model_label": model_label,
            "overall_corr_vs_v2": float(np.corrcoef(base_pred, pred)[0, 1]),
            "overall_mean_abs_diff_vs_v2": float(np.mean(np.abs(base_pred - pred))),
        }
        for year in [2022, 2023, 2024]:
            row[f"corr_vs_v2_{year}"] = float(np.corrcoef(pred_by_year[("BASE", "v2_params")][year], pred_by_year[(name, model_label)][year])[0, 1])
        corr_rows.append(row)
    corr = pd.DataFrame(corr_rows)
    summary = summary.merge(corr, on=["candidate_name", "model_label"], how="left")

    stable = summary[(summary["pseudo_2022"] > 0) & (summary["pseudo_2024"] > 0)].sort_values(
        ["worst_skill_margin", "mean_auc", "auc_2024"], ascending=[False, False, False]
    )
    discr = summary.sort_values(["mean_auc", "auc_2024"], ascending=[False, False])
    y2023 = summary[(summary["pseudo_2022"] > 0) & (summary["pseudo_2024"] > 0)].sort_values(
        ["skill_2023", "mean_auc"], ascending=[False, False]
    )
    labels = {}
    if not stable.empty:
        labels[(stable.iloc[0]["candidate_name"], stable.iloc[0]["model_label"])] = "Candidate A - fold-stable"
    if not discr.empty:
        key = (discr.iloc[0]["candidate_name"], discr.iloc[0]["model_label"])
        labels[key] = (labels.get(key, "") + "; Candidate B - highest discrimination").strip("; ")
    if not y2023.empty:
        key = (y2023.iloc[0]["candidate_name"], y2023.iloc[0]["model_label"])
        labels[key] = (labels.get(key, "") + "; Candidate C - best 2023 loss").strip("; ")
    summary["selection_note"] = [
        labels.get((r.candidate_name, r.model_label), "") for r in summary.itertuples()
    ]

    selected = set(labels)
    fi_all = pd.concat(fi_parts, ignore_index=True)
    fi_selected = fi_all[
        fi_all[["candidate_name", "model_label"]].apply(tuple, axis=1).isin(selected | {("BASE", "v2_params")})
    ].copy()
    rel_all = pd.concat([r for r in rel_parts if not r.empty], ignore_index=True) if rel_parts else pd.DataFrame()
    top_reconstructed = (
        fi_all[fi_all["is_reconstructed"]]
        .groupby("feature", as_index=False)
        .agg(mean_importance=("importance", "mean"), max_importance=("importance", "max"), count_top40=("feature", "count"))
        .sort_values(["mean_importance", "max_importance"], ascending=[False, False])
    )
    rel_all.to_csv(os.path.join(OUT_DIR, "reconstructed_feature_relations.csv"), index=False, encoding="utf-8")
    top_reconstructed.to_csv(os.path.join(OUT_DIR, "reconstructed_feature_strength.csv"), index=False, encoding="utf-8")
    fold_metrics.to_csv(os.path.join(OUT_DIR, "fold_metrics.csv"), index=False, encoding="utf-8")
    summary.to_csv(os.path.join(OUT_DIR, "candidate_summary.csv"), index=False, encoding="utf-8")
    summary.to_csv(os.path.join(OUT_DIR, "feature_family_metrics.csv"), index=False, encoding="utf-8")
    corr.to_csv(os.path.join(OUT_DIR, "prediction_correlation.csv"), index=False, encoding="utf-8")
    fi_selected.to_csv(os.path.join(OUT_DIR, "feature_importance.csv"), index=False, encoding="utf-8")
    pd.concat(runtime_parts, ignore_index=True).to_csv(os.path.join(OUT_DIR, "runtime_summary.csv"), index=False, encoding="utf-8")
    summary[["candidate_name", "model_label", "auc_2023", "skill_2023", "pseudo_2023", "pred_std_2023", "high_minus_low_actual_rate_2023", "delta_vs_v2_skill_2023"]].to_csv(
        os.path.join(OUT_DIR, "year2023_analysis.csv"), index=False, encoding="utf-8"
    )
    print("\nTop candidates")
    print(
        summary[
            [
                "candidate_name",
                "model_label",
                "mean_auc",
                "auc_2024",
                "pseudo_2022",
                "pseudo_2023",
                "pseudo_2024",
                "skill_2023",
                "delta_vs_v2_mean_auc",
                "delta_vs_v2_skill_2023",
                "overall_corr_vs_v2",
                "success_flag",
                "selection_note",
            ]
        ].head(20).to_string(index=False)
    )
    print("\nTop reconstructed features")
    print(top_reconstructed.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
