import os
from itertools import combinations

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.stats import spearmanr
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from calibration_utils import apply_calibration, fit_calibrators
from model_utils import FeatureBuilder, TARGET_COL
from validate_v3_asof_signal_reconstruction import (
    HARD_CAP,
    TEMPERATURE,
    TUNED_CATBOOST_PARAMS,
    V2_CATBOOST_PARAMS,
    add_asof_reconstruction,
    apply_v2_strength,
    target_rate_for_year,
)


OUT_DIR = "output/v3_ensemble"
ASOF_DIR = "output/v3_asof_signal_reconstruction"
TRAIN_PATH = "data/train.csv"
FOLDS = [
    (2019, 2020, 2021, 2022),
    (2019, 2021, 2022, 2023),
    (2019, 2022, 2023, 2024),
]
WEIGHTS_2MODEL = [round(x, 1) for x in np.linspace(0.0, 1.0, 11)]
EPS = 1e-6


def clip_prob(pred):
    return np.clip(np.asarray(pred, dtype=np.float64), EPS, 1.0 - EPS)


def metric_dict(y, pred):
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
        "prediction_std": float(pred.std()),
        "prediction_mean": float(pred.mean()),
    }


def read_previous_candidates():
    summary = pd.read_csv(os.path.join(ASOF_DIR, "candidate_summary.csv"))
    summary["is_reconstructed"] = summary["candidate_name"] != "BASE"
    reconstructed = summary[summary["is_reconstructed"]].copy()
    stable_pool = reconstructed[(reconstructed["pseudo_2022"] > 0) & (reconstructed["pseudo_2024"] > 0)]
    if stable_pool.empty:
        stable = None
    else:
        stable = stable_pool.sort_values(
            ["worst_skill_margin", "mean_auc", "auc_2024"],
            ascending=[False, False, False],
        ).iloc[0]
    discrimination = reconstructed.sort_values(["mean_auc", "auc_2024"], ascending=[False, False]).iloc[0]
    base = summary[(summary["candidate_name"] == "BASE") & (summary["model_label"] == "v2_params")].iloc[0]
    y2023_pool = reconstructed.copy()
    y2023_pool["skill_2023_delta_vs_v2"] = y2023_pool["skill_2023"] - float(base["skill_2023"])
    y2023 = y2023_pool.sort_values(["skill_2023_delta_vs_v2", "mean_auc"], ascending=[False, False]).iloc[0]
    return summary, base, stable, discrimination, y2023


def model_specs_from_previous():
    summary, base, stable, discrimination, y2023 = read_previous_candidates()
    specs = [
        {
            "model_id": "A_v2_base",
            "source_candidate": "BASE",
            "families": "",
            "params": V2_CATBOOST_PARAMS,
            "selection_role": "V2 baseline model",
        },
        {
            "model_id": "B_tuned_base",
            "source_candidate": "BASE",
            "families": "",
            "params": TUNED_CATBOOST_PARAMS,
            "selection_role": "V3 tuned CatBoost on V2 features",
        },
    ]
    selected = set()
    candidate_rows = []
    if stable is not None:
        candidate_rows.append(("C_stable_reconstructed", stable, "Best fold-stable reconstructed candidate"))
    candidate_rows.append(("D_high_discrimination", discrimination, "Highest discrimination reconstructed candidate"))
    if float(y2023["skill_2023_delta_vs_v2"]) > 0:
        candidate_rows.append(("E_2023_skill", y2023, "Best 2023 skill-margin reconstructed candidate"))

    for prefix, row, role in candidate_rows:
        key = (row["candidate_name"], row["model_label"])
        if key in selected:
            continue
        selected.add(key)
        params = TUNED_CATBOOST_PARAMS if row["model_label"] == "tuned_params" else V2_CATBOOST_PARAMS
        specs.append(
            {
                "model_id": f"{prefix}_{row['candidate_name']}_{row['model_label']}",
                "source_candidate": row["candidate_name"],
                "families": "" if pd.isna(row["families"]) else str(row["families"]),
                "params": params,
                "selection_role": role,
            }
        )
    selected_summary = pd.DataFrame(
        [
            {
                "role": "previous_base_anchor",
                "candidate_name": base["candidate_name"],
                "model_label": base["model_label"],
                "families": base["families"],
                "mean_auc": base["mean_auc"],
                "pseudo_2022": base["pseudo_2022"],
                "pseudo_2023": base["pseudo_2023"],
                "pseudo_2024": base["pseudo_2024"],
                "skill_2023": base["skill_2023"],
                "success_flag": base["success_flag"],
            },
            *[
                {
                    "role": role,
                    "candidate_name": row["candidate_name"],
                    "model_label": row["model_label"],
                    "families": row["families"],
                    "mean_auc": row["mean_auc"],
                    "pseudo_2022": row["pseudo_2022"],
                    "pseudo_2023": row["pseudo_2023"],
                    "pseudo_2024": row["pseudo_2024"],
                    "skill_2023": row["skill_2023"],
                    "success_flag": row["success_flag"],
                }
                for _, row, role in candidate_rows
            ],
        ]
    )
    return specs, selected_summary, summary


def prepare_fold(df, train_start, train_end, cal_year, valid_year, families):
    train_df = df[(df["season"] >= train_start) & (df["season"] <= train_end)].copy()
    cal_df = df[df["season"] == cal_year].copy()
    valid_df = df[df["season"] == valid_year].copy()
    train_df = add_asof_reconstruction(train_df, families)
    cal_df = add_asof_reconstruction(cal_df, families)
    valid_df = add_asof_reconstruction(valid_df, families)
    y_train = train_df[TARGET_COL].astype("int8")
    y_cal = cal_df[TARGET_COL].astype("int8")
    y_valid = valid_df[TARGET_COL].astype("int8")
    builder = FeatureBuilder()
    X_train = builder.fit_transform(train_df.drop(columns=[TARGET_COL]), y_train)
    X_cal = builder.transform(cal_df.drop(columns=[TARGET_COL]))
    X_valid = builder.transform(valid_df.drop(columns=[TARGET_COL]))
    return cal_df, valid_df, X_train, X_cal, X_valid, y_train, y_cal, y_valid


def build_oof(df, specs, year_rates):
    parts = []
    cal_parts = []
    for train_start, train_end, cal_year, valid_year in FOLDS:
        fold_name = f"{train_start}-{train_end}_cal{cal_year}_valid{valid_year}"
        base_frame = None
        cal_frame = None
        for spec in specs:
            cal_df, valid_df, X_train, X_cal, X_valid, y_train, y_cal, y_valid = prepare_fold(
                df, train_start, train_end, cal_year, valid_year, spec["families"]
            )
            model = CatBoostClassifier(**spec["params"])
            model.fit(X_train, y_train)
            raw_cal = clip_prob(model.predict_proba(X_cal)[:, 1])
            raw_valid = clip_prob(model.predict_proba(X_valid)[:, 1])
            calibration = fit_calibrators(raw_cal, y_cal, cal_df[["game_type"]])
            platt_valid = apply_calibration(raw_valid, valid_df[["game_type"]], calibration, "platt")
            final_valid = apply_v2_strength(platt_valid, target_rate_for_year(year_rates, valid_year))
            if base_frame is None:
                base_frame = pd.DataFrame(
                    {
                        "row_id": valid_df.index.to_numpy(),
                        "fold": fold_name,
                        "valid_year": valid_year,
                        "target": y_valid.to_numpy(dtype=np.int8),
                    }
                )
                cal_frame = pd.DataFrame(
                    {
                        "fold": fold_name,
                        "valid_year": valid_year,
                        "cal_year": cal_year,
                        "target": y_cal.to_numpy(dtype=np.int8),
                        "game_type": cal_df["game_type"].to_numpy(),
                    }
                )
            base_frame[f"{spec['model_id']}__raw"] = raw_valid
            base_frame[f"{spec['model_id']}__final"] = final_valid
            cal_frame[f"{spec['model_id']}__raw"] = raw_cal
        parts.append(base_frame)
        cal_parts.append(cal_frame)
        print(f"built OOF fold={fold_name} rows={len(base_frame)} models={len(specs)}")
    return pd.concat(parts, ignore_index=True), pd.concat(cal_parts, ignore_index=True)


def model_fold_metrics(oof, specs):
    rows = []
    for spec in specs:
        pred_col = f"{spec['model_id']}__final"
        raw_col = f"{spec['model_id']}__raw"
        for year, g in oof.groupby("valid_year"):
            rows.append(
                {
                    "candidate_id": spec["model_id"],
                    "candidate_type": "single_model",
                    "valid_year": int(year),
                    "weights": spec["model_id"],
                    "raw_prediction_std": float(g[raw_col].std()),
                    **metric_dict(g["target"], g[pred_col]),
                }
            )
    return pd.DataFrame(rows)


def pairwise_prediction_diversity(oof, specs):
    rows = []
    for left, right in combinations([s["model_id"] for s in specs], 2):
        for scope, g in [("overall", oof), *[(str(y), gy) for y, gy in oof.groupby("valid_year")]]:
            a = g[f"{left}__raw"].to_numpy(dtype=np.float64)
            b = g[f"{right}__raw"].to_numpy(dtype=np.float64)
            rows.append(
                {
                    "model_a": left,
                    "model_b": right,
                    "scope": scope,
                    "pearson_corr": float(np.corrcoef(a, b)[0, 1]),
                    "spearman_corr": float(spearmanr(a, b).correlation),
                    "prediction_diff_std": float(np.std(a - b)),
                    "mean_abs_prediction_diff": float(np.mean(np.abs(a - b))),
                }
            )
    return pd.DataFrame(rows)


def pairwise_error_diversity(oof, specs):
    rows = []
    model_ids = [s["model_id"] for s in specs]
    for left, right in combinations(model_ids, 2):
        for scope, g in [("overall", oof), *[(str(y), gy) for y, gy in oof.groupby("valid_year")]]:
            y = g["target"].to_numpy(dtype=np.float64)
            ea = (g[f"{left}__final"].to_numpy(dtype=np.float64) - y) ** 2
            eb = (g[f"{right}__final"].to_numpy(dtype=np.float64) - y) ** 2
            worse_than_constant_a = ea >= (y.mean() - y) ** 2
            worse_than_constant_b = eb >= (y.mean() - y) ** 2
            rows.append(
                {
                    "model_a": left,
                    "model_b": right,
                    "scope": scope,
                    "squared_error_corr": float(np.corrcoef(ea, eb)[0, 1]),
                    "model_a_better_row_rate": float(np.mean(ea < eb)),
                    "model_b_better_row_rate": float(np.mean(eb < ea)),
                    "tie_row_rate": float(np.mean(np.isclose(ea, eb))),
                    "both_worse_than_constant_row_rate": float(np.mean(worse_than_constant_a & worse_than_constant_b)),
                    "both_wrong_top_decile_error_rate": float(
                        np.mean((ea >= np.quantile(ea, 0.9)) & (eb >= np.quantile(eb, 0.9)))
                    ),
                }
            )
    return pd.DataFrame(rows)


def calibrate_blend(cal_g, valid_g, model_weights, target_rate):
    raw_cal = np.zeros(len(cal_g), dtype=np.float64)
    raw_valid = np.zeros(len(valid_g), dtype=np.float64)
    for model_id, weight in model_weights.items():
        raw_cal += weight * cal_g[f"{model_id}__raw"].to_numpy(dtype=np.float64)
        raw_valid += weight * valid_g[f"{model_id}__raw"].to_numpy(dtype=np.float64)
    calibration = fit_calibrators(raw_cal, cal_g["target"], cal_g[["game_type"]])
    platt_valid = apply_calibration(raw_valid, valid_g[["game_type"]] if "game_type" in valid_g else None, calibration, "platt")
    return raw_valid, apply_v2_strength(platt_valid, target_rate)


def ensemble_metrics(oof, cal_oof, specs, year_rates, diversity):
    model_ids = [s["model_id"] for s in specs]
    grid_rows = []
    fold_rows = []
    candidates = []

    pair_keys = {tuple(sorted(("A_v2_base", model_id))) for model_id in model_ids if model_id != "A_v2_base"}
    non_v2 = diversity[
        (diversity["scope"] == "overall")
        & (diversity["model_a"] != "A_v2_base")
        & (diversity["model_b"] != "A_v2_base")
    ].sort_values("pearson_corr")
    if not non_v2.empty:
        pair_keys.add(tuple(sorted((non_v2.iloc[0]["model_a"], non_v2.iloc[0]["model_b"]))))

    for left, right in sorted(pair_keys):
        for w in WEIGHTS_2MODEL:
            weights = {left: w, right: round(1.0 - w, 10)}
            candidate_id = f"blend2__{left}__{w:.1f}__{right}__{1.0 - w:.1f}"
            candidates.append((candidate_id, weights, "two_model_blend"))

    # Only one coarse 3-model check, and only among the V2 anchor plus two actual reconstructed candidates.
    reconstructed_ids = [s["model_id"] for s in specs if s["model_id"].startswith(("C_", "D_", "E_"))]
    if len(reconstructed_ids) >= 2:
        three = ["A_v2_base", reconstructed_ids[0], reconstructed_ids[1]]
        for weights in [
            {three[0]: 0.5, three[1]: 0.25, three[2]: 0.25},
            {three[0]: 0.4, three[1]: 0.3, three[2]: 0.3},
            {three[0]: 0.34, three[1]: 0.33, three[2]: 0.33},
        ]:
            suffix = "__".join([f"{k}__{v:.2f}" for k, v in weights.items()])
            candidates.append((f"blend3__{suffix}", weights, "three_model_blend_limited"))

    for candidate_id, weights, candidate_type in candidates:
        valid_pred_parts = []
        raw_pred_parts = []
        for year, valid_g in oof.groupby("valid_year", sort=True):
            cal_g = cal_oof[cal_oof["valid_year"] == year].reset_index(drop=True)
            valid_g = valid_g.reset_index(drop=True)
            raw_valid, final_valid = calibrate_blend(cal_g, valid_g, weights, target_rate_for_year(year_rates, int(year)))
            tmp = valid_g[["row_id", "fold", "valid_year", "target"]].copy()
            tmp["raw_pred"] = raw_valid
            tmp["final_pred"] = final_valid
            valid_pred_parts.append(tmp)
            raw_pred_parts.append(raw_valid)
            fold_rows.append(
                {
                    "candidate_id": candidate_id,
                    "candidate_type": candidate_type,
                    "valid_year": int(year),
                    "weights": ";".join([f"{k}:{v:.3f}" for k, v in weights.items()]),
                    "raw_prediction_std": float(raw_valid.std()),
                    **metric_dict(valid_g["target"], final_valid),
                }
            )
        pred_df = pd.concat(valid_pred_parts, ignore_index=True)
        overall = metric_dict(pred_df["target"], pred_df["final_pred"])
        grid_rows.append(
            {
                "candidate_id": candidate_id,
                "candidate_type": candidate_type,
                "weights": ";".join([f"{k}:{v:.3f}" for k, v in weights.items()]),
                "mean_auc": float(np.mean([r["auc"] for r in [metric_dict(g["target"], g["final_pred"]) for _, g in pred_df.groupby("valid_year")]])),
                "mean_brier": float(np.mean([r["model_brier"] for r in [metric_dict(g["target"], g["final_pred"]) for _, g in pred_df.groupby("valid_year")]])),
                "worst_skill_margin": float(min([r["skill_margin"] for r in [metric_dict(g["target"], g["final_pred"]) for _, g in pred_df.groupby("valid_year")]])),
                "overall_auc": overall["auc"],
                "overall_brier": overall["model_brier"],
                "overall_skill_margin": overall["skill_margin"],
            }
        )
    return pd.DataFrame(grid_rows), pd.DataFrame(fold_rows)


def summarize_candidates(fold_metrics, previous_summary):
    rows = []
    base = fold_metrics[(fold_metrics["candidate_id"] == "A_v2_base") & (fold_metrics["candidate_type"] == "single_model")]
    base_by_year = base.set_index("valid_year")
    for candidate_id, g in fold_metrics.groupby("candidate_id", sort=False):
        by_year = g.set_index("valid_year")
        if not {2022, 2023, 2024}.issubset(set(by_year.index)):
            continue
        row = {
            "candidate_id": candidate_id,
            "candidate_type": g["candidate_type"].iloc[0],
            "weights": g["weights"].iloc[0],
            "mean_auc": float(g["auc"].mean()),
            "mean_brier": float(g["model_brier"].mean()),
            "worst_skill_margin": float(g["skill_margin"].min()),
            "mean_skill_margin": float(g["skill_margin"].mean()),
            "mean_pseudo_score": float(g["pseudo_score"].mean()),
            "mean_prediction_std": float(g["prediction_std"].mean()),
            "auc_2022": float(by_year.loc[2022, "auc"]),
            "auc_2023": float(by_year.loc[2023, "auc"]),
            "auc_2024": float(by_year.loc[2024, "auc"]),
            "pseudo_2022": float(by_year.loc[2022, "pseudo_score"]),
            "pseudo_2023": float(by_year.loc[2023, "pseudo_score"]),
            "pseudo_2024": float(by_year.loc[2024, "pseudo_score"]),
            "skill_2022": float(by_year.loc[2022, "skill_margin"]),
            "skill_2023": float(by_year.loc[2023, "skill_margin"]),
            "skill_2024": float(by_year.loc[2024, "skill_margin"]),
            "pred_std_2023": float(by_year.loc[2023, "prediction_std"]),
            "delta_vs_v2_mean_auc": float(g["auc"].mean() - base["auc"].mean()),
            "delta_vs_v2_mean_brier": float(g["model_brier"].mean() - base["model_brier"].mean()),
            "delta_vs_v2_skill_2023": float(by_year.loc[2023, "skill_margin"] - base_by_year.loc[2023, "skill_margin"]),
            "delta_vs_v2_pseudo_2024": float(by_year.loc[2024, "pseudo_score"] - base_by_year.loc[2024, "pseudo_score"]),
        }
        row["selection_rule_a"] = (
            row["pseudo_2024"] >= 25.0
            and row["pseudo_2022"] > 0.0
            and row["skill_2023"] >= float(base_by_year.loc[2023, "skill_margin"])
        )
        row["selection_rule_b"] = (
            row["skill_2023"] >= float(base_by_year.loc[2023, "skill_margin"]) + 0.00005
            and row["pseudo_2022"] > 0.0
            and row["pseudo_2024"] > 0.0
        )
        row["selection_rule_c"] = row["mean_auc"] >= 0.523 and row["worst_skill_margin"] > -0.001
        row["meets_selection_rule"] = row["selection_rule_a"] or row["selection_rule_b"] or row["selection_rule_c"]
        rows.append(row)
    out = pd.DataFrame(rows).sort_values(
        ["meets_selection_rule", "mean_auc", "pseudo_2024", "skill_2023"],
        ascending=[False, False, False, False],
    )
    out["previous_signal_reconstruction_success_count"] = int(
        pd.Series(previous_summary["success_flag"]).astype(str).str.lower().eq("true").sum()
    )
    return out


def year2023_special(oof, fold_metrics):
    rows = []
    for candidate_id, g in fold_metrics[fold_metrics["valid_year"] == 2023].groupby("candidate_id", sort=False):
        metric_row = g.iloc[0].to_dict()
        if candidate_id in oof.columns:
            pred = oof[candidate_id].to_numpy(dtype=np.float64)
        rows.append(
            {
                "candidate_id": candidate_id,
                "candidate_type": metric_row["candidate_type"],
                "auc_2023": metric_row["auc"],
                "skill_margin_2023": metric_row["skill_margin"],
                "pseudo_2023": metric_row["pseudo_score"],
                "prediction_std_2023": metric_row["prediction_std"],
            }
        )
    return pd.DataFrame(rows)


def add_2023_high_low(summary, oof, cal_oof):
    # For single models use already materialized predictions; for blends this value is recomputed in a compact pass.
    rows = []
    year_rates = pd.read_csv(TRAIN_PATH).groupby("season")[TARGET_COL].mean().sort_index()
    for _, row in summary.iterrows():
        candidate_id = row["candidate_id"]
        if row["candidate_type"] == "single_model":
            g = oof[oof["valid_year"] == 2023]
            pred = g[f"{candidate_id}__final"].to_numpy(dtype=np.float64)
            y = g["target"].to_numpy(dtype=np.float64)
        else:
            weights = {}
            for token in row["weights"].split(";"):
                model_id, weight = token.rsplit(":", 1)
                weights[model_id] = float(weight)
            valid_g = oof[oof["valid_year"] == 2023].reset_index(drop=True)
            cal_g = cal_oof[cal_oof["valid_year"] == 2023].reset_index(drop=True)
            _, pred = calibrate_blend(cal_g, valid_g, weights, target_rate_for_year(year_rates, 2023))
            y = valid_g["target"].to_numpy(dtype=np.float64)
        high = pred >= np.median(pred)
        rows.append(
            {
                "candidate_id": candidate_id,
                "high_half_actual_rate_2023": float(y[high].mean()),
                "low_half_actual_rate_2023": float(y[~high].mean()),
                "high_minus_low_actual_rate_2023": float(y[high].mean() - y[~high].mean()),
            }
        )
    return summary.merge(pd.DataFrame(rows), on="candidate_id", how="left")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    specs, selected_previous, previous_summary = model_specs_from_previous()
    selected_previous.to_csv(os.path.join(OUT_DIR, "selected_previous_candidates.csv"), index=False, encoding="utf-8")
    pd.DataFrame([{k: v for k, v in spec.items() if k != "params"} for spec in specs]).to_csv(
        os.path.join(OUT_DIR, "model_specs.csv"), index=False, encoding="utf-8"
    )

    df = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig")
    year_rates = df.groupby("season")[TARGET_COL].mean().sort_index()
    oof_path = os.path.join(OUT_DIR, "oof_predictions.csv")
    cal_oof_path = os.path.join(OUT_DIR, "calibration_oof_predictions.csv")
    if os.path.exists(oof_path) and os.path.exists(cal_oof_path):
        oof = pd.read_csv(oof_path)
        cal_oof = pd.read_csv(cal_oof_path)
        print(f"reused OOF rows={len(oof)} cal_rows={len(cal_oof)}")
    else:
        oof, cal_oof = build_oof(df, specs, year_rates)
        oof.to_csv(oof_path, index=False, encoding="utf-8")
        cal_oof.to_csv(cal_oof_path, index=False, encoding="utf-8")

    model_pair_diversity = pairwise_prediction_diversity(oof, specs)
    error_diversity = pairwise_error_diversity(oof, specs)
    single_fold = model_fold_metrics(oof, specs)
    ensemble_grid, ensemble_fold = ensemble_metrics(oof, cal_oof, specs, year_rates, model_pair_diversity)
    fold_metrics = pd.concat([single_fold, ensemble_fold], ignore_index=True)
    summary = summarize_candidates(fold_metrics, previous_summary)
    summary = add_2023_high_low(summary, oof, cal_oof)

    model_pair_diversity.to_csv(os.path.join(OUT_DIR, "model_pair_diversity.csv"), index=False, encoding="utf-8")
    error_diversity.to_csv(os.path.join(OUT_DIR, "error_diversity.csv"), index=False, encoding="utf-8")
    ensemble_grid.to_csv(os.path.join(OUT_DIR, "ensemble_grid.csv"), index=False, encoding="utf-8")
    fold_metrics.to_csv(os.path.join(OUT_DIR, "fold_metrics.csv"), index=False, encoding="utf-8")
    summary.to_csv(os.path.join(OUT_DIR, "ensemble_summary.csv"), index=False, encoding="utf-8")
    summary[
        [
            "candidate_id",
            "candidate_type",
            "auc_2023",
            "high_minus_low_actual_rate_2023",
            "skill_2023",
            "pseudo_2023",
            "pred_std_2023",
            "delta_vs_v2_skill_2023",
        ]
    ].to_csv(os.path.join(OUT_DIR, "year2023_analysis.csv"), index=False, encoding="utf-8")

    value = summary[summary["meets_selection_rule"]].copy()
    verdict = "ENSEMBLE IMPROVEMENT FOUND" if not value.empty else "NO ENSEMBLE VALUE"
    pd.DataFrame(
        [
            {
                "verdict": verdict,
                "selected_candidate_count": min(2, len(value)),
                "v2_anchor_public_score": 96.253447238,
                "temperature": TEMPERATURE,
                "hard_cap": HARD_CAP,
                "previous_signal_reconstruction_success_count": int(
                    pd.Series(previous_summary["success_flag"]).astype(str).str.lower().eq("true").sum()
                ),
            }
        ]
    ).to_csv(os.path.join(OUT_DIR, "verdict.csv"), index=False, encoding="utf-8")

    print("\nPrevious selected candidates")
    print(selected_previous.to_string(index=False))
    print("\nTop diversity pairs")
    print(
        model_pair_diversity[model_pair_diversity["scope"] == "overall"]
        .sort_values("pearson_corr")
        .head(10)
        .to_string(index=False)
    )
    print("\nTop candidates")
    print(
        summary[
            [
                "candidate_id",
                "candidate_type",
                "mean_auc",
                "pseudo_2022",
                "pseudo_2023",
                "pseudo_2024",
                "skill_2023",
                "delta_vs_v2_skill_2023",
                "delta_vs_v2_mean_brier",
                "meets_selection_rule",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )
    print(f"\n{verdict}")


if __name__ == "__main__":
    main()
