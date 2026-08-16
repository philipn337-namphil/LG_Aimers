import os
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRanker, Pool
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from model_utils import FeatureBuilder, TARGET_COL
from validate_v3_asof_signal_reconstruction import V2_CATBOOST_PARAMS, apply_v2_strength, clip_prob, logit, target_rate_for_year


OUT_DIR = "output/v3_ranking_aware"
TRAIN_PATH = "data/train.csv"
FOLDS = [
    (2019, 2020, 2021, 2022),
    (2019, 2021, 2022, 2023),
    (2019, 2022, 2023, 2024),
]
RANK_PARAMS = {
    "loss_function": "QueryRMSE",
    "iterations": 220,
    "learning_rate": 0.045,
    "depth": 6,
    "l2_leaf_reg": 5.0,
    "random_seed": 42,
    "verbose": False,
    "allow_writing_files": False,
    "thread_count": -1,
}
RNG_SEED = 42


def add_context_columns(df):
    out = df.copy()
    out["count_state"] = out["balls_before"].astype(str) + "-" + out["strikes_before"].astype(str)
    return out


def group_key(df, definition):
    if definition == "season":
        return df["season"].astype(str)
    if definition == "game_type":
        return df["game_type"].astype(str)
    if definition == "pitcher_id":
        return df["pitcher_id"].astype(str)
    if definition == "pitcher_id_game_type":
        return df["pitcher_id"].astype(str) + "|" + df["game_type"].astype(str)
    if definition == "count_state":
        return df["count_state"].astype(str)
    if definition == "pitcher_id_count_state":
        return df["pitcher_id"].astype(str) + "|" + df["count_state"].astype(str)
    if definition == "pitcher_id_game_type_count_state":
        return df["pitcher_id"].astype(str) + "|" + df["game_type"].astype(str) + "|" + df["count_state"].astype(str)
    raise ValueError(definition)


def ranking_group_feasibility(df):
    rows = []
    definitions = [
        "season",
        "game_type",
        "pitcher_id",
        "pitcher_id_game_type",
        "count_state",
        "pitcher_id_count_state",
        "pitcher_id_game_type_count_state",
    ]
    for definition in definitions:
        gkey = group_key(df, definition)
        stat = pd.DataFrame({"group": gkey, "y": df[TARGET_COL].astype(int)})
        grouped = stat.groupby("group")["y"].agg(["count", "sum"])
        grouped["fail"] = grouped["count"] - grouped["sum"]
        grouped["has_both"] = (grouped["sum"] > 0) & (grouped["fail"] > 0)
        grouped["pair_count"] = grouped["sum"] * grouped["fail"]
        rows.append(
            {
                "group_definition": definition,
                "group_count": int(len(grouped)),
                "avg_rows_per_group": float(grouped["count"].mean()),
                "median_rows_per_group": float(grouped["count"].median()),
                "both_class_group_rate": float(grouped["has_both"].mean()),
                "pair_possible_group_rate": float((grouped["pair_count"] > 0).mean()),
                "estimated_positive_negative_pairs": int(grouped["pair_count"].sum()),
                "very_large_group_rate_ge5000": float((grouped["count"] >= 5000).mean()),
                "very_small_group_rate_lt20": float((grouped["count"] < 20).mean()),
            }
        )
    return pd.DataFrame(rows)


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
        f"{prefix}_ap": float(average_precision_score(y, pred)),
        f"{prefix}_logloss": float(log_loss(y, pred)),
        f"{prefix}_pred_mean": float(pred.mean()),
        f"{prefix}_pred_std": float(pred.std()),
    }


def fit_score_platt(score, y):
    model = LogisticRegression(C=1e6, solver="lbfgs", random_state=42)
    model.fit(np.asarray(score).reshape(-1, 1), np.asarray(y, dtype=np.int8))
    return model


def apply_score_platt(model, score):
    return clip_prob(model.predict_proba(np.asarray(score).reshape(-1, 1))[:, 1])


def prepare_features(train_df, cal_df, valid_df):
    y_train = train_df[TARGET_COL].astype("int8")
    y_cal = cal_df[TARGET_COL].astype("int8")
    y_valid = valid_df[TARGET_COL].astype("int8")
    builder = FeatureBuilder()
    x_train = builder.fit_transform(train_df.drop(columns=[TARGET_COL]), y_train)
    x_cal = builder.transform(cal_df.drop(columns=[TARGET_COL]))
    x_valid = builder.transform(valid_df.drop(columns=[TARGET_COL]))
    return x_train, x_cal, x_valid, y_train, y_cal, y_valid


def sort_for_ranker(x, y, groups):
    tmp = pd.DataFrame({"group": groups.to_numpy(), "pos": np.arange(len(groups))})
    tmp = tmp.sort_values(["group", "pos"]).reset_index(drop=True)
    idx = tmp["pos"].to_numpy(dtype=np.int64)
    return x.iloc[idx].reset_index(drop=True), np.asarray(y)[idx], tmp["group"].to_numpy()


def sample_pairwise_accuracy(y, score, groups, k=5, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"y": np.asarray(y, dtype=np.int8), "score": np.asarray(score), "group": np.asarray(groups)})
    correct = 0
    total = 0
    for _, g in df.groupby("group", sort=False):
        pos = g[g["y"] == 1]["score"].to_numpy()
        neg = g[g["y"] == 0]["score"].to_numpy()
        if len(pos) == 0 or len(neg) == 0:
            continue
        for ps in pos:
            take = min(k, len(neg))
            ns = rng.choice(neg, size=take, replace=False)
            correct += int((ps > ns).sum())
            correct += 0.5 * int((ps == ns).sum())
            total += take
            if total >= 250000:
                break
        if total >= 250000:
            break
    return float(correct / total) if total else np.nan, int(total)


def pair_sampling_stats(df, candidate_name, group_definition):
    rows = []
    for train_start, train_end, _, valid_year in FOLDS:
        train = df[(df["season"] >= train_start) & (df["season"] <= train_end)].copy()
        gkey = group_key(train, group_definition)
        stat = pd.DataFrame({"group": gkey, "y": train[TARGET_COL].astype(int)})
        grouped = stat.groupby("group")["y"].agg(["count", "sum"])
        grouped["fail"] = grouped["count"] - grouped["sum"]
        grouped["full_pairs"] = grouped["sum"] * grouped["fail"]
        grouped["sampled_k5_upper_bound"] = grouped["sum"] * np.minimum(grouped["fail"], 5)
        rows.append(
            {
                "candidate_name": candidate_name,
                "valid_year": valid_year,
                "group_definition": group_definition,
                "group_count": int(len(grouped)),
                "pairable_group_count": int((grouped["full_pairs"] > 0).sum()),
                "estimated_full_pair_count": int(grouped["full_pairs"].sum()),
                "sampled_k5_upper_bound": int(grouped["sampled_k5_upper_bound"].sum()),
                "max_group_rows": int(grouped["count"].max()),
            }
        )
    return pd.DataFrame(rows)


def candidate_specs():
    return [
        {"candidate_name": "A_CLASSIFIER", "kind": "classifier", "objective": "Logloss", "group_definition": "none", "calibration": "Platt + fixed V2 strength"},
        {"candidate_name": "B_QUERYRMSE_PITCHER_GAME", "kind": "ranker", "objective": "QueryRMSE", "group_definition": "pitcher_id_game_type", "calibration": "score Platt + fixed V2 strength"},
        {"candidate_name": "C_QUERYRMSE_PITCHER_COUNT", "kind": "ranker", "objective": "QueryRMSE", "group_definition": "pitcher_id_count_state", "calibration": "score Platt + fixed V2 strength"},
        {"candidate_name": "D_QUERYRMSE_PITCHER_GAME_COUNT", "kind": "ranker", "objective": "QueryRMSE", "group_definition": "pitcher_id_game_type_count_state", "calibration": "score Platt + fixed V2 strength"},
    ]


def evaluate_candidate(df, spec):
    fold_rows = []
    ranking_rows = []
    y2023_rows = []
    context_rows = []
    runtime_rows = []
    pred_parts = []
    year_rates = df.groupby("season")[TARGET_COL].mean().sort_index()
    for train_start, train_end, cal_year, valid_year in FOLDS:
        train = df[(df["season"] >= train_start) & (df["season"] <= train_end)].copy()
        cal = df[df["season"] == cal_year].copy()
        valid = df[df["season"] == valid_year].copy()
        x_train, x_cal, x_valid, y_train, y_cal, y_valid = prepare_features(train, cal, valid)
        t0 = time.time()
        if spec["kind"] == "classifier":
            model = CatBoostClassifier(**V2_CATBOOST_PARAMS)
            model.fit(x_train, y_train)
            score_cal = clip_prob(model.predict_proba(x_cal)[:, 1])
            score_valid = clip_prob(model.predict_proba(x_valid)[:, 1])
            platt = fit_score_platt(logit(score_cal), y_cal)
            platt_valid = apply_score_platt(platt, logit(score_valid))
            rank_score_valid = score_valid
            rank_score_cal = score_cal
        else:
            train_group = group_key(train, spec["group_definition"])
            x_sorted, y_sorted, group_sorted = sort_for_ranker(x_train, y_train, train_group)
            model = CatBoostRanker(**RANK_PARAMS)
            model.fit(Pool(x_sorted, y_sorted, group_id=group_sorted))
            rank_score_cal = model.predict(x_cal)
            rank_score_valid = model.predict(x_valid)
            platt = fit_score_platt(rank_score_cal, y_cal)
            platt_valid = apply_score_platt(platt, rank_score_valid)
        train_seconds = time.time() - t0
        final_valid = apply_v2_strength(platt_valid, target_rate_for_year(year_rates, valid_year))
        for mode, pred in [("platt", platt_valid), ("v2_strength", final_valid)]:
            fold_rows.append(
                {
                    "candidate_name": spec["candidate_name"],
                    "objective": spec["objective"],
                    "group_definition": spec["group_definition"],
                    "calibration_mode": mode,
                    "valid_year": valid_year,
                    **metric_dict(y_valid, pred, "metric"),
                }
            )
        valid_group = group_key(valid, spec["group_definition"]) if spec["kind"] == "ranker" else group_key(valid, "pitcher_id")
        for mode, score in [("raw_score", rank_score_valid), ("platt", platt_valid), ("v2_strength", final_valid)]:
            pa, n_pairs = sample_pairwise_accuracy(y_valid, score, valid_group, k=5, seed=RNG_SEED + valid_year)
            ranking_rows.append(
                {
                    "candidate_name": spec["candidate_name"],
                    "score_mode": mode,
                    "valid_year": valid_year,
                    "ranking_group_definition": spec["group_definition"] if spec["kind"] == "ranker" else "pitcher_id_diagnostic",
                    "roc_auc": float(roc_auc_score(y_valid, score)),
                    "average_precision": float(average_precision_score(y_valid, score)),
                    "sampled_pairwise_accuracy": pa,
                    "sampled_pair_count": n_pairs,
                }
            )
        if valid_year == 2023:
            y = y_valid.to_numpy()
            pred = final_valid
            med = np.median(pred)
            low = pred < med
            high = ~low
            y2023_rows.append(
                {
                    "candidate_name": spec["candidate_name"],
                    "auc": float(roc_auc_score(y, pred)),
                    "pairwise_accuracy": sample_pairwise_accuracy(y, pred, valid_group, k=5, seed=7)[0],
                    "brier": float(brier_score_loss(y, pred)),
                    "skill_margin": float(y.mean() * (1 - y.mean()) - brier_score_loss(y, pred)),
                    "pseudo_score": float(max(0.0, 100000.0 * (y.mean() * (1 - y.mean()) - brier_score_loss(y, pred)) / (y.mean() * (1 - y.mean())))),
                    "prediction_std": float(pred.std()),
                    "low_half_actual_rate": float(y[low].mean()),
                    "high_half_actual_rate": float(y[high].mean()),
                    "high_minus_low_actual_rate": float(y[high].mean() - y[low].mean()),
                }
            )
        valid_diag = valid.copy()
        valid_diag["pred"] = final_valid
        valid_diag["y"] = y_valid.to_numpy()
        valid_diag["pitcher_bucket"] = pd.cut(valid_diag["asof_pitcher_n"].fillna(0), bins=[-1, 50, 300, 1000, np.inf], labels=["lt50", "50_300", "300_1000", "1000_plus"])
        for ctx, col in [("game_type", "game_type"), ("count_state", "count_state"), ("pitcher_bucket", "pitcher_bucket")]:
            for key, g in valid_diag.groupby(col, observed=True):
                if len(g) < 1000 or g["y"].nunique() < 2:
                    continue
                context_rows.append(
                    {
                        "candidate_name": spec["candidate_name"],
                        "valid_year": valid_year,
                        "context": ctx,
                        "context_value": str(key),
                        "n": len(g),
                        "auc": float(roc_auc_score(g["y"], g["pred"])),
                        "prediction_std": float(g["pred"].std()),
                    }
                )
        runtime_rows.append(
            {
                "candidate_name": spec["candidate_name"],
                "valid_year": valid_year,
                "train_seconds": train_seconds,
                "inference_rows": len(valid),
                "model_size_proxy_trees": 220,
            }
        )
        pred_parts.append(
            pd.DataFrame(
                {
                    "row_id": valid["row_id"].to_numpy(),
                    "valid_year": valid_year,
                    "target": y_valid.to_numpy(),
                    "candidate_name": spec["candidate_name"],
                    "rank_score": rank_score_valid,
                    "platt_pred": platt_valid,
                    "final_pred": final_valid,
                }
            )
        )
        print(f"{spec['candidate_name']} valid={valid_year} auc={roc_auc_score(y_valid, final_valid):.6f} train_s={train_seconds:.1f}")
    return (
        pd.DataFrame(fold_rows),
        pd.DataFrame(ranking_rows),
        pd.DataFrame(y2023_rows),
        pd.DataFrame(context_rows),
        pd.DataFrame(runtime_rows),
        pd.concat(pred_parts, ignore_index=True),
    )


def summarize(fold_metrics, ranking_metrics):
    rows = []
    base = fold_metrics[(fold_metrics["candidate_name"] == "A_CLASSIFIER") & (fold_metrics["calibration_mode"] == "v2_strength")].set_index("valid_year")
    for name, g in fold_metrics[fold_metrics["calibration_mode"] == "v2_strength"].groupby("candidate_name", sort=False):
        by = g.set_index("valid_year")
        rg = ranking_metrics[(ranking_metrics["candidate_name"] == name) & (ranking_metrics["score_mode"] == "v2_strength")]
        row = {
            "candidate_name": name,
            "objective": g["objective"].iloc[0],
            "group_definition": g["group_definition"].iloc[0],
            "calibration_method": "Platt + fixed V2 strength",
            "mean_auc": float(g["metric_auc"].mean()),
            "mean_brier": float(g["metric_brier"].mean()),
            "mean_pairwise_accuracy": float(rg["sampled_pairwise_accuracy"].mean()),
            "positive_fold_count": int((g["metric_skill_margin"] > 0).sum()),
            "worst_skill_margin": float(g["metric_skill_margin"].min()),
        }
        for year in [2022, 2023, 2024]:
            row[f"auc_{year}"] = float(by.loc[year, "metric_auc"])
            row[f"pseudo_{year}"] = float(by.loc[year, "metric_pseudo_score"])
            row[f"skill_{year}"] = float(by.loc[year, "metric_skill_margin"])
            row[f"delta_auc_{year}_vs_v2"] = float(by.loc[year, "metric_auc"] - base.loc[year, "metric_auc"])
            row[f"delta_skill_{year}_vs_v2"] = float(by.loc[year, "metric_skill_margin"] - base.loc[year, "metric_skill_margin"])
            row[f"pairwise_acc_{year}"] = float(rg[rg["valid_year"] == year]["sampled_pairwise_accuracy"].iloc[0])
        row["delta_mean_auc_vs_v2"] = row["mean_auc"] - float(base["metric_auc"].mean())
        row["success_a"] = row["mean_auc"] >= 0.523
        row["success_b"] = row["delta_auc_2023_vs_v2"] >= 0.0005 and row["delta_skill_2023_vs_v2"] >= 0.0001 and row["pseudo_2022"] > 0 and row["pseudo_2024"] > 0
        row["success_c"] = row["pseudo_2024"] >= 25 and row["pseudo_2022"] > 0 and row["skill_2023"] >= float(base.loc[2023, "metric_skill_margin"])
        row["success_d"] = all(row[f"delta_auc_{year}_vs_v2"] > 0 for year in [2022, 2023, 2024])
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["mean_auc", "pseudo_2024", "skill_2023"], ascending=[False, False, False])


def prediction_correlation(preds, top_names):
    base = preds[preds["candidate_name"] == "A_CLASSIFIER"][["row_id", "valid_year", "target", "final_pred"]].rename(columns={"final_pred": "v2_pred"})
    rows = []
    for name in top_names:
        if name == "A_CLASSIFIER":
            continue
        cand = preds[preds["candidate_name"] == name][["row_id", "valid_year", "target", "final_pred"]].rename(columns={"final_pred": "cand_pred"})
        m = base.merge(cand, on=["row_id", "valid_year", "target"])
        for scope, g in [("overall", m), *[(str(y), gy) for y, gy in m.groupby("valid_year")]]:
            ev = (g["v2_pred"] - g["target"]) ** 2
            ec = (g["cand_pred"] - g["target"]) ** 2
            rows.append(
                {
                    "candidate_name": name,
                    "scope": scope,
                    "pearson_corr": float(np.corrcoef(g["v2_pred"], g["cand_pred"])[0, 1]),
                    "spearman_corr": float(spearmanr(g["v2_pred"], g["cand_pred"]).correlation),
                    "mean_abs_prediction_diff": float(np.mean(np.abs(g["v2_pred"] - g["cand_pred"]))),
                    "squared_error_corr": float(np.corrcoef(ev, ec)[0, 1]),
                }
            )
    return pd.DataFrame(rows)


def error_complementarity(preds, top_names):
    base = preds[preds["candidate_name"] == "A_CLASSIFIER"][["row_id", "valid_year", "target", "final_pred"]].rename(columns={"final_pred": "v2_pred"})
    rows = []
    for name in top_names:
        if name == "A_CLASSIFIER":
            continue
        cand = preds[preds["candidate_name"] == name][["row_id", "valid_year", "target", "final_pred"]].rename(columns={"final_pred": "cand_pred"})
        m = base.merge(cand, on=["row_id", "valid_year", "target"])
        for scope, g in [("overall", m), *[(str(y), gy) for y, gy in m.groupby("valid_year")]]:
            ev = (g["v2_pred"] - g["target"]) ** 2
            ec = (g["cand_pred"] - g["target"]) ** 2
            rows.append(
                {
                    "candidate_name": name,
                    "scope": scope,
                    "ranking_better_rate": float((ec < ev).mean()),
                    "v2_better_rate": float((ev < ec).mean()),
                    "tie_rate": float(np.isclose(ev, ec).mean()),
                    "both_bad_top_decile_rate": float(((ev >= np.quantile(ev, 0.9)) & (ec >= np.quantile(ec, 0.9))).mean()),
                }
            )
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = add_context_columns(pd.read_csv(TRAIN_PATH, encoding="utf-8-sig"))
    feasibility = ranking_group_feasibility(df)
    specs = candidate_specs()
    pair_stats = pd.concat(
        [pair_sampling_stats(df, s["candidate_name"], s["group_definition"]) for s in specs if s["kind"] == "ranker"],
        ignore_index=True,
    )
    fold_parts, rank_parts, y2023_parts, context_parts, runtime_parts, pred_parts = [], [], [], [], [], []
    for spec in specs:
        fm, rm, y23, ctx, rt, pred = evaluate_candidate(df, spec)
        fold_parts.append(fm)
        rank_parts.append(rm)
        y2023_parts.append(y23)
        context_parts.append(ctx)
        runtime_parts.append(rt)
        pred_parts.append(pred)
    fold_metrics = pd.concat(fold_parts, ignore_index=True)
    ranking_metrics = pd.concat(rank_parts, ignore_index=True)
    summary = summarize(fold_metrics, ranking_metrics)
    preds = pd.concat(pred_parts, ignore_index=True)
    top_names = summary["candidate_name"].head(3).tolist()
    corr = prediction_correlation(preds, top_names)
    err = error_complementarity(preds, top_names)
    verdict = "RANKING SIGNAL FOUND" if summary[["success_a", "success_b", "success_c", "success_d"]].any(axis=None) else "RANKING OBJECTIVE NOT USEFUL"
    feasibility.to_csv(os.path.join(OUT_DIR, "ranking_group_feasibility.csv"), index=False, encoding="utf-8")
    pair_stats.to_csv(os.path.join(OUT_DIR, "pair_sampling_stats.csv"), index=False, encoding="utf-8")
    fold_metrics.to_csv(os.path.join(OUT_DIR, "fold_metrics.csv"), index=False, encoding="utf-8")
    ranking_metrics.to_csv(os.path.join(OUT_DIR, "ranking_metrics.csv"), index=False, encoding="utf-8")
    summary.to_csv(os.path.join(OUT_DIR, "candidate_summary.csv"), index=False, encoding="utf-8")
    pd.concat(y2023_parts, ignore_index=True).to_csv(os.path.join(OUT_DIR, "year2023_analysis.csv"), index=False, encoding="utf-8")
    pd.concat(context_parts, ignore_index=True).to_csv(os.path.join(OUT_DIR, "context_ranking_analysis.csv"), index=False, encoding="utf-8")
    corr.to_csv(os.path.join(OUT_DIR, "prediction_correlation.csv"), index=False, encoding="utf-8")
    err.to_csv(os.path.join(OUT_DIR, "error_complementarity.csv"), index=False, encoding="utf-8")
    pd.concat(runtime_parts, ignore_index=True).to_csv(os.path.join(OUT_DIR, "runtime_summary.csv"), index=False, encoding="utf-8")
    pd.DataFrame([{"verdict": verdict}]).to_csv(os.path.join(OUT_DIR, "verdict.csv"), index=False, encoding="utf-8")
    preds.to_csv(os.path.join(OUT_DIR, "oof_predictions.csv"), index=False, encoding="utf-8")
    print(summary.to_string(index=False))
    print(verdict)


if __name__ == "__main__":
    main()
