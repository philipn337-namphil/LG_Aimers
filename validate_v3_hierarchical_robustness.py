import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from calibration_utils import apply_calibration, fit_calibrators, logit, sigmoid
from model_utils import TARGET_COL
from validate_v3_asof_signal_reconstruction import apply_v2_strength, clip_prob
from validate_v3_hierarchical import (
    FOLDS,
    add_context_columns,
    future_global_rate,
    group_key,
    metric_dict,
    predict_pitcher_context,
    v2_predictions,
)


OUT_DIR = "output/v3_hierarchical_robustness"
TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"
ALPHA_PITCHER_GRID = [50, 100, 200]
ALPHA_CONTEXT_GRID = [150, 300, 600]
MAIN_ALPHA_PITCHER = 100
MAIN_ALPHA_CONTEXT = 300
BOOTSTRAPS = 200
RNG_SEED = 42


def calibrate_strength(df, cal, valid, cal_pred, valid_pred, valid_year):
    calibrator = fit_calibrators(cal_pred, cal[TARGET_COL].astype(int), cal[["game_type"]])
    platt_valid = apply_calibration(valid_pred, valid[["game_type"]], calibrator, "platt")
    strength_valid = apply_v2_strength(platt_valid, future_global_rate(df, valid_year))
    return platt_valid, strength_valid


def context_native(df, train_end, cal_year, valid_year, context, alpha_pitcher, alpha_context, history_policy="all"):
    hist_cal = history_frame(df, cal_year, history_policy)
    hist_valid = history_frame(df, valid_year, history_policy)
    cal = df[df["season"] == cal_year].copy()
    valid = df[df["season"] == valid_year].copy()
    q_cal = future_global_rate(df, cal_year)
    q_valid = future_global_rate(df, valid_year)
    cal_pred, _, cal_context_count = predict_pitcher_context(hist_cal, cal, q_cal, context, alpha_pitcher, alpha_context)
    valid_pred, valid_pitcher_count, valid_context_count = predict_pitcher_context(hist_valid, valid, q_valid, context, alpha_pitcher, alpha_context)
    return cal_pred, valid_pred, valid_pitcher_count, valid_context_count, cal_context_count


def history_frame(df, prediction_year, policy):
    hist = df[df["season"] < prediction_year].copy()
    if policy == "recent2":
        return hist[hist["season"] >= prediction_year - 2].copy()
    if policy == "recent3":
        return hist[hist["season"] >= prediction_year - 3].copy()
    if policy == "all":
        return hist
    if policy == "recency_weighted":
        # Physical row duplication is avoided; weighted posterior is implemented separately.
        return hist
    raise ValueError(policy)


def recency_weighted_context(df, prediction_year, rows, context, alpha_pitcher=MAIN_ALPHA_PITCHER, alpha_context=MAIN_ALPHA_CONTEXT):
    hist = df[df["season"] < prediction_year].copy()
    if hist.empty:
        q = future_global_rate(df, prediction_year)
        return np.full(len(rows), q), np.zeros(len(rows)), np.zeros(len(rows))
    age = prediction_year - hist["season"].astype(int)
    weights = np.select([age == 1, age == 2, age == 3], [1.0, 0.7, 0.5], default=0.3)
    q = future_global_rate(df, prediction_year)
    pkey = group_key(hist, "pitcher")
    ph = pd.DataFrame({"key": pkey, "y": hist[TARGET_COL].astype(float), "w": weights})
    pg = ph.groupby("key").apply(lambda g: pd.Series({"wsum": float((g["y"] * g["w"]).sum()), "wcount": float(g["w"].sum())}), include_groups=False)
    pg["posterior"] = (pg["wsum"] + alpha_pitcher * q) / (pg["wcount"] + alpha_pitcher)
    row_pkey = group_key(rows, "pitcher")
    pitcher_pred = row_pkey.map(pg["posterior"].to_dict()).fillna(q).to_numpy(dtype=np.float64)
    pitcher_count = row_pkey.map(pg["wcount"].to_dict()).fillna(0.0).to_numpy(dtype=np.float64)
    parent = pkey.map(pg["posterior"].to_dict()).fillna(q).to_numpy(dtype=np.float64)
    ckey = group_key(hist, context)
    ch = pd.DataFrame({"key": ckey, "y": hist[TARGET_COL].astype(float), "w": weights, "parent": parent})
    cg = ch.groupby("key").apply(
        lambda g: pd.Series(
            {
                "wsum": float((g["y"] * g["w"]).sum()),
                "wcount": float(g["w"].sum()),
                "parent": float(np.average(g["parent"], weights=g["w"])),
            }
        ),
        include_groups=False,
    )
    cg["posterior"] = (cg["wsum"] + alpha_context * cg["parent"]) / (cg["wcount"] + alpha_context)
    rkey = group_key(rows, context)
    pred = rkey.map(cg["posterior"].to_dict()).fillna(pd.Series(pitcher_pred, index=rows.index)).to_numpy(dtype=np.float64)
    context_count = rkey.map(cg["wcount"].to_dict()).fillna(0.0).to_numpy(dtype=np.float64)
    return clip_prob(pred), pitcher_count, context_count


def effect_map(history, q, context, alpha_pitcher, alpha_context):
    p_pitcher, _, _ = predict_pitcher_context(history, history, q, context, alpha_pitcher, alpha_context)
    parent, _ = predict_pitcher_context(history, history, q, "pitcher_game", alpha_pitcher, alpha_context)[:2] if context == "pitcher_game" else (None, None)
    if parent is None:
        pmap, _ = predict_pitcher_only_for_rows(history, history, q, alpha_pitcher)
        parent = pmap
    eff = logit(p_pitcher) - logit(parent)
    key = group_key(history, context)
    grouped = pd.DataFrame({"key": key, "effect": eff}).groupby("key")["effect"].mean()
    counts = pd.DataFrame({"key": key}).groupby("key").size()
    return grouped.to_dict(), counts.to_dict()


def predict_pitcher_only_for_rows(history, rows, q, alpha_pitcher):
    grouped = pd.DataFrame({"key": group_key(history, "pitcher"), "y": history[TARGET_COL].astype(float)}).groupby("key")["y"].agg(["sum", "count"])
    grouped["posterior"] = (grouped["sum"] + alpha_pitcher * q) / (grouped["count"] + alpha_pitcher)
    keys = group_key(rows, "pitcher")
    return keys.map(grouped["posterior"].to_dict()).fillna(q).to_numpy(dtype=np.float64), keys.map(grouped["count"].to_dict()).fillna(0).to_numpy(dtype=np.float64)


def combined_cd_prediction(df, prediction_year, rows, w_game, w_hand, alpha_pitcher=MAIN_ALPHA_PITCHER, alpha_context=MAIN_ALPHA_CONTEXT):
    hist = df[df["season"] < prediction_year].copy()
    q = future_global_rate(df, prediction_year)
    p_pitcher, pitcher_count = predict_pitcher_only_for_rows(hist, rows, q, alpha_pitcher)
    logits = logit(q) + (logit(p_pitcher) - logit(q))
    for context, weight in [("pitcher_game", w_game), ("pitcher_batter_hand", w_hand)]:
        hist_context, hist_parent_count, hist_context_count = predict_pitcher_context(hist, hist, q, context, alpha_pitcher, alpha_context)
        hist_parent, _ = predict_pitcher_only_for_rows(hist, hist, q, alpha_pitcher)
        eff_df = pd.DataFrame(
            {
                "key": group_key(hist, context),
                "effect": logit(hist_context) - logit(hist_parent),
                "count": hist_context_count,
            }
        ).groupby("key").agg(effect=("effect", "mean"), count=("count", "max"))
        rkey = group_key(rows, context)
        row_eff = rkey.map(eff_df["effect"].to_dict()).fillna(0.0).to_numpy(dtype=np.float64)
        row_count = rkey.map(eff_df["count"].to_dict()).fillna(0.0).to_numpy(dtype=np.float64)
        reliability = row_count / (row_count + alpha_context)
        logits += weight * reliability * row_eff
    return clip_prob(sigmoid(np.clip(logits, logit(q) - 0.25, logit(q) + 0.25))), pitcher_count


def evaluate_candidate_rows(name, df, mode_builder, v2_cache):
    fold_rows, pred_parts, coverage_rows = [], [], []
    for train_start, train_end, cal_year, valid_year in FOLDS:
        cal = df[df["season"] == cal_year].copy()
        valid = df[df["season"] == valid_year].copy()
        y = valid[TARGET_COL].astype(int).to_numpy()
        v2_final = v2_cache[valid_year]["v2_strength"]
        v2_platt = v2_cache[valid_year]["platt"]
        cal_pred, native, pitcher_count, context_count, meta = mode_builder(train_end, cal_year, valid_year, cal, valid, v2_final, v2_platt)
        platt, strength = calibrate_strength(df, cal, valid, cal_pred, native, valid_year)
        for calibration_mode, pred in [("native", native), ("platt", platt), ("v2_strength", strength)]:
            fold_rows.append({"candidate_name": name, "calibration_mode": calibration_mode, "valid_year": valid_year, **metric_dict(y, pred, "metric")})
        pred_parts.append(
            pd.DataFrame(
                {
                    "row_id": valid["row_id"].to_numpy(),
                    "valid_year": valid_year,
                    "target": y,
                    "candidate_name": name,
                    "native_pred": native,
                    "platt_pred": platt,
                    "v2_strength_pred": strength,
                    "v2_pred": v2_final,
                    "pitcher_id": valid["pitcher_id"].astype(str).to_numpy(),
                    "game_type": valid["game_type"].astype(str).to_numpy(),
                    "batter_hand": valid["batter_hand"].astype(str).to_numpy(),
                    "hist_pitcher_count": pitcher_count,
                    "hist_context_count": context_count,
                }
            )
        )
        coverage_rows.append(
            {
                "candidate_name": name,
                "valid_year": valid_year,
                "hierarchy_available_rate": float((pitcher_count > 0).mean()),
                "pitcher_unseen_rate": float((pitcher_count == 0).mean()),
                "context_unseen_rate": float((context_count == 0).mean()),
                "fallback_used_rate": float(meta.get("fallback_used_rate", 0.0)),
                "full_hierarchy_used_rate": float(((pitcher_count > 0) & (context_count > 0)).mean()),
            }
        )
    return pd.DataFrame(fold_rows), pd.concat(pred_parts, ignore_index=True), pd.DataFrame(coverage_rows)


def build_v2_cache(df):
    fold_rows, pred_parts, cache = [], [], {}
    for train_start, train_end, cal_year, valid_year in FOLDS:
        train = df[(df["season"] >= train_start) & (df["season"] <= train_end)].copy()
        cal = df[df["season"] == cal_year].copy()
        valid = df[df["season"] == valid_year].copy()
        _, _, platt, _, y_valid = v2_predictions(train, cal, valid)
        strength = apply_v2_strength(platt, future_global_rate(df, valid_year))
        cache[valid_year] = {"platt": platt, "v2_strength": strength}
        for mode, pred in [("platt", platt), ("v2_strength", strength)]:
            fold_rows.append({"candidate_name": "V2_BASELINE", "calibration_mode": mode, "valid_year": valid_year, **metric_dict(y_valid, pred, "metric")})
        pred_parts.append(
            pd.DataFrame(
                {
                    "row_id": valid["row_id"].to_numpy(),
                    "valid_year": valid_year,
                    "target": y_valid.to_numpy(),
                    "candidate_name": "V2_BASELINE",
                    "native_pred": platt,
                    "platt_pred": platt,
                    "v2_strength_pred": strength,
                    "v2_pred": strength,
                    "pitcher_id": valid["pitcher_id"].astype(str).to_numpy(),
                    "game_type": valid["game_type"].astype(str).to_numpy(),
                    "batter_hand": valid["batter_hand"].astype(str).to_numpy(),
                    "hist_pitcher_count": valid["asof_pitcher_n"].fillna(0).to_numpy(),
                    "hist_context_count": np.nan,
                }
            )
        )
    return cache, pd.DataFrame(fold_rows), pd.concat(pred_parts, ignore_index=True)


def context_builder(context, alpha_pitcher=MAIN_ALPHA_PITCHER, alpha_context=MAIN_ALPHA_CONTEXT, history_policy="all"):
    def _builder(train_end, cal_year, valid_year, cal, valid, v2_final, v2_platt):
        if history_policy == "recency_weighted":
            cal_pred, _, cal_context = recency_weighted_context(pd.concat([cal.iloc[0:0], GLOBAL_DF]), cal_year, cal, context, alpha_pitcher, alpha_context)
            valid_pred, pitcher_count, context_count = recency_weighted_context(GLOBAL_DF, valid_year, valid, context, alpha_pitcher, alpha_context)
        else:
            cal_pred, valid_pred, pitcher_count, context_count, cal_context = context_native(GLOBAL_DF, train_end, cal_year, valid_year, context, alpha_pitcher, alpha_context, history_policy)
        return cal_pred, valid_pred, pitcher_count, context_count, {}

    return _builder


def hard_fallback_builder(context, threshold):
    base_builder = context_builder(context)

    def _builder(train_end, cal_year, valid_year, cal, valid, v2_final, v2_platt):
        cal_pred, native, pitcher_count, context_count, meta = base_builder(train_end, cal_year, valid_year, cal, valid, v2_final, v2_platt)
        mask = pitcher_count < threshold
        native = native.copy()
        native[mask] = v2_platt[mask]
        meta["fallback_used_rate"] = float(mask.mean())
        return cal_pred, native, pitcher_count, context_count, meta

    return _builder


def soft_blend_builder(context, k, logit_space=False):
    base_builder = context_builder(context)

    def _builder(train_end, cal_year, valid_year, cal, valid, v2_final, v2_platt):
        cal_pred, native, pitcher_count, context_count, meta = base_builder(train_end, cal_year, valid_year, cal, valid, v2_final, v2_platt)
        reliability = pitcher_count / (pitcher_count + k)
        if logit_space:
            native = clip_prob(sigmoid(reliability * logit(native) + (1.0 - reliability) * logit(v2_platt)))
        else:
            native = clip_prob(reliability * native + (1.0 - reliability) * v2_platt)
        meta["fallback_used_rate"] = float((reliability < 0.5).mean())
        return cal_pred, native, pitcher_count, context_count, meta

    return _builder


def combo_builder(w_game, w_hand):
    def _builder(train_end, cal_year, valid_year, cal, valid, v2_final, v2_platt):
        cal_pred, _ = combined_cd_prediction(GLOBAL_DF, cal_year, cal, w_game, w_hand)
        native, pitcher_count = combined_cd_prediction(GLOBAL_DF, valid_year, valid, w_game, w_hand)
        # Context count is diagnostic; take max of game and hand counts.
        _, _, game_count = predict_pitcher_context(GLOBAL_DF[GLOBAL_DF["season"] < valid_year], valid, future_global_rate(GLOBAL_DF, valid_year), "pitcher_game")
        _, _, hand_count = predict_pitcher_context(GLOBAL_DF[GLOBAL_DF["season"] < valid_year], valid, future_global_rate(GLOBAL_DF, valid_year), "pitcher_batter_hand")
        return cal_pred, native, pitcher_count, np.maximum(game_count, hand_count), {}

    return _builder


def summarize_fold_metrics(fold_metrics):
    base = fold_metrics[(fold_metrics["candidate_name"] == "V2_BASELINE") & (fold_metrics["calibration_mode"] == "v2_strength")].set_index("valid_year")
    rows = []
    for (name, mode), g in fold_metrics.groupby(["candidate_name", "calibration_mode"], sort=False):
        by = g.set_index("valid_year")
        row = {
            "candidate_name": name,
            "calibration_mode": mode,
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
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["mean_auc", "pseudo_2024", "skill_2023"], ascending=[False, False, False])


def sample_bucket_analysis(preds):
    rows = []
    base = preds[preds["candidate_name"] == "V2_BASELINE"][["row_id", "valid_year", "v2_strength_pred"]].rename(columns={"v2_strength_pred": "base_pred"})
    bins = [-1, 25, 50, 100, 300, np.inf]
    labels = ["lt25", "25_49", "50_99", "100_299", "300_plus"]
    for name, g0 in preds.groupby("candidate_name"):
        g = g0.merge(base, on=["row_id", "valid_year"], how="left")
        g["bucket"] = pd.cut(g["hist_pitcher_count"], bins=bins, labels=labels)
        for (year, bucket), b in g.groupby(["valid_year", "bucket"], observed=True):
            if len(b) < 500:
                continue
            y = b["target"].to_numpy()
            p = b["v2_strength_pred"].to_numpy()
            v2 = b["base_pred"].to_numpy()
            rows.append(
                {
                    "candidate_name": name,
                    "valid_year": year,
                    "sample_bucket": str(bucket),
                    "n": int(len(b)),
                    "v2_brier": float(brier_score_loss(y, v2)),
                    "candidate_brier": float(brier_score_loss(y, p)),
                    "delta_brier_candidate_minus_v2": float(brier_score_loss(y, p) - brier_score_loss(y, v2)),
                    "candidate_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan,
                    "skill_contribution": float(np.mean((v2 - y) ** 2 - (p - y) ** 2)),
                }
            )
    return pd.DataFrame(rows)


def stress_analysis(preds, year):
    rows = []
    base = preds[preds["candidate_name"] == "V2_BASELINE"][["row_id", "v2_strength_pred"]].rename(columns={"v2_strength_pred": "base_pred"})
    for name, g0 in preds[preds["valid_year"] == year].groupby("candidate_name"):
        g = g0.merge(base, on="row_id", how="left")
        g["sample_bucket"] = pd.cut(g["hist_pitcher_count"], bins=[-1, 25, 50, 100, 300, np.inf], labels=["lt25", "25_49", "50_99", "100_299", "300_plus"])
        g["coverage"] = np.where(g["hist_context_count"].fillna(0) > 0, "full_hierarchy", "fallback_parent")
        for col in ["sample_bucket", "game_type", "batter_hand", "coverage"]:
            for key, b in g.groupby(col, observed=True):
                if len(b) < 500:
                    continue
                y = b["target"].to_numpy()
                p = b["v2_strength_pred"].to_numpy()
                v2 = b["base_pred"].to_numpy()
                constant = y.mean() * (1 - y.mean())
                rows.append(
                    {
                        "candidate_name": name,
                        "valid_year": year,
                        "segment_type": col,
                        "segment": str(key),
                        "n": int(len(b)),
                        "skill_margin": float(constant - brier_score_loss(y, p)),
                        "delta_brier_vs_v2": float(brier_score_loss(y, p) - brier_score_loss(y, v2)),
                        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def bootstrap_summary(preds, top_candidates, cluster=False):
    rng = np.random.default_rng(RNG_SEED + (1000 if cluster else 0))
    rows = []
    base = preds[preds["candidate_name"] == "V2_BASELINE"][["row_id", "valid_year", "target", "v2_strength_pred", "pitcher_id"]].rename(columns={"v2_strength_pred": "v2_pred"})
    for cand in top_candidates:
        c = preds[preds["candidate_name"] == cand][["row_id", "valid_year", "target", "v2_strength_pred", "pitcher_id"]].rename(columns={"v2_strength_pred": "cand_pred"})
        m = base.merge(c, on=["row_id", "valid_year", "target", "pitcher_id"])
        for year, g in m.groupby("valid_year"):
            g = g.reset_index(drop=True)
            deltas = []
            if cluster:
                pitchers = g["pitcher_id"].unique()
                groups = {p: idx.to_numpy() for p, idx in g.groupby("pitcher_id").groups.items()}
                for _ in range(BOOTSTRAPS):
                    chosen = rng.choice(pitchers, size=len(pitchers), replace=True)
                    idx = np.concatenate([groups[p] for p in chosen])
                    s = g.iloc[idx]
                    deltas.append(float(np.mean((s["v2_pred"] - s["target"]) ** 2 - (s["cand_pred"] - s["target"]) ** 2)))
            else:
                n = len(g)
                for _ in range(BOOTSTRAPS):
                    idx = rng.integers(0, n, size=n)
                    s = g.iloc[idx]
                    deltas.append(float(np.mean((s["v2_pred"] - s["target"]) ** 2 - (s["cand_pred"] - s["target"]) ** 2)))
            rows.append(
                {
                    "candidate_name": cand,
                    "valid_year": year,
                    "bootstrap_type": "pitcher_cluster" if cluster else "row",
                    "mean_delta_brier_v2_minus_candidate": float(np.mean(deltas)),
                    "median_delta_brier": float(np.median(deltas)),
                    "p05_delta_brier": float(np.quantile(deltas, 0.05)),
                    "p95_delta_brier": float(np.quantile(deltas, 0.95)),
                    "p_candidate_beats_v2": float(np.mean(np.asarray(deltas) > 0)),
                }
            )
    return pd.DataFrame(rows)


def improvement_concentration(preds, top_candidates):
    rows = []
    base = preds[preds["candidate_name"] == "V2_BASELINE"][["row_id", "valid_year", "target", "v2_strength_pred", "pitcher_id"]].rename(columns={"v2_strength_pred": "v2_pred"})
    for cand in top_candidates:
        c = preds[preds["candidate_name"] == cand][["row_id", "valid_year", "target", "v2_strength_pred", "pitcher_id"]].rename(columns={"v2_strength_pred": "cand_pred"})
        m = base.merge(c, on=["row_id", "valid_year", "target", "pitcher_id"])
        m["gain"] = (m["v2_pred"] - m["target"]) ** 2 - (m["cand_pred"] - m["target"]) ** 2
        pg = m.groupby("pitcher_id")["gain"].sum().sort_values(ascending=False)
        total = float(pg.sum())
        for pct in [0.01, 0.05, 0.10]:
            k = max(1, int(np.ceil(len(pg) * pct)))
            rows.append(
                {
                    "candidate_name": cand,
                    "top_pitcher_fraction": pct,
                    "pitcher_count": k,
                    "total_gain": total,
                    "top_pitcher_gain": float(pg.head(k).sum()),
                    "top_pitcher_gain_share": float(pg.head(k).sum() / total) if total != 0 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def prediction_correlation(preds, top_candidates):
    rows = []
    base = preds[preds["candidate_name"] == "V2_BASELINE"][["row_id", "valid_year", "target", "v2_strength_pred"]].rename(columns={"v2_strength_pred": "v2_pred"})
    for cand in top_candidates:
        c = preds[preds["candidate_name"] == cand][["row_id", "valid_year", "target", "v2_strength_pred"]].rename(columns={"v2_strength_pred": "cand_pred"})
        m = base.merge(c, on=["row_id", "valid_year", "target"])
        for scope, g in [("overall", m), *[(str(y), gy) for y, gy in m.groupby("valid_year")]]:
            ev = (g["v2_pred"] - g["target"]) ** 2
            ec = (g["cand_pred"] - g["target"]) ** 2
            rows.append(
                {
                    "candidate_name": cand,
                    "scope": scope,
                    "pearson_corr": float(np.corrcoef(g["v2_pred"], g["cand_pred"])[0, 1]),
                    "spearman_corr": float(spearmanr(g["v2_pred"], g["cand_pred"]).correlation),
                    "mean_abs_prediction_diff": float(np.mean(np.abs(g["v2_pred"] - g["cand_pred"]))),
                    "squared_error_corr": float(np.corrcoef(ev, ec)[0, 1]),
                }
            )
    return pd.DataFrame(rows)


def coverage_2025(df):
    if not os.path.exists(TEST_PATH):
        return pd.DataFrame()
    test = add_context_columns(pd.read_csv(TEST_PATH, encoding="utf-8-sig"))
    hist = df[df["season"] <= 2024].copy()
    rows = []
    for name, context in [("C_PITCHER_GAME", "pitcher_game"), ("D_PITCHER_BATTER_HAND", "pitcher_batter_hand")]:
        pkey = group_key(test, "pitcher")
        ckey = group_key(test, context)
        hist_p = set(group_key(hist, "pitcher"))
        hist_c = set(group_key(hist, context))
        rows.append(
            {
                "candidate_name": name,
                "valid_year": 2025,
                "hierarchy_available_rate": float(pkey.isin(hist_p).mean()),
                "pitcher_unseen_rate": float((~pkey.isin(hist_p)).mean()),
                "context_unseen_rate": float((~ckey.isin(hist_c)).mean()),
                "fallback_used_rate": np.nan,
                "full_hierarchy_used_rate": float((pkey.isin(hist_p) & ckey.isin(hist_c)).mean()),
            }
        )
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    global GLOBAL_DF
    GLOBAL_DF = add_context_columns(pd.read_csv(TRAIN_PATH, encoding="utf-8-sig"))
    v2_cache, v2_folds, v2_preds = build_v2_cache(GLOBAL_DF)
    all_fold = [v2_folds]
    all_preds = [v2_preds]
    all_cov = []
    smoothing_rows = []
    for context_name, context in [("C_PITCHER_GAME", "pitcher_game"), ("D_PITCHER_BATTER_HAND", "pitcher_batter_hand")]:
        for ap in ALPHA_PITCHER_GRID:
            for ac in ALPHA_CONTEXT_GRID:
                name = f"{context_name}_ap{ap}_ac{ac}"
                fm, pr, cov = evaluate_candidate_rows(name, GLOBAL_DF, context_builder(context, ap, ac), v2_cache)
                all_fold.append(fm)
                all_preds.append(pr)
                all_cov.append(cov)
                smoothing_rows.append(summarize_fold_metrics(pd.concat([v2_folds, fm], ignore_index=True)).assign(alpha_pitcher=ap, alpha_context=ac, context=context_name))
                print(name)
    fallback_names = []
    for context_name, context in [("C_PITCHER_GAME", "pitcher_game"), ("D_PITCHER_BATTER_HAND", "pitcher_batter_hand")]:
        for threshold in [25, 50, 100]:
            name = f"{context_name}_fallback{threshold}"
            fm, pr, cov = evaluate_candidate_rows(name, GLOBAL_DF, hard_fallback_builder(context, threshold), v2_cache)
            all_fold.append(fm)
            all_preds.append(pr)
            all_cov.append(cov)
            fallback_names.append(name)
            print(name)
    soft_names = []
    for context_name, context in [("C_PITCHER_GAME", "pitcher_game"), ("D_PITCHER_BATTER_HAND", "pitcher_batter_hand")]:
        for k in [50, 100, 300]:
            for logit_space in [False, True]:
                name = f"{context_name}_{'logit' if logit_space else 'prob'}blend{k}"
                fm, pr, cov = evaluate_candidate_rows(name, GLOBAL_DF, soft_blend_builder(context, k, logit_space), v2_cache)
                all_fold.append(fm)
                all_preds.append(pr)
                all_cov.append(cov)
                soft_names.append(name)
                print(name)
    combo_names = []
    for w_game, w_hand in [(1.0, 0.0), (0.0, 1.0), (0.5, 0.5), (0.75, 0.25), (0.25, 0.75)]:
        name = f"CD_COMBO_g{w_game:.2f}_h{w_hand:.2f}".replace(".", "p")
        fm, pr, cov = evaluate_candidate_rows(name, GLOBAL_DF, combo_builder(w_game, w_hand), v2_cache)
        all_fold.append(fm)
        all_preds.append(pr)
        all_cov.append(cov)
        combo_names.append(name)
        print(name)
    recency_names = []
    for context_name, context in [("C_PITCHER_GAME", "pitcher_game"), ("D_PITCHER_BATTER_HAND", "pitcher_batter_hand")]:
        for policy in ["recent2", "recent3", "recency_weighted"]:
            name = f"{context_name}_{policy}"
            fm, pr, cov = evaluate_candidate_rows(name, GLOBAL_DF, context_builder(context, history_policy=policy), v2_cache)
            all_fold.append(fm)
            all_preds.append(pr)
            all_cov.append(cov)
            recency_names.append(name)
            print(name)
    fold_metrics = pd.concat(all_fold, ignore_index=True)
    preds = pd.concat(all_preds, ignore_index=True)
    coverage = pd.concat(all_cov + [coverage_2025(GLOBAL_DF)], ignore_index=True)
    summary = summarize_fold_metrics(fold_metrics)
    # Keep top candidates from distinct policy families for expensive robustness summaries.
    top_candidates = (
        summary[(summary["calibration_mode"] == "v2_strength") & (summary["candidate_name"] != "V2_BASELINE")]
        .sort_values(["mean_auc", "pseudo_2024", "skill_2023"], ascending=[False, False, False])
        ["candidate_name"]
        .head(3)
        .tolist()
    )
    smoothing_grid = pd.concat(smoothing_rows, ignore_index=True)
    fallback_metrics = summary[summary["candidate_name"].isin(fallback_names)]
    soft_blend_metrics = summary[summary["candidate_name"].isin(soft_names)]
    context_combination_metrics = summary[summary["candidate_name"].isin(combo_names)]
    recency_metrics = summary[summary["candidate_name"].isin(recency_names)]
    bucket = sample_bucket_analysis(preds[preds["candidate_name"].isin(["V2_BASELINE"] + top_candidates + fallback_names[:2])])
    y2023 = stress_analysis(preds[preds["candidate_name"].isin(["V2_BASELINE"] + top_candidates)], 2023)
    y2024 = stress_analysis(preds[preds["candidate_name"].isin(["V2_BASELINE"] + top_candidates)], 2024)
    row_boot = bootstrap_summary(preds, top_candidates, cluster=False)
    pitcher_boot = bootstrap_summary(preds, top_candidates, cluster=True)
    concentration = improvement_concentration(preds, top_candidates)
    corr = prediction_correlation(preds, top_candidates)
    v2 = summary[(summary["candidate_name"] == "V2_BASELINE") & (summary["calibration_mode"] == "v2_strength")].iloc[0]
    plateau = smoothing_grid[
        (smoothing_grid["calibration_mode"] == "v2_strength")
        & (smoothing_grid["mean_auc"] >= 0.527)
        & (smoothing_grid["skill_2023"] >= float(v2["skill_2023"]) + 0.0001)
        & (smoothing_grid["pseudo_2022"] > 0)
        & (smoothing_grid["pseudo_2024"] > 0)
    ]
    top = summary[(summary["candidate_name"].isin(top_candidates)) & (summary["calibration_mode"] == "v2_strength")].copy()
    boot_win = row_boot.groupby("candidate_name")["p_candidate_beats_v2"].mean().to_dict()
    top["row_bootstrap_mean_win_prob"] = top["candidate_name"].map(boot_win)
    top["v3_ready"] = (
        (top["mean_auc"] >= 0.527)
        & (top["pseudo_2022"] > 0)
        & (top["pseudo_2024"] > 0)
        & (top["skill_2023"] >= float(v2["skill_2023"]) + 0.0001)
        & (len(plateau) >= 4)
        & ((top["pseudo_2024"] >= 50) | (top["row_bootstrap_mean_win_prob"] >= 0.70))
    )
    verdict = "READY FOR V3 ARTIFACT" if bool(top["v3_ready"].any()) else "HIERARCHICAL SIGNAL REAL BUT NOT ROBUST ENOUGH"
    fold_metrics.to_csv(os.path.join(OUT_DIR, "fold_metrics.csv"), index=False, encoding="utf-8")
    summary.to_csv(os.path.join(OUT_DIR, "candidate_summary.csv"), index=False, encoding="utf-8")
    smoothing_grid.to_csv(os.path.join(OUT_DIR, "smoothing_grid.csv"), index=False, encoding="utf-8")
    fallback_metrics.to_csv(os.path.join(OUT_DIR, "fallback_metrics.csv"), index=False, encoding="utf-8")
    soft_blend_metrics.to_csv(os.path.join(OUT_DIR, "soft_blend_metrics.csv"), index=False, encoding="utf-8")
    context_combination_metrics.to_csv(os.path.join(OUT_DIR, "context_combination_metrics.csv"), index=False, encoding="utf-8")
    recency_metrics.to_csv(os.path.join(OUT_DIR, "recency_weighting_metrics.csv"), index=False, encoding="utf-8")
    bucket.to_csv(os.path.join(OUT_DIR, "sample_size_bucket_analysis.csv"), index=False, encoding="utf-8")
    coverage.to_csv(os.path.join(OUT_DIR, "coverage_analysis.csv"), index=False, encoding="utf-8")
    y2023.to_csv(os.path.join(OUT_DIR, "year2023_stress.csv"), index=False, encoding="utf-8")
    y2024.to_csv(os.path.join(OUT_DIR, "year2024_stress.csv"), index=False, encoding="utf-8")
    row_boot.to_csv(os.path.join(OUT_DIR, "bootstrap_summary.csv"), index=False, encoding="utf-8")
    pitcher_boot.to_csv(os.path.join(OUT_DIR, "pitcher_bootstrap_summary.csv"), index=False, encoding="utf-8")
    concentration.to_csv(os.path.join(OUT_DIR, "improvement_concentration.csv"), index=False, encoding="utf-8")
    corr.to_csv(os.path.join(OUT_DIR, "prediction_correlation.csv"), index=False, encoding="utf-8")
    top.to_csv(os.path.join(OUT_DIR, "v3_candidate_selection.csv"), index=False, encoding="utf-8")
    pd.DataFrame([{"verdict": verdict, "plateau_config_count": int(len(plateau)), "top_candidates": ";".join(top_candidates)}]).to_csv(
        os.path.join(OUT_DIR, "verdict.csv"), index=False, encoding="utf-8"
    )
    preds.to_csv(os.path.join(OUT_DIR, "oof_predictions.csv"), index=False, encoding="utf-8")
    print(top.to_string(index=False))
    print(verdict)


if __name__ == "__main__":
    main()
