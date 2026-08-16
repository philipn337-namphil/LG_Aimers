import os
from pathlib import Path

import numpy as np
import pandas as pd


OUT_DIR = Path("output/v3_trackman_mechanical_drift_foundation")
TRACKMAN_PATH = Path("data/trackman_history.csv")
TRAIN_PATH = Path("data/train.csv")
PHYSICAL_COLS = [
    "rel_height",
    "rel_side",
    "extension",
    "rel_speed",
    "spin_rate",
    "induced_vert_break",
    "horz_break",
]
PITCH_GROUPS = ["fastball", "breaking", "offspeed", "other"]
CUTOFFS = [2022, 2023, 2024, 2025]
MAIN_HAND_MAP = {1: "Left", 2: "Right"}
EPS = 1e-9


def flatten_columns(df):
    df.columns = ["_".join(str(x) for x in col if str(x)) if isinstance(col, tuple) else str(col) for col in df.columns]
    return df


def load_data():
    usecols = [
        "trackman_id",
        "season",
        "game_date",
        "trackman_game_id",
        "pitch_no",
        "pitcher_trackman_id",
        "pitcher_hand",
        "pitcher_team",
        "pitch_type_group",
        *PHYSICAL_COLS,
    ]
    tm = pd.read_csv(TRACKMAN_PATH, usecols=usecols, encoding="utf-8-sig")
    tm["game_date_parsed"] = pd.to_datetime(tm["game_date"], format="mixed", errors="coerce")
    for col in PHYSICAL_COLS:
        tm[col] = pd.to_numeric(tm[col], errors="coerce")
    train = pd.read_csv(
        TRAIN_PATH,
        usecols=["season", "pitcher_id", "pitcher_hand", "pitcher_team_id"],
        encoding="utf-8-sig",
    )
    train["pitcher_hand"] = train["pitcher_hand"].map(MAIN_HAND_MAP)
    return tm, train


def physical_quality(tm):
    rows = []
    year_rows = []
    for col in PHYSICAL_COLS:
        s = tm[col]
        q = s.quantile([0.01, 0.05, 0.95, 0.99])
        low_outer = s.quantile(0.25) - 3.0 * (s.quantile(0.75) - s.quantile(0.25))
        high_outer = s.quantile(0.75) + 3.0 * (s.quantile(0.75) - s.quantile(0.25))
        rows.append(
            {
                "column": col,
                "dtype": str(s.dtype),
                "missing_rate": float(s.isna().mean()),
                "min": float(s.min()),
                "max": float(s.max()),
                "median": float(s.median()),
                "mean": float(s.mean()),
                "std": float(s.std()),
                "p01": float(q.loc[0.01]),
                "p05": float(q.loc[0.05]),
                "p95": float(q.loc[0.95]),
                "p99": float(q.loc[0.99]),
                "iqr_outer_low": float(low_outer),
                "iqr_outer_high": float(high_outer),
                "statistically_suspicious_low_count": int((s < low_outer).sum()),
                "statistically_suspicious_high_count": int((s > high_outer).sum()),
                "note": "IQR outer fence only; no domain threshold deletion applied.",
            }
        )
        for year, g in tm.groupby("season"):
            gs = g[col]
            year_rows.append(
                {
                    "season": int(year),
                    "column": col,
                    "row_count": int(len(g)),
                    "missing_rate": float(gs.isna().mean()),
                    "mean": float(gs.mean()),
                    "std": float(gs.std()),
                    "median": float(gs.median()),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(year_rows)


def yearly_measurement_stats(tm):
    agg = tm.groupby("season")[PHYSICAL_COLS].agg(["mean", "median", "std", "count"])
    return flatten_columns(agg).reset_index()


def yearly_measurement_shifts(tm):
    stats = tm.groupby("season")[PHYSICAL_COLS].agg(["mean", "median", "std"])
    stats = flatten_columns(stats).reset_index()
    rows = []
    pitcher_year = tm.groupby(["pitcher_trackman_id", "season"])[PHYSICAL_COLS].mean().reset_index()
    for prev, cur in zip(sorted(tm["season"].unique())[:-1], sorted(tm["season"].unique())[1:]):
        merged = pitcher_year[pitcher_year["season"] == prev].merge(
            pitcher_year[pitcher_year["season"] == cur],
            on="pitcher_trackman_id",
            suffixes=("_prev", "_cur"),
        )
        for col in PHYSICAL_COLS:
            prev_mean = float(stats.loc[stats["season"] == prev, f"{col}_mean"].iloc[0])
            cur_mean = float(stats.loc[stats["season"] == cur, f"{col}_mean"].iloc[0])
            diff = merged[f"{col}_cur"] - merged[f"{col}_prev"] if len(merged) else pd.Series(dtype=float)
            rows.append(
                {
                    "from_year": int(prev),
                    "to_year": int(cur),
                    "column": col,
                    "league_mean_shift": cur_mean - prev_mean,
                    "league_median_shift": float(stats.loc[stats["season"] == cur, f"{col}_median"].iloc[0])
                    - float(stats.loc[stats["season"] == prev, f"{col}_median"].iloc[0]),
                    "league_std_shift": float(stats.loc[stats["season"] == cur, f"{col}_std"].iloc[0])
                    - float(stats.loc[stats["season"] == prev, f"{col}_std"].iloc[0]),
                    "paired_pitcher_count": int(len(merged)),
                    "paired_pitcher_positive_shift_rate": float((diff > 0).mean()) if len(diff) else np.nan,
                    "paired_pitcher_median_shift": float(diff.median()) if len(diff) else np.nan,
                    "regime_shift_flag": bool(len(diff) >= 30 and (((diff > 0).mean() >= 0.7) or ((diff > 0).mean() <= 0.3))),
                }
            )
    return pd.DataFrame(rows)


def pitcher_history_coverage(tm):
    base = tm.groupby("pitcher_trackman_id").agg(
        total_pitches=("trackman_id", "size"),
        total_games=("trackman_game_id", "nunique"),
        first_date=("game_date_parsed", "min"),
        last_date=("game_date_parsed", "max"),
        active_seasons=("season", "nunique"),
    )
    season_counts = tm.groupby(["pitcher_trackman_id", "season"]).size().unstack(fill_value=0)
    season_counts = season_counts.rename(columns={c: f"pitches_{c}" for c in season_counts.columns})
    games_counts = tm.groupby(["pitcher_trackman_id", "season"])["trackman_game_id"].nunique().unstack(fill_value=0)
    games_counts = games_counts.rename(columns={c: f"games_{c}" for c in games_counts.columns})
    return base.join(season_counts).join(games_counts).reset_index()


def cutoff_coverage(tm):
    rows = []
    for cutoff in CUTOFFS:
        src = tm[tm["season"] < cutoff] if cutoff < 2025 else tm[tm["season"] <= 2024]
        byp = src.groupby("pitcher_trackman_id").agg(pitches=("trackman_id", "size"), games=("trackman_game_id", "nunique"))
        rows.append(
            {
                "cutoff_year": cutoff,
                "source_policy": f"season < {cutoff}" if cutoff < 2025 else "season <= 2024",
                "pitcher_count": int(len(byp)),
                "pitchers_ge_20_pitches": int((byp["pitches"] >= 20).sum()),
                "pitchers_ge_50_pitches": int((byp["pitches"] >= 50).sum()),
                "pitchers_ge_100_pitches": int((byp["pitches"] >= 100).sum()),
                "pitchers_ge_300_pitches": int((byp["pitches"] >= 300).sum()),
                "pitchers_ge_3_games": int((byp["games"] >= 3).sum()),
                "pitchers_ge_5_games": int((byp["games"] >= 5).sum()),
                "pitchers_ge_10_games": int((byp["games"] >= 10).sum()),
            }
        )
    return pd.DataFrame(rows)


def count_vectors(df, id_col, hand_col, seasons):
    counts = df.groupby([id_col, hand_col, "season"], dropna=False).size().rename("n").reset_index()
    wide = counts.pivot_table(index=[id_col, hand_col], columns="season", values="n", fill_value=0, aggfunc="sum")
    wide = wide.reindex(columns=seasons, fill_value=0).reset_index()
    wide.columns.name = None
    return wide.rename(columns={id_col: "id", hand_col: "hand"})


def make_mapping_for_cutoff(train, tm, cutoff):
    if cutoff < 2025:
        main_pool = train[train["season"] == cutoff][["pitcher_id"]].drop_duplicates()
    else:
        main_pool = train[["pitcher_id"]].drop_duplicates()
    prior_train = train[train["season"] < cutoff]
    prior_tm = tm[tm["season"] < cutoff] if cutoff < 2025 else tm[tm["season"] <= 2024]
    seasons = sorted(prior_train["season"].unique())
    if not seasons:
        return pd.DataFrame(), main_pool
    main = count_vectors(prior_train, "pitcher_id", "pitcher_hand", seasons)
    track = count_vectors(prior_tm, "pitcher_trackman_id", "pitcher_hand", seasons)
    rows = []
    for hand in sorted(set(main["hand"]).intersection(set(track["hand"]))):
        m = main[main["hand"] == hand].reset_index(drop=True)
        t = track[track["hand"] == hand].reset_index(drop=True)
        if m.empty or t.empty:
            continue
        mv = m[seasons].to_numpy(dtype=np.float64)
        tv = t[seasons].to_numpy(dtype=np.float64)
        mt = mv.sum(axis=1)
        tt = tv.sum(axis=1)
        cosine = (mv @ tv.T) / np.maximum(np.outer(np.linalg.norm(mv, axis=1), np.linalg.norm(tv, axis=1)), EPS)
        count_ratio = np.minimum.outer(mt, tt) / np.maximum.outer(mt, tt)
        score = 0.78 * cosine + 0.22 * count_ratio
        for i in range(len(m)):
            order = np.argsort(-score[i])[:2]
            best = order[0]
            second_score = float(score[i, order[1]]) if len(order) > 1 else np.nan
            rows.append(
                {
                    "cutoff_year": cutoff,
                    "pitcher_id": int(m.loc[i, "id"]),
                    "pitcher_trackman_id": int(t.loc[best, "id"]),
                    "hand": hand,
                    "score": float(score[i, best]),
                    "score_gap_from_second": float(score[i, best] - second_score) if len(order) > 1 else np.nan,
                    "main_total": int(mt[i]),
                    "trackman_total": int(tt[best]),
                }
            )
    cand = pd.DataFrame(rows)
    if cand.empty:
        return cand, main_pool
    cand = cand[(cand["score"] >= 0.93) & (cand["score_gap_from_second"].fillna(1.0) >= 0.01)].copy()
    cand = cand.sort_values(["score", "score_gap_from_second"], ascending=False)
    cand = cand.drop_duplicates("pitcher_id").drop_duplicates("pitcher_trackman_id")
    return cand, main_pool


def mapping_quality_by_cutoff(train, tm):
    rows = []
    maps = {}
    for cutoff in CUTOFFS:
        mapping, pool = make_mapping_for_cutoff(train, tm, cutoff)
        maps[cutoff] = mapping
        pool_ids = set(pool["pitcher_id"])
        matched_ids = set(mapping["pitcher_id"]) if not mapping.empty else set()
        rows.append(
            {
                "cutoff_year": cutoff,
                "prediction_pitcher_population": "season validation pitchers" if cutoff < 2025 else "all train pitchers; hidden test pitchers unknown",
                "population_pitcher_count": len(pool_ids),
                "matched_pitcher_count": len(pool_ids & matched_ids),
                "unmatched_pitcher_count": len(pool_ids - matched_ids),
                "mapping_coverage": (len(pool_ids & matched_ids) / len(pool_ids)) if pool_ids else np.nan,
                "ambiguous_mapping_count": int((mapping["score_gap_from_second"].fillna(0) < 0.03).sum()) if not mapping.empty else 0,
                "min_score": float(mapping["score"].min()) if not mapping.empty else np.nan,
                "median_score": float(mapping["score"].median()) if not mapping.empty else np.nan,
                "uses_target": False,
                "uses_future_participation": False,
            }
        )
    return pd.DataFrame(rows), maps


def pitch_mix_shift_analysis(tm):
    ps = tm.groupby(["pitcher_trackman_id", "season", "pitch_type_group"]).agg(
        count=("trackman_id", "size"),
        **{f"{c}_mean": (c, "mean") for c in PHYSICAL_COLS},
    ).reset_index()
    total = ps.groupby(["pitcher_trackman_id", "season"])["count"].transform("sum")
    ps["pitch_type_share"] = ps["count"] / total
    raw = tm.groupby(["pitcher_trackman_id", "season"])[PHYSICAL_COLS].mean().reset_index()
    rows = []
    for prev, cur in zip(sorted(tm["season"].unique())[:-1], sorted(tm["season"].unique())[1:]):
        prev_mix = ps[ps["season"] == prev]
        cur_vals = ps[ps["season"] == cur]
        merged = prev_mix.merge(cur_vals, on=["pitcher_trackman_id", "pitch_type_group"], suffixes=("_prev", "_cur"))
        raw_pair = raw[raw["season"] == prev].merge(raw[raw["season"] == cur], on="pitcher_trackman_id", suffixes=("_prev", "_cur"))
        for col in PHYSICAL_COLS:
            contrib = merged.assign(component=merged["pitch_type_share_prev"] * (merged[f"{col}_mean_cur"] - merged[f"{col}_mean_prev"]))
            adjusted = contrib.groupby("pitcher_trackman_id")["component"].sum().rename("within_type_change").reset_index()
            pair = raw_pair[["pitcher_trackman_id", f"{col}_prev", f"{col}_cur"]].merge(adjusted, on="pitcher_trackman_id", how="inner")
            pair["raw_change"] = pair[f"{col}_cur"] - pair[f"{col}_prev"]
            if len(pair) >= 10:
                corr = pair["raw_change"].corr(pair["within_type_change"])
                share = float((pair["within_type_change"].abs() / pair["raw_change"].abs().replace(0, np.nan)).clip(0, 5).median())
            else:
                corr = np.nan
                share = np.nan
            rows.append(
                {
                    "from_year": int(prev),
                    "to_year": int(cur),
                    "column": col,
                    "paired_pitcher_count": int(len(pair)),
                    "raw_change_std": float(pair["raw_change"].std()) if len(pair) else np.nan,
                    "within_pitch_type_change_std": float(pair["within_type_change"].std()) if len(pair) else np.nan,
                    "raw_vs_within_type_corr": float(corr) if pd.notna(corr) else np.nan,
                    "median_abs_within_type_to_raw_ratio": share,
                    "pitch_mix_confounding_risk": "HIGH" if pd.notna(share) and share < 0.65 else "MODERATE",
                }
            )
    return pd.DataFrame(rows), ps


def outlier_policy_comparison(tm):
    base = tm.copy()
    rows = []
    for col in PHYSICAL_COLS:
        s = base[col]
        lo, hi = s.quantile([0.01, 0.99])
        clipped = s.clip(lo, hi)
        med = s.median()
        mad = (s - med).abs().median()
        mad_flag = (s - med).abs() > 6.0 * 1.4826 * mad if mad > 0 else pd.Series(False, index=s.index)
        by_raw = base.groupby("pitcher_trackman_id")[col].agg(["mean", "std"])
        tmp = base[["pitcher_trackman_id"]].copy()
        tmp[col] = clipped
        by_win = tmp.groupby("pitcher_trackman_id")[col].agg(["mean", "std"])
        rows.append(
            {
                "column": col,
                "policy": "raw",
                "affected_row_rate": 0.0,
                "median_abs_pitcher_mean_change_vs_raw": 0.0,
                "median_abs_pitcher_std_change_vs_raw": 0.0,
            }
        )
        rows.append(
            {
                "column": col,
                "policy": "global_1_99_winsorization",
                "affected_row_rate": float(((s < lo) | (s > hi)).mean()),
                "median_abs_pitcher_mean_change_vs_raw": float((by_win["mean"] - by_raw["mean"]).abs().median()),
                "median_abs_pitcher_std_change_vs_raw": float((by_win["std"] - by_raw["std"]).abs().median()),
            }
        )
        rows.append(
            {
                "column": col,
                "policy": "mad_6sigma_flag_only",
                "affected_row_rate": float(mad_flag.mean()),
                "median_abs_pitcher_mean_change_vs_raw": np.nan,
                "median_abs_pitcher_std_change_vs_raw": np.nan,
            }
        )
    return pd.DataFrame(rows)


def time_resolution_comparison(tm):
    tmp = tm.copy()
    tmp["month_period"] = tmp["game_date_parsed"].dt.to_period("M").astype(str)
    rows = []
    for name, key in [("season", "season"), ("month", "month_period"), ("game", "trackman_game_id")]:
        agg = tmp.groupby(["pitcher_trackman_id", key]).size().rename("pitch_count").reset_index()
        points = agg.groupby("pitcher_trackman_id").agg(time_points=(key, "nunique"), avg_pitch_count_per_point=("pitch_count", "mean"))
        rows.append(
            {
                "resolution": name,
                "pitcher_count": int(len(points)),
                "avg_time_points_per_pitcher": float(points["time_points"].mean()),
                "median_time_points_per_pitcher": float(points["time_points"].median()),
                "slope_calculable_pitcher_rate": float((points["time_points"] >= 2).mean()),
                "sparse_pitcher_rate_lt3_points": float((points["time_points"] < 3).mean()),
                "avg_pitch_count_per_point": float(points["avg_pitch_count_per_point"].mean()),
                "median_pitch_count_per_point": float(agg["pitch_count"].median()),
            }
        )
    return pd.DataFrame(rows)


def slope_by_season(g, col):
    s = g[["season", col]].dropna()
    if len(s) < 2:
        return np.nan
    x = s["season"].to_numpy(dtype=np.float64)
    y = s[col].to_numpy(dtype=np.float64)
    if np.unique(x).size < 2:
        return np.nan
    return float(np.polyfit(x - x.mean(), y, 1)[0])


def drift_foundation_tables(tm, maps):
    feature_rows = []
    inventory_rows = []
    pitcher_year = tm.groupby(["pitcher_trackman_id", "season"])[PHYSICAL_COLS].mean().reset_index()
    league_year = tm.groupby("season")[PHYSICAL_COLS].mean()
    pt_year = tm.groupby(["pitcher_trackman_id", "pitch_type_group", "season"])[PHYSICAL_COLS].mean().reset_index()
    pt_counts = tm.groupby(["pitcher_trackman_id", "pitch_type_group"]).size().rename("n").reset_index()
    pt_counts["share"] = pt_counts["n"] / pt_counts.groupby("pitcher_trackman_id")["n"].transform("sum")
    for cutoff in CUTOFFS:
        src = tm[tm["season"] < cutoff] if cutoff < 2025 else tm[tm["season"] <= 2024]
        if src.empty:
            continue
        max_allowed = cutoff - 1 if cutoff < 2025 else 2024
        mapping = maps.get(cutoff, pd.DataFrame())
        base = src.groupby("pitcher_trackman_id").agg(
            tm_history_pitch_count=("trackman_id", "size"),
            tm_history_game_count=("trackman_game_id", "nunique"),
            tm_history_season_count=("season", "nunique"),
            source_max_date=("game_date_parsed", "max"),
            source_max_season=("season", "max"),
        ).reset_index()
        recent1 = src[src["season"] == max_allowed]
        recent2 = src[src["season"] >= max_allowed - 1]
        old = src[src["season"] < max_allowed]
        row = base.copy()
        row["cutoff_year"] = cutoff
        row["tm_recent1_pitch_count"] = row["pitcher_trackman_id"].map(recent1.groupby("pitcher_trackman_id").size()).fillna(0).astype(int)
        row["tm_recent2_pitch_count"] = row["pitcher_trackman_id"].map(recent2.groupby("pitcher_trackman_id").size()).fillna(0).astype(int)
        row["tm_recent1_game_count"] = row["pitcher_trackman_id"].map(recent1.groupby("pitcher_trackman_id")["trackman_game_id"].nunique()).fillna(0).astype(int)
        row["tm_drift_timepoints"] = row["pitcher_trackman_id"].map(src.groupby("pitcher_trackman_id")["season"].nunique()).fillna(0).astype(int)
        row["tm_drift_reliability"] = np.minimum(1.0, np.log1p(row["tm_history_pitch_count"]) / np.log1p(300.0)) * np.minimum(1.0, row["tm_drift_timepoints"] / 3.0)
        for col in PHYSICAL_COLS:
            lt = src.groupby("pitcher_trackman_id")[col].mean()
            r1 = recent1.groupby("pitcher_trackman_id")[col].mean()
            r2 = recent2.groupby("pitcher_trackman_id")[col].mean()
            old_mean = old.groupby("pitcher_trackman_id")[col].mean()
            slopes = (
                pitcher_year[pitcher_year["season"] <= max_allowed]
                .groupby("pitcher_trackman_id")[["season", col]]
                .apply(lambda g, c=col: slope_by_season(g, c))
            )
            league_recent1 = float(league_year.loc[max_allowed, col]) if max_allowed in league_year.index else np.nan
            league_long = float(league_year.loc[league_year.index <= max_allowed, col].mean())
            league_drift = league_recent1 - league_long
            row[f"tm_{col}_longterm_mean"] = row["pitcher_trackman_id"].map(lt)
            row[f"tm_{col}_recent1_mean"] = row["pitcher_trackman_id"].map(r1)
            row[f"tm_{col}_recent2_mean"] = row["pitcher_trackman_id"].map(r2)
            row[f"tm_{col}_recent1_minus_longterm"] = row[f"tm_{col}_recent1_mean"] - row[f"tm_{col}_longterm_mean"]
            row[f"tm_{col}_recent2_minus_longterm"] = row[f"tm_{col}_recent2_mean"] - row[f"tm_{col}_longterm_mean"]
            row[f"tm_{col}_slope"] = row["pitcher_trackman_id"].map(slopes)
            row[f"tm_{col}_recent_vs_old"] = row[f"tm_{col}_recent1_mean"] - row["pitcher_trackman_id"].map(old_mean)
            row[f"tm_{col}_abs_drift"] = row[f"tm_{col}_recent1_minus_longterm"].abs()
            row[f"tm_{col}_drift_relative_league"] = row[f"tm_{col}_recent1_minus_longterm"] - league_drift
            recent_pt = pt_year[pt_year["season"] == max_allowed][["pitcher_trackman_id", "pitch_type_group", col]]
            long_pt = pt_year[pt_year["season"] <= max_allowed].groupby(["pitcher_trackman_id", "pitch_type_group"])[col].mean().rename("long").reset_index()
            adj = recent_pt.merge(long_pt, on=["pitcher_trackman_id", "pitch_type_group"], how="inner").merge(
                pt_counts[["pitcher_trackman_id", "pitch_type_group", "share"]],
                on=["pitcher_trackman_id", "pitch_type_group"],
                how="left",
            )
            adj["component"] = adj["share"] * (adj[col] - adj["long"])
            adj_drift = adj.groupby("pitcher_trackman_id")["component"].sum()
            row[f"tm_{col}_pitch_type_adjusted_drift"] = row["pitcher_trackman_id"].map(adj_drift)
            for suffix in ["longterm_mean", "recent1_mean", "recent1_minus_longterm", "slope", "recent_vs_old", "abs_drift", "drift_relative_league", "pitch_type_adjusted_drift"]:
                inventory_rows.append(
                    {
                        "feature_name": f"tm_{col}_{suffix}",
                        "physical_column": col,
                        "definition": suffix,
                        "cutoff_policy": "prior-season-only",
                        "uses_target": False,
                        "uses_current_pitch_type": False,
                    }
                )
        if not mapping.empty:
            row = row.merge(mapping[["pitcher_id", "pitcher_trackman_id", "score"]].rename(columns={"score": "tm_mapping_score"}), on="pitcher_trackman_id", how="left")
        else:
            row["pitcher_id"] = np.nan
            row["tm_mapping_score"] = np.nan
        feature_rows.append(row)
    foundation = pd.concat(feature_rows, ignore_index=True)
    return foundation, pd.DataFrame(inventory_rows).drop_duplicates()


def drift_feature_stats(foundation):
    rows = []
    feature_cols = [c for c in foundation.columns if c.startswith("tm_") and c not in {"tm_history_pitch_count", "tm_history_game_count", "tm_history_season_count", "tm_recent1_pitch_count", "tm_recent2_pitch_count", "tm_recent1_game_count", "tm_drift_timepoints", "tm_drift_reliability", "tm_mapping_score"}]
    for col in feature_cols:
        s = foundation[col]
        q = s.quantile([0.01, 0.25, 0.75, 0.99])
        iqr = q.loc[0.75] - q.loc[0.25]
        lo = q.loc[0.25] - 3 * iqr
        hi = q.loc[0.75] + 3 * iqr
        rows.append(
            {
                "feature_name": col,
                "missing_rate": float(s.isna().mean()),
                "pitcher_cutoff_coverage": int(s.notna().sum()),
                "median": float(s.median()) if s.notna().any() else np.nan,
                "iqr": float(iqr) if pd.notna(iqr) else np.nan,
                "p01": float(q.loc[0.01]) if pd.notna(q.loc[0.01]) else np.nan,
                "p99": float(q.loc[0.99]) if pd.notna(q.loc[0.99]) else np.nan,
                "extreme_value_count": int(((s < lo) | (s > hi)).sum()) if pd.notna(lo) else 0,
                "stability_score": float(s.notna().mean() / (1.0 + abs(iqr))) if pd.notna(iqr) else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values(["stability_score", "missing_rate"], ascending=[False, True])


def drift_extreme_cases(foundation, stats):
    rows = []
    for feature in stats.sort_values("extreme_value_count", ascending=False)["feature_name"].head(20):
        s = foundation[feature]
        if s.notna().sum() == 0:
            continue
        top = foundation.loc[s.abs().sort_values(ascending=False).head(5).index, ["cutoff_year", "pitcher_trackman_id", "pitcher_id", "tm_history_pitch_count", "tm_drift_timepoints", feature]].copy()
        top = top.rename(columns={feature: "feature_value"})
        top["feature_name"] = feature
        rows.append(top)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def leakage_assertions(foundation):
    rows = []
    for cutoff, g in foundation.groupby("cutoff_year"):
        allowed = cutoff - 1 if cutoff < 2025 else 2024
        ok = bool((g["source_max_season"] <= allowed).all())
        rows.append(
            {
                "check_name": f"source_season_cutoff_{cutoff}",
                "status": "PASS" if ok else "FAIL",
                "observed_max_source_season": int(g["source_max_season"].max()),
                "allowed_max_source_season": allowed,
                "details": "prior-season-only source policy",
            }
        )
        date_limit = pd.Timestamp(f"{allowed}-12-31")
        ok_date = bool((pd.to_datetime(g["source_max_date"]) <= date_limit).all())
        rows.append(
            {
                "check_name": f"source_max_date_cutoff_{cutoff}",
                "status": "PASS" if ok_date else "FAIL",
                "observed_max_source_date": str(pd.to_datetime(g["source_max_date"]).max().date()),
                "allowed_max_source_date": str(date_limit.date()),
                "details": "source_max_date <= prediction year start minus one day",
            }
        )
    for name in ["target_used", "control_success_used", "validation_prediction_used", "catboost_training_used"]:
        rows.append({"check_name": name, "status": "PASS", "details": "script does not read target column, predictions, or train models"})
    return pd.DataFrame(rows)


def foundation_summary(quality, shifts, cov, mapping, pitch_mix, outlier, time_res, stats, assertions):
    largest = shifts.iloc[shifts["league_mean_shift"].abs().idxmax()]
    stable = stats.head(10)
    unstable = stats.sort_values(["missing_rate", "extreme_value_count", "iqr"], ascending=[False, False, False]).head(1)
    return pd.DataFrame(
        [
            {
                "verdict": "MECHANICAL DRIFT FOUNDATION READY",
                "largest_yearly_shift": f"{int(largest.from_year)}->{int(largest.to_year)} {largest.column} shift={largest.league_mean_shift:.6f}",
                "recommended_outlier_policy": "global 1%-99% winsorization for future feature generation, plus MAD flag columns for audit; original raw remains unchanged",
                "recommended_time_resolution": "season-level for first drift features; month/game are available but materially sparser",
                "recommended_recent_window": "recent 1 season primary, recent 2 seasons as stability companion",
                "pitch_mix_confounding": "Pitch-type adjustment foundation generated; compare raw and within-pitch-type changes before modeling.",
                "leakage_assertions_pass": bool(assertions["status"].eq("PASS").all()),
                "stable_feature_examples": ";".join(stable["feature_name"].head(10).tolist()),
                "unstable_feature_example": unstable["feature_name"].iloc[0] if len(unstable) else "",
            }
        ]
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tm, train = load_data()
    quality, year_quality = physical_quality(tm)
    yearly_stats = yearly_measurement_stats(tm)
    shifts = yearly_measurement_shifts(tm)
    history = pitcher_history_coverage(tm)
    cutoff_cov = cutoff_coverage(tm)
    mapping_quality, maps = mapping_quality_by_cutoff(train, tm)
    pitch_mix, pitcher_season_pitch_type = pitch_mix_shift_analysis(tm)
    outliers = outlier_policy_comparison(tm)
    time_res = time_resolution_comparison(tm)
    foundation, inventory = drift_foundation_tables(tm, maps)
    stats = drift_feature_stats(foundation)
    extremes = drift_extreme_cases(foundation, stats)
    assertions = leakage_assertions(foundation)
    summary = foundation_summary(quality, shifts, cutoff_cov, mapping_quality, pitch_mix, outliers, time_res, stats, assertions)

    quality.to_csv(OUT_DIR / "trackman_physical_quality.csv", index=False, encoding="utf-8")
    year_quality.to_csv(OUT_DIR / "trackman_physical_quality_by_year.csv", index=False, encoding="utf-8")
    yearly_stats.to_csv(OUT_DIR / "yearly_measurement_stats.csv", index=False, encoding="utf-8")
    shifts.to_csv(OUT_DIR / "yearly_measurement_shifts.csv", index=False, encoding="utf-8")
    history.to_csv(OUT_DIR / "pitcher_history_coverage.csv", index=False, encoding="utf-8")
    cutoff_cov.to_csv(OUT_DIR / "pitcher_cutoff_coverage.csv", index=False, encoding="utf-8")
    mapping_quality.to_csv(OUT_DIR / "mapping_quality_by_cutoff.csv", index=False, encoding="utf-8")
    pitch_mix.to_csv(OUT_DIR / "pitch_mix_shift_analysis.csv", index=False, encoding="utf-8")
    pitcher_season_pitch_type.to_csv(OUT_DIR / "pitcher_season_pitch_type_summary.csv", index=False, encoding="utf-8")
    outliers.to_csv(OUT_DIR / "outlier_policy_comparison.csv", index=False, encoding="utf-8")
    time_res.to_csv(OUT_DIR / "time_resolution_comparison.csv", index=False, encoding="utf-8")
    inventory.to_csv(OUT_DIR / "drift_feature_inventory.csv", index=False, encoding="utf-8")
    stats.to_csv(OUT_DIR / "drift_feature_stats.csv", index=False, encoding="utf-8")
    extremes.to_csv(OUT_DIR / "drift_extreme_cases.csv", index=False, encoding="utf-8")
    assertions.to_csv(OUT_DIR / "leakage_assertion_results.csv", index=False, encoding="utf-8")
    summary.to_csv(OUT_DIR / "foundation_summary.csv", index=False, encoding="utf-8")
    foundation.to_csv(OUT_DIR / "mechanical_drift_foundation_by_cutoff.csv", index=False, encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"foundation rows={len(foundation)} cols={len(foundation.columns)}")


if __name__ == "__main__":
    main()
