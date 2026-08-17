from __future__ import annotations

import numpy as np
import pandas as pd


EPS = 1e-6
TEMPERATURE = 2.3
HARD_CAP = 0.020
ALPHA_PITCHER = 100
ALPHA_CONTEXT = 300
BLEND_K = 100
TARGET_COL = "control_success"


def clip_prob(pred):
    return np.clip(np.asarray(pred, dtype=np.float64), EPS, 1.0 - EPS)


def logit(pred):
    pred = clip_prob(pred)
    return np.log(pred / (1.0 - pred))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def add_context_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["count_state"] = out["balls_before"].astype(str) + "-" + out["strikes_before"].astype(str)
    return out


def linear_forecast_recent3(year_rates: dict[int, float] | pd.Series) -> float:
    s = pd.Series(year_rates, dtype=np.float64).dropna().sort_index()
    if len(s) > 3:
        s = s.tail(3)
    if len(s) == 1:
        return float(s.iloc[-1])
    x = s.index.to_numpy(dtype=np.float64)
    y = s.to_numpy(dtype=np.float64)
    slope, intercept = np.polyfit(x, y, 1)
    return float(np.clip(intercept + slope * (x[-1] + 1), 0.42, 0.62))


def target_rate_for_year(train_df: pd.DataFrame, prediction_year: int) -> float:
    rates = train_df[train_df["season"] < prediction_year].groupby("season")[TARGET_COL].mean().sort_index()
    return linear_forecast_recent3(rates)


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


def v2_strength_control(platt_pred, target_rate):
    adjusted = logit_mean_match(platt_pred, target_rate)
    centered = logit(adjusted) - float(logit(target_rate))
    temperature_pred = sigmoid(float(logit(target_rate)) + centered / TEMPERATURE)
    temperature_pred = logit_mean_match(temperature_pred, target_rate)
    deviation = clip_prob(temperature_pred) - target_rate
    return clip_prob(target_rate + np.clip(deviation, -HARD_CAP, HARD_CAP))


def pitcher_key(df: pd.DataFrame) -> pd.Series:
    return df["pitcher_id"].astype(str)


def pitcher_game_key(df: pd.DataFrame) -> pd.Series:
    return df["pitcher_id"].astype(str) + "|" + df["game_type"].astype(str)


def build_hierarchy_tables(train_df: pd.DataFrame, prediction_year: int, alpha_pitcher=ALPHA_PITCHER, alpha_context=ALPHA_CONTEXT):
    train_df = add_context_columns(train_df)
    history = train_df[train_df["season"] < prediction_year].copy()
    target_rate = target_rate_for_year(train_df, prediction_year)

    p_grouped = pd.DataFrame({"key": pitcher_key(history), "y": history[TARGET_COL].astype(float)}).groupby("key")["y"].agg(["sum", "count"])
    p_grouped["posterior"] = (p_grouped["sum"] + alpha_pitcher * target_rate) / (p_grouped["count"] + alpha_pitcher)
    pitcher_parent = pitcher_key(history).map(p_grouped["posterior"].to_dict()).fillna(target_rate).to_numpy(dtype=np.float64)

    pg = pd.DataFrame(
        {
            "key": pitcher_game_key(history),
            "y": history[TARGET_COL].astype(float),
            "parent": pitcher_parent,
        }
    )
    c_grouped = pg.groupby("key").agg(success=("y", "sum"), count=("y", "count"), parent=("parent", "mean"))
    c_grouped["posterior"] = (c_grouped["success"] + alpha_context * c_grouped["parent"]) / (c_grouped["count"] + alpha_context)

    return {
        "prediction_year": int(prediction_year),
        "target_rate": float(target_rate),
        "alpha_pitcher": float(alpha_pitcher),
        "alpha_context": float(alpha_context),
        "pitcher": p_grouped[["posterior", "count"]].reset_index(),
        "pitcher_game": c_grouped[["posterior", "count"]].reset_index(),
        "history_min_season": int(history["season"].min()) if len(history) else None,
        "history_max_season": int(history["season"].max()) if len(history) else None,
        "history_rows": int(len(history)),
    }


def predict_hierarchy(rows: pd.DataFrame, hierarchy_tables: dict):
    rows = add_context_columns(rows)
    target_rate = float(hierarchy_tables["target_rate"])
    ptab = hierarchy_tables["pitcher"].set_index("key")
    ctab = hierarchy_tables["pitcher_game"].set_index("key")

    p_keys = pitcher_key(rows)
    c_keys = pitcher_game_key(rows)
    p_post = p_keys.map(ptab["posterior"].to_dict()).fillna(target_rate).to_numpy(dtype=np.float64)
    p_count = p_keys.map(ptab["count"].to_dict()).fillna(0.0).to_numpy(dtype=np.float64)
    c_post = c_keys.map(ctab["posterior"].to_dict())
    c_count = c_keys.map(ctab["count"].to_dict()).fillna(0.0).to_numpy(dtype=np.float64)
    pred = c_post.fillna(pd.Series(p_post, index=rows.index)).to_numpy(dtype=np.float64)
    return clip_prob(pred), p_count, c_count


def candidate_a_native(rows: pd.DataFrame, hierarchy_tables: dict, v2_platt_pred):
    hierarchy_pred, pitcher_count, context_count = predict_hierarchy(rows, hierarchy_tables)
    reliability = pitcher_count / (pitcher_count + BLEND_K)
    native = reliability * hierarchy_pred + (1.0 - reliability) * clip_prob(v2_platt_pred)
    return clip_prob(native), pitcher_count, context_count, reliability
