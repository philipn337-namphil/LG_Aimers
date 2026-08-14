import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


EPS = 1e-6


def clip_prob(pred):
    return np.clip(np.asarray(pred, dtype=np.float64), EPS, 1.0 - EPS)


def logit(pred):
    pred = clip_prob(pred)
    return np.log(pred / (1.0 - pred))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def fit_calibrators(pred, y, meta: pd.DataFrame):
    pred = clip_prob(pred)
    y = np.asarray(y, dtype=np.int8)
    meta = meta.reset_index(drop=True)

    platt = LogisticRegression(C=1e6, solver="lbfgs", random_state=42)
    platt.fit(logit(pred).reshape(-1, 1), y)

    isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    isotonic.fit(pred, y)

    global_logit_offset = float(logit(y.mean()) - logit(pred.mean()))
    game_type_offsets = {}
    game_type_rates = {}
    for game_type, idx in meta.groupby("game_type").groups.items():
        idx = np.asarray(list(idx), dtype=np.int64)
        if len(idx) < 1000:
            continue
        game_type_offsets[str(game_type)] = float(logit(y[idx].mean()) - logit(pred[idx].mean()))
        game_type_rates[str(game_type)] = float(y[idx].mean())

    return {
        "platt": platt,
        "isotonic": isotonic,
        "global_logit_offset": global_logit_offset,
        "game_type_logit_offsets": game_type_offsets,
        "global_rate": float(y.mean()),
        "game_type_rates": game_type_rates,
    }


def apply_calibration(pred, meta: pd.DataFrame | None, calibration: dict | None, method: str | None):
    pred = clip_prob(pred)
    if not calibration or not method or method == "raw":
        return pred
    if method == "platt":
        return calibration["platt"].predict_proba(logit(pred).reshape(-1, 1))[:, 1]
    if method == "isotonic":
        return calibration["isotonic"].predict(pred)
    if method == "global_logit_offset":
        return sigmoid(logit(pred) + calibration["global_logit_offset"])
    if method == "game_type_logit_offset":
        if meta is None or "game_type" not in meta.columns:
            return sigmoid(logit(pred) + calibration["global_logit_offset"])
        offsets = meta["game_type"].astype(str).map(calibration["game_type_logit_offsets"]).fillna(
            calibration["global_logit_offset"]
        )
        return sigmoid(logit(pred) + offsets.to_numpy(dtype=np.float64))
    if method.startswith("blend_global_"):
        weight = float(method.rsplit("_", 1)[-1])
        return (1.0 - weight) * pred + weight * calibration["global_rate"]
    if method.startswith("blend_game_type_"):
        weight = float(method.rsplit("_", 1)[-1])
        if meta is None or "game_type" not in meta.columns:
            prior = np.full(len(pred), calibration["global_rate"], dtype=np.float64)
        else:
            prior = meta["game_type"].astype(str).map(calibration["game_type_rates"]).fillna(
                calibration["global_rate"]
            )
            prior = prior.to_numpy(dtype=np.float64)
        return (1.0 - weight) * pred + weight * prior
    raise ValueError(f"Unknown calibration method: {method}")
