import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from calibration_utils import apply_calibration


ID_COL = "row_id"
TARGET_COL = "control_success"
EPS = 1e-6
TEMPERATURE = 2.3
HARD_CAP = 0.020
HISTORICAL_RATES = {
    2022: 0.5289204435249241,
    2023: 0.49995723449750534,
    2024: 0.4861049201797189,
}


def clip_prob(pred):
    return np.clip(np.asarray(pred, dtype=np.float64), EPS, 1.0 - EPS)


def logit(pred):
    pred = clip_prob(pred)
    return np.log(pred / (1.0 - pred))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def linear_forecast_recent3(year_rates: dict[int, float]) -> float:
    s = pd.Series(year_rates, dtype=np.float64).dropna().sort_index()
    if len(s) > 3:
        s = s.tail(3)
    if len(s) == 1:
        return float(s.iloc[-1])
    x = s.index.to_numpy(dtype=np.float64)
    y = s.to_numpy(dtype=np.float64)
    slope, intercept = np.polyfit(x, y, 1)
    return float(np.clip(intercept + slope * (x[-1] + 1), 0.42, 0.62))


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


def candidate_a_calibration(platt_pred, target_rate):
    adjusted = logit_mean_match(platt_pred, target_rate)
    centered = logit(adjusted) - float(logit(target_rate))
    temperature_pred = sigmoid(float(logit(target_rate)) + centered / TEMPERATURE)
    temperature_pred = logit_mean_match(temperature_pred, target_rate)
    deviation = clip_prob(temperature_pred) - target_rate
    return clip_prob(target_rate + np.clip(deviation, -HARD_CAP, HARD_CAP))


def load_test(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    if ID_COL not in df.columns:
        raise ValueError(f"{ID_COL} column is missing from test data.")
    return df


def load_sample_submission(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    if list(df.columns[:2]) != [ID_COL, TARGET_COL]:
        raise ValueError(f"sample_submission columns are invalid: {list(df.columns)}")
    return df


def main():
    started = time.time()
    base_dir = Path(__file__).resolve().parent
    script_dir = str(base_dir)
    cwd_data_dir = os.path.abspath("./data")
    script_data_dir = os.path.join(script_dir, "data")
    test_dir = cwd_data_dir if os.path.exists(os.path.join(cwd_data_dir, "test.csv")) else script_data_dir
    model_path = base_dir / "model" / "model.pkl"
    out_dir = os.path.abspath("./output") if test_dir == cwd_data_dir else os.path.join(script_dir, "output")

    print(f"script_path={Path(__file__).resolve()}")
    print(f"base_dir={base_dir}")
    print(f"model_path={model_path}")
    print(f"model_path_exists={model_path.exists()}")
    if not model_path.exists():
        raise FileNotFoundError(f"model artifact not found at resolved path: {model_path}")
    print(f"model_file_size={model_path.stat().st_size}")

    target_rate = linear_forecast_recent3(HISTORICAL_RATES)
    print(f"candidate_a_temperature={TEMPERATURE}")
    print(f"candidate_a_hard_cap={HARD_CAP}")
    print(f"historical_rates={HISTORICAL_RATES}")
    print(f"estimated_target_rate_2025={target_rate:.15f}")

    t0 = time.time()
    test = load_test(os.path.join(test_dir, "test.csv"))
    sub = load_sample_submission(os.path.join(test_dir, "sample_submission.csv"))
    artifact = joblib.load(model_path)
    load_seconds = time.time() - t0

    builder = artifact["builder"]
    model = artifact["model"]
    calibration = artifact["calibration"]
    calibration_method = artifact["calibration_method"]

    t0 = time.time()
    X = builder.transform(test)
    feature_seconds = time.time() - t0

    t0 = time.time()
    raw_pred = model.predict_proba(X)[:, 1]
    platt_pred = apply_calibration(raw_pred, test[["game_type"]], calibration, calibration_method)
    inference_seconds = time.time() - t0

    t0 = time.time()
    pred = candidate_a_calibration(platt_pred, target_rate)
    calibration_seconds = time.time() - t0

    pred_map = dict(zip(test[ID_COL], pred))
    sub[TARGET_COL] = sub[ID_COL].map(pred_map).fillna(sub[TARGET_COL]).clip(0.0, 1.0)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "submission.csv")
    sub.to_csv(out_path, index=False, encoding="utf-8")

    print(
        "prediction_summary "
        f"rows={len(pred)} mean={pred.mean():.15f} std={pred.std():.15f} "
        f"min={pred.min():.15f} max={pred.max():.15f}"
    )
    print(f"max_abs_deviation_from_target={np.max(np.abs(pred - target_rate)):.15f}")
    print(
        "runtime_seconds "
        f"load={load_seconds:.3f} feature={feature_seconds:.3f} "
        f"inference={inference_seconds:.3f} calibration={calibration_seconds:.3f} "
        f"total={time.time() - started:.3f}"
    )
    print(f"Saved: {out_path} rows={len(sub)} seconds={time.time() - started:.3f}")


if __name__ == "__main__":
    main()
