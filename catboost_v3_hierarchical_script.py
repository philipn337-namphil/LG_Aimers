import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from calibration_utils import apply_calibration
from hierarchical_utils import TARGET_COL, candidate_a_native, v2_strength_control


ID_COL = "row_id"


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
    cwd_data_dir = os.path.abspath("./data")
    script_data_dir = str(base_dir / "data")
    test_dir = cwd_data_dir if os.path.exists(os.path.join(cwd_data_dir, "test.csv")) else script_data_dir
    out_dir = os.path.abspath("./output") if test_dir == cwd_data_dir else str(base_dir / "output")
    model_path = base_dir / "model" / "model.pkl"
    hierarchy_path = base_dir / "model" / "hierarchy.pkl"

    print(f"script_path={Path(__file__).resolve()}")
    print(f"base_dir={base_dir}")
    print(f"test_dir={test_dir}")
    print(f"model_path={model_path} exists={model_path.exists()}")
    print(f"hierarchy_path={hierarchy_path} exists={hierarchy_path.exists()}")
    if not model_path.exists():
        raise FileNotFoundError(f"model artifact not found: {model_path}")
    if not hierarchy_path.exists():
        raise FileNotFoundError(f"hierarchy artifact not found: {hierarchy_path}")

    t0 = time.time()
    test = load_test(os.path.join(test_dir, "test.csv"))
    sub = load_sample_submission(os.path.join(test_dir, "sample_submission.csv"))
    v2_artifact = joblib.load(model_path)
    hierarchy_artifact = joblib.load(hierarchy_path)
    load_seconds = time.time() - t0

    builder = v2_artifact["builder"]
    model = v2_artifact["model"]
    v2_calibration = v2_artifact["calibration"]
    v2_calibration_method = v2_artifact["calibration_method"]
    tables = hierarchy_artifact["hierarchy_tables"]
    target_rate = float(hierarchy_artifact["target_rate_2025"])

    t0 = time.time()
    x_test = builder.transform(test)
    feature_seconds = time.time() - t0

    t0 = time.time()
    raw_pred = model.predict_proba(x_test)[:, 1]
    v2_platt = apply_calibration(raw_pred, test[["game_type"]], v2_calibration, v2_calibration_method)
    model_seconds = time.time() - t0

    t0 = time.time()
    native, pitcher_count, context_count, reliability = candidate_a_native(test, tables, v2_platt)
    hier_platt = apply_calibration(native, test[["game_type"]], hierarchy_artifact["calibration"], hierarchy_artifact["calibration_method"])
    pred = v2_strength_control(hier_platt, target_rate)
    hierarchy_seconds = time.time() - t0

    pred_map = dict(zip(test[ID_COL], pred))
    sub[TARGET_COL] = sub[ID_COL].map(pred_map).fillna(sub[TARGET_COL]).clip(0.0, 1.0)
    if sub[TARGET_COL].isna().any():
        raise ValueError("submission contains NaN predictions")

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "submission.csv")
    t0 = time.time()
    sub.to_csv(out_path, index=False, encoding="utf-8")
    write_seconds = time.time() - t0

    quantiles = np.quantile(pred, [0.01, 0.05, 0.5, 0.95, 0.99])
    print(
        "prediction_summary "
        f"rows={len(pred)} mean={pred.mean():.15f} std={pred.std():.15f} "
        f"min={pred.min():.15f} max={pred.max():.15f} "
        f"q01={quantiles[0]:.15f} q05={quantiles[1]:.15f} q50={quantiles[2]:.15f} "
        f"q95={quantiles[3]:.15f} q99={quantiles[4]:.15f}"
    )
    print(
        "hierarchy_coverage "
        f"known_pitcher_rate={float((pitcher_count > 0).mean()):.6f} "
        f"known_pitcher_game_rate={float((context_count > 0).mean()):.6f} "
        f"mean_reliability={float(reliability.mean()):.6f}"
    )
    print(f"target_rate_2025={target_rate:.15f}")
    print(
        "runtime_seconds "
        f"load={load_seconds:.3f} feature={feature_seconds:.3f} "
        f"model={model_seconds:.3f} hierarchy_calibration={hierarchy_seconds:.3f} "
        f"write={write_seconds:.3f} total={time.time() - started:.3f}"
    )
    print(f"Saved: {out_path} rows={len(sub)} seconds={time.time() - started:.3f}")


if __name__ == "__main__":
    main()
