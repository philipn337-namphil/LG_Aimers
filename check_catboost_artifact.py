import os
import time
import zipfile

import joblib
import numpy as np
import pandas as pd

from catboost_script import ID_COL, TARGET_COL, load_sample_submission, load_test
from calibration_utils import apply_calibration


CHECK_DIR = "output/catboost_submission_check"


def describe(pred):
    q = np.quantile(pred, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return {
        "mean": float(np.mean(pred)),
        "std": float(np.std(pred)),
        "min": float(np.min(pred)),
        "max": float(np.max(pred)),
        "p01": float(q[0]),
        "p05": float(q[1]),
        "p25": float(q[2]),
        "p50": float(q[3]),
        "p75": float(q[4]),
        "p95": float(q[5]),
        "p99": float(q[6]),
    }


def main():
    os.makedirs(CHECK_DIR, exist_ok=True)
    baseline = pd.read_csv("output/baseline_test_predictions_for_catboost_compare.csv")
    cat = pd.read_csv(os.path.join(CHECK_DIR, "clean_run", "output", "submission.csv"))
    test = load_test("data/test.csv")
    sample = load_sample_submission("data/sample_submission.csv")

    merged = baseline.merge(cat, on=ID_COL, suffixes=("_baseline", "_catboost"))
    b = merged[f"{TARGET_COL}_baseline"].to_numpy(dtype=np.float64)
    c = merged[f"{TARGET_COL}_catboost"].to_numpy(dtype=np.float64)
    diff = c - b

    comparison_rows = []
    comparison_rows.append({"prediction": "baseline", **describe(b)})
    comparison_rows.append({"prediction": "catboost", **describe(c)})
    comparison_rows.append(
        {
            "prediction": "catboost_minus_baseline",
            "mean": float(np.mean(diff)),
            "std": float(np.std(diff)),
            "min": float(np.min(diff)),
            "max": float(np.max(diff)),
            "p01": float(np.quantile(diff, 0.01)),
            "p05": float(np.quantile(diff, 0.05)),
            "p25": float(np.quantile(diff, 0.25)),
            "p50": float(np.quantile(diff, 0.50)),
            "p75": float(np.quantile(diff, 0.75)),
            "p95": float(np.quantile(diff, 0.95)),
            "p99": float(np.quantile(diff, 0.99)),
        }
    )
    pred_cmp = pd.DataFrame(comparison_rows)
    pred_cmp["mean_abs_diff"] = np.nan
    pred_cmp["correlation"] = np.nan
    pred_cmp.loc[pred_cmp["prediction"] == "catboost_minus_baseline", "mean_abs_diff"] = float(np.mean(np.abs(diff)))
    pred_cmp.loc[pred_cmp["prediction"] == "catboost_minus_baseline", "correlation"] = float(np.corrcoef(b, c)[0, 1]) if len(b) > 1 else np.nan
    pred_cmp.to_csv(os.path.join(CHECK_DIR, "prediction_comparison.csv"), index=False, encoding="utf-8")

    artifact = joblib.load("catboost_model/model.pkl")
    started = time.time()
    X = artifact["builder"].transform(test)
    raw = artifact["model"].predict_proba(X)[:, 1]
    calibrated = apply_calibration(raw, test[["game_type"]], artifact["calibration"], artifact["calibration_method"])
    final = (1.0 - artifact["prior_blend_weight"]) * calibrated + artifact["prior_blend_weight"] * artifact["trend_prior_rate"]
    inference_seconds = time.time() - started

    id_order_ok = list(cat[ID_COL]) == list(sample[ID_COL])
    artifact_rows = [
        {
            "check": "catboost_artifact",
            "model_file_size_bytes": os.path.getsize("catboost_model/model.pkl"),
            "zip_size_bytes": os.path.getsize("catboost_submit.zip"),
            "zip_entries": ";".join(zipfile.ZipFile("catboost_submit.zip").namelist()),
            "submission_rows": len(cat),
            "test_rows": len(test),
            "sample_rows": len(sample),
            "id_order_ok": id_order_ok,
            "nan_count": int(cat[TARGET_COL].isna().sum()),
            "inf_count": int(np.isinf(cat[TARGET_COL].to_numpy(dtype=np.float64)).sum()),
            "prob_min": float(cat[TARGET_COL].min()),
            "prob_max": float(cat[TARGET_COL].max()),
            "range_ok": bool(((cat[TARGET_COL] >= 0) & (cat[TARGET_COL] <= 1)).all()),
            "trend_prior_rate": artifact["trend_prior_rate"],
            "prior_blend_weight": artifact["prior_blend_weight"],
            "calibration_method": artifact["calibration_method"],
            "catboost_params": repr(artifact["catboost_params"]),
        }
    ]
    pd.DataFrame(artifact_rows).to_csv(os.path.join(CHECK_DIR, "artifact_check.csv"), index=False, encoding="utf-8")

    pd.DataFrame(
        [
            {
                "check": "direct_artifact_inference",
                "rows": len(test),
                "seconds": inference_seconds,
                "raw_mean": float(raw.mean()),
                "raw_std": float(raw.std()),
                "calibrated_mean": float(calibrated.mean()),
                "calibrated_std": float(calibrated.std()),
                "final_mean": float(final.mean()),
                "final_std": float(final.std()),
            }
        ]
    ).to_csv(os.path.join(CHECK_DIR, "runtime_check.csv"), index=False, encoding="utf-8")

    print(pred_cmp.to_string(index=False))
    print(pd.DataFrame(artifact_rows).to_string(index=False))


if __name__ == "__main__":
    main()
