import argparse
import os
import pickle
import tempfile
import time
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from calibration_utils import apply_calibration, fit_calibrators
from model_utils import FeatureBuilder, TARGET_COL, make_model


FOLDS = [
    (2019, 2020, 2021, 2022),
    (2019, 2021, 2022, 2023),
    (2019, 2022, 2023, 2024),
]
TREND_WEIGHT = 0.75


def clip_prob(pred):
    return np.clip(np.asarray(pred, dtype=np.float64), 1e-6, 1.0 - 1e-6)


def linear_forecast(year_rates: pd.Series) -> float:
    s = year_rates.dropna().sort_index()
    if len(s) == 1:
        return float(s.iloc[-1])
    x = s.index.to_numpy(dtype=np.float64)
    y = s.to_numpy(dtype=np.float64)
    slope, intercept = np.polyfit(x, y, 1)
    return float(np.clip(intercept + slope * (x[-1] + 1), 0.42, 0.62))


def metrics(y, pred):
    pred = clip_prob(pred)
    qs = np.quantile(pred, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return {
        "brier": float(brier_score_loss(y, pred)),
        "logloss": float(log_loss(y, pred)),
        "auc": float(roc_auc_score(y, pred)),
        "pred_mean": float(pred.mean()),
        "pred_std": float(pred.std()),
        "pred_p01": float(qs[0]),
        "pred_p05": float(qs[1]),
        "pred_p25": float(qs[2]),
        "pred_p50": float(qs[3]),
        "pred_p75": float(qs[4]),
        "pred_p95": float(qs[5]),
        "pred_p99": float(qs[6]),
        "actual_rate": float(np.mean(y)),
    }


def model_size_mb(model) -> float:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as fp:
        path = fp.name
    try:
        with open(path, "wb") as fp:
            pickle.dump(model, fp)
        return os.path.getsize(path) / 1024 / 1024
    finally:
        if os.path.exists(path):
            os.remove(path)


def make_estimators():
    estimators = {"sklearn_hgb": make_model()}
    availability = {
        "sklearn_hgb": {"available": True, "reason": "current baseline estimator"},
    }
    try:
        from lightgbm import LGBMClassifier

        estimators["lightgbm"] = LGBMClassifier(
            objective="binary",
            n_estimators=220,
            learning_rate=0.045,
            num_leaves=31,
            min_child_samples=80,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )
        availability["lightgbm"] = {"available": True, "reason": "installed"}
    except Exception as exc:
        availability["lightgbm"] = {"available": False, "reason": repr(exc)}

    try:
        from catboost import CatBoostClassifier

        estimators["catboost"] = CatBoostClassifier(
            loss_function="Logloss",
            iterations=220,
            learning_rate=0.045,
            depth=6,
            l2_leaf_reg=5.0,
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
        )
        availability["catboost"] = {"available": True, "reason": "installed"}
    except Exception as exc:
        availability["catboost"] = {"available": False, "reason": repr(exc)}

    try:
        from xgboost import XGBClassifier

        estimators["xgboost"] = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=180,
            learning_rate=0.045,
            max_depth=4,
            min_child_weight=20,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=2.0,
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
        )
        availability["xgboost"] = {"available": True, "reason": "installed"}
    except Exception as exc:
        availability["xgboost"] = {"available": False, "reason": repr(exc)}
    return estimators, availability


def fit_predict_model(name, estimator, X_train, y_train, X_cal, X_valid):
    started = time.time()
    estimator.fit(X_train, y_train)
    train_seconds = time.time() - started

    started = time.time()
    cal_pred = estimator.predict_proba(X_cal)[:, 1]
    valid_pred = estimator.predict_proba(X_valid)[:, 1]
    inference_seconds = time.time() - started
    size_mb = model_size_mb(estimator)
    return estimator, clip_prob(cal_pred), clip_prob(valid_pred), train_seconds, inference_seconds, size_mb


def add_metric_rows(rows, fold_info, model_name, stage, y_valid, pred, trend_prior, runtime=None):
    rows.append(
        {
            **fold_info,
            "model": model_name,
            "stage": stage,
            "trend_prior": trend_prior,
            **metrics(y_valid, pred),
            **(runtime or {}),
        }
    )


def ensemble_weight_grid(names):
    if len(names) == 2:
        a, b = names
        for w in np.round(np.arange(0, 1.0001, 0.1), 2):
            yield {a: float(w), b: float(1 - w)}
    elif len(names) == 3:
        a, b, c = names
        for wa in np.round(np.arange(0, 1.0001, 0.1), 2):
            for wb in np.round(np.arange(0, 1 - wa + 0.0001, 0.1), 2):
                wc = round(float(1 - wa - wb), 2)
                if wc >= -1e-9:
                    yield {a: float(wa), b: float(wb), c: float(wc)}


def weighted_sum(preds, weights):
    out = None
    for name, weight in weights.items():
        if out is None:
            out = weight * preds[name]
        else:
            out += weight * preds[name]
    return clip_prob(out)


def run_fold(df, fold, estimators):
    train_start, train_end, cal_year, valid_year = fold
    train_df = df[(df["season"] >= train_start) & (df["season"] <= train_end)].copy()
    cal_df = df[df["season"] == cal_year].copy()
    valid_df = df[df["season"] == valid_year].copy()
    fold_info = {
        "fold": f"{train_start}-{train_end}_cal{cal_year}_valid{valid_year}",
        "train_start": train_start,
        "train_end": train_end,
        "cal_year": cal_year,
        "valid_year": valid_year,
    }
    print(f"Fold {fold_info['fold']} rows train={len(train_df)} cal={len(cal_df)} valid={len(valid_df)}")

    y_train = train_df[TARGET_COL].astype("int8")
    y_cal = cal_df[TARGET_COL].astype("int8").to_numpy()
    y_valid = valid_df[TARGET_COL].astype("int8").to_numpy()

    builder = FeatureBuilder(alpha=80.0)
    X_train = builder.fit_transform(train_df.drop(columns=[TARGET_COL]), y_train)
    X_cal = builder.transform(cal_df.drop(columns=[TARGET_COL]))
    X_valid = builder.transform(valid_df.drop(columns=[TARGET_COL]))
    trend_prior = linear_forecast(pd.concat([train_df, cal_df]).groupby("season")[TARGET_COL].mean())

    metric_rows = []
    runtime_rows = []
    cal_raw = {}
    valid_raw = {}
    valid_platt = {}
    oof = pd.DataFrame(
        {
            "row_id": valid_df["row_id"].to_numpy(),
            "fold": fold_info["fold"],
            "valid_year": valid_year,
            "target": y_valid,
        }
    )

    for name, estimator in estimators.items():
        print(f"  train {name}")
        _, cp, vp, train_s, infer_s, size_mb = fit_predict_model(name, estimator, X_train, y_train, X_cal, X_valid)
        calibration = fit_calibrators(cp, y_cal, cal_df[["game_type"]])
        platt = apply_calibration(vp, valid_df[["game_type"]], calibration, "platt")
        trend = (1.0 - TREND_WEIGHT) * platt + TREND_WEIGHT * trend_prior
        cal_platt = apply_calibration(cp, cal_df[["game_type"]], calibration, "platt")

        cal_raw[name] = cp
        valid_raw[name] = vp
        valid_platt[name] = platt
        oof[f"{name}_raw"] = vp
        oof[f"{name}_platt"] = platt
        oof[f"{name}_trend"] = trend

        runtime = {"train_seconds": train_s, "inference_seconds": infer_s, "model_size_mb": size_mb}
        runtime_rows.append({**fold_info, "model": name, **runtime})
        add_metric_rows(metric_rows, fold_info, name, "raw", y_valid, vp, trend_prior, runtime)
        add_metric_rows(metric_rows, fold_info, name, "platt", y_valid, platt, trend_prior, runtime)
        add_metric_rows(metric_rows, fold_info, name, "platt_plus_trend_w075", y_valid, trend, trend_prior, runtime)
        # Store calibration-set calibrated predictions for ensemble Platt averaging.
        cal_raw[f"{name}__platt"] = cal_platt

    ensemble_rows = []
    model_names = list(valid_raw)
    combo_sets = []
    for r in [2, 3]:
        combo_sets.extend(combinations(model_names, r))

    for combo in combo_sets:
        for weights in ensemble_weight_grid(combo):
            nonzero = {k: v for k, v in weights.items() if v > 0}
            if len(nonzero) < 2:
                continue
            label = "+".join(f"{k}:{v:.1f}" for k, v in nonzero.items())

            raw_pred = weighted_sum(valid_raw, nonzero)
            cal_ens = weighted_sum(cal_raw, nonzero)
            ens_calibration = fit_calibrators(cal_ens, y_cal, cal_df[["game_type"]])
            raw_platt_pred = apply_calibration(raw_pred, valid_df[["game_type"]], ens_calibration, "platt")
            platt_avg_pred = weighted_sum(valid_platt, nonzero)

            for stage, pred in [
                ("raw_avg", raw_pred),
                ("raw_avg_then_platt", raw_platt_pred),
                ("platt_avg", platt_avg_pred),
            ]:
                trend_pred = (1.0 - TREND_WEIGHT) * pred + TREND_WEIGHT * trend_prior
                for final_stage, final_pred in [(stage, pred), (f"{stage}_plus_trend_w075", trend_pred)]:
                    ensemble_rows.append(
                        {
                            **fold_info,
                            "ensemble": label,
                            "stage": final_stage,
                            "weights": repr(nonzero),
                            "trend_prior": trend_prior,
                            **metrics(y_valid, final_pred),
                        }
                    )
    return pd.DataFrame(metric_rows), pd.DataFrame(runtime_rows), oof, pd.DataFrame(ensemble_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", default="data/train.csv")
    parser.add_argument("--output-dir", default="output/model_comparison")
    args = parser.parse_args()

    df = pd.read_csv(args.train_path, encoding="utf-8-sig")
    os.makedirs(args.output_dir, exist_ok=True)
    estimators, availability = make_estimators()
    pd.DataFrame([{"model": k, **v} for k, v in availability.items()]).to_csv(
        os.path.join(args.output_dir, "library_availability.csv"), index=False, encoding="utf-8"
    )

    metric_parts = []
    runtime_parts = []
    oof_parts = []
    ensemble_parts = []
    for fold in FOLDS:
        m, r, o, e = run_fold(df, fold, make_estimators()[0])
        metric_parts.append(m)
        runtime_parts.append(r)
        oof_parts.append(o)
        ensemble_parts.append(e)

    model_metrics = pd.concat(metric_parts, ignore_index=True)
    runtime = pd.concat(runtime_parts, ignore_index=True)
    oof = pd.concat(oof_parts, ignore_index=True)
    ensemble_grid = pd.concat(ensemble_parts, ignore_index=True)

    current = model_metrics[(model_metrics["model"] == "sklearn_hgb") & (model_metrics["stage"] == "platt_plus_trend_w075")][
        ["valid_year", "brier"]
    ].rename(columns={"brier": "current_brier"})
    model_metrics = model_metrics.merge(current, on="valid_year", how="left")
    model_metrics["improvement_vs_current"] = model_metrics["current_brier"] - model_metrics["brier"]
    ensemble_grid = ensemble_grid.merge(current, on="valid_year", how="left")
    ensemble_grid["improvement_vs_current"] = ensemble_grid["current_brier"] - ensemble_grid["brier"]

    model_metrics.to_csv(os.path.join(args.output_dir, "model_fold_metrics.csv"), index=False, encoding="utf-8")
    runtime.to_csv(os.path.join(args.output_dir, "runtime_summary.csv"), index=False, encoding="utf-8")
    oof.to_csv(os.path.join(args.output_dir, "oof_predictions.csv"), index=False, encoding="utf-8")
    ensemble_grid.to_csv(os.path.join(args.output_dir, "ensemble_grid.csv"), index=False, encoding="utf-8")

    model_summary = (
        model_metrics.groupby(["model", "stage"])
        .agg(
            mean_brier=("brier", "mean"),
            std_brier=("brier", "std"),
            worst_brier=("brier", "max"),
            folds_improved=("improvement_vs_current", lambda s: int((s > 0).sum())),
            mean_logloss=("logloss", "mean"),
            mean_auc=("auc", "mean"),
            mean_pred_std=("pred_std", "mean"),
            mean_train_seconds=("train_seconds", "mean"),
            mean_inference_seconds=("inference_seconds", "mean"),
            mean_model_size_mb=("model_size_mb", "mean"),
        )
        .reset_index()
        .sort_values(["mean_brier", "std_brier"])
    )
    model_summary.to_csv(os.path.join(args.output_dir, "model_summary.csv"), index=False, encoding="utf-8")

    ensemble_summary = (
        ensemble_grid.groupby(["ensemble", "stage", "weights"])
        .agg(
            mean_brier=("brier", "mean"),
            std_brier=("brier", "std"),
            worst_brier=("brier", "max"),
            folds_improved=("improvement_vs_current", lambda s: int((s > 0).sum())),
            mean_logloss=("logloss", "mean"),
            mean_auc=("auc", "mean"),
            mean_pred_std=("pred_std", "mean"),
        )
        .reset_index()
        .sort_values(["mean_brier", "std_brier"])
    )
    ensemble_summary.to_csv(os.path.join(args.output_dir, "ensemble_summary.csv"), index=False, encoding="utf-8")

    print("\nModel summary")
    print(model_summary.head(30).to_string(index=False))
    print("\nEnsemble summary")
    print(ensemble_summary.head(30).to_string(index=False))
    print("\nRuntime")
    print(runtime.groupby("model").agg(train_seconds=("train_seconds", "mean"), inference_seconds=("inference_seconds", "mean"), model_size_mb=("model_size_mb", "mean")).reset_index().to_string(index=False))


if __name__ == "__main__":
    main()
