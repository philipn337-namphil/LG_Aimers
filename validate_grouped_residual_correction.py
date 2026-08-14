import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from calibration_utils import apply_calibration, fit_calibrators, logit, sigmoid
from model_utils import FeatureBuilder, TARGET_COL, make_model


FOLDS = [
    (2019, 2020, 2021, 2022),
    (2019, 2021, 2022, 2023),
    (2019, 2022, 2023, 2024),
]

BEST_WEIGHT = 0.75
LAMBDAS = [50, 100, 300, 500, 1000, 3000]
GROUP_SPECS = {
    "game_type": ["game_type"],
    "month": ["game_month"],
    "balls": ["balls_before"],
    "strikes": ["strikes_before"],
    "count_state": ["count_state"],
    "batter_side": ["batter_hand"],
    "inning_bucket": ["inning_bucket"],
}


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


def add_group_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["count_state"] = out["balls_before"].astype(str) + "-" + out["strikes_before"].astype(str)
    inning = pd.to_numeric(out["inning"], errors="coerce")
    out["inning_bucket"] = pd.cut(
        inning,
        bins=[0, 3, 6, 9, 99],
        labels=["1-3", "4-6", "7-9", "10+"],
        include_lowest=True,
    ).astype(str)
    return out


def score(y, pred):
    pred = clip_prob(pred)
    return {
        "brier": float(brier_score_loss(y, pred)),
        "logloss": float(log_loss(y, pred)),
        "pred_mean": float(pred.mean()),
        "pred_std": float(pred.std()),
        "actual_rate": float(np.mean(y)),
        "bias": float(pred.mean() - np.mean(y)),
    }


def group_key_series(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    if len(cols) == 1:
        return df[cols[0]].astype(str)
    return df[cols].astype(str).agg("|".join, axis=1)


def residual_summary(valid_df: pd.DataFrame, y, pred, group_name: str, cols: list[str]) -> pd.DataFrame:
    tmp = pd.DataFrame(
        {
            "group_value": group_key_series(valid_df, cols),
            "y": np.asarray(y, dtype=np.float64),
            "pred": clip_prob(pred),
        }
    )
    rows = []
    for value, g in tmp.groupby("group_value", dropna=False):
        rows.append(
            {
                "group_name": group_name,
                "group_value": value,
                "n": len(g),
                "actual_rate": float(g["y"].mean()),
                "pred_mean": float(g["pred"].mean()),
                "residual_actual_minus_pred": float(g["y"].mean() - g["pred"].mean()),
                "brier": float(brier_score_loss(g["y"], g["pred"])),
            }
        )
    return pd.DataFrame(rows)


def fit_offsets(cal_df: pd.DataFrame, y_cal, pred_cal, group_name: str, cols: list[str], lam: int, space: str) -> dict:
    tmp = pd.DataFrame(
        {
            "key": group_key_series(cal_df, cols),
            "y": np.asarray(y_cal, dtype=np.float64),
            "pred": clip_prob(pred_cal),
        }
    )
    offsets = {}
    for key, g in tmp.groupby("key", dropna=False):
        n = len(g)
        shrink = n / (n + lam)
        if space == "prob":
            raw_offset = float(g["y"].mean() - g["pred"].mean())
        elif space == "logit":
            raw_offset = float(logit(g["y"].mean()) - np.mean(logit(g["pred"])))
        else:
            raise ValueError(space)
        offsets[key] = shrink * raw_offset
    return offsets


def apply_offsets(valid_df: pd.DataFrame, pred, cols: list[str], offsets: dict, space: str):
    keys = group_key_series(valid_df, cols)
    offset = keys.map(offsets).fillna(0.0).to_numpy(dtype=np.float64)
    pred = clip_prob(pred)
    if space == "prob":
        return clip_prob(pred + offset)
    if space == "logit":
        return clip_prob(sigmoid(logit(pred) + offset))
    raise ValueError(space)


def run_fold(df: pd.DataFrame, fold: tuple[int, int, int, int]):
    train_start, train_end, cal_year, valid_year = fold
    train_df = add_group_columns(df[(df["season"] >= train_start) & (df["season"] <= train_end)].copy())
    cal_df = add_group_columns(df[df["season"] == cal_year].copy())
    valid_df = add_group_columns(df[df["season"] == valid_year].copy())

    print(
        f"Fold train={train_start}-{train_end} cal={cal_year} valid={valid_year} "
        f"rows train={len(train_df)} cal={len(cal_df)} valid={len(valid_df)}"
    )

    y_train = train_df[TARGET_COL].astype("int8")
    builder = FeatureBuilder(alpha=80.0)
    X_train = builder.fit_transform(train_df.drop(columns=[TARGET_COL]), y_train)
    model = make_model()
    model.fit(X_train, y_train)

    X_cal = builder.transform(cal_df.drop(columns=[TARGET_COL]))
    y_cal = cal_df[TARGET_COL].astype("int8").to_numpy()
    cal_raw = model.predict_proba(X_cal)[:, 1]
    calibration = fit_calibrators(cal_raw, y_cal, cal_df[["game_type"]])
    cal_platt = apply_calibration(cal_raw, cal_df[["game_type"]], calibration, "platt")

    X_valid = builder.transform(valid_df.drop(columns=[TARGET_COL]))
    y_valid = valid_df[TARGET_COL].astype("int8").to_numpy()
    valid_raw = model.predict_proba(X_valid)[:, 1]
    valid_platt = apply_calibration(valid_raw, valid_df[["game_type"]], calibration, "platt")

    cal_prior = linear_forecast(train_df.groupby("season")[TARGET_COL].mean())
    valid_prior = linear_forecast(pd.concat([train_df, cal_df]).groupby("season")[TARGET_COL].mean())
    cal_best = (1.0 - BEST_WEIGHT) * cal_platt + BEST_WEIGHT * cal_prior
    valid_best = (1.0 - BEST_WEIGHT) * valid_platt + BEST_WEIGHT * valid_prior

    base_metrics = []
    for method, pred in [
        ("raw_model", valid_raw),
        ("platt", valid_platt),
        ("linear_all_history_prior", np.full(len(valid_df), valid_prior)),
        ("current_best", valid_best),
    ]:
        base_metrics.append(
            {
                "train_start": train_start,
                "train_end": train_end,
                "cal_year": cal_year,
                "valid_year": valid_year,
                "method": method,
                "group_name": "",
                "lambda": np.nan,
                "space": "",
                "prior_rate": valid_prior if method in ["linear_all_history_prior", "current_best"] else np.nan,
                **score(y_valid, pred),
            }
        )

    residual_rows = []
    for group_name, cols in GROUP_SPECS.items():
        rs = residual_summary(valid_df, y_valid, valid_best, group_name, cols)
        rs.insert(0, "valid_year", valid_year)
        rs.insert(0, "cal_year", cal_year)
        residual_rows.append(rs)

    correction_rows = []
    corrected_predictions = {}
    for group_name, cols in GROUP_SPECS.items():
        for space in ["prob", "logit"]:
            for lam in LAMBDAS:
                offsets = fit_offsets(cal_df, y_cal, cal_best, group_name, cols, lam, space)
                pred = apply_offsets(valid_df, valid_best, cols, offsets, space)
                corrected_predictions[(group_name, space, lam)] = pred
                correction_rows.append(
                    {
                        "train_start": train_start,
                        "train_end": train_end,
                        "cal_year": cal_year,
                        "valid_year": valid_year,
                        "method": f"current_best_plus_{group_name}",
                        "group_name": group_name,
                        "lambda": lam,
                        "space": space,
                        "prior_rate": valid_prior,
                        **score(y_valid, pred),
                    }
                )

    return base_metrics, pd.concat(residual_rows, ignore_index=True), correction_rows, corrected_predictions, valid_df, y_valid, valid_best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", default="data/train.csv")
    parser.add_argument("--output-dir", default="output/grouped_residual")
    args = parser.parse_args()

    df = pd.read_csv(args.train_path, encoding="utf-8-sig")
    os.makedirs(args.output_dir, exist_ok=True)

    all_base = []
    all_residuals = []
    all_corrections = []
    fold_payloads = {}

    for fold in FOLDS:
        base, residuals, corrections, preds, valid_df, y_valid, valid_best = run_fold(df, fold)
        all_base.extend(base)
        all_residuals.append(residuals)
        all_corrections.extend(corrections)
        fold_payloads[fold[3]] = {
            "preds": preds,
            "valid_df": valid_df,
            "y": y_valid,
            "base": valid_best,
        }

    base_df = pd.DataFrame(all_base)
    residual_df = pd.concat(all_residuals, ignore_index=True)
    correction_df = pd.DataFrame(all_corrections)

    base_path = os.path.join(args.output_dir, "base_method_metrics.csv")
    residual_path = os.path.join(args.output_dir, "validation_group_residuals.csv")
    correction_path = os.path.join(args.output_dir, "single_group_corrections.csv")
    base_df.to_csv(base_path, index=False, encoding="utf-8")
    residual_df.to_csv(residual_path, index=False, encoding="utf-8")
    correction_df.to_csv(correction_path, index=False, encoding="utf-8")

    summary = (
        pd.concat([base_df, correction_df], ignore_index=True)
        .groupby(["method", "group_name", "space", "lambda"], dropna=False)
        .agg(
            mean_brier=("brier", "mean"),
            std_brier=("brier", "std"),
            max_brier=("brier", "max"),
            mean_logloss=("logloss", "mean"),
            mean_pred=("pred_mean", "mean"),
            mean_pred_std=("pred_std", "mean"),
            mean_actual=("actual_rate", "mean"),
            mean_bias=("bias", "mean"),
            folds=("valid_year", "nunique"),
        )
        .reset_index()
        .sort_values(["mean_brier", "std_brier"])
    )
    summary_path = os.path.join(args.output_dir, "correction_summary.csv")
    summary.to_csv(summary_path, index=False, encoding="utf-8")

    stable = correction_df.groupby(["group_name", "space", "lambda"]).agg(
        mean_brier=("brier", "mean"),
        folds_improved=("brier", lambda s: np.nan),
    )
    current_by_year = base_df[base_df["method"] == "current_best"].set_index("valid_year")["brier"]
    improved_rows = []
    for key, g in correction_df.groupby(["group_name", "space", "lambda"]):
        diffs = []
        for _, row in g.iterrows():
            diffs.append(current_by_year.loc[row["valid_year"]] - row["brier"])
        improved_rows.append(
            {
                "group_name": key[0],
                "space": key[1],
                "lambda": key[2],
                "mean_improvement": float(np.mean(diffs)),
                "min_improvement": float(np.min(diffs)),
                "folds_improved": int(np.sum(np.asarray(diffs) > 0)),
                "mean_brier": float(g["brier"].mean()),
                "std_brier": float(g["brier"].std()),
            }
        )
    stable_df = pd.DataFrame(improved_rows).sort_values(["mean_brier", "std_brier"])
    stable_path = os.path.join(args.output_dir, "stable_correction_candidates.csv")
    stable_df.to_csv(stable_path, index=False, encoding="utf-8")

    stable_two = stable_df[(stable_df["folds_improved"] >= 2) & (stable_df["mean_improvement"] > 0)]
    chosen = stable_two.sort_values(["folds_improved", "mean_improvement"], ascending=[False, False]).head(2)

    combo_rows = []
    if len(chosen) >= 2:
        correction_keys = [tuple(x) for x in chosen[["group_name", "space", "lambda"]].to_numpy()]
        for valid_year, payload in fold_payloads.items():
            pred = payload["base"].copy()
            valid_df = payload["valid_df"]
            # Rebuild sequential corrections by taking the delta from each already-computed single correction.
            # This keeps the combination conservative and avoids refitting on validation.
            for key in correction_keys:
                corrected = payload["preds"][key]
                pred = clip_prob(pred + (corrected - payload["base"]))
            combo_rows.append(
                {
                    "valid_year": valid_year,
                    "method": "combo_top2_additive_deltas",
                    "combo": " + ".join(f"{g}:{s}:{int(l)}" for g, s, l in correction_keys),
                    **score(payload["y"], pred),
                }
            )
    combo_df = pd.DataFrame(combo_rows)
    combo_path = os.path.join(args.output_dir, "combo_corrections.csv")
    combo_df.to_csv(combo_path, index=False, encoding="utf-8")

    print("\nBase metrics")
    print(base_df[["valid_year", "method", "brier", "pred_mean", "pred_std", "actual_rate", "bias"]].to_string(index=False))
    print("\nTop correction summary")
    print(summary.head(15).to_string(index=False))
    print("\nStable candidates")
    print(stable_df.head(15).to_string(index=False))
    if len(combo_df):
        print("\nCombo")
        print(combo_df.to_string(index=False))
    print(f"\nSaved: {base_path}")
    print(f"Saved: {residual_path}")
    print(f"Saved: {correction_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {stable_path}")
    print(f"Saved: {combo_path}")


if __name__ == "__main__":
    main()
