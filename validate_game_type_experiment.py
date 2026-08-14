import argparse
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from calibration_utils import apply_calibration, fit_calibrators, logit
from model_utils import FeatureBuilder, TARGET_COL, make_model


FOLDS = [
    (2019, 2020, 2021, 2022),
    (2019, 2021, 2022, 2023),
    (2019, 2022, 2023, 2024),
]
WEIGHT = 0.75
ALPHAS = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
MIN_PLATT_GROUP_N = 5000
MIN_MODEL_GROUP_N = 20000


def clip_prob(pred):
    return np.clip(np.asarray(pred, dtype=np.float64), 1e-6, 1.0 - 1e-6)


def linear_forecast(year_rates: pd.Series) -> float:
    s = year_rates.dropna().sort_index()
    if len(s) == 0:
        raise ValueError("No history for prior.")
    if len(s) == 1:
        return float(s.iloc[-1])
    x = s.index.to_numpy(dtype=np.float64)
    y = s.to_numpy(dtype=np.float64)
    slope, intercept = np.polyfit(x, y, 1)
    return float(np.clip(intercept + slope * (x[-1] + 1), 0.42, 0.62))


def game_type_priors(history_df: pd.DataFrame, alpha: float = 1.0) -> tuple[float, dict]:
    global_prior = linear_forecast(history_df.groupby("season")[TARGET_COL].mean())
    priors = {}
    for game_type in ["R", "F"]:
        sub = history_df[history_df["game_type"] == game_type]
        if sub["season"].nunique() < 2:
            group_prior = global_prior
        else:
            group_prior = linear_forecast(sub.groupby("season")[TARGET_COL].mean())
        priors[game_type] = float(alpha * group_prior + (1.0 - alpha) * global_prior)
    return global_prior, priors


def map_prior(df: pd.DataFrame, priors: dict, fallback: float) -> np.ndarray:
    return df["game_type"].map(priors).fillna(fallback).to_numpy(dtype=np.float64)


def fit_group_platt(raw_pred, y, game_type_values):
    raw_pred = clip_prob(raw_pred)
    y = np.asarray(y, dtype=np.int8)
    game_type_values = pd.Series(game_type_values).astype(str).reset_index(drop=True)
    models = {}
    for game_type in ["R", "F"]:
        idx = np.flatnonzero(game_type_values.to_numpy() == game_type)
        if len(idx) < MIN_PLATT_GROUP_N or len(np.unique(y[idx])) < 2:
            continue
        model = LogisticRegression(C=1e6, solver="lbfgs", random_state=42)
        model.fit(logit(raw_pred[idx]).reshape(-1, 1), y[idx])
        models[game_type] = model
    return models


def apply_group_platt(raw_pred, game_type_values, group_models, global_calibration):
    out = apply_calibration(raw_pred, pd.DataFrame({"game_type": game_type_values}), global_calibration, "platt")
    raw_pred = clip_prob(raw_pred)
    game_type_values = pd.Series(game_type_values).astype(str).reset_index(drop=True)
    for game_type, model in group_models.items():
        idx = np.flatnonzero(game_type_values.to_numpy() == game_type)
        if len(idx):
            out[idx] = model.predict_proba(logit(raw_pred[idx]).reshape(-1, 1))[:, 1]
    return clip_prob(out)


def score_by_type(valid_df: pd.DataFrame, y, pred, strategy: str, fold_info: dict, global_prior, priors, alpha):
    pred = clip_prob(pred)
    y = np.asarray(y, dtype=np.int8)
    rows = []
    overall = {
        "segment": "overall",
        "n": len(valid_df),
        "brier": float(brier_score_loss(y, pred)),
        "logloss": float(log_loss(y, pred)),
        "pred_mean": float(pred.mean()),
        "actual_rate": float(y.mean()),
    }
    rows.append(overall)
    for game_type in ["R", "F"]:
        idx = np.flatnonzero(valid_df["game_type"].astype(str).to_numpy() == game_type)
        if len(idx) == 0:
            continue
        rows.append(
            {
                "segment": game_type,
                "n": int(len(idx)),
                "brier": float(brier_score_loss(y[idx], pred[idx])),
                "logloss": float(log_loss(y[idx], pred[idx])),
                "pred_mean": float(pred[idx].mean()),
                "actual_rate": float(y[idx].mean()),
            }
        )
    out = pd.DataFrame(rows)
    out.insert(0, "strategy", strategy)
    out.insert(0, "alpha", alpha)
    out.insert(0, "f_prior", priors.get("F", np.nan))
    out.insert(0, "r_prior", priors.get("R", np.nan))
    out.insert(0, "global_prior", global_prior)
    for key, value in reversed(fold_info.items()):
        out.insert(0, key, value)
    return out


def yearly_stats(df: pd.DataFrame) -> pd.DataFrame:
    yearly_global = df.groupby("season")[TARGET_COL].mean().rename("global_rate")
    rows = []
    total_by_year = df.groupby("season").size().rename("year_n")
    for (season, game_type), g in df.groupby(["season", "game_type"]):
        rows.append(
            {
                "season": season,
                "game_type": game_type,
                "n": len(g),
                "control_success_rate": float(g[TARGET_COL].mean()),
                "year_share": float(len(g) / total_by_year.loc[season]),
                "global_rate": float(yearly_global.loc[season]),
                "diff_from_global": float(g[TARGET_COL].mean() - yearly_global.loc[season]),
            }
        )
    out = pd.DataFrame(rows).sort_values(["game_type", "season"])
    out["yoy_rate_change"] = out.groupby("game_type")["control_success_rate"].diff()
    return out


def train_global_predictions(train_df, cal_df, valid_df):
    y_train = train_df[TARGET_COL].astype("int8")
    builder = FeatureBuilder(alpha=80.0)
    X_train = builder.fit_transform(train_df.drop(columns=[TARGET_COL]), y_train)
    model = make_model()
    model.fit(X_train, y_train)

    X_cal = builder.transform(cal_df.drop(columns=[TARGET_COL]))
    y_cal = cal_df[TARGET_COL].astype("int8").to_numpy()
    cal_raw = model.predict_proba(X_cal)[:, 1]
    global_calibration = fit_calibrators(cal_raw, y_cal, cal_df[["game_type"]])
    cal_platt = apply_calibration(cal_raw, cal_df[["game_type"]], global_calibration, "platt")

    X_valid = builder.transform(valid_df.drop(columns=[TARGET_COL]))
    valid_raw = model.predict_proba(X_valid)[:, 1]
    valid_platt = apply_calibration(valid_raw, valid_df[["game_type"]], global_calibration, "platt")
    return {
        "builder": builder,
        "model": model,
        "global_calibration": global_calibration,
        "cal_raw": cal_raw,
        "cal_platt": cal_platt,
        "valid_raw": valid_raw,
        "valid_platt": valid_platt,
        "y_cal": y_cal,
    }


def separate_model_predictions(train_df, cal_df, valid_df, global_valid_platt):
    pred = np.asarray(global_valid_platt).copy()
    trained = {}
    for game_type in ["R", "F"]:
        tr = train_df[train_df["game_type"] == game_type].copy()
        ca = cal_df[cal_df["game_type"] == game_type].copy()
        va = valid_df[valid_df["game_type"] == game_type].copy()
        if len(tr) < MIN_MODEL_GROUP_N or len(ca) < MIN_PLATT_GROUP_N or len(va) == 0:
            trained[game_type] = False
            continue
        y_train = tr[TARGET_COL].astype("int8")
        builder = FeatureBuilder(alpha=80.0)
        X_train = builder.fit_transform(tr.drop(columns=[TARGET_COL]), y_train)
        model = make_model()
        model.fit(X_train, y_train)

        X_cal = builder.transform(ca.drop(columns=[TARGET_COL]))
        y_cal = ca[TARGET_COL].astype("int8").to_numpy()
        cal_raw = model.predict_proba(X_cal)[:, 1]
        calibration = fit_calibrators(cal_raw, y_cal, ca[["game_type"]])

        X_valid = builder.transform(va.drop(columns=[TARGET_COL]))
        valid_raw = model.predict_proba(X_valid)[:, 1]
        valid_platt = apply_calibration(valid_raw, va[["game_type"]], calibration, "platt")
        pred[valid_df["game_type"].astype(str).to_numpy() == game_type] = valid_platt
        trained[game_type] = True
    return clip_prob(pred), trained


def run_fold(df, fold):
    train_start, train_end, cal_year, valid_year = fold
    train_df = df[(df["season"] >= train_start) & (df["season"] <= train_end)].copy()
    cal_df = df[df["season"] == cal_year].copy()
    valid_df = df[df["season"] == valid_year].copy()
    history_df = pd.concat([train_df, cal_df], ignore_index=True)
    fold_info = {
        "train_start": train_start,
        "train_end": train_end,
        "cal_year": cal_year,
        "valid_year": valid_year,
    }
    print(
        f"Fold train={train_start}-{train_end} cal={cal_year} valid={valid_year} "
        f"rows train={len(train_df)} cal={len(cal_df)} valid={len(valid_df)}"
    )

    preds = train_global_predictions(train_df, cal_df, valid_df)
    y_valid = valid_df[TARGET_COL].astype("int8").to_numpy()
    global_prior, raw_group_priors = game_type_priors(history_df, alpha=1.0)

    rows = []
    prior_rows = []

    current_prior = np.full(len(valid_df), global_prior)
    current_best = (1.0 - WEIGHT) * preds["valid_platt"] + WEIGHT * current_prior
    rows.append(score_by_type(valid_df, y_valid, preds["valid_raw"], "raw_model", fold_info, global_prior, raw_group_priors, np.nan))
    rows.append(score_by_type(valid_df, y_valid, preds["valid_platt"], "global_platt", fold_info, global_prior, raw_group_priors, np.nan))
    rows.append(score_by_type(valid_df, y_valid, current_prior, "global_linear_prior", fold_info, global_prior, raw_group_priors, np.nan))
    rows.append(score_by_type(valid_df, y_valid, current_best, "current_best_global_prior_w075", fold_info, global_prior, raw_group_priors, np.nan))

    group_platt_models = fit_group_platt(preds["cal_raw"], preds["y_cal"], cal_df["game_type"])
    valid_sep_platt = apply_group_platt(preds["valid_raw"], valid_df["game_type"], group_platt_models, preds["global_calibration"])
    rows.append(score_by_type(valid_df, y_valid, valid_sep_platt, "separate_platt", fold_info, global_prior, raw_group_priors, np.nan))

    for alpha in ALPHAS:
        _, group_priors = game_type_priors(history_df, alpha=alpha)
        prior_vec = map_prior(valid_df, group_priors, global_prior)
        blended = (1.0 - WEIGHT) * preds["valid_platt"] + WEIGHT * prior_vec
        sep_platt_blended = (1.0 - WEIGHT) * valid_sep_platt + WEIGHT * prior_vec
        rows.append(score_by_type(valid_df, y_valid, prior_vec, f"game_type_prior_alpha_{alpha}", fold_info, global_prior, group_priors, alpha))
        rows.append(score_by_type(valid_df, y_valid, blended, f"global_platt_plus_game_type_prior_alpha_{alpha}", fold_info, global_prior, group_priors, alpha))
        rows.append(score_by_type(valid_df, y_valid, sep_platt_blended, f"separate_platt_plus_game_type_prior_alpha_{alpha}", fold_info, global_prior, group_priors, alpha))
        prior_rows.append(
            {
                **fold_info,
                "alpha": alpha,
                "global_prior": global_prior,
                "r_prior": group_priors["R"],
                "f_prior": group_priors["F"],
                "raw_r_prior": raw_group_priors["R"],
                "raw_f_prior": raw_group_priors["F"],
                "r_actual": float(valid_df.loc[valid_df["game_type"] == "R", TARGET_COL].mean()),
                "f_actual": float(valid_df.loc[valid_df["game_type"] == "F", TARGET_COL].mean()),
                "r_n": int((valid_df["game_type"] == "R").sum()),
                "f_n": int((valid_df["game_type"] == "F").sum()),
            }
        )

    sep_model_pred, trained = separate_model_predictions(train_df, cal_df, valid_df, preds["valid_platt"])
    rows.append(score_by_type(valid_df, y_valid, sep_model_pred, f"separate_model_platt_R{int(trained.get('R', False))}_F{int(trained.get('F', False))}", fold_info, global_prior, raw_group_priors, np.nan))
    sep_model_blend = (1.0 - WEIGHT) * sep_model_pred + WEIGHT * current_prior
    rows.append(score_by_type(valid_df, y_valid, sep_model_blend, "separate_model_platt_plus_global_prior", fold_info, global_prior, raw_group_priors, np.nan))

    return pd.concat(rows, ignore_index=True), pd.DataFrame(prior_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", default="data/train.csv")
    parser.add_argument("--output-dir", default="output/game_type_experiment")
    args = parser.parse_args()

    df = pd.read_csv(args.train_path, encoding="utf-8-sig")
    os.makedirs(args.output_dir, exist_ok=True)

    stats = yearly_stats(df)
    stats.to_csv(os.path.join(args.output_dir, "game_type_yearly_stats.csv"), index=False, encoding="utf-8")

    metric_parts = []
    prior_parts = []
    for fold in FOLDS:
        metrics, priors = run_fold(df, fold)
        metric_parts.append(metrics)
        prior_parts.append(priors)

    metrics = pd.concat(metric_parts, ignore_index=True)
    priors = pd.concat(prior_parts, ignore_index=True)

    current = metrics[(metrics["strategy"] == "current_best_global_prior_w075") & (metrics["segment"] == "overall")][
        ["valid_year", "brier"]
    ].rename(columns={"brier": "current_brier"})
    metrics = metrics.merge(current, on="valid_year", how="left")
    metrics["improvement_vs_current"] = metrics["current_brier"] - metrics["brier"]

    metrics_path = os.path.join(args.output_dir, "fold_strategy_metrics.csv")
    priors_path = os.path.join(args.output_dir, "game_type_prior_predictions.csv")
    metrics.to_csv(metrics_path, index=False, encoding="utf-8")
    priors.to_csv(priors_path, index=False, encoding="utf-8")

    overall = metrics[metrics["segment"] == "overall"]
    summary = (
        overall.groupby(["strategy", "alpha"], dropna=False)
        .agg(
            mean_brier=("brier", "mean"),
            std_brier=("brier", "std"),
            worst_brier=("brier", "max"),
            folds_improved=("improvement_vs_current", lambda s: int((s > 0).sum())),
            mean_improvement=("improvement_vs_current", "mean"),
            mean_pred=("pred_mean", "mean"),
            mean_actual=("actual_rate", "mean"),
        )
        .reset_index()
        .sort_values(["mean_brier", "std_brier"])
    )
    summary_path = os.path.join(args.output_dir, "shrink_alpha_summary.csv")
    summary.to_csv(summary_path, index=False, encoding="utf-8")

    print("\nYearly game_type stats")
    print(stats.to_string(index=False))
    print("\nTop strategy summary")
    print(summary.head(20).to_string(index=False))
    print("\nFold overall selected")
    selected = overall[
        overall["strategy"].isin(
            [
                "current_best_global_prior_w075",
                "global_platt",
                "global_linear_prior",
                "separate_platt",
                "global_platt_plus_game_type_prior_alpha_0.1",
                "global_platt_plus_game_type_prior_alpha_0.2",
                "global_platt_plus_game_type_prior_alpha_0.3",
                "separate_platt_plus_game_type_prior_alpha_0.1",
            ]
        )
    ]
    print(selected[["valid_year", "strategy", "alpha", "brier", "improvement_vs_current", "pred_mean", "actual_rate", "global_prior", "r_prior", "f_prior"]].to_string(index=False))
    print(f"\nSaved: {metrics_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {priors_path}")


if __name__ == "__main__":
    main()
