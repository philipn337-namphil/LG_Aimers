import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from calibration_utils import apply_calibration, fit_calibrators
from model_utils import FeatureBuilder, TARGET_COL, make_model


FOLDS = [
    (2019, 2020, 2021, 2022),
    (2019, 2021, 2022, 2023),
    (2019, 2022, 2023, 2024),
]
WEIGHT = 0.75
FIXED_ALPHAS = [0.05, 0.10, 0.15]
SAMPLE_LAMBDAS = [50_000, 100_000, 300_000]
VOL_KS = [5, 10, 20, 50]
COMBINED_LAMBDAS = [100_000, 300_000]
COMBINED_KS = [10, 20, 50]


def clip_prob(pred):
    return np.clip(np.asarray(pred, dtype=np.float64), 1e-6, 1.0 - 1e-6)


def linear_forecast(year_rates: pd.Series) -> tuple[float, float]:
    s = year_rates.dropna().sort_index()
    if len(s) == 0:
        raise ValueError("No rate history.")
    if len(s) == 1:
        return float(s.iloc[-1]), 0.0
    x = s.index.to_numpy(dtype=np.float64)
    y = s.to_numpy(dtype=np.float64)
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    resid_std = float(np.std(y - fitted, ddof=1)) if len(y) > 2 else float(abs(y[-1] - y[0]) / 2)
    return float(np.clip(intercept + slope * (x[-1] + 1), 0.42, 0.62)), resid_std


def group_stability(history: pd.DataFrame) -> tuple[float, dict, pd.DataFrame]:
    global_prior, global_resid_std = linear_forecast(history.groupby("season")[TARGET_COL].mean())
    rows = []
    priors = {}
    for game_type in ["R", "F"]:
        sub = history[history["game_type"] == game_type]
        yearly = sub.groupby("season").agg(
            annual_success_rate=(TARGET_COL, "mean"),
            sample_count=(TARGET_COL, "size"),
        )
        if len(yearly) >= 2:
            raw_prior, resid_std = linear_forecast(yearly["annual_success_rate"])
        else:
            raw_prior, resid_std = global_prior, global_resid_std
        yoy = yearly["annual_success_rate"].diff().abs().dropna()
        volatility = float(yoy.mean() + resid_std)
        n_total = int(yearly["sample_count"].sum())
        n_eff = float((yearly["sample_count"].sum() ** 2) / (np.square(yearly["sample_count"]).sum()))
        priors[game_type] = raw_prior
        rows.append(
            {
                "game_type": game_type,
                "history_start": int(history["season"].min()),
                "history_end": int(history["season"].max()),
                "years": int(len(yearly)),
                "total_n": n_total,
                "effective_years": n_eff,
                "rate_mean": float(yearly["annual_success_rate"].mean()),
                "rate_std": float(yearly["annual_success_rate"].std(ddof=1)) if len(yearly) > 1 else 0.0,
                "mean_abs_yoy_change": float(yoy.mean()) if len(yoy) else 0.0,
                "linear_trend_residual_std": resid_std,
                "volatility": volatility,
                "raw_group_prior": raw_prior,
                "global_prior": global_prior,
                "raw_prior_minus_global": raw_prior - global_prior,
            }
        )
    return global_prior, priors, pd.DataFrame(rows)


def alpha_specs(stability: pd.DataFrame):
    specs = []
    for a in FIXED_ALPHAS:
        specs.append((f"fixed_alpha_{a:.2f}", {"R": a, "F": a}))

    stats = stability.set_index("game_type")
    for lam in SAMPLE_LAMBDAS:
        specs.append(
            (
                f"sample_size_lambda_{lam}",
                {g: float(np.clip(stats.loc[g, "total_n"] / (stats.loc[g, "total_n"] + lam), 0, 1)) for g in ["R", "F"]},
            )
        )
    for k in VOL_KS:
        specs.append(
            (
                f"volatility_k_{k}",
                {g: float(np.clip(1.0 / (1.0 + k * stats.loc[g, "volatility"]), 0, 1)) for g in ["R", "F"]},
            )
        )
    for lam in COMBINED_LAMBDAS:
        for k in COMBINED_KS:
            specs.append(
                (
                    f"sample_vol_lambda_{lam}_k_{k}",
                    {
                        g: float(
                            np.clip(
                                (stats.loc[g, "total_n"] / (stats.loc[g, "total_n"] + lam))
                                * (1.0 / (1.0 + k * stats.loc[g, "volatility"])),
                                0,
                                1,
                            )
                        )
                        for g in ["R", "F"]
                    },
                )
            )
    return specs


def score(valid_df, y, pred, strategy, fold_info, priors_info):
    pred = clip_prob(pred)
    y = np.asarray(y, dtype=np.int8)
    rows = []
    for segment in ["overall", "R", "F"]:
        if segment == "overall":
            idx = np.arange(len(valid_df))
        else:
            idx = np.flatnonzero(valid_df["game_type"].astype(str).to_numpy() == segment)
        if len(idx) == 0:
            continue
        rows.append(
            {
                **fold_info,
                "strategy": strategy,
                "segment": segment,
                "n": int(len(idx)),
                "brier": float(brier_score_loss(y[idx], pred[idx])),
                "logloss": float(log_loss(y[idx], pred[idx])),
                "pred_mean": float(pred[idx].mean()),
                "pred_std": float(pred[idx].std()),
                "actual_rate": float(y[idx].mean()),
                **priors_info,
            }
        )
    return rows


def train_global(train_df, cal_df, valid_df):
    y_train = train_df[TARGET_COL].astype("int8")
    builder = FeatureBuilder(alpha=80.0)
    X_train = builder.fit_transform(train_df.drop(columns=[TARGET_COL]), y_train)
    model = make_model()
    model.fit(X_train, y_train)

    X_cal = builder.transform(cal_df.drop(columns=[TARGET_COL]))
    y_cal = cal_df[TARGET_COL].astype("int8").to_numpy()
    cal_raw = model.predict_proba(X_cal)[:, 1]
    calibration = fit_calibrators(cal_raw, y_cal, cal_df[["game_type"]])

    X_valid = builder.transform(valid_df.drop(columns=[TARGET_COL]))
    valid_raw = model.predict_proba(X_valid)[:, 1]
    valid_platt = apply_calibration(valid_raw, valid_df[["game_type"]], calibration, "platt")
    return valid_raw, valid_platt


def prior_vector(valid_df, global_prior, raw_priors, alphas):
    shrunk = {
        g: alphas[g] * raw_priors[g] + (1.0 - alphas[g]) * global_prior
        for g in ["R", "F"]
    }
    vec = valid_df["game_type"].map(shrunk).fillna(global_prior).to_numpy(dtype=np.float64)
    return vec, shrunk


def shift_distribution(df: pd.DataFrame, years=(2022, 2023)) -> pd.DataFrame:
    f = df[(df["game_type"] == "F") & (df["season"].isin(years))].copy()
    f["count_state"] = f["balls_before"].astype(str) + "-" + f["strikes_before"].astype(str)
    inning = pd.to_numeric(f["inning"], errors="coerce")
    f["inning_bucket"] = pd.cut(
        inning,
        bins=[0, 3, 6, 9, 99],
        labels=["1-3", "4-6", "7-9", "10+"],
        include_lowest=True,
    ).astype(str)
    specs = {
        "month": "game_month",
        "balls": "balls_before",
        "strikes": "strikes_before",
        "count_state": "count_state",
        "batter_hand": "batter_hand",
        "inning_bucket": "inning_bucket",
    }
    rows = []
    for feature, col in specs.items():
        total = f.groupby("season").size()
        for (season, value), g in f.groupby(["season", col], dropna=False):
            rows.append(
                {
                    "feature": feature,
                    "value": value,
                    "season": int(season),
                    "n": len(g),
                    "share": float(len(g) / total.loc[season]),
                    "success_rate": float(g[TARGET_COL].mean()),
                }
            )
    out = pd.DataFrame(rows)
    wide = out.pivot_table(index=["feature", "value"], columns="season", values=["n", "share", "success_rate"], fill_value=0)
    wide.columns = [f"{metric}_{season}" for metric, season in wide.columns]
    wide = wide.reset_index()
    if "share_2022" in wide.columns and "share_2023" in wide.columns:
        wide["share_delta_2023_minus_2022"] = wide["share_2023"] - wide["share_2022"]
    if "success_rate_2022" in wide.columns and "success_rate_2023" in wide.columns:
        wide["rate_delta_2023_minus_2022"] = wide["success_rate_2023"] - wide["success_rate_2022"]
    return wide.sort_values(["feature", "share_delta_2023_minus_2022"], ascending=[True, False])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", default="data/train.csv")
    parser.add_argument("--output-dir", default="output/uncertainty_prior")
    args = parser.parse_args()

    df = pd.read_csv(args.train_path, encoding="utf-8-sig")
    os.makedirs(args.output_dir, exist_ok=True)

    all_metrics = []
    all_stability = []
    all_priors = []
    for train_start, train_end, cal_year, valid_year in FOLDS:
        train_df = df[(df["season"] >= train_start) & (df["season"] <= train_end)].copy()
        cal_df = df[df["season"] == cal_year].copy()
        valid_df = df[df["season"] == valid_year].copy()
        history = pd.concat([train_df, cal_df], ignore_index=True)
        fold_info = {
            "train_start": train_start,
            "train_end": train_end,
            "cal_year": cal_year,
            "valid_year": valid_year,
        }
        print(f"Fold train={train_start}-{train_end} cal={cal_year} valid={valid_year}")

        _, platt = train_global(train_df, cal_df, valid_df)
        y_valid = valid_df[TARGET_COL].astype("int8").to_numpy()
        global_prior, raw_priors, stability = group_stability(history)
        stability = stability.assign(**fold_info)
        all_stability.append(stability)

        current_vec = np.full(len(valid_df), global_prior)
        current_pred = (1.0 - WEIGHT) * platt + WEIGHT * current_vec
        current_info = {
            "global_prior": global_prior,
            "raw_r_prior": raw_priors["R"],
            "raw_f_prior": raw_priors["F"],
            "alpha_R": 0.0,
            "alpha_F": 0.0,
            "shrunk_r_prior": global_prior,
            "shrunk_f_prior": global_prior,
        }
        all_metrics.extend(score(valid_df, y_valid, current_pred, "current_best_global_prior_w075", fold_info, current_info))

        for name, alphas in alpha_specs(stability):
            pvec, shrunk = prior_vector(valid_df, global_prior, raw_priors, alphas)
            pred = (1.0 - WEIGHT) * platt + WEIGHT * pvec
            info = {
                "global_prior": global_prior,
                "raw_r_prior": raw_priors["R"],
                "raw_f_prior": raw_priors["F"],
                "alpha_R": alphas["R"],
                "alpha_F": alphas["F"],
                "shrunk_r_prior": shrunk["R"],
                "shrunk_f_prior": shrunk["F"],
            }
            all_metrics.extend(score(valid_df, y_valid, pred, name, fold_info, info))
            all_priors.append({**fold_info, "strategy": name, **info})

    metrics = pd.DataFrame(all_metrics)
    current = metrics[(metrics["strategy"] == "current_best_global_prior_w075") & (metrics["segment"] == "overall")][
        ["valid_year", "brier"]
    ].rename(columns={"brier": "current_brier"})
    metrics = metrics.merge(current, on="valid_year", how="left")
    metrics["improvement_vs_current"] = metrics["current_brier"] - metrics["brier"]

    stability_df = pd.concat(all_stability, ignore_index=True)
    priors_df = pd.DataFrame(all_priors)
    shift_df = shift_distribution(df)

    metrics.to_csv(os.path.join(args.output_dir, "fold_strategy_metrics.csv"), index=False, encoding="utf-8")
    stability_df.to_csv(os.path.join(args.output_dir, "game_type_stability_by_fold.csv"), index=False, encoding="utf-8")
    priors_df.to_csv(os.path.join(args.output_dir, "prior_alpha_predictions.csv"), index=False, encoding="utf-8")
    shift_df.to_csv(os.path.join(args.output_dir, "f_2022_2023_shift_analysis.csv"), index=False, encoding="utf-8")

    overall = metrics[metrics["segment"] == "overall"]
    fseg = metrics[metrics["segment"] == "F"][["valid_year", "strategy", "brier"]].rename(columns={"brier": "f_brier"})
    summary = (
        overall.groupby("strategy")
        .agg(
            mean_brier=("brier", "mean"),
            std_brier=("brier", "std"),
            worst_brier=("brier", "max"),
            folds_improved=("improvement_vs_current", lambda s: int((s > 0).sum())),
            mean_improvement=("improvement_vs_current", "mean"),
            mean_alpha_R=("alpha_R", "mean"),
            mean_alpha_F=("alpha_F", "mean"),
            std_alpha_R=("alpha_R", "std"),
            std_alpha_F=("alpha_F", "std"),
        )
        .reset_index()
    )
    b2023 = overall[overall["valid_year"] == 2023][["strategy", "brier"]].rename(columns={"brier": "brier_2023"})
    f2023 = fseg[fseg["valid_year"] == 2023][["strategy", "f_brier"]].rename(columns={"f_brier": "f_brier_2023"})
    summary = summary.merge(b2023, on="strategy", how="left").merge(f2023, on="strategy", how="left")
    summary = summary.sort_values(["mean_brier", "worst_brier"])
    summary.to_csv(os.path.join(args.output_dir, "strategy_summary.csv"), index=False, encoding="utf-8")

    print("\nTop summary")
    print(summary.head(20).to_string(index=False))
    print("\nStability")
    print(stability_df.to_string(index=False))
    print("\nF 2022->2023 largest share shifts")
    print(shift_df.reindex(shift_df["share_delta_2023_minus_2022"].abs().sort_values(ascending=False).index).head(20).to_string(index=False))


if __name__ == "__main__":
    main()
