import argparse
import os
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from pandas.util import hash_pandas_object
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from calibration_utils import apply_calibration, fit_calibrators
from model_utils import BASE_NUMERIC_COLS, CAT_CODE_COLS, FeatureBuilder, TARGET_COL


FOLDS = [
    (2019, 2020, 2021, 2022),
    (2019, 2021, 2022, 2023),
    (2019, 2022, 2023, 2024),
]
CATBOOST_PARAMS = {
    "loss_function": "Logloss",
    "iterations": 220,
    "learning_rate": 0.045,
    "depth": 6,
    "l2_leaf_reg": 5.0,
    "random_seed": 42,
    "verbose": False,
    "allow_writing_files": False,
    "thread_count": -1,
}

PITCHER_ASOF = [
    "asof_pitcher_n",
    "asof_pitcher_success_rate",
    "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate",
    "asof_pitcher_pitchmix_n",
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
]
BATTER_ASOF = ["asof_batter_n", "asof_batter_success_rate", "asof_batter_middle_rate"]
CONTEXT_COLS = [
    "home_win_expectancy",
    "away_win_expectancy",
    "li",
    "score_diff_pitcher_team",
    "num_runners_on",
    "base_state",
    "game_type",
    "inning",
]
COUNT_COLS = ["balls_before", "strikes_before", "outs_before"]

FEATURE_SETS = {
    "current_baseline": [],
    "pitcher_asof_momentum": ["pitcher_momentum"],
    "batter_asof_deltas": ["batter_deltas"],
    "context_asof_interactions": ["context_asof"],
    "count_interactions": ["count_interactions"],
    "identity_context_interactions": ["identity_context"],
    "all_safe_official_interactions": [
        "pitcher_momentum",
        "batter_deltas",
        "context_asof",
        "count_interactions",
        "identity_context",
    ],
}


def clip_prob(pred):
    return np.clip(np.asarray(pred, dtype=np.float64), 1e-6, 1.0 - 1e-6)


def stable_code(df: pd.DataFrame, cols) -> pd.Series:
    key = df[list(cols)].astype("string").fillna("__NA__").agg("|".join, axis=1)
    return (hash_pandas_object(key, index=False).to_numpy(dtype=np.uint64) % 1_000_003).astype("float32")


def add_basic(df: pd.DataFrame) -> pd.DataFrame:
    return df.copy()


def add_feature_blocks(df: pd.DataFrame, blocks) -> pd.DataFrame:
    out = df.copy()
    blocks = set(blocks)
    if "pitcher_momentum" in blocks:
        out["rf_prev1_minus_prev5_success"] = out["asof_pitcher_prev1_game_success_rate"] - out["asof_pitcher_prev5_game_success_rate"]
        out["rf_prev3_minus_prev5_success"] = out["asof_pitcher_prev3_game_success_rate"] - out["asof_pitcher_prev5_game_success_rate"]
        out["rf_prev1_minus_prev3_success"] = out["asof_pitcher_prev1_game_success_rate"] - out["asof_pitcher_prev3_game_success_rate"]
        out["rf_pitcher_success_minus_prev5"] = out["asof_pitcher_success_rate"] - out["asof_pitcher_prev5_game_success_rate"]
        out["rf_prev1_minus_prev5_middle"] = out["asof_pitcher_prev1_game_middle_rate"] - out["asof_pitcher_prev5_game_middle_rate"]
        out["rf_prev3_minus_prev5_middle"] = out["asof_pitcher_prev3_game_middle_rate"] - out["asof_pitcher_prev5_game_middle_rate"]
    if "batter_deltas" in blocks:
        out["rf_batter_success_minus_pitcher_success"] = out["asof_batter_success_rate"] - out["asof_pitcher_success_rate"]
        out["rf_batter_middle_minus_pitcher_middle"] = out["asof_batter_middle_rate"] - out["asof_pitcher_middle_rate"]
        out["rf_batter_n_log1p"] = np.log1p(pd.to_numeric(out["asof_batter_n"], errors="coerce"))
        out["rf_pitcher_n_log1p"] = np.log1p(pd.to_numeric(out["asof_pitcher_n"], errors="coerce"))
    if "context_asof" in blocks:
        out["rf_li_x_pitcher_success"] = out["li"] * out["asof_pitcher_success_rate"]
        out["rf_late_x_pitcher_success"] = (out["inning"] >= 7).astype("int8") * out["asof_pitcher_success_rate"]
        out["rf_runner_x_pitcher_success"] = (out["num_runners_on"] > 0).astype("int8") * out["asof_pitcher_success_rate"]
        out["rf_score_abs_x_pitcher_success"] = out["score_diff_pitcher_team"].abs() * out["asof_pitcher_success_rate"]
        out["rf_pitchmix_fast_minus_break"] = out["asof_pitcher_fastball_rate"] - out["asof_pitcher_breaking_rate"]
        out["rf_pitchmix_fast_minus_offspeed"] = out["asof_pitcher_fastball_rate"] - out["asof_pitcher_offspeed_rate"]
    if "count_interactions" in blocks:
        balls = pd.to_numeric(out["balls_before"], errors="coerce")
        strikes = pd.to_numeric(out["strikes_before"], errors="coerce")
        out["rf_count_state_code"] = stable_code(out.assign(count_state=balls.astype("Int64").astype(str) + "-" + strikes.astype("Int64").astype(str)), ["count_state"])
        out["rf_balls_x_strikes"] = balls * strikes
        out["rf_count_pressure"] = balls - strikes
        out["rf_three_ball_x_pitcher_success"] = (balls == 3).astype("int8") * out["asof_pitcher_success_rate"]
        out["rf_two_strike_x_pitcher_success"] = (strikes == 2).astype("int8") * out["asof_pitcher_success_rate"]
        out["rf_count_x_prev3_success"] = (balls * 3 + strikes) * out["asof_pitcher_prev3_game_success_rate"]
    if "identity_context" in blocks:
        inning_bucket = pd.cut(
            pd.to_numeric(out["inning"], errors="coerce"),
            bins=[-np.inf, 3, 6, np.inf],
            labels=["early", "mid", "late"],
        ).astype("string")
        temp = out.assign(inning_bucket=inning_bucket)
        temp = temp.assign(count_state=temp["balls_before"].astype(str) + "-" + temp["strikes_before"].astype(str))
        out["rf_code_pitcher_game_type"] = stable_code(temp, ["pitcher_id", "game_type"])
        out["rf_code_pitcher_count"] = stable_code(temp, ["pitcher_id", "count_state"])
        out["rf_code_pitcher_batter_hand"] = stable_code(temp, ["pitcher_id", "batter_hand"])
        out["rf_code_inning_bucket_game_type"] = stable_code(temp, ["inning_bucket", "game_type"])
    return out


def pseudo_score(brier, actual_rate):
    denom = actual_rate * (1.0 - actual_rate)
    return max(0.0, 100000.0 * (1.0 - brier / denom))


def metric(y, pred):
    pred = clip_prob(pred)
    actual_rate = float(np.mean(y))
    brier = float(brier_score_loss(y, pred))
    oracle = actual_rate * (1.0 - actual_rate)
    return {
        "brier": brier,
        "oracle_constant_brier": float(oracle),
        "skill_margin": float(oracle - brier),
        "pseudo_score": pseudo_score(brier, actual_rate),
        "auc": float(roc_auc_score(y, pred)),
        "logloss": float(log_loss(y, pred)),
        "pred_mean": float(pred.mean()),
        "pred_std": float(pred.std()),
        "actual_rate": actual_rate,
        "pred_minus_actual": float(pred.mean() - actual_rate),
    }


def run_fit_eval(train_df, cal_df, valid_df, blocks):
    tr = add_feature_blocks(train_df, blocks)
    ca = add_feature_blocks(cal_df, blocks)
    va = add_feature_blocks(valid_df, blocks)
    y_train = tr[TARGET_COL].astype("int8")
    y_cal = ca[TARGET_COL].astype("int8").to_numpy()
    y_valid = va[TARGET_COL].astype("int8").to_numpy()

    builder = FeatureBuilder(alpha=80.0)
    X_train = builder.fit_transform(tr.drop(columns=[TARGET_COL]), y_train)
    X_cal = builder.transform(ca.drop(columns=[TARGET_COL]))
    X_valid = builder.transform(va.drop(columns=[TARGET_COL]))

    model = CatBoostClassifier(**CATBOOST_PARAMS)
    started = time.time()
    model.fit(X_train, y_train)
    train_seconds = time.time() - started
    raw_cal = model.predict_proba(X_cal)[:, 1]
    raw_valid = model.predict_proba(X_valid)[:, 1]
    calibration = fit_calibrators(raw_cal, y_cal, ca[["game_type"]])
    platt = apply_calibration(raw_valid, va[["game_type"]], calibration, "platt")

    rows = []
    for stage, pred in [("raw", raw_valid), ("platt", platt)]:
        rows.append({"stage": stage, "train_seconds": train_seconds, **metric(y_valid, pred)})
    fi = pd.DataFrame({"feature": X_train.columns, "importance": model.get_feature_importance()})
    return rows, fi


def group_for_feature(col):
    if col in CAT_CODE_COLS:
        return "H_categorical"
    if col in PITCHER_ASOF:
        return "E_official_asof_pitcher"
    if col in BATTER_ASOF:
        return "E_official_asof_batter"
    if col in ["pitcher_id", "pitcher_hand", "pitcher_team_id"]:
        return "A_pitcher_identity_profile"
    if col in ["batter_id", "batter_hand", "batter_team_id"]:
        return "B_batter_identity_profile"
    if col in CONTEXT_COLS:
        return "C_game_context"
    if col in COUNT_COLS:
        return "D_count"
    if col in ["season", "game_month", "game_dayofweek"]:
        return "F_temporal"
    if col in BASE_NUMERIC_COLS:
        return "G_other_numeric"
    return "other"


def target_separation(df, col):
    s = df[col]
    y = df[TARGET_COL].astype(float)
    if pd.api.types.is_numeric_dtype(s):
        x = pd.to_numeric(s, errors="coerce")
        if x.notna().sum() < 10 or x.nunique(dropna=True) < 2:
            return np.nan
        lo = x <= x.quantile(0.2)
        hi = x >= x.quantile(0.8)
        return float(y[hi].mean() - y[lo].mean())
    rates = df.groupby(col, dropna=False)[TARGET_COL].mean()
    counts = df.groupby(col, dropna=False)[TARGET_COL].size()
    if len(rates) < 2:
        return np.nan
    valid = counts >= 100
    if valid.sum() < 2:
        return np.nan
    return float(rates[valid].max() - rates[valid].min())


def make_inventory(df, baseline_importance):
    rows = []
    for col in [c for c in df.columns if c != TARGET_COL]:
        rows.append(
            {
                "feature": col,
                "group": group_for_feature(col),
                "used_by_current_feature_builder": bool(col in BASE_NUMERIC_COLS or col in CAT_CODE_COLS),
                "dtype": str(df[col].dtype),
                "missing_rate": float(df[col].isna().mean()),
                "unique_count": int(df[col].nunique(dropna=True)),
                "target_separation": target_separation(df, col),
                "catboost_importance": float(baseline_importance.get(col, np.nan)),
            }
        )
    return pd.DataFrame(rows)


def asof_analysis(df, baseline_importance):
    cols = [c for c in df.columns if "asof" in c or "prev" in c or "success_rate" in c]
    inv = make_inventory(df[cols + [TARGET_COL]], baseline_importance)
    return inv.sort_values(["target_separation", "catboost_importance"], ascending=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", default="data/train.csv")
    parser.add_argument("--output-dir", default="output/score_positive_features")
    parser.add_argument("--feature-sets", default=",".join(FEATURE_SETS.keys()))
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = add_basic(pd.read_csv(args.train_path, encoding="utf-8-sig"))
    partial_path = os.path.join(args.output_dir, "feature_set_fold_metrics_partial.csv")
    fi_partial_path = os.path.join(args.output_dir, "feature_importance_partial.csv")
    if os.path.exists(partial_path):
        fold_rows = pd.read_csv(partial_path).to_dict("records")
    else:
        fold_rows = []
    done = {
        (r["feature_set"], int(r["valid_year"]))
        for r in fold_rows
        if r.get("stage") == "platt"
    }
    fi_rows = []
    if os.path.exists(fi_partial_path):
        fi_rows.append(pd.read_csv(fi_partial_path))
    baseline_importance = {}
    selected_feature_sets = [x for x in args.feature_sets.split(",") if x]

    for feature_set in selected_feature_sets:
        blocks = FEATURE_SETS[feature_set]
        print(f"feature_set={feature_set}")
        for fold in FOLDS:
            train_start, train_end, cal_year, valid_year = fold
            if (feature_set, valid_year) in done:
                print(f"  skip completed valid={valid_year}")
                continue
            train_df = df[(df["season"] >= train_start) & (df["season"] <= train_end)].copy()
            cal_df = df[df["season"] == cal_year].copy()
            valid_df = df[df["season"] == valid_year].copy()
            print(f"  fold train={train_start}-{train_end} cal={cal_year} valid={valid_year}")
            rows, fi = run_fit_eval(train_df, cal_df, valid_df, blocks)
            fi["feature_set"] = feature_set
            fi["valid_year"] = valid_year
            fi_rows.append(fi)
            if feature_set == "current_baseline":
                for _, row in fi.iterrows():
                    baseline_importance[row["feature"]] = max(
                        float(row["importance"]), baseline_importance.get(row["feature"], 0.0)
                    )
            for row in rows:
                fold_rows.append(
                    {
                        "feature_set": feature_set,
                        "train_start": train_start,
                        "train_end": train_end,
                        "cal_year": cal_year,
                        "valid_year": valid_year,
                        **row,
                    }
                )
            pd.DataFrame(fold_rows).to_csv(os.path.join(args.output_dir, "feature_set_fold_metrics_partial.csv"), index=False)
            pd.concat(fi_rows, ignore_index=True).to_csv(fi_partial_path, index=False)

    fold_metrics = pd.DataFrame(fold_rows)
    fold_metrics.to_csv(os.path.join(args.output_dir, "feature_set_fold_metrics.csv"), index=False, encoding="utf-8")
    feature_importance = pd.concat(fi_rows, ignore_index=True)
    feature_importance.to_csv(os.path.join(args.output_dir, "feature_importance.csv"), index=False, encoding="utf-8")

    platt = fold_metrics[fold_metrics["stage"] == "platt"].copy()
    summary = (
        platt.groupby("feature_set")
        .agg(
            mean_brier=("brier", "mean"),
            std_brier=("brier", "std"),
            worst_brier=("brier", "max"),
            mean_pseudo_score=("pseudo_score", "mean"),
            positive_score_folds=("skill_margin", lambda s: int((s > 0).sum())),
            worst_skill_margin=("skill_margin", "min"),
            mean_auc=("auc", "mean"),
            mean_logloss=("logloss", "mean"),
            mean_pred_std=("pred_std", "mean"),
            brier_2024=("brier", lambda s: float(s[platt.loc[s.index, "valid_year"] == 2024].iloc[0])),
            skill_margin_2024=("skill_margin", lambda s: float(s[platt.loc[s.index, "valid_year"] == 2024].iloc[0])),
            pseudo_score_2024=("pseudo_score", lambda s: float(s[platt.loc[s.index, "valid_year"] == 2024].iloc[0])),
            auc_2024=("auc", lambda s: float(s[platt.loc[s.index, "valid_year"] == 2024].iloc[0])),
        )
        .reset_index()
        .sort_values(["positive_score_folds", "skill_margin_2024", "mean_auc"], ascending=[False, False, False])
    )
    summary.to_csv(os.path.join(args.output_dir, "feature_set_summary.csv"), index=False, encoding="utf-8")

    inv = make_inventory(df, baseline_importance)
    inv.to_csv(os.path.join(args.output_dir, "feature_inventory.csv"), index=False, encoding="utf-8")
    asof = asof_analysis(df, baseline_importance)
    asof.to_csv(os.path.join(args.output_dir, "asof_feature_analysis.csv"), index=False, encoding="utf-8")

    print("\nFeature set summary")
    print(summary.to_string(index=False))
    best = summary.iloc[0]["feature_set"]
    print(f"\nTop importances for {best}")
    print(
        feature_importance[feature_importance["feature_set"] == best]
        .groupby("feature")["importance"]
        .mean()
        .sort_values(ascending=False)
        .head(30)
        .to_string()
    )


if __name__ == "__main__":
    main()
