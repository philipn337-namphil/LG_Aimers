import argparse
import os

import numpy as np
import pandas as pd


NUMERIC_COLS = [
    "rel_speed",
    "spin_rate",
    "induced_vert_break",
    "horz_break",
    "extension",
    "rel_height",
    "rel_side",
    "zone_speed",
]

PITCH_GROUPS = ["fastball", "breaking", "offspeed", "other"]


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = ["_".join(str(x) for x in col if str(x)) if isinstance(col, tuple) else str(col) for col in df.columns]
    return df


def summarize_trackman(trackman: pd.DataFrame, id_col: str, prefix: str) -> pd.DataFrame:
    for col in NUMERIC_COLS:
        trackman[col] = pd.to_numeric(trackman[col], errors="coerce")

    base = trackman.groupby(id_col)[NUMERIC_COLS].agg(["mean", "std", "median"])
    base = flatten_columns(base).reset_index()
    base = base.rename(columns={c: f"{prefix}_{c}" for c in base.columns if c != id_col})

    n = trackman.groupby(id_col).size().rename(f"{prefix}_n").reset_index()
    out = n.merge(base, on=id_col, how="left")

    group_counts = (
        trackman.groupby([id_col, "pitch_type_group"])
        .size()
        .rename("n")
        .reset_index()
        .pivot(index=id_col, columns="pitch_type_group", values="n")
        .fillna(0)
    )
    group_total = group_counts.sum(axis=1).replace(0, np.nan)
    for group in PITCH_GROUPS:
        if group not in group_counts.columns:
            group_counts[group] = 0
        out = out.merge(
            (group_counts[group] / group_total).rename(f"{prefix}_pitch_group_rate_{group}").reset_index(),
            on=id_col,
            how="left",
        )

    by_group = trackman[trackman["pitch_type_group"].isin(PITCH_GROUPS)]
    grouped_numeric = by_group.groupby([id_col, "pitch_type_group"])[["rel_speed", "spin_rate", "induced_vert_break", "horz_break"]].agg(["mean", "std"])
    grouped_numeric = flatten_columns(grouped_numeric).reset_index()
    grouped_wide = grouped_numeric.pivot(index=id_col, columns="pitch_type_group")
    grouped_wide = flatten_columns(grouped_wide).reset_index()
    grouped_wide = grouped_wide.rename(columns={c: f"{prefix}_bygroup_{c}" for c in grouped_wide.columns if c != id_col})
    out = out.merge(grouped_wide, on=id_col, how="left")

    out[f"{prefix}_release_consistency"] = (
        out.get(f"{prefix}_rel_height_std", 0)
        + out.get(f"{prefix}_rel_side_std", 0)
        + out.get(f"{prefix}_extension_std", 0)
    )
    out[f"{prefix}_movement_consistency"] = (
        out.get(f"{prefix}_induced_vert_break_std", 0)
        + out.get(f"{prefix}_horz_break_std", 0)
    )
    out[f"{prefix}_velocity_consistency"] = out.get(f"{prefix}_rel_speed_std", np.nan)
    return out


def attach_matches(features: pd.DataFrame, matches_path: str, main_col: str, trackman_col: str, prefix: str, min_score: float) -> pd.DataFrame:
    matches = pd.read_csv(matches_path)
    matches = matches[matches["score"] >= min_score].copy()
    cols = ["main_id", "trackman_id", "score", "score_gap_from_second", "count_ratio"]
    matches = matches[cols].rename(
        columns={
            "main_id": main_col,
            "trackman_id": trackman_col,
            "score": f"{prefix}_match_score",
            "score_gap_from_second": f"{prefix}_match_gap",
            "count_ratio": f"{prefix}_match_count_ratio",
        }
    )
    out = matches.merge(features, on=trackman_col, how="left")
    out[f"{prefix}_matched"] = 1
    return out.drop(columns=[trackman_col])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trackman-path", default="data/trackman_history.csv")
    parser.add_argument("--match-dir", default="output/trackman_matching")
    parser.add_argument("--output-path", default="output/trackman_features/player_trackman_features.csv")
    parser.add_argument("--min-score", type=float, default=0.98)
    args = parser.parse_args()

    print("Load trackman...")
    usecols = [
        "pitcher_trackman_id",
        "batter_trackman_id",
        "pitch_type_group",
        *NUMERIC_COLS,
    ]
    trackman = pd.read_csv(args.trackman_path, usecols=usecols, encoding="utf-8-sig")

    print("Build pitcher features...")
    pitcher_features = summarize_trackman(trackman.copy(), "pitcher_trackman_id", "tm_p")
    pitcher_features = attach_matches(
        pitcher_features,
        os.path.join(args.match_dir, "pitcher_trackman_matches.csv"),
        "pitcher_id",
        "pitcher_trackman_id",
        "tm_p",
        args.min_score,
    )

    print("Build batter features...")
    batter_features = summarize_trackman(trackman.copy(), "batter_trackman_id", "tm_b")
    batter_features = attach_matches(
        batter_features,
        os.path.join(args.match_dir, "batter_trackman_matches.csv"),
        "batter_id",
        "batter_trackman_id",
        "tm_b",
        args.min_score,
    )

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    pitcher_path = os.path.join(os.path.dirname(args.output_path), "pitcher_trackman_features.csv")
    batter_path = os.path.join(os.path.dirname(args.output_path), "batter_trackman_features.csv")
    pitcher_features.to_csv(pitcher_path, index=False, encoding="utf-8")
    batter_features.to_csv(batter_path, index=False, encoding="utf-8")
    print(f"Saved: {pitcher_path} rows={len(pitcher_features)}")
    print(f"Saved: {batter_path} rows={len(batter_features)}")


if __name__ == "__main__":
    main()
