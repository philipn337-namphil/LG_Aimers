import argparse
import os

import numpy as np
import pandas as pd


SEASONS = [2019, 2020, 2021, 2022, 2023, 2024]
MAIN_HAND_MAP = {1: "Left", 2: "Right"}


PLAYER_CONFIG = {
    "pitcher": {
        "main_id": "pitcher_id",
        "main_hand": "pitcher_hand",
        "main_team": "pitcher_team_id",
        "trackman_id": "pitcher_trackman_id",
        "trackman_hand": "pitcher_hand",
        "trackman_team": "pitcher_team",
    },
    "batter": {
        "main_id": "batter_id",
        "main_hand": "batter_hand",
        "main_team": "batter_team_id",
        "trackman_id": "batter_trackman_id",
        "trackman_hand": "batter_hand",
        "trackman_team": "batter_team",
    },
}


def load_main_counts(train_path: str, player_type: str) -> pd.DataFrame:
    cfg = PLAYER_CONFIG[player_type]
    cols = [cfg["main_id"], cfg["main_hand"], cfg["main_team"], "season"]
    df = pd.read_csv(train_path, usecols=cols, encoding="utf-8-sig")
    df[cfg["main_hand"]] = df[cfg["main_hand"]].map(MAIN_HAND_MAP)
    team_mode = (
        df.groupby([cfg["main_id"], cfg["main_hand"]], dropna=False)[cfg["main_team"]]
        .agg(lambda s: s.value_counts().index[0])
        .rename("main_team")
        .reset_index()
    )
    counts = (
        df.groupby([cfg["main_id"], cfg["main_hand"], "season"], dropna=False)
        .size()
        .rename("n")
        .reset_index()
    )
    wide = (
        counts.pivot_table(
            index=[cfg["main_id"], cfg["main_hand"]],
            columns="season",
            values="n",
            fill_value=0,
            aggfunc="sum",
        )
        .reindex(columns=SEASONS, fill_value=0)
        .reset_index()
    )
    wide.columns.name = None
    wide = wide.merge(team_mode, on=[cfg["main_id"], cfg["main_hand"]], how="left")
    return wide.rename(
        columns={
            cfg["main_id"]: "main_id",
            cfg["main_hand"]: "hand",
        }
    )


def load_trackman_counts(trackman_path: str, player_type: str) -> pd.DataFrame:
    cfg = PLAYER_CONFIG[player_type]
    cols = [cfg["trackman_id"], cfg["trackman_hand"], cfg["trackman_team"], "season"]
    df = pd.read_csv(trackman_path, usecols=cols, encoding="utf-8-sig")
    team_mode = (
        df.groupby([cfg["trackman_id"], cfg["trackman_hand"]], dropna=False)[cfg["trackman_team"]]
        .agg(lambda s: s.value_counts().index[0])
        .rename("trackman_team")
        .reset_index()
    )
    counts = (
        df.groupby([cfg["trackman_id"], cfg["trackman_hand"], "season"], dropna=False)
        .size()
        .rename("n")
        .reset_index()
    )
    wide = (
        counts.pivot_table(
            index=[cfg["trackman_id"], cfg["trackman_hand"]],
            columns="season",
            values="n",
            fill_value=0,
            aggfunc="sum",
        )
        .reindex(columns=SEASONS, fill_value=0)
        .reset_index()
    )
    wide.columns.name = None
    wide = wide.merge(team_mode, on=[cfg["trackman_id"], cfg["trackman_hand"]], how="left")
    return wide.rename(
        columns={
            cfg["trackman_id"]: "trackman_id",
            cfg["trackman_hand"]: "hand",
        }
    )


def vector_matrix(df: pd.DataFrame) -> np.ndarray:
    return df[SEASONS].to_numpy(dtype=np.float32)


def make_candidates(main: pd.DataFrame, trackman: pd.DataFrame, top_k: int) -> pd.DataFrame:
    rows = []
    for hand in sorted(set(main["hand"]).intersection(set(trackman["hand"]))):
        m = main[main["hand"] == hand].reset_index(drop=True)
        t = trackman[trackman["hand"] == hand].reset_index(drop=True)
        mv = vector_matrix(m)
        tv = vector_matrix(t)

        m_total = mv.sum(axis=1)
        t_total = tv.sum(axis=1)
        m_norm = np.linalg.norm(mv, axis=1)
        t_norm = np.linalg.norm(tv, axis=1)
        cosine = (mv @ tv.T) / np.maximum(np.outer(m_norm, t_norm), 1e-6)
        count_ratio = np.minimum.outer(m_total, t_total) / np.maximum.outer(m_total, t_total)
        abs_total_diff = np.abs(np.subtract.outer(m_total, t_total))
        score = 0.78 * cosine + 0.22 * count_ratio

        k = min(top_k, len(t))
        top_idx = np.argpartition(-score, kth=np.arange(k), axis=1)[:, :k]
        for i in range(len(m)):
            ordered = top_idx[i][np.argsort(-score[i, top_idx[i]])]
            best_score = float(score[i, ordered[0]]) if len(ordered) else np.nan
            second_score = float(score[i, ordered[1]]) if len(ordered) > 1 else np.nan
            for rank, j in enumerate(ordered, start=1):
                rows.append(
                    {
                        "main_id": m.loc[i, "main_id"],
                        "trackman_id": t.loc[j, "trackman_id"],
                        "hand": hand,
                        "main_team": m.loc[i, "main_team"],
                        "trackman_team": t.loc[j, "trackman_team"],
                        "rank": rank,
                        "score": float(score[i, j]),
                        "score_gap_from_second": best_score - second_score if rank == 1 and len(ordered) > 1 else np.nan,
                        "cosine": float(cosine[i, j]),
                        "count_ratio": float(count_ratio[i, j]),
                        "main_total": int(m_total[i]),
                        "trackman_total": int(t_total[j]),
                        "abs_total_diff": int(abs_total_diff[i, j]),
                        **{f"main_{s}": int(mv[i, si]) for si, s in enumerate(SEASONS)},
                        **{f"trackman_{s}": int(tv[j, si]) for si, s in enumerate(SEASONS)},
                    }
                )
    return pd.DataFrame(rows).sort_values(["main_id", "rank"]).reset_index(drop=True)


def greedy_one_to_one(candidates: pd.DataFrame, min_score: float, min_gap: float) -> pd.DataFrame:
    top = candidates[
        (candidates["rank"] == 1)
        & (candidates["score"] >= min_score)
        & (candidates["score_gap_from_second"].fillna(1.0) >= min_gap)
    ].copy()
    top = top.sort_values(["score", "score_gap_from_second", "count_ratio"], ascending=False)

    used_main = set()
    used_trackman = set()
    chosen = []
    for row in top.to_dict("records"):
        if row["main_id"] in used_main or row["trackman_id"] in used_trackman:
            continue
        used_main.add(row["main_id"])
        used_trackman.add(row["trackman_id"])
        chosen.append(row)
    return pd.DataFrame(chosen).sort_values("main_id").reset_index(drop=True)


def make_team_summary(matched: pd.DataFrame, min_score: float = 0.98) -> pd.DataFrame:
    high = matched[matched["score"] >= min_score].copy()
    if high.empty:
        return pd.DataFrame(
            columns=["main_team", "trackman_team", "n", "share_within_main_team", "high_conf_matches"]
        )

    total = high.groupby("main_team").size().rename("high_conf_matches")
    pairs = (
        high.groupby(["main_team", "trackman_team"])
        .size()
        .rename("n")
        .reset_index()
        .merge(total, on="main_team", how="left")
    )
    pairs["share_within_main_team"] = pairs["n"] / pairs["high_conf_matches"]
    return pairs.sort_values(["main_team", "n", "trackman_team"], ascending=[True, False, True])


def run_player_type(args, player_type: str):
    print(f"Build {player_type} count vectors...")
    main = load_main_counts(args.train_path, player_type)
    trackman = load_trackman_counts(args.trackman_path, player_type)
    print(f"{player_type}: main={len(main)} trackman={len(trackman)}")

    candidates = make_candidates(main, trackman, args.top_k)
    matched = greedy_one_to_one(candidates, args.min_score, args.min_gap)

    os.makedirs(args.output_dir, exist_ok=True)
    cand_path = os.path.join(args.output_dir, f"{player_type}_trackman_candidates.csv")
    match_path = os.path.join(args.output_dir, f"{player_type}_trackman_matches.csv")
    team_path = os.path.join(args.output_dir, f"{player_type}_team_mapping_summary.csv")
    candidates.to_csv(cand_path, index=False, encoding="utf-8")
    matched.to_csv(match_path, index=False, encoding="utf-8")
    make_team_summary(matched).to_csv(team_path, index=False, encoding="utf-8")

    print(
        f"{player_type}: candidates={len(candidates)} matches={len(matched)} "
        f"saved={match_path}"
    )
    if len(matched):
        print(matched.head(10)[["main_id", "trackman_id", "score", "score_gap_from_second", "main_total", "trackman_total"]])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", default="data/train.csv")
    parser.add_argument("--trackman-path", default="data/trackman_history.csv")
    parser.add_argument("--output-dir", default="output/trackman_matching")
    parser.add_argument("--player-type", choices=["pitcher", "batter", "both"], default="both")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-score", type=float, default=0.93)
    parser.add_argument("--min-gap", type=float, default=0.01)
    args = parser.parse_args()

    player_types = ["pitcher", "batter"] if args.player_type == "both" else [args.player_type]
    for player_type in player_types:
        run_player_type(args, player_type)


if __name__ == "__main__":
    main()
