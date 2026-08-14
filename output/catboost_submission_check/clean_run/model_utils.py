from dataclasses import dataclass, field
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier


warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

TARGET_COL = "control_success"


BASE_NUMERIC_COLS = [
    "season",
    "game_month",
    "game_dayofweek",
    "inning",
    "balls_before",
    "strikes_before",
    "outs_before",
    "run_top_before",
    "run_bot_before",
    "run_total_before",
    "score_diff_home",
    "score_diff_pitcher_team",
    "runner_on_1b",
    "runner_on_2b",
    "runner_on_3b",
    "num_runners_on",
    "home_win_expectancy",
    "away_win_expectancy",
    "li",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team_id",
    "batter_team_id",
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
    "asof_batter_n",
    "asof_batter_success_rate",
    "asof_batter_middle_rate",
    "asof_pitcher_pitchmix_n",
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
]


CAT_CODE_COLS = ["top_bottom", "game_type", "base_state", "pitcher_id", "batter_id"]


TE_GROUPS = [
    ("pitcher_id",),
    ("batter_id",),
    ("pitcher_team_id",),
    ("batter_team_id",),
    ("game_type",),
    ("season",),
    ("pitcher_id", "balls_before", "strikes_before"),
    ("pitcher_id", "batter_hand"),
    ("pitcher_id", "base_state"),
    ("pitcher_id", "game_type"),
    ("batter_id", "pitcher_hand"),
    ("balls_before", "strikes_before"),
    ("balls_before", "strikes_before", "game_type"),
    ("pitcher_hand", "batter_hand"),
    ("base_state", "outs_before"),
]


@dataclass
class FeatureBuilder:
    alpha: float = 80.0
    global_rate_: float = 0.5
    extra_numeric_cols_: list = field(default_factory=list)
    category_maps_: dict = field(default_factory=dict)
    te_maps_: dict = field(default_factory=dict)
    feature_names_: list = field(default_factory=list)
    fill_values_: dict = field(default_factory=dict)

    def fit(self, df: pd.DataFrame, y: pd.Series):
        self.global_rate_ = float(y.mean())
        self.extra_numeric_cols_ = sorted([c for c in df.columns if c.startswith("tm_")])

        for col in CAT_CODE_COLS:
            values = pd.Series(df[col].astype("string").fillna("__NA__").unique())
            self.category_maps_[col] = {v: i + 1 for i, v in enumerate(values)}

        work = df.copy()
        work[TARGET_COL] = y.to_numpy()
        for group in TE_GROUPS:
            key = self._group_name(group)
            grouped = work.groupby(list(group), dropna=False)[TARGET_COL].agg(["sum", "count"])
            enc = (grouped["sum"] + self.alpha * self.global_rate_) / (grouped["count"] + self.alpha)
            self.te_maps_[key] = enc.astype("float32").to_dict()

        features = self._build(df)
        self.feature_names_ = list(features.columns)
        self.fill_values_ = features.median(numeric_only=True).fillna(self.global_rate_).to_dict()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        features = self._build(df)
        for col in self.feature_names_:
            if col not in features.columns:
                features[col] = np.nan
        features = features[self.feature_names_]
        return features.fillna(self.fill_values_).astype("float32")

    def fit_transform(self, df: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        self.fit(df, y)
        return self.transform(df)

    def _build(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)

        for col in BASE_NUMERIC_COLS:
            out[col] = pd.to_numeric(df[col], errors="coerce") if col in df.columns else np.nan
        for col in self.extra_numeric_cols_:
            out[col] = pd.to_numeric(df[col], errors="coerce") if col in df.columns else np.nan

        for col in CAT_CODE_COLS:
            if col in df.columns:
                vals = df[col].astype("string").fillna("__NA__")
            else:
                vals = pd.Series("__NA__", index=df.index)
            out[f"{col}_code"] = vals.map(self.category_maps_.get(col, {})).fillna(0).astype("float32")

        balls = pd.to_numeric(df["balls_before"], errors="coerce")
        strikes = pd.to_numeric(df["strikes_before"], errors="coerce")
        score_diff = pd.to_numeric(df["score_diff_pitcher_team"], errors="coerce")
        runners = pd.to_numeric(df["num_runners_on"], errors="coerce")

        out["count_index"] = balls * 3 + strikes
        out["two_strike"] = (strikes == 2).astype("int8")
        out["three_ball"] = (balls == 3).astype("int8")
        out["full_count"] = ((balls == 3) & (strikes == 2)).astype("int8")
        out["pitcher_count"] = (strikes > balls).astype("int8")
        out["hitter_count"] = (balls > strikes).astype("int8")
        out["even_count"] = (balls == strikes).astype("int8")
        out["runner_on"] = (runners > 0).astype("int8")
        out["scoring_position"] = (
            (pd.to_numeric(df["runner_on_2b"], errors="coerce") == 1)
            | (pd.to_numeric(df["runner_on_3b"], errors="coerce") == 1)
        ).astype("int8")
        out["abs_score_diff_pitcher_team"] = score_diff.abs()
        out["pitcher_team_leading"] = (score_diff > 0).astype("int8")
        out["pitcher_team_trailing"] = (score_diff < 0).astype("int8")
        out["late_inning"] = (pd.to_numeric(df["inning"], errors="coerce") >= 7).astype("int8")

        for group in TE_GROUPS:
            key = self._group_name(group)
            out[f"te_{key}"] = self._map_group(df, group, self.te_maps_.get(key, {}))
            out[f"te_{key}_delta"] = out[f"te_{key}"] - self.global_rate_

        out["asof_pitcher_success_minus_global"] = out["asof_pitcher_success_rate"] - self.global_rate_
        out["asof_batter_success_minus_global"] = out["asof_batter_success_rate"] - self.global_rate_
        out["pitcher_success_x_two_strike"] = out["asof_pitcher_success_rate"] * out["two_strike"]
        out["pitcher_success_x_runner_on"] = out["asof_pitcher_success_rate"] * out["runner_on"]
        out["pitcher_fastball_x_hitter_count"] = out["asof_pitcher_fastball_rate"] * out["hitter_count"]
        return out

    def _map_group(self, df: pd.DataFrame, group: tuple, mapping: dict) -> pd.Series:
        if len(group) == 1:
            vals = df[group[0]]
        else:
            vals = list(map(tuple, df[list(group)].to_numpy()))
        return pd.Series(vals, index=df.index).map(mapping).fillna(self.global_rate_).astype("float32")

    @staticmethod
    def _group_name(group: tuple) -> str:
        return "__".join(group)


def make_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.055,
        max_iter=220,
        max_leaf_nodes=31,
        l2_regularization=0.03,
        min_samples_leaf=40,
        early_stopping=True,
        validation_fraction=0.12,
        n_iter_no_change=25,
        random_state=42,
    )
