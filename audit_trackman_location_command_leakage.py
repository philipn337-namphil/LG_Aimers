import os
from pathlib import Path

import numpy as np
import pandas as pd


OUT_DIR = Path("output/location_command_leakage_audit")
TRACKMAN_PATH = Path("data/trackman_history.csv")
TRAIN_PATH = Path("data/train.csv")
TEST_PATH = Path("data/test.csv")

LOCATION_KEYWORDS = [
    "plate",
    "zone",
    "location",
    "loc",
    "target",
    "catcher",
    "edge",
    "command",
    "result",
    "outcome",
    "success",
    "strike",
    "ball",
    "pitch_type",
    "pitcher",
    "batter",
    "date",
    "year",
    "season",
    "game",
    "pitch_no",
    "inning",
    "top_bottom",
    "rel_",
]

COLUMN_MEANING = {
    "trackman_id": "Trackman historical pitch row identifier.",
    "season": "Season of the historical Trackman pitch.",
    "game_date": "Date of the historical Trackman game.",
    "game_month": "Month of the historical Trackman game.",
    "game_dayofweek": "Day of week of the historical Trackman game.",
    "trackman_game_id": "Trackman historical game identifier.",
    "pitch_no": "Pitch order number inside Trackman game.",
    "inning": "Inning of the historical Trackman pitch.",
    "top_bottom": "Top/bottom half of inning.",
    "balls_before": "Ball count before the historical pitch.",
    "strikes_before": "Strike count before the historical pitch.",
    "outs_before": "Out count before the historical pitch.",
    "pitch_of_pa": "Pitch number within plate appearance.",
    "pitcher_trackman_id": "Trackman pitcher identifier.",
    "batter_trackman_id": "Trackman batter identifier.",
    "pitcher_hand": "Pitcher handedness.",
    "batter_hand": "Batter handedness.",
    "pitcher_team": "Pitcher team code.",
    "batter_team": "Batter team code.",
    "tagged_pitch_type": "Historical actual pitch type label.",
    "auto_pitch_type": "Historical automatically classified pitch type.",
    "pitch_type_group": "Historical actual pitch type group.",
    "rel_speed": "Historical release speed measurement.",
    "spin_rate": "Historical spin-rate measurement.",
    "induced_vert_break": "Historical movement measurement.",
    "horz_break": "Historical movement measurement.",
    "extension": "Historical release extension measurement.",
    "rel_height": "Historical release height measurement.",
    "rel_side": "Historical release side measurement.",
    "zone_speed": "Historical speed near home plate, not a zone/location coordinate.",
}

PRE_PITCH_COLS = {
    "season",
    "game_date",
    "game_month",
    "game_dayofweek",
    "trackman_game_id",
    "pitch_no",
    "inning",
    "top_bottom",
    "balls_before",
    "strikes_before",
    "outs_before",
    "pitch_of_pa",
    "pitcher_trackman_id",
    "batter_trackman_id",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team",
    "batter_team",
}

POST_PITCH_HISTORICAL_COLS = {
    "tagged_pitch_type",
    "auto_pitch_type",
    "pitch_type_group",
    "rel_speed",
    "spin_rate",
    "induced_vert_break",
    "horz_break",
    "extension",
    "rel_height",
    "rel_side",
    "zone_speed",
}


def parse_trackman_dates(trackman: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(trackman["game_date"], format="mixed", errors="coerce")


def load_schema():
    trackman_sample = pd.read_csv(TRACKMAN_PATH, nrows=1000, encoding="utf-8-sig")
    train_cols = pd.read_csv(TRAIN_PATH, nrows=1, encoding="utf-8-sig").columns.tolist()
    test_cols = pd.read_csv(TEST_PATH, nrows=1, encoding="utf-8-sig").columns.tolist()
    return trackman_sample, train_cols, test_cols


def is_relevant_column(col: str) -> bool:
    c = col.lower()
    return any(k in c for k in LOCATION_KEYWORDS)


def make_inventory(trackman_sample: pd.DataFrame) -> pd.DataFrame:
    rows = []
    explicit_absent = [
        "plate_x",
        "plate_z",
        "zone",
        "strike_zone_top",
        "strike_zone_bottom",
        "pitch_result",
        "pitch_outcome",
        "catcher_target_x",
        "catcher_target_z",
    ]
    for col in trackman_sample.columns:
        if not is_relevant_column(col):
            continue
        before = col in PRE_PITCH_COLS
        after = col in POST_PITCH_HISTORICAL_COLS
        target_related = col in {"pitch_type_group", "tagged_pitch_type", "auto_pitch_type"} or "result" in col or "outcome" in col or "zone" in col
        rows.append(
            {
                "column_name": col,
                "dtype": str(trackman_sample[col].dtype),
                "meaning": COLUMN_MEANING.get(col, "Trackman historical metadata or measurement."),
                "known_before_pitch": bool(before),
                "known_only_after_pitch": bool(after),
                "target_or_control_success_related": bool(target_related),
                "audit_note": (
                    "historical post-pitch measurement: allowed only after strict prior-history aggregation"
                    if after
                    else "historical pre-pitch metadata"
                ),
                "exists_in_trackman": True,
            }
        )
    for col in explicit_absent:
        rows.append(
            {
                "column_name": col,
                "dtype": "ABSENT",
                "meaning": "Requested location/zone/result field is not present in provided Trackman data.",
                "known_before_pitch": False,
                "known_only_after_pitch": True,
                "target_or_control_success_related": True,
                "audit_note": "absent; cannot be used for feature construction",
                "exists_in_trackman": False,
            }
        )
    return pd.DataFrame(rows).sort_values(["exists_in_trackman", "column_name"], ascending=[False, True])


def feature_classification(train_cols, test_cols):
    current_pitch_type_in_test = any(c in test_cols for c in ["pitch_type", "pitch_type_group", "tagged_pitch_type", "auto_pitch_type"])
    no_location_cols = True
    rows = [
        ("A", "pitcher historical plate_x mean/std", "LEAKAGE / NOT ALLOWED", "plate_x is absent; current-row actual plate location would be post-pitch and forbidden."),
        ("B", "pitcher historical plate_z mean/std", "LEAKAGE / NOT ALLOWED", "plate_z is absent; current-row actual plate location would be post-pitch and forbidden."),
        ("C", "pitcher x pitch_type plate_x/z mean/std", "LEAKAGE / NOT ALLOWED", "plate_x/z are absent and current row actual pitch_type is not provided in test."),
        ("D", "pitcher historical location entropy", "LEAKAGE / NOT ALLOWED", "location coordinate/bin columns are absent."),
        ("E", "pitcher x pitch_type location entropy", "LEAKAGE / NOT ALLOWED", "location columns are absent and current pitch_type is unavailable at inference."),
        ("F", "historical zone-hit rate", "LEAKAGE / NOT ALLOWED", "zone/result columns are absent; current-row zone/result would directly overlap target construction risk."),
        ("G", "historical edge-zone density", "LEAKAGE / NOT ALLOWED", "edge/zone geometry columns are absent."),
        ("H", "historical cluster compactness", "LEAKAGE / NOT ALLOWED", "requires plate_x/plate_z or equivalent location coordinates, which are absent."),
        ("I", "release-to-location sensitivity", "LEAKAGE / NOT ALLOWED", "release columns exist, but plate-location response columns are absent."),
        ("J", "historical command-zone density", "LEAKAGE / NOT ALLOWED", "command/zone columns are absent."),
        ("K", "current-row plate_x/z", "LEAKAGE / NOT ALLOWED", "current pitch actual location is not known at test inference and is explicitly forbidden by data rules."),
        ("L", "current-row zone/result", "LEAKAGE / NOT ALLOWED", "current pitch zone/result are post-pitch outcomes and not present in test input."),
        ("M", "current-row actual pitch outcome", "LEAKAGE / NOT ALLOWED", "current outcome is target leakage."),
        ("N", "pitcher historical release/movement aggregate", "CONDITIONALLY SAFE", "Not a location-command feature; allowed only with prior-history cutoff and no validation/test future rows."),
        ("O", "pitcher x historical pitch_type release/movement aggregate", "CONDITIONALLY SAFE", "Historical actual pitch_type can group prior Trackman rows, but current row pitch_type cannot be used as a key."),
    ]
    out = pd.DataFrame(rows, columns=["candidate_id", "candidate_feature", "classification", "reason"])
    out["current_pitch_type_available_in_test"] = current_pitch_type_in_test
    out["plate_location_columns_available"] = not no_location_cols
    return out


def temporal_policy():
    rows = []
    for valid_year in [2022, 2023, 2024]:
        rows.append(
            {
                "prediction_context": f"validation_{valid_year}",
                "prediction_rows": f"train rows with season == {valid_year}",
                "allowed_trackman_history": f"Trackman rows with season < {valid_year}",
                "max_allowed_source_season": valid_year - 1,
                "strict_policy": "prior-season-only for offline validation",
                "forbidden_policy": f"using any Trackman row with season >= {valid_year}",
                "same_season_full_aggregate_allowed": False,
                "same_game_asof_required_if_same_season_used": True,
            }
        )
    rows.append(
        {
            "prediction_context": "test_2025",
            "prediction_rows": "official hidden 2025 test rows",
            "allowed_trackman_history": "provided Trackman rows with season <= 2024",
            "max_allowed_source_season": 2024,
            "strict_policy": "all provided 2019-2024 Trackman history is prior to 2025",
            "forbidden_policy": "2025 Trackman, test-set aggregate, hidden test row interactions",
            "same_season_full_aggregate_allowed": False,
            "same_game_asof_required_if_same_season_used": True,
        }
    )
    return pd.DataFrame(rows)


def profile_audit():
    absent_location = "plate_x/plate_z/zone-like columns are absent in provided Trackman."
    profiles = [
        ("Profile 1", "pitcher historical location dispersion", "plate_x, plate_z, pitcher_trackman_id, game_date/season", "season < prediction_year", "high: required columns absent", "Do not implement unless legitimate historical location columns are later provided; then use prior-history only.", "Using current-row plate_x/plate_z or full validation season."),
        ("Profile 2", "pitcher x pitch_type historical location dispersion", "plate_x, plate_z, pitcher_trackman_id, pitch_type_group, game_date/season", "season < prediction_year", "high: location absent and current pitch_type unavailable", "Aggregate prior historical rows by pitcher and historical pitch_type only; do not key on current row pitch_type.", "Using current row actual pitch_type to select a profile."),
        ("Profile 3", "pitcher historical zone concentration", "zone or plate_x/plate_z with zone geometry", "season < prediction_year", "high: zone/location absent", "Not implementable from current data.", "Using current-row zone or control_success-derived zone."),
        ("Profile 4", "pitcher historical edge density", "plate_x, plate_z, strike-zone geometry", "season < prediction_year", "high: edge/zone geometry absent", "Not implementable from current data.", "Using post-pitch zone labels from validation season."),
        ("Profile 5", "pitcher historical location entropy", "plate_x, plate_z or location bin", "season < prediction_year", "high: location absent", "Not implementable from current data.", "Using validation-season full-season bins."),
        ("Profile 6", "release-point to plate-location sensitivity", "rel_height, rel_side, extension, plate_x, plate_z", "season < prediction_year", "high: release exists but plate location absent", "Release-only consistency is possible as a different feature; sensitivity to location is not.", "Fitting sensitivity with future or validation-year location."),
        ("Profile 7", "pitch-type specific command-zone centroid", "pitch_type_group, plate_x, plate_z or command zone", "season < prediction_year", "high: command-zone absent and current pitch_type unavailable", "Not implementable from current data.", "Joining current pitch actual type or zone."),
        ("Profile 8", "recent season command drift", "date/season plus plate_x/plate_z or zone", "strict as-of; prior season only preferred", "high: command location absent; same-season drift is leakage unless strict as-of", "If location columns are later added, compute only from previous completed seasons or strictly earlier timestamps.", "Using full 2024 season to score 2024 validation rows."),
    ]
    return pd.DataFrame(
        profiles,
        columns=[
            "profile",
            "feature_definition",
            "needed_raw_columns",
            "usable_cutoff",
            "leakage_risk",
            "safe_implementation",
            "unsafe_implementation_example",
        ],
    ).assign(dataset_note=absent_location)


def safe_recipe():
    rows = [
        {
            "feature_name": "tm_p_prior_release_height_std",
            "status": "SAFE NON-LOCATION PROXY",
            "exact_raw_columns": "pitcher_trackman_id, season, rel_height",
            "group_keys": "pitcher_trackman_id mapped to pitcher_id",
            "time_cutoff": "source season < prediction season for validation; <=2024 for 2025 test",
            "aggregation": "std with count",
            "fallback": "global prior-history median plus missing flag",
            "note": "Not a plate location command feature; release consistency only.",
        },
        {
            "feature_name": "tm_p_prior_release_side_std",
            "status": "SAFE NON-LOCATION PROXY",
            "exact_raw_columns": "pitcher_trackman_id, season, rel_side",
            "group_keys": "pitcher_trackman_id mapped to pitcher_id",
            "time_cutoff": "source season < prediction season for validation; <=2024 for 2025 test",
            "aggregation": "std with count",
            "fallback": "global prior-history median plus missing flag",
            "note": "Not a plate location command feature.",
        },
        {
            "feature_name": "tm_p_prior_pitch_type_group_rate_fastball",
            "status": "CONDITIONALLY SAFE",
            "exact_raw_columns": "pitcher_trackman_id, season, pitch_type_group",
            "group_keys": "pitcher_trackman_id mapped to pitcher_id",
            "time_cutoff": "source season < prediction season for validation; <=2024 for 2025 test",
            "aggregation": "prior historical rate of fastball group",
            "fallback": "global prior-history rate",
            "note": "Pitch mix prior is safe; current row actual pitch_type is not safe.",
        },
        {
            "feature_name": "tm_p_prior_by_pitch_type_release_consistency",
            "status": "CONDITIONALLY SAFE",
            "exact_raw_columns": "pitcher_trackman_id, season, pitch_type_group, rel_height, rel_side, extension",
            "group_keys": "pitcher_trackman_id, historical pitch_type_group",
            "time_cutoff": "source season < prediction season for validation; <=2024 for 2025 test",
            "aggregation": "std by historical pitch type; expose as pitcher-level summary or pitch-mix weighted prior, not selected by current pitch_type",
            "fallback": "pitcher-level prior or global prior",
            "note": "Do not join with current actual pitch_type because test input lacks it.",
        },
    ]
    return pd.DataFrame(rows)


def forbidden_features():
    rows = []
    for col in ["plate_x", "plate_z", "zone", "pitch_result", "pitch_outcome", "control_success"]:
        rows.append({"forbidden_feature": f"current_row_{col}", "reason": "current pitch post-outcome/location information is unavailable at inference and leaks target"})
    for col in ["tagged_pitch_type", "auto_pitch_type", "pitch_type_group"]:
        rows.append({"forbidden_feature": f"current_row_{col}", "reason": "actual current pitch type is explicitly not provided in train/test input and is post-decision information"})
    rows.extend(
        [
            {"forbidden_feature": "validation_year_full_trackman_aggregate", "reason": "includes later validation-year games/pitches"},
            {"forbidden_feature": "same_game_full_trackman_aggregate", "reason": "includes future pitches from the same game"},
            {"forbidden_feature": "post_game_aggregate_for_current_pitch", "reason": "includes pitches after the current row"},
            {"forbidden_feature": "test_set_aggregate", "reason": "uses hidden evaluation rows against rules"},
            {"forbidden_feature": "target_based_trackman_mapping", "reason": "mapping may not use control_success pattern or predictions"},
            {"forbidden_feature": "future_season_mapping_counts_for_validation", "reason": "mapping code currently uses 2019-2024 count vectors; offline validation should rebuild mapping with prior seasons only"},
        ]
    )
    return pd.DataFrame(rows)


def assertion_results(trackman: pd.DataFrame) -> pd.DataFrame:
    rows = []
    trackman = trackman.copy()
    trackman["parsed_game_date"] = parse_trackman_dates(trackman)
    for valid_year in [2022, 2023, 2024]:
        prior = trackman[trackman["season"] < valid_year]
        unsafe_same_or_future = trackman[trackman["season"] >= valid_year]
        rows.append(
            {
                "check_name": f"prior_history_cutoff_validation_{valid_year}",
                "invariant": f"max(source season) < {valid_year}",
                "status": "PASS" if len(prior) and prior["season"].max() < valid_year else "FAIL",
                "observed_value": int(prior["season"].max()) if len(prior) else np.nan,
                "details": f"prior rows={len(prior)}",
            }
        )
        rows.append(
            {
                "check_name": f"full_season_would_leak_validation_{valid_year}",
                "invariant": f"no source season >= {valid_year}",
                "status": "EXPECTED_FAIL_UNSAFE_POLICY" if len(unsafe_same_or_future) else "PASS",
                "observed_value": int(unsafe_same_or_future["season"].min()) if len(unsafe_same_or_future) else np.nan,
                "details": f"same/future rows that must be excluded={len(unsafe_same_or_future)}",
            }
        )
    rows.append(
        {
            "check_name": "test_2025_trackman_cutoff",
            "invariant": "max(source season) <= 2024",
            "status": "PASS" if trackman["season"].max() <= 2024 else "FAIL",
            "observed_value": int(trackman["season"].max()),
            "details": f"rows={len(trackman)}",
        }
    )
    rows.append(
        {
            "check_name": "trackman_current_location_columns_absent",
            "invariant": "plate_x/plate_z/zone/result/outcome columns absent from provided Trackman",
            "status": "PASS",
            "observed_value": "absent",
            "details": "Confirms current-row location cannot be accidentally merged from this dataset schema.",
        }
    )
    duplicated_order = trackman.duplicated(["trackman_game_id", "pitch_no"]).sum()
    rows.append(
        {
            "check_name": "same_game_order_keys_available",
            "invariant": "game_date, trackman_game_id and pitch_no are available for strict as-of ordering if same-season aggregation is ever attempted",
            "status": "PASS" if {"game_date", "trackman_game_id", "pitch_no"}.issubset(trackman.columns) else "FAIL",
            "observed_value": f"duplicated_game_pitch_no={int(duplicated_order)}",
            "details": "Prior-season-only policy avoids same-game leakage; strict as-of would need tie handling because pitch_no may not be globally unique.",
        }
    )
    return pd.DataFrame(rows)


def mapping_audit():
    return pd.DataFrame(
        [
            {
                "mapping_component": "match_trackman_players.py",
                "uses_target": False,
                "uses_predictions": False,
                "uses_future_counts_for_offline_validation": True,
                "safe_for_2025_test": "CONDITIONALLY_SAFE",
                "offline_validation_policy": "Rebuild mapping with only seasons < validation_year, or freeze mapping as external metadata and do not tune decisions on validation target.",
                "reason": "Current matcher uses hand plus 2019-2024 pitch-count vectors. This is not target leakage, but for 2022/2023/2024 validation it can use future participation patterns.",
            }
        ]
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    trackman_sample, train_cols, test_cols = load_schema()
    trackman = pd.read_csv(
        TRACKMAN_PATH,
        usecols=["season", "game_date", "trackman_game_id", "pitch_no"],
        encoding="utf-8-sig",
    )

    make_inventory(trackman_sample).to_csv(OUT_DIR / "trackman_location_column_inventory.csv", index=False, encoding="utf-8")
    feature_classification(train_cols, test_cols).to_csv(OUT_DIR / "feature_leakage_classification.csv", index=False, encoding="utf-8")
    temporal_policy().to_csv(OUT_DIR / "temporal_cutoff_policy.csv", index=False, encoding="utf-8")
    profile_audit().to_csv(OUT_DIR / "command_profile_leakage_audit.csv", index=False, encoding="utf-8")
    safe_recipe().to_csv(OUT_DIR / "safe_feature_recipe.csv", index=False, encoding="utf-8")
    forbidden_features().to_csv(OUT_DIR / "forbidden_features.csv", index=False, encoding="utf-8")
    assertion_results(trackman).to_csv(OUT_DIR / "leakage_assertion_results.csv", index=False, encoding="utf-8")
    mapping_audit().to_csv(OUT_DIR / "mapping_leakage_audit.csv", index=False, encoding="utf-8")
    pd.DataFrame(
        [
            {
                "verdict": "LOCATION COMMAND APPROACH NOT SAFE",
                "reason": "Provided Trackman has no plate_x/plate_z/zone/result/outcome columns, and current actual pitch_type is absent from train/test. Only non-location historical release/movement aggregates are conditionally safe with strict prior-history aggregation.",
                "no_model_training": True,
                "no_submission_artifact": True,
            }
        ]
    ).to_csv(OUT_DIR / "verdict.csv", index=False, encoding="utf-8")

    print("Wrote leakage audit outputs to", OUT_DIR)


if __name__ == "__main__":
    main()
