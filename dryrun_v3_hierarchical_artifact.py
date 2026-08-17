import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

import validate_v3_hierarchical_robustness as robust
from build_v3_hierarchical_artifact import build_artifact
from calibration_utils import apply_calibration, fit_calibrators
from hierarchical_utils import (
    ALPHA_CONTEXT,
    ALPHA_PITCHER,
    BLEND_K,
    TARGET_COL,
    build_hierarchy_tables,
    candidate_a_native,
    predict_hierarchy,
    target_rate_for_year,
    v2_strength_control,
)
from validate_v3_hierarchical import FOLDS, add_context_columns, metric_dict, v2_predictions


OUT_DIR = Path("output/v3_artifact_dryrun")
STAGING_DIR = Path("catboost_v3_staging_tmp")
ZIP_PATH = Path("catboost_submit_v3_dryrun.zip")
V2_ZIP_PATH = Path("catboost_submit_v2.zip")
TRAIN_PATH = Path("data/train.csv")
TEST_PATH = Path("data/test.csv")
SAMPLE_PATH = Path("data/sample_submission.csv")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def constant_metrics(y, pred):
    actual = float(np.mean(y))
    constant = actual * (1.0 - actual)
    brier = float(brier_score_loss(y, pred))
    skill = constant - brier
    return {
        "auc": float(roc_auc_score(y, pred)),
        "brier": brier,
        "constant_brier": constant,
        "skill_margin": float(skill),
        "pseudo_score": float(max(0.0, 100000.0 * skill / constant)),
        "pred_mean": float(np.mean(pred)),
        "pred_std": float(np.std(pred)),
    }


def validation_equivalence(train):
    rows = []
    replay_rows = []
    robust.GLOBAL_DF = train
    for train_start, train_end, cal_year, valid_year in FOLDS:
        train_fold = train[(train["season"] >= train_start) & (train["season"] <= train_end)].copy()
        cal = train[train["season"] == cal_year].copy()
        valid = train[train["season"] == valid_year].copy()
        _, _, v2_platt_valid, _, y_valid = v2_predictions(train_fold, cal, valid)
        source_builder = robust.soft_blend_builder("pitcher_game", 100, False)
        source_cal, source_valid, _, _, _ = source_builder(train_end, cal_year, valid_year, cal, valid, None, v2_platt_valid)

        table_cal = build_hierarchy_tables(train, cal_year, ALPHA_PITCHER, ALPHA_CONTEXT)
        table_valid = build_hierarchy_tables(train, valid_year, ALPHA_PITCHER, ALPHA_CONTEXT)
        impl_cal, _, _ = predict_hierarchy(cal, table_cal)
        impl_valid, _, _, _ = candidate_a_native(valid, table_valid, v2_platt_valid)

        cal_diff = float(np.max(np.abs(source_cal - impl_cal)))
        valid_diff = float(np.max(np.abs(source_valid - impl_valid)))
        rows.append(
            {
                "valid_year": valid_year,
                "calibration_rows": len(cal),
                "validation_rows": len(valid),
                "calibration_max_abs_diff": cal_diff,
                "validation_max_abs_diff": valid_diff,
            }
        )
        if cal_diff > 1e-12 or valid_diff > 1e-12:
            raise RuntimeError(f"implementation equivalence failed for {valid_year}: cal={cal_diff} valid={valid_diff}")

        calibrator = fit_calibrators(impl_cal, cal[TARGET_COL].astype("int8"), cal[["game_type"]])
        platt_valid = apply_calibration(impl_valid, valid[["game_type"]], calibrator, "platt")
        final_valid = v2_strength_control(platt_valid, target_rate_for_year(train, valid_year))
        replay_rows.append({"valid_year": valid_year, **constant_metrics(y_valid, final_valid)})
    return pd.DataFrame(rows), pd.DataFrame(replay_rows)


def prepare_staging(train):
    root = Path.cwd().resolve()
    staging = (root / STAGING_DIR).resolve()
    if staging.exists():
        if root not in staging.parents:
            raise RuntimeError(f"refusing to delete staging outside repo: {staging}")
        shutil.rmtree(staging)
    (staging / "model").mkdir(parents=True)

    shutil.copy2("catboost_v3_hierarchical_script.py", staging / "script.py")
    for name in ["model_utils.py", "calibration_utils.py", "hierarchical_utils.py"]:
        shutil.copy2(name, staging / name)
    shutil.copy2("catboost_requirements.txt", staging / "requirements.txt")
    with zipfile.ZipFile(V2_ZIP_PATH) as z:
        z.extract("model/model.pkl", staging)
    hierarchy = build_artifact(TRAIN_PATH, V2_ZIP_PATH, staging / "model" / "hierarchy.pkl")
    return hierarchy


def make_zip():
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path in STAGING_DIR.rglob("*"):
            if path.is_file():
                z.write(path, path.relative_to(STAGING_DIR).as_posix())
    with zipfile.ZipFile(ZIP_PATH) as z:
        bad = z.testzip()
        members = z.namelist()
    if bad is not None:
        raise RuntimeError(f"zip test failed at {bad}")
    return members


def run_clean_room(non_root=False):
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        app = td / "app"
        with zipfile.ZipFile(ZIP_PATH) as z:
            z.extractall(app)
        if non_root:
            (app / "data").mkdir()
            shutil.copy2(TEST_PATH, app / "data" / "test.csv")
            shutil.copy2(SAMPLE_PATH, app / "data" / "sample_submission.csv")
            cwd = td / "cwd_elsewhere"
            cwd.mkdir()
        else:
            cwd = td
            (cwd / "data").mkdir()
            shutil.copy2(TEST_PATH, cwd / "data" / "test.csv")
            shutil.copy2(SAMPLE_PATH, cwd / "data" / "sample_submission.csv")
        started = time.time()
        proc = subprocess.run([sys.executable, str(app / "script.py")], cwd=cwd, text=True, capture_output=True, timeout=120)
        elapsed = time.time() - started
        if proc.returncode != 0:
            raise RuntimeError(f"clean-room failed rc={proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
        out_path = (cwd / "output" / "submission.csv") if not non_root else (app / "output" / "submission.csv")
        sub = pd.read_csv(out_path)
        return {
            "non_root": bool(non_root),
            "elapsed_seconds": elapsed,
            "stdout_tail": proc.stdout[-2000:],
            "rows": int(len(sub)),
            "columns": "|".join(sub.columns),
            "pred_mean": float(sub[TARGET_COL].mean()),
            "pred_std": float(sub[TARGET_COL].std(ddof=0)),
            "pred_min": float(sub[TARGET_COL].min()),
            "pred_max": float(sub[TARGET_COL].max()),
            "nan_count": int(sub[TARGET_COL].isna().sum()),
        }


def direct_test_predictions():
    test = pd.read_csv(TEST_PATH, encoding="utf-8-sig")
    with zipfile.ZipFile(V2_ZIP_PATH) as z:
        with tempfile.TemporaryDirectory() as td:
            z.extract("model/model.pkl", td)
            v2 = joblib.load(Path(td) / "model" / "model.pkl")
    hier = joblib.load(STAGING_DIR / "model" / "hierarchy.pkl")
    x = v2["builder"].transform(test)
    raw = v2["model"].predict_proba(x)[:, 1]
    v2_platt = apply_calibration(raw, test[["game_type"]], v2["calibration"], v2["calibration_method"])
    v2_final = v2_strength_control(v2_platt, hier["target_rate_2025"])
    native, p_count, c_count, reliability = candidate_a_native(test, hier["hierarchy_tables"], v2_platt)
    platt = apply_calibration(native, test[["game_type"]], hier["calibration"], hier["calibration_method"])
    v3_final = v2_strength_control(platt, hier["target_rate_2025"])
    corr = np.nan if len(test) < 2 else float(np.corrcoef(v2_final, v3_final)[0, 1])
    spear = np.nan if len(test) < 2 else float(pd.Series(v2_final).corr(pd.Series(v3_final), method="spearman"))
    return pd.DataFrame(
        [
            {
                "rows": len(test),
                "v2_mean": float(v2_final.mean()),
                "v2_std": float(v2_final.std()),
                "v3_mean": float(v3_final.mean()),
                "v3_std": float(v3_final.std()),
                "v3_min": float(v3_final.min()),
                "v3_max": float(v3_final.max()),
                "pearson": corr,
                "spearman": spear,
                "mean_abs_diff": float(np.mean(np.abs(v2_final - v3_final))),
                "max_abs_diff": float(np.max(np.abs(v2_final - v3_final))),
                "known_pitcher_rate": float((p_count > 0).mean()),
                "known_pitcher_game_rate": float((c_count > 0).mean()),
                "mean_reliability": float(reliability.mean()),
            }
        ]
    )


def fallback_smoke_tests():
    test = pd.read_csv(TEST_PATH, encoding="utf-8-sig")
    hier = joblib.load(STAGING_DIR / "model" / "hierarchy.pkl")
    base = test.iloc[[0]].copy()
    cases = []
    known = base.copy()
    known["case"] = "known_pitcher_known_game_type"
    cases.append(known)
    unseen_game = base.copy()
    unseen_game["game_type"] = "UNSEEN_GAME_TYPE"
    unseen_game["case"] = "known_pitcher_unseen_game_type"
    cases.append(unseen_game)
    unseen_pitcher = base.copy()
    unseen_pitcher["pitcher_id"] = "__UNSEEN_PITCHER__"
    unseen_pitcher["case"] = "unseen_pitcher"
    cases.append(unseen_pitcher)
    missing_game = base.copy()
    missing_game["game_type"] = np.nan
    missing_game["case"] = "missing_game_type"
    cases.append(missing_game)
    zero_history = base.copy()
    zero_history["pitcher_id"] = "__ZERO_HISTORY__"
    zero_history["game_type"] = "R"
    zero_history["case"] = "zero_history_count_edge"
    cases.append(zero_history)
    rows = pd.concat(cases, ignore_index=True)
    native, p_count, c_count, reliability = candidate_a_native(rows, hier["hierarchy_tables"], np.full(len(rows), 0.5))
    return pd.DataFrame(
        {
            "case": rows["case"],
            "prediction": native,
            "pitcher_count": p_count,
            "context_count": c_count,
            "reliability": reliability,
            "finite": np.isfinite(native),
            "in_range": (native >= 0) & (native <= 1),
        }
    )


def zip_integrity(members):
    rows = []
    with zipfile.ZipFile(ZIP_PATH) as z:
        for member in members:
            info = z.getinfo(member)
            rows.append({"member": member, "size": info.file_size, "compressed_size": info.compress_size})
    text_scan = []
    for rel in ["script.py", "model_utils.py", "calibration_utils.py", "hierarchical_utils.py", "requirements.txt"]:
        text = (STAGING_DIR / rel).read_text(encoding="utf-8", errors="ignore")
        text_scan.append(
            {
                "file": rel,
                "has_windows_abs_path": bool(("C:\\" in text) or ("C:/" in text)),
                "has_secret_word": any(token in text.lower() for token in ["token", "password", "secret", "github_pat"]),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(text_scan)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train = add_context_columns(pd.read_csv(TRAIN_PATH, encoding="utf-8-sig"))
    equivalence, replay = validation_equivalence(train)
    hierarchy = prepare_staging(train)
    members = make_zip()
    clean = run_clean_room(non_root=False)
    non_root = run_clean_room(non_root=True)
    test_compare = direct_test_predictions()
    smoke = fallback_smoke_tests()
    member_df, scan_df = zip_integrity(members)

    hierarchy_tables = hierarchy["hierarchy_tables"]
    table_summary = pd.DataFrame(
        [
            {
                "target_rate_2025": hierarchy["target_rate_2025"],
                "year_rates": json.dumps(hierarchy["year_rates"], sort_keys=True),
                "pitcher_rows": len(hierarchy_tables["pitcher"]),
                "pitcher_game_rows": len(hierarchy_tables["pitcher_game"]),
                "history_rows": hierarchy_tables["history_rows"],
                "history_min_season": hierarchy_tables["history_min_season"],
                "history_max_season": hierarchy_tables["history_max_season"],
                "hierarchy_pkl_size": (STAGING_DIR / "model" / "hierarchy.pkl").stat().st_size,
                "model_pkl_size": (STAGING_DIR / "model" / "model.pkl").stat().st_size,
                "zip_size": ZIP_PATH.stat().st_size,
                "zip_sha256": sha256_file(ZIP_PATH),
                "model_sha256": sha256_file(STAGING_DIR / "model" / "model.pkl"),
                "hierarchy_sha256": sha256_file(STAGING_DIR / "model" / "hierarchy.pkl"),
            }
        ]
    )
    clean_df = pd.DataFrame([clean, non_root])
    verdict = "V3 DRY-RUN ARTIFACT READY"
    if not smoke["finite"].all() or not smoke["in_range"].all():
        verdict = "V3 ARTIFACT IMPLEMENTATION ISSUE"
    if replay["auc"].mean() < 0.5313 or abs(float(replay[replay["valid_year"] == 2023]["skill_margin"].iloc[0]) - (-0.000398)) > 5e-5:
        verdict = "V3 ARTIFACT IMPLEMENTATION ISSUE"
    pd.DataFrame([{"verdict": verdict}]).to_csv(OUT_DIR / "verdict.csv", index=False)
    equivalence.to_csv(OUT_DIR / "validation_equivalence.csv", index=False)
    replay.to_csv(OUT_DIR / "historical_replay.csv", index=False)
    table_summary.to_csv(OUT_DIR / "artifact_table_summary.csv", index=False)
    clean_df.to_csv(OUT_DIR / "clean_room_results.csv", index=False)
    test_compare.to_csv(OUT_DIR / "test_prediction_sanity.csv", index=False)
    smoke.to_csv(OUT_DIR / "fallback_smoke_tests.csv", index=False)
    member_df.to_csv(OUT_DIR / "zip_members.csv", index=False)
    scan_df.to_csv(OUT_DIR / "artifact_scan.csv", index=False)
    print(verdict)
    print(replay.to_string(index=False))
    print(table_summary.to_string(index=False))


if __name__ == "__main__":
    main()
