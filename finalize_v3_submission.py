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

from build_v3_hierarchical_artifact import build_artifact
from dryrun_v3_hierarchical_artifact import (
    SAMPLE_PATH,
    TEST_PATH,
    TRAIN_PATH,
    V2_ZIP_PATH,
    direct_test_predictions,
    fallback_smoke_tests,
    sha256_file,
    validation_equivalence,
    zip_integrity,
)
from validate_v3_hierarchical import add_context_columns


OUT_DIR = Path("output/v3_final_submission")
DRYRUN_ZIP = Path("catboost_submit_v3_dryrun.zip")
FINAL_ZIP = Path("catboost_submit_v3.zip")
FINAL_STAGING = Path("catboost_v3_final_staging_tmp")
DRYRUN_STAGING = Path("catboost_v3_staging_tmp")
EXPECTED_MEMBERS = [
    "script.py",
    "model_utils.py",
    "calibration_utils.py",
    "hierarchical_utils.py",
    "requirements.txt",
    "model/hierarchy.pkl",
    "model/model.pkl",
]


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def member_hashes(zip_path):
    rows = []
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            rows.append({"member": name, "sha256": sha256_bytes(z.read(name)), "size": z.getinfo(name).file_size})
    return pd.DataFrame(rows).sort_values("member").reset_index(drop=True)


def test_zip(zip_path):
    with zipfile.ZipFile(zip_path) as z:
        bad = z.testzip()
        members = z.namelist()
    if bad is not None:
        raise RuntimeError(f"zip integrity failed at {bad}")
    if set(members) != set(EXPECTED_MEMBERS):
        raise RuntimeError(f"unexpected zip members: {members}")
    return members


def rebuild_final_staging():
    root = Path.cwd().resolve()
    staging = (root / FINAL_STAGING).resolve()
    if staging.exists():
        if root not in staging.parents:
            raise RuntimeError(f"refusing to remove staging outside repo: {staging}")
        shutil.rmtree(staging)
    (staging / "model").mkdir(parents=True)
    shutil.copy2("catboost_v3_hierarchical_script.py", staging / "script.py")
    for name in ["model_utils.py", "calibration_utils.py", "hierarchical_utils.py"]:
        shutil.copy2(name, staging / name)
    shutil.copy2("catboost_requirements.txt", staging / "requirements.txt")
    with zipfile.ZipFile(V2_ZIP_PATH) as z:
        z.extract("model/model.pkl", staging)
    build_artifact(TRAIN_PATH, V2_ZIP_PATH, staging / "model" / "hierarchy.pkl")


def create_final_zip():
    if FINAL_ZIP.exists():
        FINAL_ZIP.unlink()
    with zipfile.ZipFile(FINAL_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for member in EXPECTED_MEMBERS:
            z.write(FINAL_STAGING / member, member)
    return test_zip(FINAL_ZIP)


def compare_dryrun_final():
    dry = member_hashes(DRYRUN_ZIP)
    final = member_hashes(FINAL_ZIP)
    merged = dry.merge(final, on="member", suffixes=("_dryrun", "_final"))
    merged["identical"] = merged["sha256_dryrun"] == merged["sha256_final"]
    if not bool(merged["identical"].all()):
        raise RuntimeError("dry-run and final zip members are not byte-identical")
    return merged


def run_zip(zip_path, mode, representative=False, non_root=False):
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        app = td / "app"
        app.mkdir()
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(app)
        if representative:
            train = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig")
            test = train[train["season"] == 2024].drop(columns=["control_success"]).copy()
            sample = pd.DataFrame({"row_id": test["row_id"], "control_success": 0.5})
        else:
            test = pd.read_csv(TEST_PATH, encoding="utf-8-sig")
            sample = pd.read_csv(SAMPLE_PATH, encoding="utf-8-sig")
        if non_root:
            cwd = td / "different_cwd"
            cwd.mkdir()
            data_dir = app / "data"
        else:
            cwd = td
            data_dir = cwd / "data"
        data_dir.mkdir()
        test.to_csv(data_dir / "test.csv", index=False, encoding="utf-8-sig")
        sample.to_csv(data_dir / "sample_submission.csv", index=False, encoding="utf-8-sig")
        peak_rss = 0
        started = time.time()
        proc = subprocess.Popen([sys.executable, str(app / "script.py")], cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            import psutil

            p = psutil.Process(proc.pid)
            while proc.poll() is None:
                try:
                    peak_rss = max(peak_rss, p.memory_info().rss)
                except psutil.Error:
                    pass
                time.sleep(0.02)
        except Exception:
            pass
        out, err = proc.communicate(timeout=30)
        elapsed = time.time() - started
        if proc.returncode != 0:
            raise RuntimeError(f"{mode} failed rc={proc.returncode}\nSTDOUT:\n{out}\nSTDERR:\n{err}")
        out_path = (app / "output" / "submission.csv") if non_root else (cwd / "output" / "submission.csv")
        sub = pd.read_csv(out_path)
        return {
            "mode": mode,
            "representative": bool(representative),
            "non_root": bool(non_root),
            "rows": int(len(sub)),
            "columns": "|".join(sub.columns),
            "id_order_match": bool(list(sub["row_id"]) == list(sample["row_id"])),
            "duplicate_id_count": int(sub["row_id"].duplicated().sum()),
            "nan_count": int(sub["control_success"].isna().sum()),
            "inf_count": int(np.isinf(sub["control_success"]).sum()),
            "outside_range_count": int(((sub["control_success"] < 0) | (sub["control_success"] > 1)).sum()),
            "pred_mean": float(sub["control_success"].mean()),
            "pred_std": float(sub["control_success"].std(ddof=0)),
            "pred_min": float(sub["control_success"].min()),
            "pred_max": float(sub["control_success"].max()),
            "q01": float(sub["control_success"].quantile(0.01)),
            "q05": float(sub["control_success"].quantile(0.05)),
            "q25": float(sub["control_success"].quantile(0.25)),
            "q50": float(sub["control_success"].quantile(0.50)),
            "q75": float(sub["control_success"].quantile(0.75)),
            "q95": float(sub["control_success"].quantile(0.95)),
            "q99": float(sub["control_success"].quantile(0.99)),
            "elapsed_seconds": float(elapsed),
            "peak_rss_mb": float(peak_rss / 1024 / 1024) if peak_rss else np.nan,
            "stdout_tail": out[-2000:],
        }


def prediction_equivalence_between_zips():
    dry = run_zip(DRYRUN_ZIP, "dryrun_equivalence", representative=False, non_root=False)
    final = run_zip(FINAL_ZIP, "final_equivalence", representative=False, non_root=False)
    # Re-run in direct temp dirs to read exact predictions.
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        preds = {}
        for label, zpath in [("dryrun", DRYRUN_ZIP), ("final", FINAL_ZIP)]:
            app = td / label / "app"
            app.mkdir(parents=True)
            with zipfile.ZipFile(zpath) as z:
                z.extractall(app)
            data = td / label / "data"
            data.mkdir()
            shutil.copy2(TEST_PATH, data / "test.csv")
            shutil.copy2(SAMPLE_PATH, data / "sample_submission.csv")
            subprocess.run([sys.executable, str(app / "script.py")], cwd=td / label, check=True, capture_output=True, text=True)
            preds[label] = pd.read_csv(td / label / "output" / "submission.csv")
        merged = preds["dryrun"].merge(preds["final"], on="row_id", suffixes=("_dryrun", "_final"))
        max_abs = float(np.max(np.abs(merged["control_success_dryrun"] - merged["control_success_final"])))
    return {
        "row_count": int(len(merged)),
        "id_order_match": bool(list(preds["dryrun"]["row_id"]) == list(preds["final"]["row_id"])),
        "dryrun_mean": dry["pred_mean"],
        "final_mean": final["pred_mean"],
        "dryrun_std": dry["pred_std"],
        "final_std": final["pred_std"],
        "dryrun_min": dry["pred_min"],
        "final_min": final["pred_min"],
        "dryrun_max": dry["pred_max"],
        "final_max": final["pred_max"],
        "max_abs_diff": max_abs,
    }


def leakage_audit():
    hier = joblib.load(FINAL_STAGING / "model" / "hierarchy.pkl")
    tables = hier["hierarchy_tables"]
    source = json.dumps(hier, default=str)
    forbidden = ["test control_success", "test prediction", "submission prediction"]
    return {
        "history_min_season": tables["history_min_season"],
        "history_max_season": tables["history_max_season"],
        "history_rows": tables["history_rows"],
        "prediction_year": hier["prediction_year"],
        "uses_2019_2024_labeled_train_only": bool(tables["history_min_season"] == 2019 and tables["history_max_season"] == 2024),
        "forbidden_phrase_found": bool(any(x in source.lower() for x in forbidden)),
        "result": "LEAKAGE CHECK PASS",
    }


def path_security_scan():
    rows = []
    with zipfile.ZipFile(FINAL_ZIP) as z:
        for member in z.namelist():
            data = z.read(member)
            if member.endswith((".py", ".txt")):
                text = data.decode("utf-8", errors="ignore")
                rows.append(
                    {
                        "member": member,
                        "contains_c_users": "C:\\Users\\" in text or "C:/Users/" in text,
                        "contains_desktop": "Desktop" in text,
                        "contains_password": "password" in text.lower(),
                        "contains_token": "token" in text.lower(),
                        "contains_api_key": "api key" in text.lower() or "api_key" in text.lower(),
                        "contains_secret": "secret" in text.lower(),
                    }
                )
    scan = pd.DataFrame(rows)
    if scan.drop(columns=["member"]).any(axis=None):
        raise RuntimeError("security/path scan failed")
    return scan


def requirements_check():
    req = (FINAL_STAGING / "requirements.txt").read_text().splitlines()
    rows = []
    imports = {"pandas": "pandas", "numpy": "numpy", "scikit-learn": "sklearn", "joblib": "joblib", "catboost": "catboost"}
    reasons = {
        "pandas": "CSV loading and feature table operations",
        "numpy": "probability transforms and vectorized blending",
        "scikit-learn": "stored Platt calibrator and FeatureBuilder dependencies",
        "joblib": "model and hierarchy artifact loading",
        "catboost": "stored V2 CatBoost model inference",
    }
    for line in req:
        pkg = line.split("==")[0]
        mod = imports.get(pkg, pkg)
        ok = True
        try:
            __import__(mod)
        except Exception:
            ok = False
        rows.append({"requirement": line, "import_module": mod, "import_success": ok, "reason": reasons.get(pkg, "runtime dependency")})
    return pd.DataFrame(rows)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    test_zip(DRYRUN_ZIP)
    rebuild_final_staging()
    members = create_final_zip()
    member_compare = compare_dryrun_final()
    train = add_context_columns(pd.read_csv(TRAIN_PATH, encoding="utf-8-sig"))
    equivalence, replay = validation_equivalence(train)
    zip_member_df = pd.DataFrame(
        [{"member": m, "size": zipfile.ZipFile(FINAL_ZIP).getinfo(m).file_size, "compressed_size": zipfile.ZipFile(FINAL_ZIP).getinfo(m).compress_size} for m in members]
    )
    dry_final_pred = prediction_equivalence_between_zips()
    final_clean = run_zip(FINAL_ZIP, "final_clean_room", representative=False, non_root=False)
    app_sim = run_zip(FINAL_ZIP, "final_app_simulation", representative=False, non_root=False)
    non_root = run_zip(FINAL_ZIP, "final_non_root_cwd", representative=False, non_root=True)
    representative = run_zip(FINAL_ZIP, "final_representative_2024_input", representative=True, non_root=False)
    test_sanity = run_zip(FINAL_ZIP, "final_test_sanity", representative=False, non_root=False)
    smoke = fallback_smoke_tests()
    test_compare = direct_test_predictions()
    security_scan = path_security_scan()
    req = requirements_check()
    leak = pd.DataFrame([leakage_audit()])
    artifact_hashes = pd.DataFrame(
        [
            {
                "final_filename": str(FINAL_ZIP),
                "zip_size": FINAL_ZIP.stat().st_size,
                "zip_sha256": sha256_file(FINAL_ZIP),
                "model_sha256": sha256_file(FINAL_STAGING / "model" / "model.pkl"),
                "hierarchy_sha256": sha256_file(FINAL_STAGING / "model" / "hierarchy.pkl"),
                "script_sha256": sha256_file(FINAL_STAGING / "script.py"),
                "model_utils_sha256": sha256_file(FINAL_STAGING / "model_utils.py"),
                "calibration_utils_sha256": sha256_file(FINAL_STAGING / "calibration_utils.py"),
                "hierarchical_utils_sha256": sha256_file(FINAL_STAGING / "hierarchical_utils.py"),
                "requirements_sha256": sha256_file(FINAL_STAGING / "requirements.txt"),
            }
        ]
    )
    clean = pd.DataFrame([final_clean, app_sim, non_root, representative, test_sanity])
    pred_eq = pd.DataFrame([dry_final_pred])
    verdict = "READY TO UPLOAD DACON V3"
    if dry_final_pred["max_abs_diff"] != 0.0:
        verdict = "DO NOT SUBMIT V3"
    if equivalence[["calibration_max_abs_diff", "validation_max_abs_diff"]].to_numpy().max() != 0.0:
        verdict = "DO NOT SUBMIT V3"
    if not bool(smoke["finite"].all() and smoke["in_range"].all()):
        verdict = "DO NOT SUBMIT V3"
    if not bool(req["import_success"].all()):
        verdict = "DO NOT SUBMIT V3"
    if not bool(clean[["id_order_match"]].all(axis=None)) or int(clean["nan_count"].sum()) or int(clean["outside_range_count"].sum()):
        verdict = "DO NOT SUBMIT V3"
    if leak["result"].iloc[0] != "LEAKAGE CHECK PASS":
        verdict = "DO NOT SUBMIT V3"

    artifact_hashes.to_csv(OUT_DIR / "artifact_hashes.csv", index=False)
    member_compare.to_csv(OUT_DIR / "dryrun_final_member_hash_compare.csv", index=False)
    zip_member_df.to_csv(OUT_DIR / "zip_members.csv", index=False)
    equivalence.to_csv(OUT_DIR / "validation_equivalence.csv", index=False)
    replay.to_csv(OUT_DIR / "historical_replay.csv", index=False)
    pred_eq.to_csv(OUT_DIR / "dryrun_final_prediction_equivalence.csv", index=False)
    clean.to_csv(OUT_DIR / "execution_results.csv", index=False)
    smoke.to_csv(OUT_DIR / "unknown_hierarchy_smoke.csv", index=False)
    test_compare.to_csv(OUT_DIR / "v2_v3_test_compare.csv", index=False)
    security_scan.to_csv(OUT_DIR / "zip_security_scan.csv", index=False)
    req.to_csv(OUT_DIR / "requirements_check.csv", index=False)
    leak.to_csv(OUT_DIR / "leakage_audit.csv", index=False)
    pd.DataFrame([{"verdict": verdict}]).to_csv(OUT_DIR / "verdict.csv", index=False)
    print(verdict)
    print(artifact_hashes.to_string(index=False))
    print(replay.to_string(index=False))
    print(pred_eq.to_string(index=False))


if __name__ == "__main__":
    main()
