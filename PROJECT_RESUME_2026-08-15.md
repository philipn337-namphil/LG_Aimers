# LG Aimers Resume Note - 2026-08-15

## Git Baseline Already Saved

- GitHub repo: `https://github.com/philipn337-namphil/LG_Aimers.git`
- Branch: `main`
- `submission-v1`: original CatBoost submission checkpoint
  - commit: `c2e9d8b1354f5bb2e2cda4b2e308a337e4614f0a`
- `submission-v1-hardened`: cwd-independent hardened artifact checkpoint
  - commit: `85eb400046004604f02125471d375938b9bf3fc0`

## Submission Artifacts

- `catboost_submit.zip`
  - Original CatBoost + Platt + global linear trend prior artifact.
  - Do not modify.
- `catboost_submit_hardened.zip`
  - Added cwd-independent path handling.
  - Submitted to DACON and failed with server error:
    - `FileNotFoundError: [Errno 2] No such file or directory: '/app/model/model.pkl'`
- `catboost_submit_daconfix.zip`
  - New artifact created after server error.
  - Zip member listing verified exactly:
    - `script.py`
    - `model_utils.py`
    - `calibration_utils.py`
    - `requirements.txt`
    - `model/model.pkl`
  - `model/model.pkl` exists in zip, size `598167`.
  - model SHA256: `3048DC5D2AB0756BAC0408C52E3E0A2C0BDB1C1A3FAAF4B940004B971753960D`
  - Local `/app`-style root and non-root cwd inference both passed.
  - Prediction compared to `catboost_submit_hardened.zip` on 5-row test:
    - `max_abs_diff = 0.0`
  - Not committed yet.
- `catboost_submit_v2.zip`
  - Tag: `submission-v2`
  - Commit: `5c5ad0dff61332ead008d81a10b5d4508af689f0`
  - Submitted on 2026-08-15.
  - Official leaderboard anchor from this point forward.
  - Model/calibration:
    - CatBoost + Platt
    - `linear_trend_recent3` target-rate estimator
    - logit intercept mean matching
    - temperature `T=2.3`
    - symmetric hard cap `+/-0.020`
  - DACON Public Score: `96.253447238`

## DACON Public Result

- `catboost_submit_daconfix.zip` submitted and code execution succeeded.
- Public Score = `0`.
- Official score:
  - `Score = max(0, 100000 * (1 - Brier / (r*(1-r))))`
- Interpretation:
  - Public test Brier was at least as bad as constant mean-rate baseline.

## Leaderboard Zero Diagnosis

Script:
- `diagnose_leaderboard_zero.py`

Output:
- `output/leaderboard_zero_diagnosis/`
  - `variant_fold_metrics.csv`
  - `pseudo_scores.csv`
  - `yearly_global_rates.csv`
  - `variant_summary.csv`

Key validation rates:
- 2022 actual rate: `0.528920`
- 2023 actual rate: `0.499957`
- 2024 actual rate: `0.486105`
- 2024 constant-rate Brier: about `0.249807`

2024 results:
- CatBoost + trend `w=0.75`: Brier `0.249839`, pseudo score `0`
- CatBoost + trend `w=0.50`: Brier `0.249868`, pseudo score `0`
- CatBoost + trend `w=0.30`: Brier `0.249912`, pseudo score `0`
- CatBoost + trend `w=0.10`: Brier `0.249975`, pseudo score `0`
- CatBoost + Platt: Brier `0.250014`, pseudo score `0`
- CatBoost raw: Brier `0.255516`, pseudo score `0`

Conclusion:
- Problem is not only trend prior weight.
- Current row-level discrimination is not strong enough to reliably beat constant-rate baseline under DACON scoring.

## V3 Ensemble Validation - 2026-08-16

Script:
- `validate_v3_ensemble.py`

Output:
- `output/v3_ensemble/`
  - `selected_previous_candidates.csv`
  - `model_specs.csv`
  - `model_pair_diversity.csv`
  - `error_diversity.csv`
  - `ensemble_grid.csv`
  - `fold_metrics.csv`
  - `ensemble_summary.csv`
  - `year2023_analysis.csv`
  - `verdict.csv`

Large local-only files, intentionally not committed:
- `output/v3_ensemble/oof_predictions.csv`
- `output/v3_ensemble/calibration_oof_predictions.csv`

Previous as-of reconstruction status:
- `previous_signal_reconstruction_success_count = 0`
- No candidate met the prior `success_flag`.

Actual previous best candidates used:
- Fold-stable reconstructed: `A_recent_vs_longterm`, `v2_params`, families `A`
- Highest discrimination reconstructed: `A_C_skill_combo_tuned_catboost`, `tuned_params`, families `AC`
- Best 2023 skill-margin reconstructed: `A_recent_vs_longterm`, same as fold-stable candidate

Compared models:
- `A_v2_base`: V2 CatBoost params + V2 feature set
- `B_tuned_base`: tuned CatBoost params + V2 feature set
- `C_stable_reconstructed_A_recent_vs_longterm_v2_params`
- `D_high_discrimination_A_C_skill_combo_tuned_catboost_tuned_params`

Key results:
- Most diverse prediction pair vs V2: `A_v2_base` vs `C_stable_reconstructed_A_recent_vs_longterm_v2_params`
  - Pearson correlation `0.9908975`
  - Spearman correlation `0.9887960`
  - mean abs prediction difference `0.0063212`
- Lowest squared-error correlation pair: `A_v2_base` vs `C_stable_reconstructed_A_recent_vs_longterm_v2_params`
  - squared-error correlation `0.9967285`
- Best mean AUC candidate was the single reconstructed tuned model:
  - `D_high_discrimination_A_C_skill_combo_tuned_catboost_tuned_params`
  - mean AUC `0.5210465`
  - 2022 pseudo `122.4409`
  - 2023 pseudo `0`
  - 2024 pseudo `18.1113`
  - 2023 skill margin `-0.0005734`
  - mean Brier improvement vs V2 `-0.00002966`
- Best 2023 skill-margin reconstructed model:
  - `C_stable_reconstructed_A_recent_vs_longterm_v2_params`
  - 2023 skill margin `-0.0005524`
  - delta vs V2 2023 skill `+0.0000156`
  - 2024 pseudo only `14.7889`
- No 2-model or limited 3-model blend satisfied V3 selection criteria.

Verdict:
- `NO ENSEMBLE VALUE`
- Do not create `submission-v3` from this ensemble result.
- Next recommended direction: Trackman second-stage feature engineering.

## Workspace Cleanup for Trackman V3 - 2026-08-16

Goal:
- Keep only V3 research essentials in the current working tree.
- Preserve historical V1/V2/failed-experiment artifacts through Git history and tags.

Preserved:
- `data/`
  - `train.csv`
  - `test.csv`
  - `sample_submission.csv`
  - `trackman_history.csv`
- `.git/`, `.gitignore`
- `model/`
- `catboost_submit_v2.zip`
- V2/V3 core code:
  - `model_utils.py`
  - `calibration_utils.py`
  - `catboost_v2_script.py`
  - `train_catboost_submit_model.py`
  - `validate_v3_catboost_tuning.py`
  - `validate_v3_asof_signal_reconstruction.py`
  - `validate_v3_ensemble.py`
- Trackman code:
  - `match_trackman_players.py`
  - `build_trackman_features.py`
  - `trackman_feature_utils.py`
- Trackman derived reference outputs:
  - `output/trackman_matching/`
  - `output/trackman_features/`
- V3 reference outputs:
  - `output/v3_catboost_tuning/`
  - `output/v3_asof_signal_reconstruction/`
  - `output/v3_ensemble/`

Deleted from current workspace:
- staging/tmp directories
- duplicate `catboost_model/`
- old submission zips except `catboost_submit_v2.zip`
- old V1/V2 diagnostic scripts no longer needed for active V3 work
- old experiment output directories
- large regenerated OOF files:
  - `output/model_comparison/oof_predictions.csv`
  - `output/v3_ensemble/oof_predictions.csv`
  - `output/v3_ensemble/calibration_oof_predictions.csv`

Review, kept intentionally:
- `RECENT_FORM_RESUME.md`
- `catboost_requirements.txt`

Trackman raw:
- `data/trackman_history.csv`
- seasons `2019-2024`
- rows are not modified.

Tags preserved:
- `submission-v1`
- `submission-v1-hardened`
- `submission-v2`

## Score-Positive Feature Experiment

Script:
- `validate_score_positive_features.py`

Output:
- `output/score_positive_features/`
  - `feature_inventory.csv`
  - `feature_set_fold_metrics.csv`
  - `feature_set_summary.csv`
  - `feature_importance.csv`
  - `asof_feature_analysis.csv`

Important correction:
- Initial run accidentally added `rf_count_state_code` to baseline.
- Script was corrected.
- Final files in `output/score_positive_features/` were copied from corrected results.

Findings:
- Official as-of/history features are already used by current `FeatureBuilder`.
- Strongest as-of target separation:
  - `asof_pitcher_success_rate`: `+0.1216`
  - `asof_pitcher_reverse_rate`: `-0.1145`
  - `asof_pitcher_prev5_game_success_rate`: `+0.1112`
  - `asof_pitcher_prev3_game_success_rate`: `+0.1094`
- Best 2024 feature set was `count_interactions`, but still failed:
  - 2024 Brier `0.249997`
  - constant Brier `0.249807`
  - skill margin `-0.000190`
  - pseudo score `0`
- No feature set achieved 2024 positive score.
- Do not make a submission from these feature sets yet.

## Holdout-Aware Calibration Experiment

Script:
- `validate_holdout_aware_calibration.py`

Output:
- `output/holdout_aware_calibration/`
  - `rate_estimator_fold_metrics.csv`
  - `rate_estimator_summary.csv`
  - `mean_shift_metrics.csv`
  - `damped_trend_grid.csv`
  - `yearly_global_rates.csv`

Setup:
- Base prediction fixed to CatBoost + Platt.
- No feature/model/hyperparameter changes.
- Only global mean adjustment compared.
- Validation actual rate was not used for mean matching.

Best 2024 estimator:
- `G_linear_trend_recent3 + probability_shift`
- estimated 2024 rate: `0.487742`
- actual 2024 rate: `0.486105`
- rate error: `+0.001637`
- Brier: `0.249824`
- constant Brier: `0.249807`
- skill margin: `-0.000017`
- pseudo score: `0`

Conclusion:
- Even nearly correct global mean did not create positive 2024 score.
- Calibration-only is not enough with current discrimination.

## Current Untracked Work

Current `git status` showed untracked:
- `catboost_daconfix_staging_tmp/`
- `catboost_submit_daconfix.zip`
- `diagnose_leaderboard_zero.py`
- `validate_score_positive_features.py`
- `validate_holdout_aware_calibration.py`
- `output/leaderboard_zero_diagnosis/`
- `output/score_positive_features/`
- `output/score_positive_features_baseline_fi/`
- `output/score_positive_features_corrected/`
- `output/holdout_aware_calibration/`

Do not commit yet unless explicitly requested.

## Recommended Next Step

Next experiment should not be more global mean calibration. It should test whether controlled prediction-strength/shrinkage can beat constant baseline:

- Base: CatBoost + Platt
- Mean target: conservative `linear_trend_recent3`
- Apply logit temperature scaling or shrink-to-mean:
  - `p_adj = target_rate + lambda * (p_mean_matched - target_rate)`
  - compare `lambda` values like `0.1, 0.2, 0.3, 0.5, 0.75, 1.0`
- Primary success criterion:
  - 2024 `skill_margin > 0`
  - positive-score folds >= 2/3

No submission zip should be created until a method produces positive 2024 pseudo score offline.

## V3 CatBoost Tuning Experiment

Script:
- `validate_v3_catboost_tuning.py`

Output:
- `output/v3_catboost_tuning/`

Setup:
- FeatureBuilder and feature set unchanged.
- CatBoost parameters only changed.
- V2 calibration policy fixed for every candidate:
  - Platt
  - `linear_trend_recent3`
  - logit intercept mean matching
  - temperature `T=2.3`
  - symmetric hard cap `+/-0.020`
- No test prediction/distribution tuning.

Result:
- 35 total configurations.
- Baseline V2 replay mean AUC: `0.519446`.
- Best mean AUC: `0.520376`.
- Best 2024 AUC: `0.511992`.
- Best 2024 pseudo score: `18.300`, but 2023 skill margin worsened vs V2.
- Most fold-stable variant: `random_strength=2.0`
  - improves 2023 loss slightly
  - but lowers 2024 pseudo vs V2.

Conclusion:
- CatBoost hyperparameter tuning alone appears saturated.
- Do not create `submission-v3` from this experiment.
- Next experiment should move to official as-of feature signal reconstruction, especially long-term vs recent pitcher/count/context signals.

## V3 Official As-Of Signal Reconstruction Experiment

Script:
- `validate_v3_asof_signal_reconstruction.py`

Output:
- `output/v3_asof_signal_reconstruction/`

Setup:
- Only official provided `asof_*` columns were deterministically transformed.
- No new target aggregates.
- No validation/test target-derived rolling features.
- No Trackman.
- V2 CatBoost parameters and V2 calibration policy were kept fixed for family comparisons.

Important source signals:
- `asof_pitcher_success_rate`
- `asof_pitcher_reverse_rate`
- `asof_pitcher_prev1_game_success_rate`
- `asof_pitcher_prev3_game_success_rate`
- `asof_pitcher_prev5_game_success_rate`
- `asof_batter_success_rate`
- official pitcher pitchmix rates.

Best V2-param feature family:
- `A_C_skill_combo`
  - recent-vs-career pitcher success deltas
  - pitcher success/reverse/middle decomposition
  - mean AUC: `0.520134`
  - delta vs V2 mean AUC: `+0.000689`
  - 2024 pseudo: `17.530`
  - 2023 skill margin: `-0.000560`

Tuned CatBoost one-shot with best family:
- `A_C_skill_combo_tuned_catboost`
  - mean AUC: `0.521047`
  - 2024 pseudo: `18.111`
  - 2023 skill margin: `-0.000573`

Most 2023-helpful family:
- `A_recent_vs_longterm`
  - 2023 skill improvement vs V2: `+0.0000156`
  - 2024 pseudo: `14.789`

Conclusion:
- Reconstructed official as-of features receive non-zero model importance.
- Improvements are small and do not meet success thresholds:
  - mean AUC did not reach `0.523`
  - 2024 pseudo did not reach `25`
  - 2023 skill improvement was too small to treat as a breakthrough.
- Do not create `submission-v3` from this experiment.
- Next step should evaluate OOF error diversity/ensemble only if combining V2, tuned CatBoost, and reconstructed-signal models can reduce fold-specific errors; otherwise revisit Trackman at pitch-context level.
