# Recent Form Experiment Resume Note

Date: 2026-08-14

## User Request

Continue the LG Aimers control success probability project with a recent-form / rolling historical feature experiment.

## Constraints

- Do not use Trackman.
- Do not modify `catboost_submit.zip`.
- Keep the current CatBoost + `FeatureBuilder` + global Platt + global `linear_all_history` trend prior + prior weight `0.75` structure.
- Do not combine the previous cumulative historical target aggregate experiment with this rolling experiment.
- Avoid validation leakage. Because `data_description.md` says test rows cannot use other test rows for rolling/expanding features, validation-year rolling features should be conservative: use only train + calibration history, not validation-year targets.

## Current Baseline

Existing stable baseline from `output/model_comparison/model_fold_metrics.csv`:

- 2022 Brier: `0.2488975149153299`
- 2023 Brier: `0.25015880346033137`
- 2024 Brier: `0.2498392201633735`
- Mean Brier: about `0.249632`

## Files Changed

- `model_utils.py`
  - `FeatureBuilder.fit()` now includes `rf_` prefixed columns as extra numeric features.

- `validate_recent_form_features.py`
  - New script for recent-form rolling feature validation.
  - Creates pitch-level pitcher rolling windows: `5, 10, 20, 50, 100`.
  - Uses `shift(1)` rolling inside training folds.
  - Uses static history from train/cal only for calibration/validation rows to avoid validation-year target accumulation.
  - Adds long-term pitcher rate and form deltas versus long-term/global rates.
  - Adds limited contextual rolling features for pitcher x `game_type`, pitcher x `count_state`, pitcher x `batter_hand` using window 50.
  - Outputs intended files under `output/recent_form/`.

## Partial Run

A smoke run was started:

```powershell
python validate_recent_form_features.py --output-dir output/recent_form_smoke --alphas 20 --feature-sets baseline,recent_20
```

It timed out after 120 seconds. Partial output exists:

- `output/recent_form_smoke/fold_metrics_partial.csv`

Only the first baseline fold appears to have completed before timeout:

- alpha `20`
- feature_set `baseline`
- valid year `2022`
- stages: `raw`, `platt`, `platt_plus_trend_w075`
- train seconds: about `18.39`

## Next Step

When resuming, first inspect whether any background Python process is still running. Then decide whether to:

1. Optimize `validate_recent_form_features.py` runtime, likely by reducing repeated baseline fits and narrowing the feature grid.
2. Run a smaller staged set first:

```powershell
python validate_recent_form_features.py --output-dir output/recent_form_stage1 --alphas 20 --feature-sets baseline,recent_5,recent_10,recent_20,recent_50,recent_20_50_100,recent_delta
```

3. If stage 1 shows a candidate, run alpha comparison only for that candidate plus baseline:

```powershell
python validate_recent_form_features.py --output-dir output/recent_form --alphas 5,10,20,50,100 --feature-sets baseline,recent_20,recent_50,recent_20_50_100,recent_delta,best_candidate
```

Avoid touching `catboost_submit.zip`.
