# GridGuard — Model Card

**Model**: Gradient-boosted trees (LightGBM) over 22 engineered features from
90-day smart-meter consumption windows.

## Data

- **Source**: SGCC (State Grid Corporation of China) electricity-theft
  benchmark — daily kWh of 42,372 consumers, 1 Jan 2014 – 31 Oct 2016
  (1,034 days), with 3,615 consumers (8.53%) flagged for electricity theft
  by the utility's inspection process. Public mirrors:
  Kaggle `bensalem14/sgcc-dataset`; originally released with
  Zheng et al., *"Electricity Theft Detection Using Generative-Adversarial
  Networks"* (Wide & Deep CNN variants, 2017) and used by Buzau et al.,
  *"Hybrid Deep Neural Networks for Detection of Non-Technical Losses in
  Electricity Smart Meters"* (IEEE TSG 2019).
- **Labels are utility inspection outcomes, not ground truth**: a "flagged"
  consumer was *caught* stealing; unflagged consumers are "not (yet) caught".
  Some label noise is inherent (undetected thieves among the honest class;
  consumers who stopped stealing among the flagged class).
- **Missingness**: 25.6% of daily readings are missing overall. Imputation:
  clip negatives → linear interpolation over interior gaps → nearest-value
  edge fill → 0 for fully-dead meters. No target information is used.
- **Geographic scope**: Chinese urban/rural distribution grid. Cross-country
  transfer (e.g., Indian DISCOM data) requires retraining + recalibration;
  consumption magnitudes and theft behaviours differ.

## Features (22, per 90-day window)

Level (mean/std/min/Q1/median/Q3/max/CV), shape (population skew/kurtosis),
zero-behaviour (zero-ratio, longest zero run, normalized), temporal
(weekday/weekend means, weekend ratio, month-bucket spread), drift
(first-vs-second-half ratio, OLS trend slope), volatility (mean abs daily
diff, std of daily diffs, mean abs pct change).

## Training

- Windowing: 12 windows of 90 days per consumer (10 non-overlapping +
  2 end-anchored), imputed on the full series then sliced.
- Split: **by consumer** — 80% train / 20% test, stratified by flag.
  A consumer never appears on both sides.
- Model selection: 5-fold **StratifiedGroupKFold** CV (grouped by consumer),
  PR-AUC as the criterion (imbalanced data; accuracy is misleading here).
  Candidates: logistic regression (scaled), random forest, LightGBM —
  see `cv_table` in `models/model_meta.json` for numbers.
- Class imbalance handled via `class_weight="balanced"`; no synthetic
  oversampling (widely shown to overfit on this dataset when combined with
  consumer-level leakage; we avoid both the leakage and the SMOTE).
- Thresholds: `balanced` = best-F1 on out-of-fold train predictions;
  `high` = smallest threshold reaching ≥35% precision on out-of-fold
  predictions (tuned for a costly field-inspection workflow).

## Evaluation (hold-out consumers, window scores averaged)

Reported in `reports/metrics.json`; headline numbers also in the live app
header. Metrics: ROC-AUC, PR-AUC, precision/recall at both thresholds,
precision@top-1% / top-5% (inspection-budget view), lift@top-1%.

**Failure modes** known:
- Honest consumers with genuinely vacated/seasonal premises can look
  "theft-like" (long zero runs) — false positives.
- Intermittent, low-magnitude theft within normal variability is hard to
  separate — false negatives.
- Concept drift: theft patterns evolve (e.g., after inspector visits);
  retraining cadence recommended (quarterly, or on PSI drift alerts).

## Deployment

- `models/model.onnx` — float32 input `[N, 22]`, outputs probabilities
  (`zipmap=False`, `nocl=True`). Verified: max |p_onnx − p_sklearn| < 1e-3
  on hold-out windows and on the committed fixture (pytest gates this).
- In-browser inference via onnxruntime-web (WASM, single-threaded); the TS
  feature engine is asserted equal to Python via committed goldens
  (`tests/data/parity_cases.json`).
- Scoring is **screening**: rank consumers, let humans decide. The app
  labels itself decision-support, not enforcement (see ETHICS.md).

## Citing the underlying data

- Zheng, K. et al. (2017). *Electricity Theft Detection Using Generative
  Adversarial Networks*. arXiv:1710.09288. (dataset release paper family)
- Buzau, M. M., Shapour, A., Aguado, J. A. (2019). *Hybrid Deep Neural
  Networks for Detection of Non-Technical Losses in Electricity Smart
  Meters*. IEEE Transactions on Smart Grid 10(3).
