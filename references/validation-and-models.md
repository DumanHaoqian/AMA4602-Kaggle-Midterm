# Validation and model routing

## Split reflects the prediction question

| Data mechanism | Initial split | Check |
|---|---|---|
| Independent classification observations | StratifiedKFold, shuffled with a saved seed | Smallest class supports the fold count |
| Independent regression observations | KFold, shuffled with a saved seed | Target range and extreme observations |
| Repeat patients/sites/devices | GroupKFold or StratifiedGroupKFold | No group crosses train/validation |
| Future observations | TimeSeriesSplit after explicit time ordering | Train precedes validation; gap if outcome windows overlap |
| Group and time simultaneously | Explicit custom indices/adapter | Neither a generic group nor time splitter alone guarantees both constraints |

Do not infer IID from a missing group column. Inspect metadata and repeated observations. Duplicate IDs are not automatically duplicate people; choose the correct observation key and entity key. Do not “fix” duplicate patients by dropping them without a scientific reason.

The bundled script supports the first four modes. For a custom split, implement and audit the adapter before fitting; never silently fall back to random folds. Time validation may leave an initial training block without OOF predictions. Record covered rows, temporal range and metric denominator.

Freeze the actual index arrays and row IDs, not only random_state. Class stratification does not protect against batch/site leakage. All candidates must use the same folds and target semantics. If comparing separate script runs, compare the saved row IDs and actual split arrays.

## Pipeline boundary

Fit imputation, learned variance filtering, scaling, ANOVA/MI selection and PCA on fold-training rows only. Feature-selection labels must also be fold-training labels. Apply only transform/predict to validation and test. Do the same for target encoding, probability calibration, learned thresholds and oversampling.

For exploratory figures or hypothesis testing, full labelled training data are available. Their results must not become a fixed supervised feature list before CV. A predeclared external feature set may be used if permitted and independent of this task's labels; document that source.

Missingness indicators can be tried if plausible. Preserve actual zero expression. Do not automatically delete all rows containing a missing predictor. Drop all-missing/constant columns within each fold, with consistent handling at prediction.

## Route by dimensionality and output

| Data | Fast starting point | One useful challenger |
|---|---|---|
| Numeric, small n and p >> n, classification | Median → variance filter → ANOVA top ≤1000 → scaler → L2 logistic | Same selector + linear SVM, or scaler → randomized PCA ≤50 → logistic |
| Numeric, small n and p >> n, regression | Median → variance filter → scaler → Ridge | PCA + Ridge; optional training-fold f_regression selector |
| Ordinary low/moderate-dimensional tabular | Regularized linear model / simple tree baseline | Existing installed histogram boosting, LightGBM or CatBoost if appropriate |
| Mixed numeric/categorical | ColumnTransformer with separate numeric/categorical imputers and train-fitted one-hot encoding | Suitable categorical tree model if already available |
| Text/image/sequences | Task-specific small adapter and metric-correct baseline | Only upgrade within remaining time |

Nonlinear challengers in evaluate_models.py (`--models all`, fixed settings, same folds): random forest (500 trees), ANOVA top-1000 + RBF SVM (C = 1, gamma = scale, Platt probabilities), ANOVA top-1000 + MLP (one hidden layer of 64, alpha 1e-3, early stopping on a 10% split of the training fold); regression random forest, RBF SVR, MLP. Report them as a check of whether a linear rule leaves signal unused, together with the train-validation gap; do not tune them on the same folds and then claim the tuned score.

Holdout: with `"holdout": {"fraction": 0.2}` the holdout is drawn before the folds, respecting stratification, groups or time order, and is scored once with the selected pipeline. It is the only estimate in the run not used for selection. `--folds` reuse is refused together with a holdout.

The supplied model runner deliberately supports numeric classification/regression, not universal AutoML. If adapting to mixed data, do not silently coerce strings to NaN or delete categorical columns. Use handle_unknown="ignore" or equivalent train-fitted encoding; avoid dense one-hot expansion if cardinality is large.

Cap SelectKBest by actual post-filter feature count. Cap PCA by both training-fold rows and available features, with a conservative margin for centered data; avoid a fixed 50 when a training fold has only 30 rows. With 50 or fewer features, PCA keeps every component, which is only a rotation and leaves an L2 model on standardised features unchanged; evaluate_models.py records the PCA candidates as skipped_redundant instead of reporting a fake tie. Randomized PCA avoids forming a huge p-by-p covariance matrix. Thread limits prevent every fold/model from oversubscribing the laptop.

Do not treat historical PANCAN near-perfect scores as evidence that every high-dimensional problem is easy. If performance is weak, inspect the task/split and a small set of meaningful candidates; do not spend the entire budget tuning one seed.

## Metric contract

Config `metric` names implemented by evaluate_models.py and leaderboard.py:

| Official metric | `metric` | Needs |
|---|---|---|
| Accuracy / balanced accuracy | accuracy / balanced_accuracy | labels |
| Binary F1 of one class | f1 | binary, positive_class |
| Macro / weighted F1 | macro_f1 / weighted_f1 | labels |
| Matthews correlation | mcc | labels |
| Quadratic weighted kappa | qwk | numeric ordinal labels (sorted order = rating order) |
| Log loss | log_loss | probabilities |
| Binary ROC AUC | roc_auc | binary, positive_class, probabilities |
| Multiclass ROC AUC | roc_auc_ovr (macro), roc_auc_ovr_weighted, roc_auc_ovo (macro) | probabilities; every validation fold holds every class |
| RMSE / MAE / R² | rmse / mae / r2 | continuous |
| RMSLE | rmsle | nonnegative target; target_transform defaults to log1p |
| MAPE | mape | continuous (sklearn fraction, not percent) |

- Accuracy/macro F1/balanced accuracy: preserve original class labels. Use macro F1 when officially required; do not optimize a different metric because it looks better.
- Binary F1, MCC and kappa: labels are the arg-max of class probabilities. Threshold tuning for F1 is not automatic; if used, tune on OOF predictions only (see leaderboard.md).
- Log loss: use class probabilities, with an explicit column-to-class map and all required classes. No decision scores disguised as probabilities.
- Binary ROC AUC: state the positive class and use that score/probability; label predictions discard ranking information. positive_class 1 and 1.0 match each other; nothing else is guessed.
- Multiclass AUC: confirm OVR/OVO and macro/weighted settings from the competition and choose the matching name above. Any other variant needs an explicit evaluator.
- Regression RMSE/MAE: preserve target units. With `"target_transform": "log1p"` the model fits log1p(y) and predictions are back-transformed with expm1 (negatives projected to 0) before scoring on the original scale.
- R²: negative scores are possible. Higher is better; RMSE, MAE, RMSLE, MAPE and log loss are lower-is-better.
- Competition-specific weighted or grouped metrics: reproduce the definition and verify a small hand-computable fixture before model comparisons.

OOF means each reported prediction came from a model that did not train on that row. Save row ID, fold, true target, label/continuous output and probability columns where applicable. Write class_order into evidence; align probability columns using model.classes_.

Averaging fold metrics and computing one metric over pooled OOF are different operations, especially for nonlinear metrics or unequal folds. Label both. Use a consistent selection rule for all candidates; report exactly which one was used. Fold SD describes variation across these folds and is not a confidence interval.

Hyperparameter tuning/model choice makes the chosen CV score optimistic as an estimate of future performance. Call it selection CV. An untouched holdout or nested design is a stronger evaluation when sample size and time support it; it is not a compulsory expensive detour during this exam.

## Improvements with a budget

Begin with a dummy baseline and one real pipeline. Make a valid early prediction file before broad comparison. A few model/regularization/feature-count choices are sufficient initial experiments; log how many were tried. Stop when the differences are smaller than the uncertainty or error count permits judging.

Keep converged simpler models when ties remain. Repeated 100% results warrant audits, not artificial extra tuning. Inspect the actual misclassified observations and minority-class performance without repeatedly redesigning the model around the same holdout.

If using an ensemble, validate it on aligned OOF predictions. Average compatible probability predictions for probability tasks; require class alignment. Fit any meta-model using OOF base predictions and evaluate it with appropriate additional separation, not the same data it fitted. Do not default to complex stacking, pseudo-labeling or model distillation in 110 minutes.

## Reproducibility and failure handling

Every run has a unique directory. Keep full parameters, library versions, config, input paths and sizes, fold indices and row IDs, timing, warnings and exception details. Preserve an earlier validated output before starting a new candidate.

The runner checks budget between fits. A fit already in progress may overrun. Pilot one fit to estimate cost; do not launch expensive work inside the finalization buffer. A final-fit failure must be reported, not replaced with unlabelled stale predictions.

A clean environment matters more than the newest version. Use the installed versions that passed the rehearsal. Do not install or upgrade TensorFlow, CUDA, a large notebook research stack or scikit-learn merely because an upstream skill describes them.

## If a neural network is actually used

Treat MLP as an optional candidate for an appropriate dataset and budget. Record hidden-layer sizes, activation, optimizer, learning rate, batch size, epoch/max-iteration budget, regularization, early-stopping/validation design and seed. Justify them against sample size, feature dimension and runtime. Keep preprocessing within each fold and inspect convergence. Do not paste the historical assignment's example settings into a report unless those exact settings were executed.
