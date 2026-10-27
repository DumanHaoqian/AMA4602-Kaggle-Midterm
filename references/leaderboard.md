# Leaderboard mode: maximise the hidden-test score without fooling yourself

## Contents

1. When to switch to leaderboard mode
2. What scripts/leaderboard.py does
3. Running it and reading the output
4. Submission discipline and the submission log
5. When CV and the leaderboard disagree (worked example: Don't Overfit! II)
6. Manual techniques from winning solutions, and when they apply
7. What not to do in 110 minutes

## 1. When to switch to leaderboard mode

Use it when the grade or ranking depends on the score of predictions for a hidden test set (Kaggle public/private leaderboard). Do not use it when the brief rewards analysis, explanation or a report; there, evaluate_models.py and the statistics scripts are the right tools and a simpler, interpretable model is worth more than the last 0.002 of AUC.

Order of work stays the same as in SKILL.md: task contract, data audit, frozen validation, then a valid first submission from evaluate_models.py (or from a quick leaderboard run with a small budget) before any long search.

## 2. What scripts/leaderboard.py does

All learned steps stay inside training folds; the public leaderboard is never used for selection.

| Step | Behaviour |
|---|---|
| Adversarial validation | Trains a LightGBM train-vs-test classifier on features only. AUC ≈ 0.5: no shift. 0.6–0.75: moderate. > 0.75: strong; inspect the top shifting features (IDs, timestamps, leakage) and consider validating on the most test-like training rows. Recorded in evidence.json. |
| Preprocessing | Mixed types handled per fold: numeric median imputation (+ scaling for linear models); low-cardinality categories one-hot (linear) or ordinal (trees); categories with > 30 levels use sklearn TargetEncoder (internal cross-fitting) for linear models; CatBoost uses native categorical features. Optional `--transductive` standardises numeric features on train + test features (no labels), only if the rules allow using test features; helpful mainly when n is small. |
| Model families | Classification: L1 / L2 / elastic-net logistic, shrinkage LDA, RBF SVM, random forest, extra trees, LightGBM (GBDT or DART), XGBoost, CatBoost. Regression: ridge, lasso, elastic net, RBF SVR, random forest, extra trees, LightGBM, XGBoost, CatBoost. Default routing: n < 2000 favours regularised linear models plus a few trees; larger n favours boosted trees. |
| Baseline | DummyClassifier(prior) or DummyRegressor(mean) on the same folds, reported as `dummy`; never selected or blended. |
| Search | Round-robin: one sensible default per family first, then random configurations within bounded ranges, until the budget (minus a 25% reserve for blending and refit) runs out or `--max-trials` is reached. A family's cost is learned per tree/iteration, so a 900-tree configuration is budgeted at roughly six times a 150-tree one (DART three times more). |
| Budget | Checked before and after every fold. When the projected finish of a trial would enter the reserve, the trial is aborted (`aborted_budget` in trials, partial folds discarded). The final refit lowers the seed count, then falls back from the blend to the best single configuration, if its estimated time exceeds what is left. A single fit already running is not interrupted. |
| Validation | Repeated stratified/plain k-fold (5 repeats if n < 1000, 3 if n < 10,000, else 1) to reduce selection noise; group/time splits use the same safeguards as evaluate_models with one repeat. Every trial reports mean and repeat-level SE. |
| Blend | Greedy forward selection with replacement (Caruana) over the best `--blend-pool` configurations, rank-averaged for AUC and probability-averaged otherwise. It is chosen by nested CV: members are selected on k−1 folds and scored on the held fold, against the single best configuration selected the same way. The blend is used only if it wins. |
| Final refit | Selected configurations refitted on all training rows; stochastic families averaged over `--seeds` seeds. |
| SVM ranking | For binary AUC, RBF SVM uses its decision function, because SVC's internal Platt scaling can invert the order on small data. |

Outputs in `<output_dir>/lb_<UTC>_<id>/`: evidence.json (trials, adversarial validation, blend decision, selection, final_refit_plan, budget_exceeded), trials.csv, fold_indices.json, oof_<family>.csv for the baseline and the best configuration of each family (plus the blend when it is selected), and test_predictions.csv. Each OOF row averages that sample's predictions over the repeats; figures and reports say "k-fold × r repeats" and state this averaging. The blend appears among the experiments only when the nested-CV rule selects it, because its OOF reuses members chosen on all rows and would otherwise outrank the selected model in figures. These are compatible with validate_submission.py, plot_results.py and build_reports.py.

## 3. Running it and reading the output

~~~sh
$PY -B $SK/leaderboard.py --config config.json --budget-seconds 600
$PY -B $SK/leaderboard.py --config config.json --families logreg_l1,logreg_enet,lightgbm --max-trials 20
$PY -B $SK/validate_submission.py --config config.json --predictions RUN/test_predictions.csv --evidence RUN/evidence.json --output-dir final/vN
~~~

Read, in this order: adversarial_validation, the per-family best in trials.csv, blend.nested_cv_blend versus blend.nested_cv_single, and selection. The best trial mean is optimistic (winner's curse over many configurations); differences smaller than about two repeat-SEs are noise. Prefer the simpler or more regularised configuration when scores tie.

Budget guide for a 110-minute exam: a 5–10 minute leaderboard run after the first valid submission is usually enough; give it at most 25 minutes in total. The script checks its budget between fits, so leave margin.

## 4. Submission discipline and the submission log

- Keep a submissions_log.csv in the workspace with: time, submission ref, file name, file path and size, run directory, selected model, local CV (with its kind), public score, private score (if shown), note. Record every submission, including failed ones.
- Submit to learn, not to search: each submission should test a hypothesis (new family, blend versus single, shift handling). Never tune hyperparameters against the public score.
- Public leaderboards are often computed on a small share of the test set; a public gain smaller than the CV SE is noise.
- Final selection (when the platform asks for one or two final submissions): one with the best local CV, one that is the most robust (blend or most regularised model), not the best public score.
- Reread the competition's daily limit before submitting and keep the last validated file intact.

## 5. When CV and the leaderboard disagree

A large gap is a bug or a data problem until proven otherwise. Audit in this order and record each check:

1. Submission format: ID alignment against the sample submission, the probability column of the positive class, row order. validate_submission.py checks these.
2. Independent reference: refit the simplest reasonable model by hand outside the pipeline, compare its test predictions with the pipeline's (correlation), and submit it.
3. Single-feature sanity check: submit the strongest single feature (or its negation) as the score.
4. Adversarial validation: a high AUC explains a gap through distribution shift; an AUC near 0.5 means train and test features look alike.
5. Only then question the validation design itself (groups, time, duplicates).

Worked example, Don't Overfit! II (27 Oct 2026; 250 training rows, 300 anonymous numeric features, binary target, AUC, 19,750 test rows):

| Submission | Local CV (repeated 5×5) | Public | Private |
|---|---|---|---|
| leaderboard.py blend of L1 / elastic-net logistic | 0.755 (nested-CV blend) | 0.505 | 0.501 |
| Plain L1 logistic, C = 0.1, balanced (independent reference) | ≈ 0.75 | 0.507 | 0.501 |
| Raw feature 33 as the score (historically the strongest feature) | – | 0.472 | 0.496 |

The pipeline and the independent reference agreed (prediction correlation 0.96) and IDs were aligned, yet every submission scored ≈ 0.5. Adversarial validation gave AUC 0.525 (no feature shift). The strongest training feature is now 127, not 33, and the data files are dated 2019-12-11, after the competition closed. Conclusion: the data were re-uploaded and no longer match the scoring solution, so late submissions to this competition cannot measure model quality. The right response was to stop submitting and report this, not to tune further. Use a competition whose leaderboard still responds (check with a quick reference submission) to evaluate leaderboard techniques.

## 6. Manual techniques from winning solutions

Distilled from wenmin-wu/ds-skills (tabular domain). Apply only when the condition holds and time allows; each must be validated on the same frozen folds.

| Technique | Use when | How, briefly | Risk |
|---|---|---|---|
| Null-importance feature selection | Wide, noisy tables where trees overfit | Train LightGBM on real and ~50 shuffled targets; keep features whose gain beats the null distribution; pick the cutoff by CV inside folds | Selecting on all rows leaks; do it inside folds |
| Pseudo-labelling | Large unlabeled test set, strong confident model, rules allow it | Add test rows with predictions > 0.995 or < 0.005, retrain, keep only if OOF on the original labels improves | Confirmation bias; never for small or shifted data |
| Threshold optimisation | F1, kappa or accuracy on imbalanced or ordinal targets | Optimise class thresholds (Nelder–Mead) on OOF predictions, apply to test | Overfits if tuned on in-fold predictions |
| Probability clipping | Log loss | Clip to [ε, 1−ε]; only tune harder clipping on OOF | Confident wrong predictions are expensive |
| Log-odds averaging | Averaging boosted-tree probabilities across folds or seeds | Average logits, then apply the sigmoid | Not needed for AUC (rank average instead) |
| OOF meta-features / stacking | Several diverse strong models and time left | Use OOF predictions as features for a simple meta-model, evaluated with nested CV | In-fold predictions leak; keep the meta-model simple |
| Frequency encoding | High-cardinality categories for trees | Replace categories by their frequency (train + test counts allowed if rules permit) | Exclude synthetic test rows if present |
| Synthetic-row check | Test sets padded with generated rows | Rows with no value that is unique in its column are likely synthetic; exclude them from count features | Specific to some competitions |

Not adopted: TabPFN (needs a new package during the exam), deep tabular networks, GPU pipelines, competition-specific feature engineering (games, recommendation, molecules, geometry).

## 7. What not to do in 110 minutes

Long Optuna studies, large stacking towers, many-seed sweeps chosen on the public score, unbounded polling, and any change of the validation split after seeing leaderboard results. A valid, well-validated submission early beats a perfect one after the deadline.
