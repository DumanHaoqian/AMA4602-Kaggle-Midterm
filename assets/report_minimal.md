---
# Prediction-only report source for any evaluate_models.py or leaderboard.py run
# (classification or regression, no structure analysis needed).
# Copy it into the workspace (e.g. <workspace>/reports/report.md), then:
#   - take author and student_id from the current user/config (or explicit workspace identity rules);
#   - point config/run at the current task's files (relative to the copied file);
#   - replace every REPLACE_ marker (the build refuses while one remains) and rewrite
#     the prose for the actual task (see references/academic-writing.md);
#   - keep numbers as {{facts}}; list them with build_reports.py --list-facts.
# Optional facts: classification runs add n_classes, class_list, class_counts, oof_correct,
# oof_total, oof_errors; leaderboard runs add n_trials, nested_cv_blend, nested_cv_single,
# blend_decision, adversarial_auc, adversarial_shift. Add `::: table class_recall` for
# classification. Remove a sentence rather than typing a number by hand.
title: REPLACE_WITH_TITLE
author: REPLACE_WITH_FULL_NAME
student_id: REPLACE_WITH_STUDENT_ID
affiliation: AMA4602 Midterm Project
config: ../config.json
run: ../artifacts/REPLACE_WITH_RUN_DIR
out_root: .
slug: report
---

# Abstract

We predict REPLACE_WITH_TARGET_DESCRIPTION from {{n_features}} features for {{n_samples}} labelled samples and evaluate every candidate pipeline under the official metric, {{metric}} ({{metric_direction}}). All data-dependent preprocessing is refitted inside each training fold, so no validation sample influences the model that predicts it. We compare {{n_candidates}} pipelines on identical {{cv_design}} splits. The selected pipeline, {{selected_model}}, reaches a selection-CV {{metric_lower}} of {{selected_score}}, against a baseline of {{baseline_score}}.

# Introduction {#sec:intro}

REPLACE_WITH_TWO_SENTENCES_ON_THE_TASK_AND_WHY_IT_IS_HARD. The question for this report is how well the target can be predicted for unseen samples, estimated without leakage from preprocessing or model choice.

- We freeze one {{cv_design}} split before fitting any candidate and evaluate every pipeline on exactly those folds.
- We compare {{n_candidates}} pipelines and a baseline under the official metric and select by the mean fold score.
- We produce test-set predictions by refitting the selected pipeline on all {{n_samples}} labelled samples.

# Data {#sec:data}

Each sample has {{n_features}} predictors; the feature matrix contains {{n_missing}} missing cells, which are imputed inside each training fold, and {{n_constant}} constant features. Excluded columns: {{excluded_columns}}. @tab:data_summary summarises the inputs.

::: table data_summary

# Methods {#sec:methods}

The validation design is {{cv_design}} cross-validation with random seed {{seed}}. Within each fold, the training samples alone determine imputation values, encodings, scaling and any feature selection; the validation samples are only transformed and predicted. The candidates are {{candidate_list}}. The pipeline with the best mean fold {{metric_lower}} is selected. Because the same folds are used to choose the pipeline, its score is a selection-CV estimate and is optimistic for the selected model. Models use scikit-learn {{sklearn_version}} [@pedregosa2011].

# Results {#sec:results}

@fig:models compares the candidates on the frozen folds: {{best_model}} attains the best mean {{metric_lower}} ({{best_score}}), and the selected pipeline is {{selected_model}} with a fold standard deviation of {{selected_sd}}. @tab:cv_results lists the fold-level results, and @fig:confusion shows the out-of-fold predictions of the selected pipeline.

::: figure models
::: table cv_results
::: figure confusion

# Discussion and limitations {#sec:discussion}

The selected pipeline improves clearly on the baseline under a protocol that keeps every learned step inside the training folds. Its score is a selection-CV estimate: it was used to choose the model and is therefore optimistic, and the fold standard deviation describes variation across these folds rather than a confidence interval. The hidden test set may differ from the training sample; a public leaderboard score covers only part of that test set.

# Conclusion {#sec:conclusion}

{{selected_model}} reaches a selection-CV {{metric_lower}} of {{selected_score}} ({{cv_design}}), against a baseline of {{baseline_score}}. The test predictions come from this pipeline refitted on all labelled samples; an independent test set is required to turn the selection-CV score into a generalisation estimate.

# References

[pedregosa2011] F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, et al. Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12:2825–2830, 2011.
