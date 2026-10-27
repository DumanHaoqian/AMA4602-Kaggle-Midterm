---
# Example report source written for the historical PANCAN rehearsal.
# Copy it into the workspace (e.g. <workspace>/reports/report.md), then:
#   - take author and student_id from the current user/config (or explicit workspace identity rules);
#   - point config/run/structure at the current task's files (relative to the copied file);
#   - rewrite the title and every paragraph for the actual task (see references/academic-writing.md);
#   - keep numbers as {{facts}}; list them with build_reports.py --list-facts.
title: Leakage-Controlled Classification of Five Tumour Types from High-Dimensional Gene Expression
author: REPLACE_WITH_FULL_NAME
student_id: REPLACE_WITH_STUDENT_ID
affiliation: AMA4602 Midterm Project
config: ../config.json
run: ../artifacts/REPLACE_WITH_RUN_DIR
structure: ../artifacts/REPLACE_WITH_STRUCTURE_DIR
out_root: .
slug: report
---

# Abstract

We classify {{n_samples}} tumour samples into {{n_classes}} cancer types from {{n_features}} gene-expression features and test how much of this class structure survives a strictly leakage-controlled evaluation. The setting is difficult because the features far outnumber the samples, the classes are imbalanced, and any feature selection fitted outside the validation folds inflates accuracy. We compare regularised linear pipelines in which imputation, variance filtering, ANOVA feature selection, standardisation and principal component analysis are refitted inside every training fold, and we separate this predictive evaluation from an exploratory association screen and a label-free clustering analysis. On identical {{n_folds}}-fold stratified splits, {{best_model}} reach a selection-CV {{metric_lower}} of {{best_score}}, against a majority-class baseline of {{baseline_score}}. The selected pipeline predicts {{oof_correct}} of {{oof_total}} samples correctly out of fold, while unsupervised clustering recovers the classes only partly (ARI {{ari}}), so the classes are separable by a supervised linear rule even where they overlap in the leading principal components.

# Introduction {#sec:intro}

Gene-expression profiles carry strong tissue-of-origin signals, which makes tumour-type classification a natural benchmark for high-dimensional supervised learning. With {{n_samples}} samples and {{n_features}} features, almost any classifier can fit the training data perfectly; the question is whether the estimated accuracy reflects generalisation or leakage from preprocessing fitted on the full data set.

The main design decision of this study is therefore procedural. Every step that learns from the data, including the ANOVA-based selection of features, is fitted on the training part of each fold only and applied unchanged to the held-out part. Association tests and clustering on the full data are reported separately and labelled exploratory, because their results would otherwise leak into the model comparison.

Our contributions are:

- A comparison of {{n_candidates}} regularised linear pipelines on one frozen set of {{n_folds}}-fold stratified splits, with all supervised preprocessing refitted inside each fold.
- An out-of-fold error analysis of the selected pipeline, reported per class with its denominator.
- An exploratory one-way ANOVA screen over all {{n_tested}} non-constant features with Benjamini–Hochberg control of the false discovery rate, summarised by effect size.
- A label-free PCA and k-means analysis that measures how much of the class structure is visible without supervision.

# Data {#sec:data}

The data consist of {{n_samples}} labelled samples, each described by {{n_features}} anonymous expression features (`gene_0` to `gene_20530`); the labels were joined to the features by sample identifier rather than by row position. The classes are imbalanced: {{class_counts}}, from {{largest_class_n}} samples in {{largest_class}} to {{smallest_class_n}} in {{smallest_class}}. The feature matrix contains {{n_missing}} missing cells, which we impute rather than deleting the affected samples, and {{n_constant}} features that are constant across all samples. The clinical column {{excluded_columns}} was excluded so that the analysis uses expression features only. @tab:data_summary summarises the inputs.

::: table data_summary

# Methods {#sec:methods}

## Validation protocol

We fixed one stratified {{n_folds}}-fold split (random seed {{seed}}) before fitting any candidate and stored its indices, so that every pipeline is evaluated on exactly the same training and validation samples. Within each fold, the training samples alone determine the median imputation values, the variance filter, the ANOVA feature ranking, the standardisation parameters and the principal components; the validation samples are only transformed and predicted. We report the mean and the spread of the fold scores and the {{metric_lower}} of the pooled out-of-fold predictions. Because the same folds are used to choose the pipeline, these scores are selection-CV estimates and are optimistic for the selected model.

## Candidate pipelines

All pipelines start with median imputation and removal of zero-variance features. The candidates then differ as follows:

- **ANOVA top-1000 + logistic.** The 1,000 features with the largest ANOVA F statistic in the training fold, standardised, followed by L2-regularised multinomial logistic regression ($C = 1$).
- **ANOVA top-1000 + linear SVM.** The same selector and standardisation, followed by a linear support vector machine ($C = 0.1$).
- **PCA-50 + logistic.** Standardisation of all non-constant features, randomised PCA to at most 50 components, followed by L2-regularised logistic regression.

A majority-class classifier serves as the baseline. The pipeline with the highest mean fold {{metric_lower}} is selected; ties are broken by the declared candidate order, and we report ties explicitly.

## Exploratory analyses

For the association screen we test each non-constant feature with a one-way ANOVA across the {{n_classes}} classes on all labelled samples and control the false discovery rate over the complete family of {{n_tested}} tests with the Benjamini–Hochberg procedure [@benjamini1995]. We report the effect size

$$\eta^2 = \frac{SS_{\text{between}}}{SS_{\text{total}}},$$

the share of a feature's variance explained by class membership. For the structure analysis we standardise the features, project them onto {{pca_components}} principal components [@halko2011] and run k-means with $k = {{k_clusters}}$, fixed in advance to the number of classes. Labels are used only after fitting, to compute the adjusted Rand index (ARI) [@hubert1985] and the normalised mutual information (NMI). All models and tests use scikit-learn {{sklearn_version}} [@pedregosa2011] and SciPy.

# Results {#sec:results}

## Model selection

@fig:models shows that the two leading pipelines are indistinguishable on these folds: {{best_model}} both reach a mean {{metric_lower}} of {{best_score}}, and ANOVA top-1000 + logistic follows at {{score_select_logistic}}. All three are far above the majority-class baseline of {{baseline_score}}. Following the declared tie-break, we select {{selected_model}}. @tab:cv_results lists the fold-level results.

::: figure models
::: table cv_results

## Out-of-fold errors

The selected pipeline classifies {{oof_correct}} of {{oof_total}} samples correctly when each sample is predicted by a model that never saw it during training (@fig:confusion and @tab:class_recall). Recall is therefore equal across classes, including the smallest class, {{smallest_class}}. With only {{n_folds}} folds and no independent test set, this result shows that the classes are linearly separable in this sample under a leakage-free protocol; it does not guarantee error-free prediction on new data.

::: figure confusion
::: table class_recall

## Class-associated features

The class signal is spread across most of the feature space: {{n_significant}} of {{n_tested}} tested features ({{significant_share}}) differ across classes at BH q < {{alpha}} (@fig:associations). Statistical significance is nearly universal at this sample size, so effect size is the more informative quantity. The strongest feature, {{top_feature}}, has $\eta^2 = {{top_eta2}}$, and the leading features in @tab:top_features each separate a single class from the rest, as the class-mean profiles in @fig:associations and the raw distributions in @fig:distributions show.

::: figure associations
::: table top_features
::: figure distributions

## Label-free structure

Without labels, PCA followed by k-means recovers much but not all of the class structure (ARI {{ari}}, NMI {{nmi}}; @fig:structure). The leading two components explain {{pc1_variance}} and {{pc2_variance}} of the variance; KIRC forms a separate group, whereas BRCA, LUAD and PRAD overlap in this projection and one k-means cluster mixes LUAD with part of BRCA. The contrast with the supervised result is expected: a linear classifier can use directions that separate the classes even when those directions carry little of the total variance.

::: figure structure

# Discussion and limitations {#sec:discussion}

The main finding is that a simple, leakage-controlled linear pipeline separates the five tumour types in this sample without out-of-fold errors, and that two quite different representations, a supervised top-1000 feature subset and an unsupervised 50-component projection, reach the same result. This agreement suggests that the separability does not depend on one particular preprocessing choice.

Several limitations bound this conclusion. First, the scores are selection-CV estimates on {{n_samples}} samples from one source; they are optimistic for the selected pipeline and say nothing about samples from other laboratories or sequencing batches. Second, with {{n_folds}} folds the fold standard deviation is a coarse stability indicator rather than a confidence interval. Third, the ANOVA screen uses all labelled samples and is exploratory; its feature ranking was not used to choose the predictive features, and its omnibus test does not identify which class differs. Fourth, the features are anonymous, so we draw no biological conclusions from individual features. Finally, k was fixed to the number of classes, which favours agreement between clusters and labels.

# Conclusion {#sec:conclusion}

Under a protocol that refits every data-dependent step inside the validation folds, {{selected_model}} classifies all {{oof_total}} samples correctly out of fold ({{n_folds}}-fold selection CV, {{metric_lower}} {{selected_score}}), against a majority baseline of {{baseline_score}}. Class differences are large and widespread in the expression data, yet only partly visible without supervision. An independent test set would be required to turn this selection-CV result into a generalisation estimate.

# Appendix

## Expression patterns of the leading features

@fig:heatmap shows the {{n_samples}} samples, grouped by their reference label, for the features with the largest ANOVA F statistic. The block structure partly reflects this label-based ordering and is descriptive only.

::: figure heatmap

```python
# Fold sizes and scores of the selected pipeline on the frozen split (notebook only).
ev = run["evidence"]
pd.DataFrame([{"fold": f["fold"], "n_train": f["n_train"], "n_validation": f["n_validation"], "accuracy": f["score"]}
              for f in next(e for e in ev["experiments"] if e["name"] == ev["selection"]["model"])["folds"]])
```

# References

[pedregosa2011] F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, et al. Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12:2825–2830, 2011.
[benjamini1995] Y. Benjamini and Y. Hochberg. Controlling the false discovery rate: a practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society: Series B*, 57(1):289–300, 1995.
[halko2011] N. Halko, P.-G. Martinsson and J. A. Tropp. Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions. *SIAM Review*, 53(2):217–288, 2011.
[hubert1985] L. Hubert and P. Arabie. Comparing partitions. *Journal of Classification*, 2(1):193–218, 1985.
