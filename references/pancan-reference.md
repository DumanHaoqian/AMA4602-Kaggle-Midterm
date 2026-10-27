# Historical PANCAN rehearsal, not the 2026 task contract

Use these facts only when the supplied files match the historical exercise. Do not assume this year's task has the same target, metric, classes, columns or report requirements.

## Data supplied in the historical exercise

Inspection of the local uploaded data.csv and labels.csv found 801 samples. The raw data CSV has 20,533 columns: one ID column named Unnamed: 0, Edema and 20,531 anonymous gene_* features. The 20,532 predictors in the assignment description include Edema. Labels are in the separate Class column and must be joined by sample ID.

The five class counts are BRCA 300, KIRC 146, LUAD 141, PRAD 136 and COAD 78. There are 10 missing gene-expression cells and 267 all-zero constant genes. Zero expression is not automatically missing. Edema has values 0/1/2 and no missing entries in these supplied files. These are observed file properties, not universal TCGA metadata.

Use the historical config only as a template; provide paths to the actual files. Its gene-only branch explicitly excludes Edema. If the current task permits Edema and it is available at prediction time, compare a gene-only baseline and a clearly specified additional-clinical-feature branch. Check its meaning and measurement time; do not declare it leakage just because it is a clinical feature.

## What a strong analysis changes

- Keep partially observed patients using fold-fitted imputation instead of deleting all rows with any missing gene.
- Fit supervised feature selection and PCA inside every training fold.
- Keep full-family exploratory ANOVA/BH separate from feature selection used in predictive validation.
- For genuine unsupervised evidence, choose/transform features without class labels and only then compare clusters with labels.
- Report PC importance as PC importance unless a justified mapping back through loadings/scaling is performed.
- Treat gene_9175 and similar identifiers as anonymous variables until an actual mapping is provided.
- Derive every result and caption from executed outputs. Do not mix a template's accuracies with a different notebook run.
- Produce an explicit prediction CSV path and submission audit for the Kaggle version; last year's figures do not establish submission correctness.

## Prior clean rehearsal findings

An earlier run on the supplied 801 labelled rows used a fixed shuffled five-fold stratified split (seed 42) with all preprocessing fitted in training folds and Edema excluded. ANOVA top 1000 + scaled L2 logistic made one OOF error; the fixed linear-SVM and PCA-logistic candidates made no OOF errors on that split. This is a retrospective rehearsal result, not a guarantee for a new split, private test or this year's exam.

A separate label-independent PCA/KMeans analysis gave less-than-perfect agreement with labels. This is consistent with different objectives: supervised classification can be excellent without a five-cluster unsupervised partition matching every tumour type. Never claim one result invalidates the other without examining method differences.

The historical reference is intentionally not a scoring assertion in code. Recompute results from the supplied data, record the current config/split/run, and report new measurements. Do not hard-code expected perfect scores or a fixed number of significant genes into tests.

## Modelling priorities

For roughly 800 rows and 20,000 numeric features, start with a regularized linear classifier, small supervised feature-selection branch and a PCA branch. A dense 20,531-by-20,531 float64 matrix alone requires about 3.37 GB before algorithm overhead, so avoid full covariance/correlation construction. Standardize in folds, cap component counts by fold shape and limit BLAS parallelism.

For the five-class association question, ordinary one-way ANOVA is an exploratory option with assumptions, followed by BH across the complete prespecified test family. It is not a raw-count RNA-seq differential-expression pipeline. Distinguish statistical significance, effect size, predictive utility and biological validation.
