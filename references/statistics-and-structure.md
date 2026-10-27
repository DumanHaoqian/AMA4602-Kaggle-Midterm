# Statistics and structure: keep the questions separate

## Contents

1. Analysis branches and feature scope
2. Numeric associations and full-family multiple testing
3. Expression scales, anonymous genes, and clinical variables
4. Unsupervised structure and external label evaluation
5. Configuration and outputs
6. Reporting and exam-time priorities

## 1. Analysis branches and feature scope

Use three separate branches. Share raw input files and verified sample identity, not fitted transformations.

| Branch | Role of labels | Permitted conclusion |
|---|---|---|
| Prediction | Fit only on the relevant training fold; validation labels are used for evaluation | Estimated predictive performance under the specified validation design |
| Association testing | Labels define explicitly stated groups in a declared exploratory sample | Evidence of group differences under the stated testing assumptions |
| Unsupervised exploration | Fit the representation and clustering using X only; annotate or compare labels after fitting | Descriptive structure in the chosen feature representation |

Do not import a gene ranking, imputer, scaler, or PCA fitted on all labeled data into prediction cross-validation. Fit each learned preprocessing step inside the training fold. An association analysis on all labeled data is legitimate as declared exploration; it is not independent validation of a prediction model trained on its selected genes.

Define the hypothesis family before inspecting label associations. The structure script defaults to all predictor columns returned by the shared loader, after the configured target, ID, group/time keys, and drop_columns have been removed. It does not silently select gene_* columns or remove Edema. Use structure_feature_columns for a prespecified scope or structure_exclude_columns for explicit exclusions. Record why the scope was selected. A list obtained by selecting the smallest p-values on the same data is not a prespecified full family.

The bundled structure runner supports numeric predictors and a classification target. If the declared scope contains strings or other nonnumeric predictors, it reports that the bounded runner is not applicable. Define a scientifically justified numeric scope or use an appropriate mixed-data method; do not manufacture numeric distances by arbitrary category numbering. Prediction can still use the separate mixed-type modeling workflow.

## 2. Numeric associations and full-family multiple testing

### The question and test

For each eligible feature, ordinary one-way ANOVA tests whether all class means are equal. Its omnibus alternative is that at least one mean differs. It does not identify the differing class, establish a class-specific biomarker, or test a particular pairwise fold change.

Check the design before using ordinary ANOVA:

- Observations must be independent for its ordinary inference. Repeated patients, clustered sampling, or dependence in time may require paired, mixed-effects, blocked, or otherwise appropriate procedures. The runner declines ordinary ANOVA when a configured group has repeated observations.
- The usual F inference assumes a suitable within-group error model, including common within-group variance. Inspect distributions and unequal spreads, especially with unequal class sizes. Welch-type ANOVA or other methods may be preferable when assumptions fail; the bounded runner does not silently switch tests.
- A minimum count of two observed values per feature per class is only a computational eligibility rule, not proof of adequate power or valid assumptions. Report class sizes and limitations.
- Retain genuine extremes unless a documented measurement problem justifies correction. Do not remove points merely to improve significance or prediction scores.

The script uses the supplied numeric scale. It omits missing observations separately for each feature and reports its effective sample size and per-class counts. It does not discard a whole patient because one unrelated gene is missing, and does not silently impute measurements before a significance test.

### Family construction and correction

Exclude all-missing features and globally constant observed features without consulting labels. Record columns that lack the declared minimum number of observations in any group; those tests are not computable under the specified rule. Save every exclusion and its reason.

Test every remaining feature once. Apply scipy.stats.false_discovery_control(method="bh") to the entire declared tested family. Do not first select the top 2,000 genes by a label-dependent test and then adjust only those 2,000 p-values.

For ordered p-values, Benjamini-Hochberg adjusted p-values require both the m/i factor and a reverse cumulative minimum, capped at one. A hand-written m*p/rank loop without that monotonicity step is incomplete. Use the established implementation rather than a "simplified BH" fallback. The bounded runner needs SciPy >= 1.11; prepare the environment before the exam.

If a numerical test still fails after eligibility checks, the runner retains its place in the family with p=1 for correction and flags the failure. It preserves the raw test result separately; it does not quietly reduce the number of hypotheses after seeing their statistics.

BH's formal FDR guarantee depends on valid null p-values and independence or suitable positive dependence. A successful computation does not verify those assumptions. For exploratory correlated gene-expression data, describe the method and its assumptions without claiming that the run proves all discoveries true.

### Effect sizes and numerical extremes

Report effect sizes as well as p/q values. For ordinary K-group ANOVA, eta-squared equals the between-group sum of squares divided by the total sum of squares. It can be computed from F as:

    eta_squared = F * (K - 1) / (F * (K - 1) + N_feature - K)

The script evaluates the equivalent expression stably and reports per-class means. Eta-squared is an in-sample descriptive effect size, not a cross-validated R-squared or a causal effect.

Very strong finite F statistics can produce floating-point p=0. Report a numerical-underflow note or a justified display bound, not "the null hypothesis has zero probability." Multiple genes may tie at displayed p/q=0; rank with a stated secondary criterion such as F or effect size.

Infinite F can occur when every class is internally constant but class means differ. The runner retains the computed F/p, sets eta-squared to one, and marks this degenerate case explicitly. Investigate rounding, encoding, or measurement design before interpreting it as population evidence.

### Contrasts and visualizations

If the question requires a particular class or direction, specify a pairwise or class-versus-rest contrast and its complete testing family. Do not copy one omnibus significant-gene count into five class-specific bars. For many contrasts, state whether multiplicity is controlled per contrast or across all tested gene-contrast combinations.

An ordinary volcano plot uses a signed effect such as a justified log fold change on the horizontal axis and a stated significance measure on the vertical axis. F versus minus-log10(q) is a different statistical plot and must be named accordingly. Do not create a volcano plot merely because the word appears in an example.

A gene-by-sample expression heatmap and a gene-by-gene correlation heatmap answer different questions. If samples are sorted by known labels, state that ordering; the resulting blocks do not prove unsupervised recovery.

## 3. Expression scales, anonymous genes, and clinical variables

Read the data dictionary and inspect ranges before transforming expression data. Raw RNA-seq counts, library-normalized expression, already-log-transformed measurements, and gene-wise standardized values are not interchangeable.

- Raw counts may require a count-aware differential-expression workflow with the appropriate design and normalization. Do not promise that ordinary ANOVA on raw counts is a substitute for such a workflow.
- Already transformed continuous expression may support exploratory continuous-data methods, subject to their assumptions. Do not automatically take log1p a second time.
- Differences on a verified log2 expression scale can have a log-ratio interpretation only to the extent justified by that measurement's definition and transformation. Do not infer the scale from values alone.
- A ratio of z-scores is not fold change. A difference in z-scores is a standardized difference, not an original-expression ratio.
- Anonymous names such as gene_18 do not identify a known gene symbol. Without a reliable mapping, report candidate anonymous features; do not invent gene names, pathways, literature matches, or clinical utility.
- A principal component is a weighted direction, not a single gene. PC importance cannot be relabeled as gene importance without the relevant loadings and transformations. For an exam, a separate interpretable model or an honestly named PC-level explanation is often sufficient.

Treat Edema and other clinical variables separately from genes when the scientific question is gene expression. State the encoding, whether ordinality and equal spacing are justified, and whether the measurement would be available at prediction time. Its presence alone does not prove leakage. Set an explicit feature scope for gene-only analysis; the runner flags an included Edema name but does not silently remove it. A genes-only versus genes-plus-clinical comparison is optional when relevant and affordable.

## 4. Unsupervised structure and external label evaluation

Use only X to fit the representation:

    median imputation -> zero-variance filtering -> standardization
    -> randomized PCA -> KMeans

The runner caps PCA at 50 components and further limits it by the available feature count and n-1. It fixes the random seed and uses ten KMeans initializations. These are bounded starter choices, not a claim of optimality. Check whether scaling fits the measurement units and purpose; adapt the method when a scientifically justified alternative is needed.

Do not use ANOVA, mutual information against the label, RFE using the label, or a classifier's importance to select clustering features and then describe the whole workflow as label-free discovery. Such a workflow can be useful, but call it supervised-guided exploration.

Prefer a prespecified n_clusters. If omitted, the runner uses the observed number of label classes and explicitly records that it used this known-k prior. It did not discover how many clusters exist. If choosing k or representation size from several alternatives, distinguish internal unsupervised criteria from labels used for external evaluation.

Fit first, then compute ARI and NMI and plot labels as annotations. Cluster numbers are arbitrary; a cluster labeled "2" has no intrinsic correspondence to target class "2". ARI and NMI do not require manually matching cluster IDs. A displayed cross-tabulation is not automatically a classification confusion matrix.

Silhouette is computed with Euclidean distance in the retained, unwhitened PCA-score space. For large sample counts the runner uses a fixed-seed sample capped by structure_silhouette_max_samples and records its size. A silhouette value is not prediction accuracy; scores from different representations should not be interpreted as if the geometry were unchanged.

Explained variance concerns the processed feature space. "PCA explains 90% of variance" does not mean 90% of predictive information, biological meaning, or the variance in a different set of original features. A separated two-PC scatter plot is descriptive and cannot validate a classifier.

For a label-isolation diagnostic, fix n_clusters and the random seed, permute y, and refit the unsupervised pipeline. The representation and cluster assignments should stay unchanged; post-fit ARI/NMI should generally change. The CLI permutation flag writes into a separate diagnostic directory and marks the outputs as unsuitable for exam submission.

## 5. Configuration and outputs

Use the shared exam JSON configuration. The required classifier configuration and paths are handled by exam_io.py. Optional structure settings are:

| Setting | Default | Meaning |
|---|---|---|
| structure_feature_columns | All loaded predictors | Exact, prespecified feature scope |
| structure_exclude_columns | Empty list | Explicit exclusions from that scope |
| structure_min_group_samples | 2 | Minimum observed samples per class per tested feature |
| structure_fdr_alpha | 0.05 | Threshold reported after full-family BH |
| n_clusters | Observed class count | Fixed k, with the source of prior knowledge disclosed |
| structure_pca_components | 50 | Requested count, capped at 50 and feasible rank bounds |
| structure_silhouette_max_samples | 2000 | Bound on silhouette computation |
| structure_plot | true | Small optional PNG; missing matplotlib does not trigger installation |

Run:

    python scripts/analyze_structure.py --config config.json --mode both
    python scripts/analyze_structure.py --config config.json --mode associations
    python scripts/analyze_structure.py --config config.json --mode clustering

Each invocation creates a unique output_dir/structure_TIMESTAMP_RUNID directory and writes structure_summary.json with input paths and sizes, configuration provenance, software versions, scope information, status, and the requested branch results. Read that summary for the current run; do not treat unrelated pre-existing files as fresh results.

Association files include:

- feature_scope.csv: the declared scope and explicit exclusions.
- association_family.csv: eligibility and exclusion reasons, effective counts, and per-class counts.
- associations.csv: all tested features, F, raw p, p actually used for BH, q, eta-squared, effective n, test status, class counts, and means.
- associations_summary.json: class-index mapping, family size, summary counts, warnings, and limitations.

Clustering files include:

- clustering_scores.csv: sample IDs, observed labels, clusters, and retained PC scores.
- clustering_feature_scope.csv: all-missing/constant exclusions and used columns.
- clustering_variance.csv: component-wise and cumulative explained variance.
- cluster_label_counts.csv: post-fit label-by-cluster counts.
- clustering_summary.json: pipeline, k source, geometry, metrics, limitations, optional plot status, `ward` (Ward hierarchical clustering at the same k in the same PCA space: silhouette, ARI/NMI after fitting, agreement with k-means) and `k_sweep` (best k by silhouette for each method).
- clustering_k_sweep.csv: k = 2 to structure_k_max (default 10) with k-means silhouette and inertia and Ward silhouette, all computed without labels. cluster_label_counts_ward.csv: label × Ward cluster counts; clustering_scores.csv gains a ward_cluster column.

Ward needs an n × n distance matrix and is skipped above 5,000 samples (recorded with a reason). The sweep answers whether the data favour the preset k; if the silhouette peaks elsewhere (PANCAN: k = 7 for k-means, 8 for Ward, against the preset 5), report that the preset k came from the known classes and that the internal criterion prefers a finer split, rather than presenting k = 5 as discovered.
- clustering_pca.png: a small descriptive visualization when available.

Class mean columns use class_0, class_1, and so on; retrieve the exact class mapping from the association summary rather than guessing label order. Diagnostic label permutation uses a unique output_dir/structure_permuted_SEED_TIMESTAMP_RUNID directory and is prominently marked.

The script does not infer a new prediction target, tune predictive models, search for biological annotations, produce Kaggle predictions, or upload files. Use the appropriate separate workflow for those tasks.

## 6. Reporting and exam-time priorities

Use actual saved outputs as evidence. State the number of tested features, exclusions and reasons, sample sizes, adjustment method, and effect-size definition. Distinguish observed result, interpretation, and limitation.

Prioritize a correct prediction baseline and valid Kaggle submission when those are required. Run association and clustering branches only when the question asks for them or when they answer a concrete analysis need. They should not delay the first valid prediction submission.

Produce a few informative figures and tables, not every available chart. Full-family tests are usually cheap; a 20,000-by-20,000 correlation matrix, exhaustive pair plots, large parameter sweeps, and new package installations are not default exam work.

Do not write "perfect generalization," "confirmed biomarker," "no overfitting," or a numerical improvement unless the actual experiment and evaluation design support that precise claim. A model may honestly perform very well on an easy dataset; excellent scores warrant checking the procedure, not automatic rejection.
