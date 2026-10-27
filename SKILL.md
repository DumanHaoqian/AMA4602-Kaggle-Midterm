---
name: ama4602-kaggle-exam
description: Prepare for or run an AMA4602-style tabular Kaggle assessment from the current brief and local CSV files. Use for timed classification or regression, leakage-safe validation, high-dimensional or mixed-table model selection, identity-correct submission files, and optional evidence-based figures and reports. Historical PANCAN examples guide rehearsal; the current assessment rules define the task.
---

# AMA4602 Kaggle Exam

Turn the latest brief, the marking rules and the remaining time into an executable analysis. Secure a valid prediction and submission path first, then let validation evidence decide where further time goes. Carry authorised work through to completion; do not turn every step into a confirmation request.

## Portable setup

Read [README.md](README.md) when installing, preparing dependencies or rehearsing. Use Python 3.10+ and a verified virtual environment: `requirements.txt` is the core; `requirements-reports.txt` adds figures/reports; `requirements-boosting.txt` adds optional tree libraries. Install during preparation, not mid-exam. No upstream skills, personal workspace rules, global notebook kernel or Kaggle credentials are required for local modelling. Take identity from the current user/config; an AGENTS.md identity section is optional.

Rehearse offline with `scripts/create_demo.py --output-dir NEW_DIR`, or run `scripts/smoke_test.py` (add `--reports` with report dependencies). The demo identity is fictional and must never be uploaded. Record source paths, sizes, configuration, shapes, versions and saved row IDs/fold indices; do not compute content hashes or SHA/SHA256 checks.

## Bundled files

| Path | Purpose |
|---|---|
| scripts/exam_io.py | Shared loading (ID-based label joins) and atomic writes |
| scripts/profile_data.py | Bounded data audit for wide tables |
| scripts/evaluate_models.py | Fold-safe candidate comparison, OOF predictions, evidence, final refit |
| scripts/leaderboard.py | Leaderboard mode: adversarial validation, mixed types, 10 model families, budgeted search, repeated CV, nested-CV blend, seed-averaged refit |
| scripts/analyze_structure.py | Exploratory ANOVA/BH association screen and label-free PCA/k-means |
| scripts/validate_submission.py | Assembles and validates the identity-named submission CSV |
| scripts/viz_style.py | Single source of the figure style (palette, type, sizes, export) |
| scripts/plot_results.py | Publication figures, captions and manifest from finished runs |
| scripts/build_reports.py (+ report_source.py, report_html.py) | NeurIPS PDF, executed notebook and interactive HTML from one Markdown source |
| assets/exam_config.example.json, assets/pancan_config.example.json | Config templates |
| assets/neurips_2026/ | NeurIPS 2026 style file and official template used for the PDF report |
| assets/report_minimal.md | Prediction-only report source for any run (classification or regression, evaluate or leaderboard) |
| assets/report_example.md | Full PANCAN-style report source (prediction + associations + clustering) to copy and rewrite |
| references/ | Validation, statistics, Kaggle, leaderboard mode, visualisation, reporting, academic writing, provenance |

Run every script with the verified interpreter and `-B` so no `__pycache__` is written into the skill folder. Script paths are relative to this skill directory; outputs go to the user's workspace, never into the skill.

## Exam-day runbook

Copy, set the three variables, and run in order from the workspace. Every command writes a new run directory; nothing is overwritten. Read each printed status line before the next step.

~~~sh
PY=/absolute/path/.venv/bin/python; SK=/absolute/path/ama4602-kaggle-exam/scripts; KG=/absolute/path/.venv/bin/kaggle
cp $SK/../assets/exam_config.example.json config.json        # fill from the brief: task, target, id, metric, positive_class, prediction, identity
$PY -B $SK/profile_data.py --config config.json                # shape, missingness, nonnumeric columns, alerts
# First valid prediction (minute 10-25). Numeric-only data:
$PY -B $SK/evaluate_models.py --config config.json --budget-seconds 300
# Any nonnumeric/categorical column (evaluate_models refuses these), or as the quick tree/linear baseline:
$PY -B $SK/leaderboard.py --config config.json --families logreg_l2,lightgbm --max-trials 1 --budget-seconds 90
$PY -B $SK/validate_submission.py --config config.json --predictions RUN/test_predictions.csv --evidence RUN/evidence.json --output-dir final/v1
$KG competitions submit SLUG -f final/v1/StudentID_FullName.csv -m v1 && $KG competitions submissions SLUG | head -5
# Improve (leaderboard mode, 5-10 min), then validate into final/v2 and submit only if local CV improved:
$PY -B $SK/leaderboard.py --config config.json --budget-seconds 600
# Figures and report only if required (after a validated submission exists):
$PY -B $SK/plot_results.py --run RUN [--structure STRUCTURE]
mkdir -p reports
cp "$SK/../assets/report_minimal.md" reports/report.md            # edit, then:
$PY -B $SK/build_reports.py --source reports/report.md --formats tex
~~~

Supported metrics (set `metric` exactly; anything else must be implemented and fixture-tested before modelling): classification accuracy, balanced_accuracy, f1 (binary, needs positive_class), macro_f1, weighted_f1, mcc, qwk (numeric ordinal labels), log_loss, roc_auc (binary, needs positive_class), roc_auc_ovr, roc_auc_ovr_weighted, roc_auc_ovo; regression rmse, mae, r2, rmsle, mape. Labels are the arg-max of class probabilities (no threshold tuning).

## 1. Establish the task contract first

Read the brief, data dictionary, train/test files, sample submission and the Kaggle rules/evaluation pages that are actually accessible. Record the conclusions in task_contract.md in the workspace:

- Unit of observation, target, classification or regression, official metric and its direction, whether labels, scores or probabilities are required.
- How train/test were generated; repeated patients, batch/site/group or time structure; which columns are available at prediction time.
- ID and label-join keys, class encoding and probability column order, sample-submission column names and row order.
- Competition slug, CSV or code competition, deadline and time zone, submission quota, final-selection mechanism.
- What must be handed in this year: predictions only, or also code, explanation, statistical tests, clustering, figures, a report.
- Student ID, official English name as registered, filename rule; take them from information provided now, never guess from history.
- Rules on external data, public notebooks/discussions, internet use, collaboration and tools, as stated this year.
- Current time, real minutes remaining, available CPU/RAM, installed packages, API or manual upload route.

Write unknown for anything not established; never invent. Resolve what the files and brief can answer before asking. Ask a precise question only when a key ambiguity (target, metric, identity, permissions) genuinely blocks the next step. Missing Kaggle authentication does not block local modelling.

Historical context only: the 2026 announcement described 27 Oct 2026 16:30–18:20 HKT, allowed LLMs, used Kaggle and required StudentID_FullName.csv. Never treat that date, tool permission or task as the current assessment. Read the current brief and set the actual deadline with a time zone in config; leave it unknown until established. The bundled submission validator uses the AMA4602 identity filename convention; adapt it explicitly if the current rules differ.

## 2. Plan by the time actually left

Work to the real deadline; never restart a fresh 110-minute clock midway. Default budget for a full 110 minutes, adjusted to the required deliverables and machine speed:

| Minutes | Result that must exist |
|---|---|
| 0–10 | Task contract, data/ID audit, correct metric, frozen validation design |
| 10–25 | Quick baseline and a first locally validated prediction; submit early if allowed and possible |
| 25–60 | Two or three genuinely different candidates on the same folds, with OOF scores and run times |
| 60–85 | Error and stability checks; statistics, clustering, interpretation and key figures if the brief requires them |
| 85–100 | Freeze the approach, refit on all training rows, validate the final CSV, upload and confirm status |
| 100–110 | Handle submission problems, check identity and final files, assemble required code/report |

With only 30 minutes left: 5 minutes reading and auditing, 10 minutes for one trustworthy pipeline, 10 minutes for prediction validation and submission, 5 minutes buffer. Keep the last valid version. Do not start an experiment expected to run into the final-fit/upload buffer; the scripts check their budget only at fit boundaries and cannot interrupt an estimator.

Default to the local CPU and installed dependencies. During the exam, do not upgrade the environment, install whole upstream repositories, create GPU notebook pipelines, scrape high-scoring solutions or run long searches.

### Preparation and agent roles

During preparation, rehearse reading, CV, prediction and file validation end to end on offline data and record the working environment; confirm the Kaggle login and upload route in advance. Without a competition URL, do not invent a task or submission results; finish runnable preparation instead.

The main agent owns the task contract, the clock, the frozen folds and the final files. One parallel review task may independently recompute the metric and check leakage and submission column order; figures and write-up can be delegated when a report is required. Do not let several agents train wide-table models at once and starve RAM/CPU, and never let different agents change the validation split. State the expected gain, parameters and cost of each new experiment before running it; keep or stop it on evidence.

## 3. Data audit and script entry points

Copy [exam_config.example.json](assets/exam_config.example.json) into the workspace and fill in the real fields from the brief. All paths resolve relative to that JSON file. The example values are not auto-detected. For numeric PANCAN practice see [pancan_config.example.json](assets/pancan_config.example.json).

Run scripts/profile_data.py --config /absolute/path/config.json with the verified Python interpreter; do not assume the system python matches the notebook kernel.

Always check: rows/columns, types, missingness, constants, infinities, class imbalance, ID uniqueness, one-to-one train/label join, train/test column alignment, duplicate observations and p/n. Save per-column summaries instead of printing twenty thousand columns into the chat. Never build a raw p×p correlation matrix.

- Let scripts/exam_io.py reorder external labels by ID; never silently join by row position.
- ID/target/group/time keys are not default predictors. Check the measurement time and origin of any suspicious post-outcome variable.
- Keep true zeros; do not treat zero as missing by default and do not dropna whole patients by default.
- Do not declare leakage from a very high score alone, and do not ignore it either: check subjects crossing folds, target proxies, preprocessing boundaries, duplicate rows and class mappings.
- The package reads CSV/TSV into memory. evaluate_models.py is numeric-only and stops on any string column. For mixed numeric/categorical tables use leaderboard.py from the first baseline on (see the runbook): it one-hot/ordinal/target-encodes categories inside each fold. Text, images or sequences still need a small adapter as described in [validation-and-models.md](references/validation-and-models.md); never silently drop columns.
- Confirm from the documentation whether inputs are raw RNA counts, log/normalised expression or another scale. Do not log twice and do not force biologically meaningful back-transformations.

## 4. Freeze validation before choosing models

Read [validation-and-models.md](references/validation-and-models.md) and choose stratified, kfold, group, stratified_group or time. Save the folds; compare models on the same split with the same official metric.

Every step that learns from the data distribution or labels goes inside the training fold: imputation, variance filtering, scaling, ANOVA/MI selection, PCA, target encoding, calibration, resampling. Exploratory test results on the full training set must not fix the CV feature set. Refit on all labelled training rows only after the final approach is chosen.

Numeric, high-dimensional, small-sample data: start with regularised linear models, then a small supervised-selection or PCA branch. Do not default to deep nets, large random-forest searches or recursive elimination over twenty thousand features. Ordinary low-dimensional mixed tables can use a fold-local ColumnTransformer; try a few tree models if installed and suitable.

Run scripts/evaluate_models.py --config config.json [--models name1,name2|all] [--budget-seconds N]. It provides dummy, numeric classification and regression baselines, identical folds, OOF predictions, evidence and final predictions. Read the actual --help and evidence.json to confirm supported models and outputs; a script name is not proof of success.

- Defaults are linear (classification: select_logistic, linear_svm, pca_logistic; regression: ridge, pca_ridge). `--models all` adds fixed-setting nonlinear challengers on the same folds (random_forest, select_rbf_svm, select_mlp; regression random_forest, rbf_svr, mlp). Use them to show whether a nonlinear model adds anything; they are not tuned, and their settings are recorded in evidence.json. On PANCAN all seven run in about a minute.
- Every fold also records the training-row score (`train_score`), an overfitting diagnostic that never enters selection.
- Linear, PCA and forest candidates write importance_<model>.csv: fold-averaged weights on standardised original features (PCA coefficients back-projected through the loadings; forests use impurity importance) and the share of folds whose selector kept each feature.
- `"holdout": {"fraction": 0.2}` carves an untouched holdout (stratified, whole groups or latest rows, matching the split) before any fold is drawn. CV and selection use the development rows only; the selected pipeline is then fitted on all development rows and scored once on the holdout (holdout_predictions.csv, evidence.holdout). Test predictions still come from a refit on all labelled rows. Use it when the brief asks for a held-out evaluation or a holdout confusion matrix.

- Select by the official metric and its direction; record default parameters or the range tried. Do not fish for random seeds on the public leaderboard.
- Probability metrics need models with usable probabilities; a LinearSVC decision function is not a probability.
- Take the AUC/F1 positive class and multiclass averaging (roc_auc_ovr, roc_auc_ovr_weighted, roc_auc_ovo; macro_f1 vs weighted_f1) from the rules. Define and implement any unsupported metric before running; never substitute accuracy.
- Skewed nonnegative regression targets: set `"target_transform": "log1p"` (fit in log space, predictions back-transformed and kept ≥ 0). It is the default for rmsle; compare with and without it on the same folds for other metrics.
- CV scores used for tuning or model choice are selection CV; never call them independent test performance. The fold SD is not a confidence interval.
- Near-perfect scores: prioritise stability, interpretability and delivery. An independent hold-out or a small label permutation can serve as a diagnostic; one permutation is not a formal significance test.
- Try simple averaging only when OOF evidence supports it and time allows; align IDs and class columns. A stacking meta-model must not be trained on in-fold predictions.
- Keep every successful run; a failure or timeout must not overwrite the last valid file.

### Leaderboard mode (when the ranking is the hidden-test score)

Read [leaderboard.md](references/leaderboard.md). After the first valid submission, run scripts/leaderboard.py --config config.json --budget-seconds N (5–10 minutes is usually enough; at most about 25 in total). It handles mixed numeric/categorical data, scores a prior/mean baseline, searches regularised linear models, SVM, random forest, extra trees, LightGBM (GBDT/DART), XGBoost and CatBoost within the budget on repeated CV, checks train/test shift with adversarial validation, and keeps a blend only if it beats the best single configuration under nested CV. The budget is checked between folds: a trial projected to run into the 25% blend/refit reserve is aborted (status aborted_budget), and the final refit drops seeds or falls back from the blend to fit the remaining time. A single fit that is already running still cannot be interrupted, so keep the budget well inside the exam buffer. Validate its test_predictions.csv with validate_submission.py as usual.

- Record every submission in submissions_log.csv (ref, file path/size, run, local CV and its kind, public/private score, note).
- Each submission tests a hypothesis; never tune against the public score. Final picks: best local CV plus the most robust candidate.
- If local CV and the leaderboard disagree badly, audit before tuning: format/ID alignment, an independent hand-written reference model, a single-feature sanity submission, adversarial validation. leaderboard.md documents a real case (Don't Overfit! II) where the competition data no longer matched its scoring solution.

## 5. Add analyses the brief asks for

Read [statistics-and-structure.md](references/statistics-and-structure.md) only when the brief requires it or it clearly earns marks, and run scripts/analyze_structure.py --config config.json --mode associations|clustering|both.

- Associations: define the complete test family first, test each feature, then apply BH across the whole family; report effect sizes, missing-data handling and assumptions. A multiclass omnibus ANOVA plot is not a standard volcano plot.
- Genuinely unsupervised analysis: feature screening and fitting for clustering/PCA use no y. Colour by y or compute ARI/NMI only after fitting; if k comes from the known number of classes, say that k was preset. The clustering branch also fits Ward hierarchical clustering at the same k and sweeps k = 2–10 (silhouette for k-means and Ward, no labels); report where the silhouette peaks even when it disagrees with the preset k.
- Teaching-style exploration may use all training data, but it must be explained separately from the CV pipeline used to estimate generalisation.
- Anonymous gene_* features without a mapping are reported by name and statistical association only; never invent gene identities, pathways, clinical causality or validated biomarkers.
- Model coefficients after PCA live at the PC level. To discuss original-feature contributions, explain the projection and scaling; never relabel a PC-importance bar chart as gene importance.
- Make only the figures that support conclusions: confusion matrix/OOF errors, holdout confusion matrix, gene-level feature importance, PCA or clustering plot, essential effect-size plots or heatmaps. Read reported numbers from run outputs; never type a "typical accuracy" by hand.

### Figures (when needed)

Read [visualization.md](references/visualization.md). Produce standard figures with scripts/plot_results.py --run RUN_DIR [--structure STRUCTURE_DIR] from finished runs, without refitting (run figures: models, confusion, holdout, importance, gap; structure figures: structure, variance, robustness, associations, distributions, heatmap; each is skipped when its inputs are absent). Output: vector PDF + 300 dpi PNG, English captions.md and a manifest with input paths and sizes. For paper-style documents add --no-titles so the caption carries the title.

- All figure text is English. The title states the finding computed from the results, the grey subtitle states the method, the caption states the evidence and its limits.
- The house style follows Anthropic's transformer-circuits articles: ivory panels, warm greys, near-black ink, Claude clay as the single accent with other series grey, an orange activation ramp, small upper-case panel captions.
- Take colours, type and sizes only from scripts/viz_style.py: the validated fixed-order categorical palette, one colour per class across all figures, marker shapes from the fourth class onward, a single-hue sequential map for magnitude and a diverging map only for z-scores.
- Supplementary figures drawn by hand in a notebook also call vs.apply_style() and use vs.category_style, vs.header and vs.save_figure; never start a second style.
- Open every PNG after generation and check for overlapping or clipped labels and for titles that match the data before using it in a report.

### Reports (when the brief asks for one)

Read [reporting.md](references/reporting.md) and [academic-writing.md](references/academic-writing.md). Write the report only after a valid submission exists. Copy [report_minimal.md](assets/report_minimal.md) (prediction only, works for any run) or [report_example.md](assets/report_example.md) (with association and clustering analyses) into the workspace (e.g. reports/report.md), set the identity from the current user/config or explicit workspace rules and the paths of the current runs, replace every REPLACE_ marker (the build refuses while one remains), rewrite the prose for the actual task, then run scripts/build_reports.py --source report.md. One English source produces:

- tex/: a compiled PDF in the NeurIPS 2026 style from assets/neurips_2026 (paper form, no code; preprint mode with the first-page notice emptied, so no line numbers and no "Submitted to … Do not distribute");
- jupyter/: an executed .ipynb (Markdown prose, code that loads the runs and draws every figure and table, and its outputs);
- html/: a single-file interactive report (embedded Plotly figures and contents sidebar work offline; KaTeX maths and Google Fonts use network CDNs).

Rules:

- Every number in the prose is a {{fact}} placeholder (see --list-facts), never typed by hand; unknown placeholders, citations or cross-references stop the build.
- Cite only real, verifiable works; label selection CV, OOF and exploratory results as such; state limitations with concrete consequences, following academic-writing.md.
- After building, render and inspect the PDF pages, confirm the notebook has no errors and the HTML interactions work, then deliver.

The [historical PANCAN reference](references/pancan-reference.md) is for routing and self-checks; never hard-code its data shape, classes, Edema handling or past scores as this year's answer.

## 6. Final predictions and the Kaggle loop

Read [kaggle-operations.md](references/kaggle-operations.md). The actual sample submission and rules are authoritative; never guess column names or probability order.

Use scripts/validate_submission.py --config config.json --predictions run/test_predictions.csv --evidence run/evidence.json --output-dir final_run. For a CSV written by other code, use --submission path/StudentID_FullName.csv instead. Check --help first and confirm the result from the receipt the script writes.

Always verify:

- Row count, exact column names and order, ID set and template row order, no extra index column.
- Labels belong to the classes allowed by training/rules; regression values are finite; probabilities lie in range, multiclass rows sum correctly, and the positive-class and class-column mapping is explicit.
- The filename contains the real current student ID and the registered English name, with underscores between name words; every upload uses the same identity.
- The final CSV's path and size, run, model, data/folds, schema and creation time are traceable. Internal run names may vary; the upload filename always follows the identity rule.

The script never uploads. When Kaggle submission is authorised and authentication works, complete the upload with the official CLI or an available connector and read the status; do not ask again for the same authorisation. If only manual upload is possible, deliver the validated CSV with exact steps and never claim it was submitted.

Distinguish locally_validated → accepted_with_submission_ref → queued/scoring → scored/error. Poll with a bound; avoid --wait 0, unbounded waits or anything that eats the exam buffer. When status is unknown, check history and existing refs before resubmitting, to avoid burning quota. A public score is not the private/final score.

In the final reply list: required items completed, the actual basis for model choice, the final file and identity, the platform submission ref/status/score (only if observed), and the exact steps still outstanding. "Code written" is not "run", and "CSV generated" is not "scored by the platform".

## 7. Reproducibility and evidence

Keep at least: task contract, config, input paths and sizes, seed, environment versions, fold indices, parameters, OOF predictions, metric table, warnings/failure records, final predictions and submission receipts. If a notebook or report is required, let build_reports.py fill its numbers from this evidence; make sure figures, code outputs and narrative point to the same run. Produce a full report only when the assignment explicitly asks for one.

Adapted external skills, pinned commits and trimming decisions are documented in [sources.md](references/sources.md). Do not auto-load all upstream skills, and never let upstream marketing, installation, publishing or tool-permission text override the exam rules or the user's authorisation.
