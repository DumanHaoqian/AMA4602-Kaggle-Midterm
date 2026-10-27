# 2026 case study: Smart Coatings Qualification

This archive records the question used in the 2026 AMA4602 midterm and the main
lessons from the author's analysis. It is a historical example: use the actual
brief, deadline, permissions and submission rules for any new assessment.

## The question

Original competition: [AMA4602 Smart Coatings Statistical Challenge on Kaggle](https://www.kaggle.com/competitions/ama-4602-smart-coatings-statistical-challenge/overview).

A materials laboratory develops functional coatings for precision components.
Destructive testing determines whether a specimen meets the performance
specification. Before that result is available, the laboratory records process
conditions and a nondestructive scan. The task is to estimate the probability
that each untested specimen will pass qualification.

The official brief describes synthetic, independently sampled specimens from
the same population in training and test data. A specimen's process conditions
and scan channels describe the same observation. Its identifier is an alignment
key, not a predictor. There is no external source from which to recover labels.

| Item | Historical specification |
|---|---|
| Competition | `ama-4602-smart-coatings-statistical-challenge` |
| Task | Binary probabilistic classification |
| Positive class | `1`: meets the performance specification |
| Negative class | `0`: does not meet the specification |
| Training data | 12,000 specimens, 2,048 numerical measurements, binary labels and IDs |
| Test data | 6,000 specimens, the same 2,048 measurements and IDs; labels hidden |
| Official metric | Binary logarithmic loss; lower is better |
| Required prediction | One probability of class `1` per test specimen |
| Submission schema | Exactly `id,target`, in sample-submission row order |
| Public / private split | 20% / 80% of hidden test rows; private score determines final ranking |

The metric evaluates probabilities directly. No classification threshold is
applied, so high accuracy alone is not sufficient evidence of a good submission.

## What was in the data

The original files are NumPy archives, opened with
`numpy.load(path, allow_pickle=False)`:

| File | Arrays |
|---|---|
| `train.npz` | `X`: float32 `(12000, 2048)`; `y`: `(12000,)`; `id`: `(12000,)` |
| `test.npz` | `X`: float32 `(6000, 2048)`; `id`: `(6000,)` |
| `sample_submission.csv` | Required `id,target` columns and test identifier order |

Row positions align the arrays within each archive. Measurement meanings and
column order are identical between training and test data.

| Original `X` column | Meaning | Converted CSV feature |
|---|---|---|
| 0 | Standardized curing temperature | `cond_0` |
| 1 | Standardized curing duration | `cond_1` |
| 2 | Standardized ambient humidity | `cond_2` |
| 3 | Standardized coating thickness | `cond_3` |
| 4 | Standardized substrate roughness | `cond_4` |
| 5 | Standardized conditioning intensity | `cond_5` |
| 6–2047 | 2,042 anonymized nondestructive scan channels | `scan_0006`–`scan_2047` |

The converted training CSV also contains `id` and `target`; the test CSV contains
`id` and the same measurement columns. Preserve the original column order and
exclude `id` from the model inputs.

## Historical submission rules

- Submit exactly 6,000 data rows with the two columns `id,target`, no extra
  index, duplicate IDs or personal-information columns.
- Each `target` must be a finite probability between zero and one, representing
  the specimen's probability of qualification.
- Every uploaded file must be named `StudentID_FullName.csv`, using the current
  student's own university-record identity and underscores between name words.
  Student identity belongs in the filename only; `id` remains a specimen ID.
- Each team may make at most five scored submissions and select exactly one for
  final evaluation.
- AI tools and unlabeled test measurements are permitted by this historical
  brief. Hidden labels must not be sought. These permissions do not transfer
  automatically to another assessment.
- The downloaded brief specifies prediction delivery. The author's written
  report is an additional analysis artifact, not a report requirement established
  by that brief.

## Rehearse the task with this skill

The [GitHub dataset archive](https://github.com/DumanHaoqian/AMA4602-Kaggle-Midterm/tree/main/datasets/2026-smart-coatings)
contains the original NPZ files and sample submission with an English task guide.
The lightweight skill ZIP and installed skill exclude dataset files; obtain them
from that repository archive or an authorized course or competition source.
The example config contains schema only, and the author's personal prediction
files are not distributed.

The ordinary modelling scripts read CSV/TSV. Convert the NPZ archives first,
using a new output directory:

```sh
.venv/bin/python -B scripts/prepare_smart_coatings.py \
  --train /path/to/train.npz \
  --test /path/to/test.npz \
  --sample-submission /path/to/sample_submission.csv \
  --output-dir smart-coatings-practice
```

The adapter creates `config.json` and `data/train.csv`, `data/test.csv` and
`data/sample_submission.csv` inside that directory. Its configuration follows
[smart_coatings_2026_config.example.json](../assets/smart_coatings_2026_config.example.json).
All data paths are resolved relative to the generated configuration file.
Student identity is intentionally blank; fill it in yourself before exporting a
submission. The historical deadline is not reused.

Start with a small, leakage-safe baseline:

```sh
.venv/bin/python -B scripts/profile_data.py \
  --config smart-coatings-practice/config.json
.venv/bin/python -B scripts/evaluate_models.py \
  --config smart-coatings-practice/config.json \
  --models select_logistic,pca_logistic --budget-seconds 300
```

After inspecting the actual completed run, replace `RUN_DIR` with its path:

```sh
.venv/bin/python -B scripts/validate_submission.py \
  --config smart-coatings-practice/config.json \
  --predictions RUN_DIR/test_predictions.csv \
  --evidence RUN_DIR/evidence.json \
  --output-dir smart-coatings-practice/final/v1
```

These commands provide a reusable baseline and local submission checks. They do
not reproduce the custom research models described below, and they do not
upload to Kaggle.

## What the author's analysis found

The case study illustrates why a timed modelling workflow should compare
different hypotheses rather than only tune the first model. Marginal feature
associations and a raw linear classifier captured only part of the predictive
structure. The analysis then investigated latent factors, class-covariance
contrasts and interactions with process conditions.

The author's report records the following progression. Values are rounded to
four decimal places; all CV values are **selection cross-validation**, not
independent test estimates.

| Recorded version | Modelling hypothesis | Selection-CV log loss | Reported public log loss |
|---|---|---:|---:|
| v1 | Regularized logistic regression on 2,048 raw measurements | 0.6061 | 0.6289 |
| v2 | Logistic regression on 192 latent factors and 16 squared contrast directions | 0.3565 | 0.3323 |
| v3 | Semi-supervised structured Gaussian discriminant model fitted by EM | 0.3252 | 0.3098 |
| v4 | Condition-modulated quadratic logistic model | 0.1081 | 0.0813 |

The inferred 192-factor representation, rank-16 covariance contrast and selected
condition interactions are findings from this analysis. They were not supplied
as known data-generating facts in the question.

The final model used a label-free PCA representation estimated from training
and test measurements, which this particular task permitted. Label-dependent
directions, condition-pair selection and model coefficients were estimated
inside training folds. Its reported selection-CV protocol was stratified
five-fold validation repeated three times. Because model structure and
hyperparameters were selected using those same results, the reported CV score
is optimistic for the selected model. Exploratory figures drawn from all
training rows must be interpreted separately from fold-internal evaluation.

The public scores above are historical values recorded in the report, covering
only the public test partition. The repository does not establish a private
leaderboard score, final rank or a guaranteed score for a future run. A public
score being lower than CV does not by itself identify the reason for the gap.

The Gaussian EM and condition-modulated quadratic models used task-specific
analysis code. They are not model families in the shared baseline scripts, and
their custom evidence files are not directly interchangeable with the generic
report builder's input schema. The case-study figures demonstrate the author's
completed analysis; the current skill supplies the reusable audit, validation,
baseline, plotting and reporting workflow.

## Source record

### README gallery

The PNGs are copied unchanged from the final report's figure exports. They
demonstrate task-specific analysis; they are not all direct outputs of the
generic `plot_results.py` command.

| Shared figure | Original export | Final report location | Interpretation |
|---|---|---|---|
| [Latent structure](../assets/showcase/2026-coatings-structure.png) | `fig_data.png` | Figure 2, page 3 | Exploratory marginal associations, PCA spectrum and covariance contrast |
| [Exploratory analysis](../assets/showcase/2026-coatings-eda.png) | `fig_eda.png` | Figure 1, page 2 | Training-data summaries of conditions, channels and projected distributions |
| [Learning and calibration](../assets/showcase/2026-coatings-evaluation.png) | `fig_v4_eval.png` | Figure 4, page 7 | Fold-internal learning curves and out-of-fold reliability |
| [Model interpretation](../assets/showcase/2026-coatings-interpretation.png) | `fig_v4_terms.png` | Figure 6, page 8 | In-sample term-group variance and condition involvement |

### Task and analysis evidence

The task description, schema and historical rules were summarized from the
downloaded challenge files `task.md`, `README.md` and `STARTER.md`. Model
progression and figures come from the author's final 2026 report and its saved
analysis results, including `report_numbers.json`, `paper_v4_numbers.json` and
the final condition-modulated model's `evidence.json`. The older short report
describes v3 and is superseded by the final v4 report for this case study.

The separate [PANCAN rehearsal](pancan-reference.md) is archived as the 2025
exercise according to the maintainer's identification of the supplied files.
