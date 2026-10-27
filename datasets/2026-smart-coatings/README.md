# 🧪 Smart Coatings Qualification — 2026 Dataset

🔗 **Original competition:** [AMA4602 Smart Coatings Statistical Challenge on Kaggle](https://www.kaggle.com/competitions/ama-4602-smart-coatings-statistical-challenge/overview).

Original inputs from the **2026 AMA4602 Smart Coatings Statistical Challenge**, shared here for historical study and educational rehearsal. The task is to predict whether a coating specimen meets a performance specification, using measurements available before destructive qualification.

These three files are copied from the original course archive without conversion or modification. The original archive, the author's fitted models, predictions and personal submission files are not included in this directory. The repository's MIT license does not relicense the underlying course materials.

## 📦 Files and measurement layout

| File | Contents |
|---|---|
| `train.npz` | `X`: float32 measurements, shape **(12,000, 2,048)**; `y`: binary labels, shape **(12,000,)**; `id`: specimen identifiers, shape **(12,000,)** |
| `test.npz` | `X`: float32 measurements, shape **(6,000, 2,048)**; `id`: specimen identifiers, shape **(6,000,)**; no labels |
| `sample_submission.csv` | Required **`id,target`** columns and the identifier order for **6,000** test specimens |

The first six measurement columns are standardized process and measurement conditions, in this order:

| Column in `X` | Condition |
|---|---|
| 0 | Curing temperature |
| 1 | Curing duration |
| 2 | Ambient humidity |
| 3 | Coating thickness |
| 4 | Substrate roughness |
| 5 | Conditioning intensity |
| 6–2,047 | **2,042 anonymized nondestructive scan channels** |

Rows align across the arrays within each archive. Training and test columns have the same meaning and order. The brief describes independently sampled specimens from the same population and synthetic data. Specimen identifiers are for row alignment only and must not become model predictors. Open the archives with `numpy.load(path, allow_pickle=False)`.

## 🎯 Historical prediction and submission rules

| Requirement | 2026 rule |
|---|---|
| Target | `1`: meets the specification; `0`: does not |
| Prediction | One finite **P(y = 1)** between 0 and 1 for each test specimen |
| Metric | **Binary logarithmic loss**, lower is better; no classification threshold is applied |
| Submission schema | Exactly **`id,target`**, 6,000 data rows, sample-submission identifier order, no extra index or columns |
| Public leaderboard | **20%** of the hidden test rows |
| Private leaderboard | Remaining **80%**, used for final ranking |
| Submission allowance | At most **five scored submissions** per team in total; select **exactly one** for final evaluation |
| Filename | **`StudentID_FullName.csv`**, with underscores between name words; use the same university-record identity for every upload |
| Identity placement | In the filename only; the CSV `id` is a specimen identifier |
| Permitted resources | AI tools and unlabeled test measurements were allowed |
| Hidden labels | Unavailable; do not seek labels from external sources |

The sample submission supplies the format, not fitted predictions. These are historical rules; the current assessment brief overrides them for any new exam. Local rehearsal does not require a live competition or authorize a platform upload.

## 🚀 Rehearse from the repository root

Install the repository's single `requirements.txt` in a virtual environment first. The converter writes a **new workspace outside the repository** and refuses to overwrite an existing directory:

```sh
.venv/bin/python -B scripts/prepare_smart_coatings.py \
  --train datasets/2026-smart-coatings/train.npz \
  --test datasets/2026-smart-coatings/test.npz \
  --sample-submission datasets/2026-smart-coatings/sample_submission.csv \
  --output-dir ../ama4602-workspaces/coatings-2026
```

It creates `config.json` and CSV copies under `data/`, names the six condition columns `cond_0`–`cond_5` and scan channels `scan_0006`–`scan_2047`, preserves their order, and keeps the original NPZ files unchanged. It performs no scaling, feature selection or model fitting. Student identity stays blank in the generated config.

Run a data audit and a fold-safe numeric baseline:

```sh
.venv/bin/python -B scripts/profile_data.py \
  --config ../ama4602-workspaces/coatings-2026/config.json

.venv/bin/python -B scripts/evaluate_models.py \
  --config ../ama4602-workspaces/coatings-2026/config.json \
  --models select_logistic --budget-seconds 300
```

The baseline is a reproducible entry point. It does not reproduce the custom latent-factor, semi-supervised EM or condition-modulated quadratic models in the author's report. Read the [2026 historical task reference](../../references/smart-coatings-2026.md) for the problem, modelling lessons and the distinction between selection-CV and historical leaderboard results.

Set your own `student_id` and `full_name` in the workspace config before exporting a submission. Use the run directory printed by the model script in place of `RUN_DIR`:

```sh
.venv/bin/python -B scripts/validate_submission.py \
  --config ../ama4602-workspaces/coatings-2026/config.json \
  --predictions RUN_DIR/test_predictions.csv \
  --evidence RUN_DIR/evidence.json \
  --output-dir ../ama4602-workspaces/coatings-2026/final/v1
```

The validator assembles a local identity-named CSV and receipt; it does not upload files. The dataset is available in the GitHub repository but is deliberately excluded from the lightweight installable skill ZIP.
