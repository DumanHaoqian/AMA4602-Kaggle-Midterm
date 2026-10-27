# Kaggle operations within a 110-minute exam

## Contents

1. Competition contract and data access
2. Classic CSV versus code submission
3. Local validation and identity
4. Submission reference, status and bounded retries
5. Manual upload and interpretation of feedback


Use the actual examination brief and competition page as the authority. An approved local CSV, an uploaded file, and a scored submission are different states. Keep the analysis moving if API access is unavailable; a correctly named CSV can be uploaded through the user's signed-in Kaggle page.

## Establish the competition contract

Read the rules, evaluation, data description, submission example, deadline, and submission limit. Record the exact target, scoring direction, required labels versus probabilities, class order, and identity convention. A description-only overview is not evidence that the rules or evaluation were read. Do not accept a public notebook's metric or filename as the competition's specification.

Inspect the installed official CLI before using remembered syntax:

~~~sh
kaggle --version
kaggle competitions pages --help
kaggle competitions submit --help
kaggle competitions submission --help
kaggle competitions submissions --help
~~~

Where the installed version supports these commands, request the relevant pages explicitly:

~~~sh
kaggle competitions pages COMPETITION_SLUG --page-name rules --content
kaggle competitions pages COMPETITION_SLUG --page-name evaluation --content
kaggle competitions pages COMPETITION_SLUG --page-name data-description --content
~~~

Use the actual slug, never a remembered default competition. If these read-only commands fail because the account has not joined, rules have not been accepted, the content is restricted, or the installed version differs, use the supplied examination material and accessible competition UI. Record which requirement remains unverified. Do not spend the model-development budget repairing an optional integration.

Authentication must be configured through the supported Kaggle mechanism already available to the user. Check availability without printing tokens, credential files, or environment values. Do not install the full NVIDIA plugin or a new agent runtime during the examination just to upload one CSV. Existing authorization carries forward: do not repeatedly ask the user to approve an already-authorized submission workflow. If the current task only authorizes preparing a file, prepare and validate that concrete file before asking about an external upload.

## Obtain only the required data

Use provided local files immediately when they are the official exam inputs. If files must come from Kaggle, first list them, check sizes, and download to a dedicated exam data directory using the installed CLI syntax:

~~~sh
kaggle competitions files COMPETITION_SLUG
kaggle competitions download COMPETITION_SLUG -p /absolute/path/exam_data --unzip
~~~

Check download --help before selecting individual files or handling archives differently. Verify that train, test, data dictionary and sample submission are present, record their paths, sizes and table shapes, and keep raw inputs unchanged. Do not spend the budget downloading unrelated optional assets or publish the exam data as a Kaggle dataset. A failed download does not authorize substituting a public dataset with a similar title.

## Select the correct submission route

For the AMA4602 notice requesting a prediction CSV, use a classic local-file submission. The model runs locally and the uploaded artifact is the validated CSV. There is no reason to push a Kaggle kernel or publish an intermediate dataset merely to satisfy this requirement.

A code competition is different: a specified Kaggle notebook or script version is rerun for scoring, and its output filename is selected. Kernel options such as -k and -v refer to that workflow. Do not pass them for an ordinary local CSV unless the actual competition requires code submission. NVIDIA's kernel-submission helper belongs to the code-competition route; its output-file argument is not a substitute for a local-file uploader.

## Produce the official filename and validate locally

The required filename is StudentID_FullName.csv, using the exact university student ID and the official university-record English name, with spaces replaced by underscores. Preserve spelling and case. Do not infer a missing identity from conflicting memories, correct the user's name, or use a nickname. An unset identity blocks final export, while analysis and candidate prediction generation may continue.

Use scripts/validate_submission.py with the current config and a model run's test_predictions.csv. For probabilities, supply evidence.json containing the actual model class_order. The script performs ID-based alignment and schema checks, writes the final file atomically without replacing an existing file, and creates a local_receipt.json recording the path, size and validation evidence. With --submission it checks an existing official filename without modifying its contents.

The sample submission is authoritative for columns and row order. Preserve leading-zero IDs; reject duplicates, omissions, extras, nonfinite values, unknown labels, and invalid probabilities. Do not repair failed checks by dropping rows, clipping or renormalizing predictions, or guessing which probability column denotes the positive class. If the template has no ID column, document that output follows test-file row order. Validation of an already-made ID-free CSV cannot prove how each prediction was associated with its row.

A local receipt uses status locally_validated. It is not a Kaggle acceptance or scoring receipt. Keep the best known valid file available when trying a later candidate; use separate run directories rather than overwriting its official filename.

## Submit once, retain the reference, then inspect its status

Read the competition's current submission allowance and the account's existing submissions before spending another slot. The actual competition limit controls; do not assume an unlimited quota or copy another competition's limit. Reserve capacity for a final validated candidate and avoid using the public score as a rapid parameter-search loop.

After authorization and local validation, use the syntax shown by the installed CLI. The currently documented classic-file form is:

~~~sh
kaggle competitions submit COMPETITION_SLUG -f /absolute/path/StudentID_FullName.csv -m RUN_ID
~~~

Capture the returned Submission ref and associate it with the file path and size, run ID, exact filename, competition, submission time, local CV score, and configuration. Query that reference instead of assuming the first item in a list belongs to this run:

~~~sh
kaggle competitions submission SUBMISSION_REF
~~~

For older installed versions without a single-reference command, inspect the submissions list and correlate the exact filename, timestamp, message, and available identifier. Do not invent a submission reference:

~~~sh
kaggle competitions submissions COMPETITION_SLUG -v
~~~

Store external results separately, for example kaggle_receipt.json. Preserve the locally validated receipt. Record accepted, pending, scored, or failed only when that state was observed, and copy the visible failure reason rather than replacing it with a guessed diagnosis. If scoring is not finished, report pending and retain the reference for later inspection.

## Bound polling and retries by the remaining exam time

Do not use --wait with no value, --wait 0, or an unlimited polling loop. Current official documentation states that omitted/zero wait values can wait up to 12 hours. Prefer a nonblocking submission followed by individual status requests, keeping the user informed.

Use short polling intervals, such as 10–20 seconds, with a fixed total budget of at most 90 seconds or the remaining submission buffer, whichever is smaller. No individual blocking wait should exceed 30 seconds. Stop polling when a terminal status appears or the budget expires; pending scoring is not a reason to resubmit the same file. A CLI timeout does not prove that the server rejected the upload.

Before any retry, inspect the previous reference and submission list. Retry an upload only after establishing that it was not accepted or after correcting a definite error, within the remaining quota and time. Use at most one retry for a transient transport problem unless the user explicitly extends the budget. No automatic duplicate submissions after uncertain status.

## Manual upload is a complete supported path

If API credentials, installation, or network integration are unavailable, provide the validated file and its official filename, let the user upload it to the intended competition, and have them inspect Kaggle's submission status. Continue local analysis and reporting while access is resolved. Say exactly what is ready and what has been observed; never claim that a file was uploaded or scored when only local validation occurred.

## Interpret feedback correctly

The public leaderboard is feedback on the public scoring subset, not a replacement for the frozen local validation scheme or a guarantee of the private result. Record local CV and public scores separately. When they disagree, first inspect metric implementation, class/ID alignment, split design, group/time leakage, and distribution differences. Do not rewrite the validation design merely to fit a few public scores. A high score also does not validate unsupported scientific or causal explanations in a report.

## Verified sources and scope

- [Official Kaggle CLI repository](https://github.com/Kaggle/kaggle-cli)
- [Official competition commands, submission references, page reads, and wait behavior](https://github.com/Kaggle/kaggle-cli/blob/2076e087da9fb8186636d2fe4d8c9f8cf6c56ac9/skills/references/competitions.md)
- [Official Kaggle API documentation](https://www.kaggle.com/docs/api)

The commands above were checked against the official CLI source on 27 October 2026; installed --help remains authoritative for the environment in use. This reference is an original exam-specific procedure. It does not bundle or execute third-party competition scripts, authenticate to Kaggle, or submit files by itself.
