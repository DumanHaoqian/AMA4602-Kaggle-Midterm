# Contributing and releasing

Keep changes focused on a reusable AMA4602 workflow. Use synthetic fixtures;
never contribute course inputs, student identities, credentials or submissions.

## Local checks

Use Python 3.10+ in a virtual environment and install `requirements-dev.txt`.

```sh
python -B scripts/check_release.py
python -B scripts/smoke_test.py --reports
```

If an installed skill-creator is available, also run its `quick_validate.py`
against this directory. The repository's own checks do not require that skill.
Report and modelling changes should be checked against observable output, not
just matching text. Tectonic PDF compilation is a separate optional check; it
may download packages and is not part of the offline smoke test.

## Release

1. Update `VERSION` and `CHANGELOG.md`; keep the README version and archive
   example consistent. Review attribution if adding third-party material.
2. Run both checks above. Review `git diff` and `git status --short` for unintended
   files. `scripts/release_files.py` defines the installation/release inventory;
   add new file types there deliberately.
3. Run `python -B scripts/build_release.py`. It refuses an existing ZIP and
   includes one top-level skill directory. Review the ZIP file list before
   sharing it.
4. Commit, create a matching `vVERSION` tag, and push when ready to publish.
   The workflow checks the tag against `VERSION` and uploads a build artifact.
5. Create a GitHub Release for that tag and attach the ZIP. Copy the matching
   changelog entry into the release notes, and include any untested platforms or
   optional components. Creating a release is a deliberate maintainer action.

Do not introduce SHA/SHA256 checks or scan inputs to compute content digests.
Keep run paths, shapes, actual versions, saved row IDs/folds and exit status as
the evidence needed for reproduction. Historical upstream commit references in
`references/sources.md` are attribution only and are not recalculated.

The first release was exercised on macOS with Python 3.13, including numerical
classification/regression, mixed categorical inputs, identity-preserving CSV
delivery and executed notebook/HTML/TeX reports. Python 3.10 syntax is checked
locally; Linux/Python 3.10 and 3.13 execution is configured in CI and should be
confirmed after pushing. Windows is documented but has not been exercised.
