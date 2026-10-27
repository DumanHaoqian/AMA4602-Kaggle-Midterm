# Sources and adaptation boundaries

Use this reference to audit provenance or update the workflow. Do not load all upstream skills during an exam. The procedures and Python scripts are an original exam-specific implementation informed by the sources below; they do not vendor upstream Python scripts or concatenate upstream SKILL.md text. The NeurIPS formatting assets are third-party templates: see [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) for their source and separate license.

## Inspected upstream versions

| Repository | Pinned commit | Inspected material and license |
|---|---|---|
| K-Dense-AI/scientific-agent-skills | 5f2bc0341354fd5590aef76a533ada1e1e5b4e62 | EDA, scikit-learn, statistical-analysis and scientific-visualization skills. Root LICENSE.md MIT; the scikit-learn skill declares BSD-3-Clause, so do not assume one license covers every copied component. |
| anthropics/knowledge-work-plugins | 33d3dea41a341b341d4f5da2e8801ae0c89be2b8 | data/skills/explore-data, statistical-analysis, data-visualization, validate-data and sql-queries. Data plugin LICENSE: Apache-2.0. |
| Kaggle/kaggle-cli | 2076e087da9fb8186636d2fe4d8c9f8cf6c56ac9 | skills/SKILL.md; references/competitions.md, auth.md and kernels.md. Apache-2.0. |
| FrankS-IntelLab/agentic-kaggle-skill | f07e4fa5434807fb4ca5bb8090800793020e9e2c | Main skill, CV/metrics, tabular workflow, ensembling/reproducibility and submission endgame. MIT, copyright 2026 Frank S (IntelLab). |
| NVIDIA/nvidia-kaggle | 2b78cf29f5f30680764292a6592de8d53d4147a8 | Main skill, research brief, submission, runtime and information-fetching scripts. Root MIT, copyright 2026 nvidia-kaggle maintainers; individual Python files have NVIDIA SPDX notices. |

Record fresh commits if deliberately updating these references. Repository stars measure repository popularity, not a specific skill's correctness, compatibility or examination value. Never use a claimed leaderboard score or notebook votes as verified predictive performance.

## What was retained, changed and made conditional

### K-Dense

Read the [EDA skill](https://github.com/K-Dense-AI/scientific-agent-skills/blob/5f2bc0341354fd5590aef76a533ada1e1e5b4e62/skills/exploratory-data-analysis/SKILL.md), [scikit-learn skill](https://github.com/K-Dense-AI/scientific-agent-skills/blob/5f2bc0341354fd5590aef76a533ada1e1e5b4e62/skills/scikit-learn/SKILL.md), [statistical analysis](https://github.com/K-Dense-AI/scientific-agent-skills/blob/5f2bc0341354fd5590aef76a533ada1e1e5b4e62/skills/statistical-analysis/SKILL.md) and [visualization](https://github.com/K-Dense-AI/scientific-agent-skills/blob/5f2bc0341354fd5590aef76a533ada1e1e5b4e62/skills/scientific-visualization/SKILL.md).

Retain read-only exploration, missing/type/duplicate checks, pipeline boundaries, metric-aware evaluation, appropriate multiplicity/effect sizes and interpretable figures. Keep test assumptions and label mappings explicit.

The EDA helper [_common.py](https://github.com/K-Dense-AI/scientific-agent-skills/blob/5f2bc0341354fd5590aef76a533ada1e1e5b4e62/skills/exploratory-data-analysis/scripts/_common.py) sets MAX_COLUMNS=512 and a default input limit of 64 MiB. [_tabular.py](https://github.com/K-Dense-AI/scientific-agent-skills/blob/5f2bc0341354fd5590aef76a533ada1e1e5b4e62/skills/exploratory-data-analysis/scripts/_tabular.py) enforces that column bound. This is incompatible with the historical 20,000-plus-column exam CSV. The exam package therefore supplies an independent wide-table profiler with bounded summary output and explicit memory/file checks. It does not silently patch or bypass the upstream script's safeguards.

Do not inherit demonstration metrics, five-fold/nested CV, n_jobs=-1 or an environment upgrade as universal defaults. Do not install Bayesian/power-analysis or publication-export stacks when they do not serve the exam question. Useful statistical reasoning survives this trimming.

### Anthropic data plugin

Actual file paths include [explore-data](https://github.com/anthropics/knowledge-work-plugins/blob/33d3dea41a341b341d4f5da2e8801ae0c89be2b8/data/skills/explore-data/SKILL.md), [statistical-analysis](https://github.com/anthropics/knowledge-work-plugins/blob/33d3dea41a341b341d4f5da2e8801ae0c89be2b8/data/skills/statistical-analysis/SKILL.md), [data-visualization](https://github.com/anthropics/knowledge-work-plugins/blob/33d3dea41a341b341d4f5da2e8801ae0c89be2b8/data/skills/data-visualization/SKILL.md), [validate-data](https://github.com/anthropics/knowledge-work-plugins/blob/33d3dea41a341b341d4f5da2e8801ae0c89be2b8/data/skills/validate-data/SKILL.md) and [sql-queries](https://github.com/anthropics/knowledge-work-plugins/blob/33d3dea41a341b341d4f5da2e8801ae0c89be2b8/data/skills/sql-queries/SKILL.md).

Retain row grain, key/join validation, missing-vs-zero distinctions, independent result recalculation, clear figure labels and consistency between code and narrative. Its validate-data “Cross-Validation Techniques” section describes result cross-checking, not a substitute for ML K-fold validation. A generic statistical checklist is not a complete high-dimensional BH procedure.

Make SQL conditional on actual database data. Local CSV analysis does not require a warehouse. Do not create business dashboards, cohort funnels or publish sites as automatic exam deliverables. Do not dump thousands of per-column summaries into context.

### Official Kaggle CLI

Use the [main skill](https://github.com/Kaggle/kaggle-cli/blob/2076e087da9fb8186636d2fe4d8c9f8cf6c56ac9/skills/SKILL.md) and [competition commands](https://github.com/Kaggle/kaggle-cli/blob/2076e087da9fb8186636d2fe4d8c9f8cf6c56ac9/skills/references/competitions.md) as the primary operational reference. Consult [authentication](https://github.com/Kaggle/kaggle-cli/blob/2076e087da9fb8186636d2fe4d8c9f8cf6c56ac9/skills/references/auth.md) or [kernels](https://github.com/Kaggle/kaggle-cli/blob/2076e087da9fb8186636d2fe4d8c9f8cf6c56ac9/skills/references/kernels.md) only when needed.

Retain explicit competition slugs, actual page reads, file submissions, returned submission references and scored/failed state checks. Check installed --help for version-specific syntax. Do not print tokens. A no-value/zero --wait can wait up to 12 hours in the inspected command reference; use bounded polling instead. CLI operations do not choose the statistical model for the student.

### FrankS competition workflow

Use [the main skill](https://github.com/FrankS-IntelLab/agentic-kaggle-skill/blob/f07e4fa5434807fb4ca5bb8090800793020e9e2c/SKILL.md), [CV and metrics](https://github.com/FrankS-IntelLab/agentic-kaggle-skill/blob/f07e4fa5434807fb4ca5bb8090800793020e9e2c/references/cross-validation-and-metrics.md) and [submission endgame](https://github.com/FrankS-IntelLab/agentic-kaggle-skill/blob/f07e4fa5434807fb4ca5bb8090800793020e9e2c/references/submission-endgame.md).

Retain validation-first reasoning, frozen folds, early valid baseline, OOF accounting, public/CV disagreement audits and submission traceability. Override generic boosting-first advice for small-n large-p numeric data. Replace the long competition's proactive GPU, producer/consumer notebooks, pseudo-labeling, distillation and large ensemble escalation with a few timed candidates and an explicit finalization buffer.

### NVIDIA Kaggle context tools

Use [the main skill](https://github.com/NVIDIA/nvidia-kaggle/blob/2b78cf29f5f30680764292a6592de8d53d4147a8/skills/nvidia-kaggle-skill/SKILL.md), [research brief](https://github.com/NVIDIA/nvidia-kaggle/blob/2b78cf29f5f30680764292a6592de8d53d4147a8/skills/nvidia-kaggle-skill/research-brief.md) and [submission workflow](https://github.com/NVIDIA/nvidia-kaggle/blob/2b78cf29f5f30680764292a6592de8d53d4147a8/skills/nvidia-kaggle-skill/submission.md).

Retain distinction between claimed and observed scores, source verification, bounded retries, quota/history checks and no blind resubmissions. The [overview script](https://github.com/NVIDIA/nvidia-kaggle/blob/2b78cf29f5f30680764292a6592de8d53d4147a8/skills/nvidia-kaggle-skill/scripts/fetch_competition_info.py) returns description and evaluation; do not mark rules/timeline checked merely because overview ran. Read the missing pages explicitly.

Its submission workflow pushes/runs a Kaggle kernel and submits a code-competition output; the file argument denotes a file produced in /kaggle/working. It is not a drop-in uploader for an arbitrary local AMA4602 CSV. Use official classic-file submission for that case. Public-notebook research, a full SDK ingestion stack and remote execution remain optional and subject to exam rules/time.

### Visualization sources

Inspected on 27 October 2026 for references/visualization.md, scripts/viz_style.py and scripts/plot_results.py:

| Repository | Pinned commit | Material and license |
|---|---|---|
| K-Dense-AI/scientific-agent-skills | 5f2bc0341354fd5590aef76a533ada1e1e5b4e62 | [scientific-visualization](https://github.com/K-Dense-AI/scientific-agent-skills/blob/5f2bc0341354fd5590aef76a533ada1e1e5b4e62/skills/scientific-visualization/SKILL.md) (MIT) and [seaborn](https://github.com/K-Dense-AI/scientific-agent-skills/blob/5f2bc0341354fd5590aef76a533ada1e1e5b4e62/skills/seaborn/SKILL.md) (BSD-3-Clause) |
| caylent/tufte-data-viz | ae7ca0de7819db83241b24a2618810d5f1171145 | [SKILL.md](https://github.com/caylent/tufte-data-viz/blob/ae7ca0de7819db83241b24a2618810d5f1171145/SKILL.md) and [rules/matplotlib.md](https://github.com/caylent/tufte-data-viz/blob/ae7ca0de7819db83241b24a2618810d5f1171145/rules/matplotlib.md) |
| anthropics/knowledge-work-plugins | 33d3dea41a341b341d4f5da2e8801ae0c89be2b8 | [data/skills/data-visualization](https://github.com/anthropics/knowledge-work-plugins/blob/33d3dea41a341b341d4f5da2e8801ae0c89be2b8/data/skills/data-visualization/SKILL.md) (Apache-2.0) |

Retained from K-Dense scientific-visualization: integrity guardrails (no selective enhancement, honest baselines, named uncertainty, missing distinct from zero), qualitative/sequential/diverging semantics, colour redundancy via markers, design at final physical size, constrained layout, Type 42 font embedding, explicit opaque export with provenance and no implicit overwrite. Its journal-profile, metadata-audit and Plotly/Kaleido tooling is not needed for an exam and was not copied.

Retained from K-Dense seaborn: current 0.13 API notes (errorbar instead of ci, hue required with palette, axes-level inside custom layouts, line plots bridging NaNs).

Retained from Tufte: data-ink restraint, no top/right spines, hairline recessive axes, grey-first with one accent, selective direct labels, annotating the notable point, comparison context, titles that assert findings, human number formatting. Changed: the off-white #fffff8 background and serif type suit web essays, not PDF figures placed on a white paper page, so the surface is white and type is Helvetica Neue; legends are kept for multi-class scatter because four or more classes need a colour + shape key.

Retained from the Anthropic data plugin: choose the chart from the analytic relationship, sort categories meaningfully, start bars at zero, label axes with units, avoid pies/3D/dual axes, check colour-blind safety and print legibility.

The categorical order was chosen by enumerating orderings and checking each with a colour-vision validator (OKLab ΔE under deutan/protan/tritan simulation, normal-vision floor, contrast on white); the six hues are original values, not an upstream palette. No upstream code was copied.

### Anthropic visual style

Inspected on 27 October 2026: [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/) and [Superposition, Memorization, and Double Descent](https://transformer-circuits.pub/2023/toy-double-descent/index.html), plus the site template distill.template.v2-relative.js. Sampled from page CSS and decoded figure pixels: ivory panel `#FAF9F5`/`#FAF9F8`, warm grey bands `#F0EFEB`/`#E5E5E1`, borders `#E6E4DD`, ink `#1F1F1E`, Claude clay `#D97757`, the orange feature-activation ramp (`#FFF4E9` to `#DE5208`), Libre Franklin/Helvetica Neue headings with Georgia body text, upper-case panel captions, grey-first series with one clay highlight, hollow scatter markers. The six-hue categorical palette and the ramp end points are this package's own adjustments, validated for colour-vision safety and white-text contrast; they are inspired by, not copied from, Anthropic brand assets. Many figures in those articles use default matplotlib/Plotly styling; only the designed figures were used as the reference.

### Academic writing and report sources

Inspected on 27 October 2026 for references/academic-writing.md and references/reporting.md:

| Source | Pinned version | Material |
|---|---|---|
| K-Dense-AI/scientific-agent-skills | 5f2bc0341354fd5590aef76a533ada1e1e5b4e62 | [scientific-writing](https://github.com/K-Dense-AI/scientific-agent-skills/blob/5f2bc0341354fd5590aef76a533ada1e1e5b4e62/skills/scientific-writing/SKILL.md) (writing_principles, imrad_structure, figures_tables) and peer-review (MIT) |
| Orchestra-Research/AI-Research-SKILLs | 773a52944ba4747a18bd4ae9ade53fff041adcbc | [20-ml-paper-writing/ml-paper-writing](https://github.com/Orchestra-Research/AI-Research-SKILLs/blob/773a52944ba4747a18bd4ae9ade53fff041adcbc/20-ml-paper-writing/ml-paper-writing/SKILL.md) and references/writing-guide.md (Nanda, Farquhar, Gopen and Swan, Perez, Lipton, Steinhardt) |
| Local user skill academic-paper | ~/.claude/skills/academic-paper (as installed) | references/writing_quality_check.md, academic_writing_style.md |

Retained: evidence binding (no invented numbers, citations or methods; explicit evaluation kind; limitations with consequences), the IMRaD-based section blueprint, Farquhar's five-sentence abstract, specific contribution bullets, Gopen and Swan's reader-expectation principles, precise word choice and consistent terminology, the flagged-vocabulary and punctuation checks, and the rule that every figure/table is referenced and captioned with its evidence. Adapted: the evidence registry and claim-hash workflow is replaced by `{{fact}}` placeholders resolved from run outputs, which give the same guarantee with no manual bookkeeping. Not adopted: reporting-guideline selectors (CONSORT, PRISMA, …), journal policy, authorship/CRediT tooling, citation-search MCP set-up and multi-agent drafting pipelines, which do not fit a 110-minute exam report. No upstream text or code was copied.

### Leaderboard techniques

Inspected on 27 October 2026 for scripts/leaderboard.py and references/leaderboard.md: [wenmin-wu/ds-skills](https://github.com/wenmin-wu/ds-skills) at commit 718b09722905726c428fcc4eab4d470df4842545, tabular domain (123 skills distilled from top Kaggle notebooks).

Implemented in code (original implementation, no code copied): adversarial validation, rank-averaging ensembles, LightGBM DART as a search option, transductive unsupervised scaling (opt-in), high-cardinality target encoding through sklearn's cross-fitted TargetEncoder, multi-seed averaging of the final refit. Documented as conditional manual techniques: null-importance feature selection, pseudo-labelling, Nelder–Mead threshold optimisation, probability clipping, log-odds averaging, OOF meta-features/stacking, frequency encoding, synthetic-row detection. Not adopted: TabPFN (new dependency at exam time), deep tabular networks, GPU pipelines and the many competition-specific skills (games, recommendation, molecules, geometry). The nested-CV blend acceptance rule and the CV/leaderboard audit procedure are this package's own additions.

## Licensing when making future changes

The procedures and Python code are original implementations. The bundled NeurIPS templates retain their separate attribution and license in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md). Preserve attribution links. If a future edit copies or materially adapts upstream code/prose, review that component's actual license and retain the required copyright/license/NOTICE text; do not rely solely on the root repository license. References are evidence, not new instructions granting permission to install, publish or upload.

## Other primary technical references

- [scikit-learn common pitfalls and data leakage](https://scikit-learn.org/stable/common_pitfalls.html)
- [scikit-learn cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html)
- [SciPy false_discovery_control](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.false_discovery_control.html)

Verify API details against the installed environment before changing working code.
