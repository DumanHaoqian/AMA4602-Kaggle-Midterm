# Third-party notices

Original scripts, skill instructions and project documentation are licensed under
[MIT](LICENSE). Third-party materials retain their own terms.

## Historical teaching datasets

`datasets/2024-pancan/` contains the user-supplied Gene/PANCAN teaching table
(losslessly compressed) and its labels. The accompanying course description
identifies the measurements as a subset of TCGA PANCAN RNA-Seq data with an
additional Edema variable. The dataset README distinguishes the original task
from the skill's recommended analysis choices.

`datasets/2026-smart-coatings/` contains the original synthetic training and test
archives and sample submission for the AMA4602 Smart Coatings Statistical
Challenge. Source: [the Kaggle competition](https://www.kaggle.com/competitions/ama-4602-smart-coatings-statistical-challenge/overview)
and its downloaded challenge materials.

These data are shared as historical teaching materials at the maintainer's
request. The project's MIT license does not relicense the underlying dataset
materials. Their provenance and historical use conditions are recorded in each
dataset README. Dataset files are excluded from the lightweight skill ZIP and
installed skill. The shared archive does not include other students' reports or
the author's personal identity configuration and prediction submissions.

## Author's report figures

The four PNG files in `assets/showcase/` are unchanged copies of figures from
Haoqian Du's final 2026 report, *Hidden Contrasts: Condition-Modulated Quadratic
Discrimination for Smart-Coating Qualification*. They are shared by the author
as examples of the historical analysis. Figure mapping and analytical scope
are recorded in [the case study](references/smart-coatings-2026.md). These figures
contain no visible student identifier and do not establish a private score or
final competition rank.

## NeurIPS 2026 formatting template

Files: `assets/neurips_2026/neurips_2026.sty` and `neurips_2026.tex`.

- Attribution: NeurIPS 2026 Program Chairs; the style header also credits Roman
  Garnett and the authors of `nips15submit_e.sty`. Original headers are retained.
- Source: [official NeurIPS 2026 formatting instructions](https://neurips.cc/Conferences/2026/CallForPapers)
  and [formatting archive](https://media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip).
- The [official template listing on Overleaf](https://www.overleaf.com/latex/templates/formatting-instructions-for-neurips-2026/bjdwqfdkyftc)
  identifies NeurIPS 2026 Program Chairs as author and lists
  [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
  This attribution and license link accompany redistributed copies. These files
  are excluded from this project's MIT license. The files themselves do not
  contain a separate license block.
- The bundled template files are unchanged by this release. Generated reports
  use preprint mode and suppress the notice in the generated document, without
  modifying the bundled style file. This is a course-report format and does not
  imply NeurIPS submission, acceptance or endorsement.

## Workflow references

[references/sources.md](references/sources.md) records the upstream publications
and skills that informed this workflow, including historical version references.
Those links are attribution and context; the referenced projects are not required
installations. No upstream Python implementation is bundled from those projects.

## Report dependencies

Generated HTML embeds the Plotly JavaScript supplied by the installed Plotly
package. Its upstream license notices remain in that distribution. The report
also links to Google Fonts and KaTeX CDN assets; these assets are not vendored in
the skill release. Custom fonts and mathematical typesetting need those network
resources, while the embedded Plotly interactions work offline.
