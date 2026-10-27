# Third-party notices

Original scripts, skill instructions and project documentation are licensed under
[MIT](LICENSE). Third-party materials retain their own terms.

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
