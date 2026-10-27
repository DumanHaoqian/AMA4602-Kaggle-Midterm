#!/usr/bin/env python3
"""Build a NeurIPS-style PDF, an executed Jupyter notebook and an interactive HTML
report from one Markdown source and finished exam runs.

    python -B build_reports.py --source report.md --out-root reports
    python -B build_reports.py --source report.md --list-facts

The PDF is compiled with tectonic from the NeurIPS 2026 style shipped in this
skill (assets/neurips_2026/), copied next to the generated .tex, using the
non-anonymous preprint mode with the first-page notice removed, so it has no
line numbers and no "Submitted to ... Do not distribute" footer. Nothing is
refitted: figures and tables are drawn from the run outputs named in the source.
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

import nbformat  # noqa: E402

import plot_results as pr  # noqa: E402
import report_source as rs  # noqa: E402
import viz_style as vs  # noqa: E402
from viz_style import plt  # noqa: E402

NEURIPS_STY = SCRIPTS.parent / "assets" / "neurips_2026" / "neurips_2026.sty"
FORMATS = ("tex", "ipynb", "html")
DEFAULT_WIDTH = {"models": "0.86", "confusion": "0.62"}


def make_figure(name: str, ctx: rs.Context, titles: bool) -> dict:
    if name in ("models", "confusion", "holdout", "importance", "gap"):
        rs.need(ctx.run, f"figure {name} needs a run directory")
    if name in ("structure", "variance", "robustness", "associations", "distributions", "heatmap"):
        rs.need(ctx.structure_dir, f"figure {name} needs a structure directory")
    if name in ("distributions", "heatmap"):
        rs.need(ctx.raw is not None, f"figure {name} needs the config (raw data)")
    result = {"models": lambda: pr.fig_models(ctx.run, None, titles),
              "confusion": lambda: pr.fig_confusion(ctx.run, ctx.run_dir, None, titles),
              "holdout": lambda: pr.fig_holdout(ctx.run, ctx.run_dir, titles),
              "importance": lambda: pr.fig_importance(ctx.run, ctx.run_dir, titles),
              "gap": lambda: pr.fig_gap(ctx.run, titles),
              "variance": lambda: pr.fig_variance(ctx.structure_dir, titles),
              "robustness": lambda: pr.fig_cluster_robustness(ctx.structure_dir, titles),
              "structure": lambda: pr.fig_structure(ctx.structure_dir, titles),
              "associations": lambda: pr.fig_associations(ctx.structure_dir, titles),
              "distributions": lambda: pr.fig_distributions(ctx.structure_dir, ctx.raw, titles),
              "heatmap": lambda: pr.fig_heatmap(ctx.structure_dir, ctx.raw, titles)}[name]()
    rs.need(result is not None, f"figure {name}: required run outputs are missing")
    return result


def png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return buf.getvalue()


# ------------------------------------------------------------------- LaTeX

def build_tex(src: rs.Source, ctx: rs.Context, num: rs.Numbering, figs: dict, tex_dir: Path, slug: str) -> Path:
    def tx(text):
        return rs.inline(text, "tex", ctx.facts, num, src.references)

    meta = src.meta
    author = [rs.tex_escape(meta["author"])]
    if meta.get("student_id"):
        author.append(f"Student ID: {rs.tex_escape(meta['student_id'])}")
    if meta.get("affiliation"):
        author.append(rs.tex_escape(meta["affiliation"]))
    lines = [
        r"\documentclass{article}",
        r"\PassOptionsToPackage{numbers,compress}{natbib}",
        r"% preprint = non-anonymous and no line numbers; the notice box is emptied below.",
        r"\usepackage[preprint]{neurips_2026}",
        r"\makeatletter\renewcommand{\@notice}{}\makeatother",
        r"\usepackage[utf8]{inputenc}", r"\usepackage[T1]{fontenc}", r"\usepackage{hyperref}", r"\usepackage{url}",
        r"\usepackage{booktabs}", r"\usepackage{amsmath,amssymb}", r"\usepackage{nicefrac}", r"\usepackage{microtype}",
        r"\usepackage{xcolor}", r"\usepackage{graphicx}", r"\usepackage{array}",
        r"\definecolor{accent}{HTML}{C4511B}",
        r"\hypersetup{colorlinks=true,linkcolor=accent,citecolor=accent,urlcolor=accent}",
        r"\graphicspath{{" + f"{slug}_figures/" + r"}}",
        f"\\title{{{tx(meta['title'])}}}",
        "\\author{" + r" \\ ".join(author) + "}",
        r"\begin{document}", r"\maketitle",
        r"\begin{abstract}", tx(src.abstract), r"\end{abstract}", ""]
    for b in src.blocks:
        if b.kind == "heading":
            if b.level == 1 and b.text.lower() == "appendix":
                lines += [r"\appendix", ""]
                continue
            cmd = {1: "section", 2: "subsection", 3: "subsubsection"}[b.level]
            label = f"\\label{{sec:{b.label.split(':', 1)[-1]}}}" if b.label else ""
            lines += [f"\\{cmd}{{{tx(b.text)}}}{label}", ""]
        elif b.kind == "para":
            lines += [tx(b.text), ""]
        elif b.kind == "list":
            env = "enumerate" if b.ordered else "itemize"
            lines += [f"\\begin{{{env}}}"] + [f"  \\item {tx(i)}" for i in b.items] + [f"\\end{{{env}}}", ""]
        elif b.kind == "math":
            lines += [r"\[", rs.fill(b.text, ctx.facts), r"\]", ""]
        elif b.kind == "figure":
            f = figs[b.text]
            width = b.options.get("width", DEFAULT_WIDTH.get(b.text, "1.0"))
            lines += [r"\begin{figure}[t]", r"  \centering",
                      f"  \\includegraphics[width={width}\\linewidth]{{{f['stem']}.pdf}}",
                      f"  \\caption{{\\textbf{{{rs.tex_escape(f['title'])}.}} {rs.tex_escape(f['caption'])}}}",
                      f"  \\label{{fig:{b.text}}}", r"\end{figure}", ""]
        elif b.kind == "table":
            t = rs.table(ctx, b.text)
            spec = "".join("p{0.5\\linewidth}" if a == "p" else a for a in t["align"])
            lines += [r"\begin{table}[t]", f"  \\caption{{{rs.tex_escape(t['caption'])}}}", f"  \\label{{tab:{b.text}}}",
                      r"  \centering", r"  \small", f"  \\begin{{tabular}}{{{spec}}}", r"    \toprule",
                      "    " + " & ".join(f"\\textbf{{{rs.tex_escape(c)}}}" for c in t["columns"]) + r" \\", r"    \midrule"]
            for i, row in enumerate(t["rows"]):
                cells = [rs.tex_escape(c) for c in row]
                if i in t["bold"]:
                    cells = [f"\\textbf{{{c}}}" for c in cells]
                lines.append("    " + " & ".join(cells) + r" \\")
            lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
        # code blocks are notebook-only: the paper carries no code.
    if src.references:
        lines += [r"{\small", f"\\begin{{thebibliography}}{{{len(src.references)}}}"]
        for key, text in src.references.items():
            lines.append(f"\\bibitem{{{key}}} {rs.inline(text, 'tex', ctx.facts, num, src.references)}")
        lines += [r"\end{thebibliography}", "}"]
    lines.append(r"\end{document}")
    path = tex_dir / f"{slug}.tex"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def compile_tex(tex_path: Path) -> dict:
    tectonic = shutil.which("tectonic")
    if not tectonic:
        return {"status": "not_compiled", "reason": "tectonic not found on PATH"}
    proc = subprocess.run([tectonic, "--keep-logs", tex_path.name], cwd=tex_path.parent,
                          capture_output=True, text=True, timeout=600)
    pdf = tex_path.with_suffix(".pdf")
    log = (proc.stdout + proc.stderr).splitlines()
    warnings = [l for l in log if "warning" in l.lower() and ("undefined" in l.lower() or "overfull" in l.lower())]
    if proc.returncode != 0 or not pdf.exists():
        return {"status": "failed", "returncode": proc.returncode, "log_tail": log[-25:]}
    return {"status": "compiled", "pdf": str(pdf), "warnings": warnings[:20]}


# --------------------------------------------------------------- notebook

FIG_CALL = {"models": "pr.fig_models(run, None, titles=True)",
            "confusion": "pr.fig_confusion(run, RUN_DIR, None, titles=True)",
            "holdout": "pr.fig_holdout(run, RUN_DIR, titles=True)",
            "importance": "pr.fig_importance(run, RUN_DIR, titles=True)",
            "gap": "pr.fig_gap(run, titles=True)",
            "variance": "pr.fig_variance(STRUCTURE_DIR, titles=True)",
            "robustness": "pr.fig_cluster_robustness(STRUCTURE_DIR, titles=True)",
            "structure": "pr.fig_structure(STRUCTURE_DIR, titles=True)",
            "associations": "pr.fig_associations(STRUCTURE_DIR, titles=True)",
            "distributions": "pr.fig_distributions(STRUCTURE_DIR, raw, titles=True)",
            "heatmap": "pr.fig_heatmap(STRUCTURE_DIR, raw, titles=True)"}


def build_notebook(src: rs.Source, ctx: rs.Context, num: rs.Numbering, figs: dict, nb_dir: Path, slug: str,
                   execute: bool) -> tuple[Path, dict]:
    def md(text):
        return rs.inline(text, "md", ctx.facts, num, src.references)

    meta = src.meta
    cells = []
    byline = f"**{meta['author']}**" + (f" · Student ID {meta['student_id']}" if meta.get("student_id") else "") + (
        f" · {meta['affiliation']}" if meta.get("affiliation") else "")
    cells.append(nbformat.v4.new_markdown_cell(f"# {md(meta['title'])}\n\n{byline}\n\n**Abstract.** {md(src.abstract)}"))
    paths = {"RUN_DIR": ctx.run_dir, "STRUCTURE_DIR": ctx.structure_dir, "CONFIG": ctx.config_path}
    setup = ["import io, sys", "sys.dont_write_bytecode = True  # keep the shared skill folder free of __pycache__",
             "from pathlib import Path", "import numpy as np", "import pandas as pd",
             "from IPython.display import Image, display", "",
             f'sys.path.insert(0, {str(SCRIPTS)!r})  # exam skill scripts',
             "import viz_style as vs, plot_results as pr, report_source as rs", "vs.apply_style()", ""]
    setup += [f'{k} = Path({str(v)!r})' if v else f"{k} = None" for k, v in paths.items()]
    setup += ["", "ctx = rs.load_context(RUN_DIR, STRUCTURE_DIR, CONFIG)  # reads finished runs; nothing is refitted",
              "run, raw = ctx.run, ctx.raw", "", "def show(result):",
              '    """Render a plot_results figure inline at print resolution."""',
              "    buf = io.BytesIO()",
              '    result["fig"].savefig(buf, format="png", dpi=170, bbox_inches="tight", pad_inches=0.03)',
              '    vs.plt.close(result["fig"])', "    display(Image(buf.getvalue(), width=820))"]
    cells.append(nbformat.v4.new_markdown_cell("## Setup\n\nPaths point to the finished runs this report describes. "
                                               "All numbers below are recomputed from those files."))
    cells.append(nbformat.v4.new_code_cell("\n".join(setup)))
    if ctx.raw is not None:
        cells.append(nbformat.v4.new_code_cell(
            "X, y = raw\nprint(f\"Samples: {X.shape[0]:,} | predictors: {X.shape[1]:,} | missing cells: "
            "{int(X.isna().to_numpy().sum()):,}\")\npd.Series(y, name=\"n\").value_counts().rename_axis(\"Class\").to_frame()"))
    buffer = []

    def flush():
        if buffer:
            cells.append(nbformat.v4.new_markdown_cell("\n\n".join(buffer)))
            buffer.clear()

    for b in src.blocks:
        if b.kind == "heading":
            if b.level == 1 and b.text.lower() == "appendix":
                buffer.append("## Appendix")
                continue
            hashes = "#" * (b.level + 1)
            anchor = f"<a id=\"sec-{b.label.split(':', 1)[-1]}\"></a>" if b.label else ""
            buffer.append(f"{hashes} {b.options.get('number', '')} {md(b.text)}{anchor}")
        elif b.kind == "para":
            buffer.append(md(b.text))
        elif b.kind == "list":
            buffer.append("\n".join(f"{i + 1}. {md(t)}" if b.ordered else f"- {md(t)}" for i, t in enumerate(b.items)))
        elif b.kind == "math":
            buffer.append("$$\n" + rs.fill(b.text, ctx.facts) + "\n$$")
        elif b.kind == "figure":
            flush()
            cells.append(nbformat.v4.new_code_cell(f"show({FIG_CALL[b.text]})"))
            f = figs[b.text]
            buffer.append(f"**Figure {num.figures[b.text]}.** {f['caption']}")
        elif b.kind == "table":
            t = rs.table(ctx, b.text)
            buffer.append(f"**Table {num.tables[b.text]}.** {t['caption']}")
            flush()
            cells.append(nbformat.v4.new_code_cell(f'rs.table_frame(ctx, "{b.text}")'))
        elif b.kind == "code":
            flush()
            cells.append(nbformat.v4.new_code_cell(b.text))
    if src.references:
        buffer.append("## References\n\n" + "\n".join(
            f"{i}. {md(t)}" for i, t in enumerate(src.references.values(), 1)))
    commands = (ctx.run["evidence"].get("command") if ctx.run else None) or []
    if commands:
        buffer.append("## Reproduction\n\nThe model comparison was produced by:\n\n```\n" + " ".join(map(str, commands)) + "\n```")
    flush()
    nb = nbformat.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
        "language_info": {"name": "python"}})
    status = {"status": "not_executed"}
    if execute:
        from nbclient import NotebookClient
        from jupyter_client import AsyncKernelManager
        from jupyter_client.kernelspec import KernelSpecManager
        try:
            # A private, temporary kernelspec uses this interpreter without modifying
            # the user's Jupyter installation or selecting another environment's Python.
            with tempfile.TemporaryDirectory(prefix="ama4602-kernel-") as kernel_root:
                spec_dir = Path(kernel_root) / "python3"
                spec_dir.mkdir()
                (spec_dir / "kernel.json").write_text(json.dumps({
                    "argv": [sys.executable, "-B", "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                    "display_name": "Python 3", "language": "python",
                }), encoding="utf-8")
                manager = AsyncKernelManager(kernel_name="python3", kernel_spec_manager=KernelSpecManager(
                    kernel_dirs=[kernel_root], ensure_native_kernel=False))
                NotebookClient(nb, km=manager, timeout=900, kernel_name="python3",
                               resources={"metadata": {"path": str(nb_dir)}}).execute(cleanup_kc=True)
            status = {"status": "executed", "python_executable": sys.executable}
        except Exception as exc:  # keep the partially executed notebook for inspection
            status = {"status": "execution_failed", "error": f"{type(exc).__name__}: {str(exc)[-800:]}"}
    path = nb_dir / f"{slug}.ipynb"
    nbformat.write(nb, path)
    return path, status


# -------------------------------------------------------------------- HTML

HTML_CSS = """
:root{--ink:#1F1F1E;--ink2:#5E5D59;--ink3:#8C8B86;--rule:#E6E4DD;--accent:#D97757;--tint:#F0EEE6;--bg:#FAF9F5;
  --sans:"Libre Franklin","Helvetica Neue",Helvetica,Arial,sans-serif;--serif:Georgia,"Iowan Old Style","Times New Roman",serif}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:17px/1.7 var(--serif);-webkit-font-smoothing:antialiased}
header h1,h2.sec,h3.sec,h4.sec,nav.toc,.byline,.meta,.abstract h2,figcaption,caption,table,summary,footer{font-family:var(--sans)}
.layout{display:grid;grid-template-columns:230px minmax(0,820px);gap:56px;max-width:1160px;margin:0 auto;padding:56px 24px 96px}
nav.toc{position:sticky;top:40px;align-self:start;font-size:13.5px;line-height:1.5}
nav.toc .label{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--ink3);margin-bottom:10px;font-weight:500}
nav.toc a{display:block;color:var(--ink2);text-decoration:none;padding:4px 0 4px 12px;border-left:2px solid var(--rule)}
nav.toc a.l2{padding-left:24px;font-size:13px}
nav.toc a:hover{color:var(--accent);border-left-color:var(--accent)}
header h1{font-size:34px;line-height:1.2;font-weight:700;letter-spacing:-.01em;margin:0 0 14px}
.byline{color:var(--ink2);font-size:15px}.byline strong{color:var(--ink);font-weight:500}
.meta{display:flex;flex-wrap:wrap;gap:8px;margin:18px 0 28px}
.chip{font-size:12.5px;color:var(--ink2);border:1px solid var(--rule);border-radius:999px;padding:3px 11px}
.abstract{background:var(--tint);border-radius:10px;padding:20px 24px;margin:0 0 36px}
.abstract h2{font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--accent);margin:0 0 6px;font-weight:600}
.abstract p{margin:0}
h2.sec{font-size:24px;font-weight:700;margin:48px 0 12px;letter-spacing:-.005em}
h3.sec{font-size:19px;font-weight:600;margin:32px 0 8px}h4.sec{font-size:16px;font-weight:600;margin:24px 0 6px}
.num{color:var(--ink3);font-weight:500;margin-right:10px}
p{margin:0 0 14px}a{color:var(--accent)}
figure{margin:30px 0 34px}figure img{width:100%;height:auto;display:block}
figcaption{font-size:14px;line-height:1.55;color:var(--ink2);margin-top:10px}
figcaption b{color:var(--ink)}
.tablewrap{overflow-x:auto;margin:26px 0 30px}
table{border-collapse:collapse;width:100%;font-size:14.5px;font-variant-numeric:tabular-nums}
caption{caption-side:top;text-align:left;font-size:14px;color:var(--ink2);margin-bottom:10px;line-height:1.5}
caption b{color:var(--ink)}
thead th{border-top:1.5px solid var(--ink);border-bottom:1px solid var(--ink);font-weight:600;padding:7px 12px;text-align:left}
tbody td{padding:6px 12px;border-bottom:1px solid var(--rule)}tbody tr:last-child td{border-bottom:1.5px solid var(--ink)}
td.r,th.r{text-align:right}tr.best td{font-weight:600}
details{margin:14px 0;border:1px solid var(--rule);border-radius:8px;padding:8px 14px}
summary{cursor:pointer;color:var(--ink2);font-size:14px}
pre{background:#F0EEE6;border-radius:6px;padding:12px;overflow-x:auto;font-size:13px}
code{font-family:"SF Mono",Menlo,Consolas,monospace;font-size:.88em}
ol.refs{font-size:14px;color:var(--ink2);padding-left:22px}ol.refs li{margin-bottom:6px}
.math-display{overflow-x:auto;margin:14px 0}
footer{margin-top:56px;padding-top:18px;border-top:1px solid var(--rule);font-size:12.5px;color:var(--ink3)}
@media (max-width:980px){.layout{grid-template-columns:minmax(0,1fr);gap:0;padding:28px 16px 64px}nav.toc{display:none}
  header h1{font-size:27px}}
@media print{nav.toc{display:none}.layout{display:block;padding:0}}
"""


def build_html(src: rs.Source, ctx: rs.Context, num: rs.Numbering, figs: dict, png: dict, html_dir: Path,
               slug: str) -> Path:
    import plotly.offline
    import report_html as rh

    def h(text):
        return rs.inline(text, "html", ctx.facts, num, src.references)

    interactive = {
        "models": lambda: rh.models(ctx.run), "confusion": lambda: rh.confusion(ctx.run, ctx.run_dir),
        "structure": lambda: rh.structure(ctx.structure_dir), "associations": lambda: rh.associations(ctx.structure_dir),
        "distributions": lambda: rh.distributions(ctx.structure_dir, ctx.raw)}
    meta = src.meta
    toc, body = [], []
    for b in src.blocks:
        if b.kind == "heading":
            if b.level == 1 and b.text.lower() == "appendix":
                body.append('<h2 class="sec">Appendix</h2>')
                toc.append('<a href="#appendix">Appendix</a>')
                continue
            anchor = f"sec-{b.label.split(':', 1)[-1]}" if b.label else f"h-{len(toc)}"
            n = b.options.get("number", "")
            body.append(f'<h{b.level + 1} class="sec" id="{anchor}"><span class="num">{n}</span>{h(b.text)}</h{b.level + 1}>')
            if b.level <= 2:
                toc.append(f'<a class="l{b.level}" href="#{anchor}">{n}&nbsp; {h(b.text)}</a>')
        elif b.kind == "para":
            body.append(f"<p>{h(b.text)}</p>")
        elif b.kind == "list":
            tag = "ol" if b.ordered else "ul"
            body.append(f"<{tag}>" + "".join(f"<li>{h(i)}</li>" for i in b.items) + f"</{tag}>")
        elif b.kind == "math":
            body.append(f'<div class="math-display">$${html.escape(rs.fill(b.text, ctx.facts))}$$</div>')
        elif b.kind == "figure":
            f = figs[b.text]
            builder = interactive.get(b.text)
            built = builder() if builder else None
            plot, caption = built if built else (None, None)
            caption = caption or f["caption"]  # interactive views show fewer panels, so they carry their own caption
            if plot is not None:
                inner = plot.to_html(full_html=False, include_plotlyjs=False,
                                     config={"displaylogo": False, "responsive": True,
                                             "toImageButtonOptions": {"format": "png", "scale": 3}})
            else:
                inner = f'<img alt="{html.escape(f["title"])}" src="data:image/png;base64,{base64.b64encode(png[b.text]).decode()}">'
            body.append(f'<figure id="fig-{b.text}">{inner}<figcaption><b>Figure {num.figures[b.text]}. '
                        f'{html.escape(f["title"])}.</b> {html.escape(caption)}</figcaption></figure>')
        elif b.kind == "table":
            t = rs.table(ctx, b.text)
            aligns = ["r" if a == "r" else "" for a in t["align"]]
            head = "".join(f'<th class="{a}">{html.escape(c)}</th>' for a, c in zip(aligns, t["columns"]))
            rows = "".join(
                f'<tr class="{"best" if i in t["bold"] else ""}">' +
                "".join(f'<td class="{a}">{html.escape(c)}</td>' for a, c in zip(aligns, r)) + "</tr>"
                for i, r in enumerate(t["rows"]))
            body.append(f'<div class="tablewrap" id="tab-{b.text}"><table><caption><b>Table {num.tables[b.text]}.</b> '
                        f'{html.escape(t["caption"])}</caption><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>')
        elif b.kind == "code":
            body.append(f"<details><summary>Show code</summary><pre><code>{html.escape(b.text)}</code></pre></details>")
    if src.references:
        toc.append('<a class="l1" href="#references">References</a>')
        body.append('<h2 class="sec" id="references">References</h2><ol class="refs">' + "".join(
            f'<li id="ref-{k}">{h(t)}</li>' for k, t in src.references.items()) + "</ol>")
    chips = [f"Student ID {html.escape(meta['student_id'])}"] if meta.get("student_id") else []
    if ctx.run:
        chips.append(f"Run {html.escape(ctx.run_dir.name)}")
    chips.append(f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    provenance = []
    if ctx.run_dir:
        ev = ctx.run_dir / "evidence.json"
        provenance.append(f"evidence.json: {ev} ({ev.stat().st_size} bytes)")
        provenance.append(", ".join(f"{k} {v}" for k, v in ctx.run["evidence"]["versions"].items()))
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(meta['title'])}</title>
<style>{HTML_CSS}</style>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Libre+Franklin:wght@400;500;600;700&display=swap">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"
 onload="renderMathInElement(document.body,{{delimiters:[{{left:'$$',right:'$$',display:true}},{{left:'$',right:'$',display:false}}],ignoredClasses:['js-plotly-plot']}})"></script>
<script>{plotly.offline.get_plotlyjs()}</script>
</head><body><div class="layout">
<nav class="toc"><div class="label">Contents</div>{''.join(toc)}</nav>
<main><header><h1>{h(meta['title'])}</h1>
<div class="byline"><strong>{html.escape(meta['author'])}</strong>{(' · ' + html.escape(meta['affiliation'])) if meta.get('affiliation') else ''}</div>
<div class="meta">{''.join(f'<span class="chip">{c}</span>' for c in chips)}</div></header>
<section class="abstract"><h2>Abstract</h2><p>{h(src.abstract)}</p></section>
{''.join(body)}
<footer>Figures are interactive: hover for exact values, use the buttons to switch views, drag to zoom and double-click to reset.
All numbers are read from the run outputs. {' · '.join(html.escape(p) for p in provenance)}</footer>
</main></div></body></html>"""
    path = html_dir / f"{slug}.html"
    path.write_text(page, encoding="utf-8")
    return path


# -------------------------------------------------------------------- main

def unreplaced_markers(source: Path) -> list[str]:
    """REPLACE_* markers left from a template, ignoring front-matter comment lines."""
    found, in_front = [], False
    for number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip() == "---" and (number == 1 or in_front):
            in_front = number == 1  # opens on line 1, closes at the next ---
            continue
        if in_front and line.lstrip().startswith("#"):
            continue
        found += [f"{number}: {m}" for m in re.findall(r"REPLACE_[A-Z0-9_]+", line)]
    return found


def resolve(base: Path, value: str | None) -> Path | None:
    if not value:
        return None
    p = Path(value).expanduser()
    return (p if p.is_absolute() else base / p).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True, help="Report Markdown source (front matter + sections).")
    parser.add_argument("--out-root", type=Path, help="Folder containing tex/, jupyter/ and html/ (default: front matter out_root).")
    parser.add_argument("--formats", default=",".join(FORMATS), help="Comma-separated subset of tex,ipynb,html.")
    parser.add_argument("--run", help="Override the run directory from the front matter.")
    parser.add_argument("--structure", help="Override the structure directory.")
    parser.add_argument("--config", help="Override the exam config (needed for raw-data figures/facts).")
    parser.add_argument("--slug", help="Output file stem (default: front matter slug or source file stem).")
    parser.add_argument("--neurips-sty", type=Path, default=NEURIPS_STY,
                        help="Style file copied next to the .tex (default: the skill's assets/neurips_2026).")
    parser.add_argument("--list-facts", action="store_true", help="Print the available {{facts}} and exit.")
    parser.add_argument("--no-compile", action="store_true", help="Write the .tex but do not run tectonic.")
    parser.add_argument("--no-execute", action="store_true", help="Write the notebook without executing it.")
    parser.add_argument("--overwrite", action="store_true", help="Replace this slug's existing report files.")
    args = parser.parse_args()

    src = rs.parse_source(args.source)
    base = args.source.resolve().parent
    meta = src.meta
    # Command-line overrides resolve against the current directory, front-matter paths against the source file.
    cwd = Path.cwd()
    ctx = rs.load_context(resolve(cwd, args.run) if args.run else resolve(base, meta.get("run")),
                          resolve(cwd, args.structure) if args.structure else resolve(base, meta.get("structure")),
                          resolve(cwd, args.config) if args.config else resolve(base, meta.get("config")))
    if args.list_facts:
        print(json.dumps(ctx.facts, indent=2, ensure_ascii=False))
        return 0
    leftovers = unreplaced_markers(args.source)
    if leftovers:
        print(f"ERROR: template markers still present (line: marker): {leftovers[:8]}. Replace them before building.",
              file=sys.stderr)
        return 2
    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    if set(formats) - set(FORMATS):
        parser.error(f"Unknown formats {sorted(set(formats) - set(FORMATS))}")
    out_root = (args.out_root or resolve(base, meta.get("out_root")) or base).resolve()
    slug = args.slug or meta.get("slug") or args.source.stem
    dirs = {"tex": out_root / "tex", "ipynb": out_root / "jupyter", "html": out_root / "html"}
    targets = {"tex": [dirs["tex"] / f"{slug}.tex", dirs["tex"] / f"{slug}.pdf", dirs["tex"] / f"{slug}_figures"],
               "ipynb": [dirs["ipynb"] / f"{slug}.ipynb"], "html": [dirs["html"] / f"{slug}.html"]}
    existing = [str(p) for f in formats for p in targets[f] if p.exists()]
    if existing and not args.overwrite:
        parser.error("Report files already exist (pass --overwrite to replace this slug): " + ", ".join(existing))

    vs.apply_style()
    num = rs.number(src)
    # Validate every prose string, citation and cross-reference before writing anything.
    for text in [meta["title"], src.abstract] + [b.text for b in src.blocks if b.kind in ("heading", "para")] + \
            [i for b in src.blocks if b.kind == "list" for i in b.items] + list(src.references.values()):
        rs.inline(text, "md", ctx.facts, num, src.references)
    cited = {k for t in [src.abstract] + [b.text for b in src.blocks] + [i for b in src.blocks for i in b.items]
             for k in rs.cited_keys(t)}
    uncited = [k for k in src.references if k not in cited]
    tables = [b.text for b in src.blocks if b.kind == "table"]
    for name in tables:
        rs.table(ctx, name)

    figure_names = list(dict.fromkeys(b.text for b in src.blocks if b.kind == "figure"))
    figs, png, summary = {}, {}, {"slug": slug, "created_utc": datetime.now(timezone.utc).isoformat(),
                                  "source": str(args.source.resolve()), "facts": ctx.facts, "outputs": {}}
    if "tex" in formats:
        dirs["tex"].mkdir(parents=True, exist_ok=True)
        rs.need(args.neurips_sty.exists(), f"{args.neurips_sty} not found; pass --neurips-sty")
        shutil.copy2(args.neurips_sty, dirs["tex"] / "neurips_2026.sty")
        fig_dir = dirs["tex"] / f"{slug}_figures"
        if fig_dir.exists():
            shutil.rmtree(fig_dir)
        for name in figure_names:
            result = make_figure(name, ctx, titles=False)
            figs[name] = {"stem": result["stem"], "title": result["title"], "caption": result["caption"]}
            vs.save_figure(result["fig"], fig_dir, result["stem"])
    for name in figure_names:
        result = make_figure(name, ctx, titles=True)
        figs.setdefault(name, {"stem": result["stem"], "title": result["title"], "caption": result["caption"]})
        png[name] = png_bytes(result["fig"])

    if "tex" in formats:
        tex_path = build_tex(src, ctx, num, figs, dirs["tex"], slug)
        summary["outputs"]["tex"] = {"tex": str(tex_path)}
        summary["outputs"]["tex"].update({"status": "not_compiled"} if args.no_compile else compile_tex(tex_path))
    if "ipynb" in formats:
        dirs["ipynb"].mkdir(parents=True, exist_ok=True)
        path, status = build_notebook(src, ctx, num, figs, dirs["ipynb"], slug, not args.no_execute)
        summary["outputs"]["ipynb"] = {"notebook": str(path), **status}
    if "html" in formats:
        dirs["html"].mkdir(parents=True, exist_ok=True)
        summary["outputs"]["html"] = {"html": str(build_html(src, ctx, num, figs, png, dirs["html"], slug))}
    if uncited:
        summary["warnings"] = [f"Reference listed but never cited: {k}" for k in uncited]
    vs.write_json(out_root / f".{slug}_build.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "facts"}, indent=2, ensure_ascii=False))
    failed = [f for f, o in summary["outputs"].items() if o.get("status") in ("failed", "execution_failed")]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
