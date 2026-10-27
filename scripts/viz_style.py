"""Single source of truth for the exam figure style (palette, type, sizes, export).

Import from a script or notebook:

    import sys; sys.path.insert(0, "/path/to/ama4602-kaggle-exam/scripts")
    from viz_style import apply_style, CATEGORICAL, save_figure

Every value here was chosen for print-size academic figures (NeurIPS/ICML/Nature
column widths) and the categorical order was validated for colour-vision safety;
see references/visualization.md before changing it.
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

if not os.environ.get("MPLCONFIGDIR"):
    # A stable writable cache: the font cache is built once, not on every run.
    _cache = Path(tempfile.gettempdir()) / "ama4602-mpl"
    _cache.mkdir(exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(_cache)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.transforms import offset_copy

# Anthropic-style palette, sampled from transformer-circuits.pub articles (ivory
# panels, warm greys, Claude clay accent, orange activation ramp) and widened to
# six earth tones that pass the colour-vision validator on the ivory surface:
# adjacent CVD dE >= 10.2, normal-vision dE >= 17.9, first three pass all-pairs.
# Scatter plots always add marker shapes; clay and ochre sit just under 3:1
# contrast, so they always appear with a legend or direct label.
CATEGORICAL = ["#D97757", "#4F8FCF", "#A65A8C", "#6B8E3A", "#6C63B8", "#C08A2A"]
CATEGORICAL_NAMES = ["clay", "sky", "fig", "olive", "violet", "ochre"]
MARKERS = ["o", "s", "^", "D", "v", "P"]

INK = "#1F1F1E"          # titles, primary text (Anthropic near-black)
INK_2 = "#5E5D59"        # subtitles, axis labels, panel captions
INK_3 = "#8C8B86"        # tick labels, notes
RULE = "#B9B7AE"         # axis spines (warm)
GRID = "#E6E4DD"         # hairline grid and borders (warm)
NEUTRAL = "#A8A69D"      # de-emphasised series: grey first, clay for the one that matters
NEUTRAL_LIGHT = "#D9D7CE"
ACCENT = CATEGORICAL[0]  # Claude clay
SURFACE = "#FAF9F5"      # ivory figure background
BAND = ("#F0EFEB", "#E5E5E1")  # warm grey shading for regions/regimes

# Sequential magnitude: the orange "feature activation" ramp. Diverging: sky <->
# warm grey <-> clay, with a neutral midpoint that reads as "nothing".
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "ama_seq", ["#FBF6EF", "#FDE3CB", "#FBC293", "#F39A62", "#E2703A", "#C4511B"])
DIVERGING = LinearSegmentedColormap.from_list(
    "ama_div", ["#2A5E95", "#4F8FCF", "#A9C8E8", "#F0EFEB", "#F2C1A8", "#D97757", "#B5502A"])
SEQUENTIAL.set_bad("#F0EFEB")
DIVERGING.set_bad("#B9B7AE")

# Physical sizes in inches: design at final size, never rescale afterwards.
# FULL_WIDTH is the NeurIPS text width, so \includegraphics[width=\linewidth]
# places figures at 100% and the font sizes below are the printed sizes.
FULL_WIDTH = 5.5
HALF_WIDTH = 2.7

FONT_STACK = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]
# Type roles, all weights shipped with macOS Helvetica Neue: figure titles Bold
# (700), panel titles and axis titles Medium (500), ticks/legends/notes Regular
# (400) in dark grey, math in STIX.
TITLE_WEIGHT = "bold"
LABEL_WEIGHT = "medium"
# Font sizes in points at final print size (NeurIPS body text is 10 pt).
FS = {"title": 11, "subtitle": 8.5, "panel": 9.5, "label": 9, "body": 8.5, "note": 8, "tiny": 7.5}

RC = {
    "font.family": "sans-serif",
    "font.sans-serif": FONT_STACK,
    "font.size": FS["body"],
    "font.weight": "normal",
    "mathtext.fontset": "stix",
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.labelsize": FS["label"],
    "axes.labelweight": LABEL_WEIGHT,
    "axes.titlesize": FS["panel"],
    "axes.titleweight": LABEL_WEIGHT,
    "axes.titlelocation": "left",
    "axes.titlepad": 6,
    "axes.titlecolor": INK,
    "axes.edgecolor": RULE,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "axes.axisbelow": True,
    "axes.facecolor": SURFACE,
    "axes.prop_cycle": matplotlib.cycler(color=CATEGORICAL),
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "xtick.color": RULE,
    "ytick.color": RULE,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "xtick.labelsize": FS["note"],
    "ytick.labelsize": FS["note"],
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.pad": 2.5,
    "ytick.major.pad": 2.5,
    "lines.linewidth": 1.5,
    "lines.markersize": 4,
    "patch.linewidth": 0,
    "legend.frameon": False,
    "legend.fontsize": FS["note"],
    "legend.handletextpad": 0.3,
    "legend.borderaxespad": 0.2,
    "legend.labelcolor": INK,
    "figure.facecolor": SURFACE,
    "figure.dpi": 150,
    "figure.constrained_layout.use": True,
    "figure.constrained_layout.h_pad": 0.04,
    "figure.constrained_layout.w_pad": 0.04,
    "savefig.facecolor": SURFACE,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "image.cmap": "ama_seq",
}


def apply_style() -> None:
    """Apply the exam style globally (call once per process or notebook)."""
    for cmap in (SEQUENTIAL, DIVERGING):
        if cmap.name not in matplotlib.colormaps:
            matplotlib.colormaps.register(cmap)
    plt.rcParams.update(RC)


def category_style(index: int) -> dict:
    """Colour and marker for categorical slot `index`; colour is never reused."""
    if index >= len(CATEGORICAL):
        raise ValueError(f"More than {len(CATEGORICAL)} categories: fold the tail into 'Other' or use small multiples.")
    return {"color": CATEGORICAL[index], "marker": MARKERS[index]}


def header(fig, title: str | None, subtitle: str | None = None) -> None:
    """Left-aligned English title (the finding) above a grey subtitle (method).

    Both sit just above the figure canvas; save_figure's tight bbox expands the
    page to include them, so they never collide with panels or panel labels.
    """
    if not title:
        return
    y = 5.0
    if subtitle:
        fig.text(0.0, 1.0, subtitle, ha="left", va="bottom", fontsize=FS["subtitle"], color=INK_2,
                 transform=offset_copy(fig.transFigure, fig=fig, y=y, units="points"))
        y += FS["subtitle"] + 5.0
    fig.text(0.0, 1.0, title, ha="left", va="bottom", fontsize=FS["title"], fontweight=TITLE_WEIGHT,
             color=INK, transform=offset_copy(fig.transFigure, fig=fig, y=y, units="points"))


def panel_title(ax, text: str) -> None:
    """Small upper-case grey panel caption, as in transformer-circuits figures."""
    ax.set_title(text.upper(), fontsize=FS["tiny"], fontweight=LABEL_WEIGHT, color=INK_2, loc="left", pad=6)


def panel_label(ax, letter: str) -> None:
    """Bold lower-case panel tag outside the axes, e.g. '(a)'."""
    ax.text(-0.02, 1.02, f"({letter})", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=FS["panel"] + 0.5, fontweight="bold", color=INK)


def soften(ax, grid_axis: str | None = None) -> None:
    """Recessive axes: no top/right spines, optional hairline grid on one axis."""
    ax.spines[["top", "right"]].set_visible(False)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.5)


def save_figure(fig, out_dir: Path, stem: str, provenance: dict | None = None) -> dict:
    """Write vector PDF + 300 dpi PNG atomically; refuse to overwrite."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for ext in ("pdf", "png"):
        final = out_dir / f"{stem}.{ext}"
        if final.exists():
            raise FileExistsError(f"{final} exists; write figures to a new run directory.")
        tmp = out_dir / f".{stem}.{os.getpid()}.tmp.{ext}"
        fig.savefig(tmp, format=ext, bbox_inches="tight", pad_inches=0.03)
        os.replace(tmp, final)
        written[ext] = str(final)
    record = {"stem": stem, "files": written,
              "size_inches": [round(v, 3) for v in fig.get_size_inches()],
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "provenance": provenance or {}}
    plt.close(fig)
    return record


def write_json(path: Path, obj) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(tmp, path)
