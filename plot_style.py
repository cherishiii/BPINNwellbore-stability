"""
Global plotting style for this project (paper/journal friendly).

Goal
----
- One place to control font family, font sizes, line widths, tick styles, and save defaults.
- Default font: Times New Roman, 14 pt, bold.
- Math font: STIX (close to journal look while staying Matplotlib-native).

Usage
-----
from plot_style import set_paper_style
set_paper_style()
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def set_paper_style(
    *,
    base_fontsize: float = 14.0,
    dpi: int = 600,
    seaborn_style: str = "white",
    use_seaborn: bool = True,
    allow_cjk_fallback: bool = True,
) -> Dict[str, Any]:
    """
    Apply a consistent Matplotlib (and optional Seaborn) style.

    Parameters
    ----------
    base_fontsize:
        Base font size in points (pt). This controls rcParams['font.size'] and derived sizes.
    dpi:
        Default save/display DPI for raster outputs.
    seaborn_style:
        Seaborn style name (e.g. 'white', 'ticks').
    use_seaborn:
        If True and seaborn is installed, call seaborn.set_theme with our rc overrides.
    allow_cjk_fallback:
        If True, keep a CJK-capable fallback font at the end of the serif list so Chinese
        text does not render as tofu. Latin/Greek will still prefer Times New Roman.

    Returns
    -------
    The rcParams dict applied (useful for debugging/tests).
    """
    import matplotlib as mpl
    import matplotlib.font_manager as fm

    serif_stack = ["Times New Roman", "Times", "DejaVu Serif"]
    if allow_cjk_fallback:
        serif_stack += ["SimSun", "SimHei", "Microsoft YaHei"]

    rc: Dict[str, Any] = {
        # Fonts
        "font.family": "serif",
        "font.serif": serif_stack,
        "font.weight": "bold",
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        # Core sizing (pt) — tick/legend enlarged for readability
        "font.size": base_fontsize,
        "axes.titlesize": base_fontsize + 2,
        "axes.labelsize": base_fontsize,
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
        "xtick.labelsize": base_fontsize,
        "ytick.labelsize": base_fontsize,
        "legend.fontsize": base_fontsize,
        "legend.title_fontsize": base_fontsize,
        "figure.titlesize": base_fontsize + 2,
        "figure.titleweight": "bold",
        # Lines / axes
        "axes.linewidth": 1.4,
        "lines.linewidth": 1.8,
        "lines.markersize": 6.0,
        # Ticks (journal-like, always visible)
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 5.0,
        "ytick.major.size": 5.0,
        "xtick.minor.size": 3.0,
        "ytick.minor.size": 3.0,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "xtick.minor.width": 1.0,
        "ytick.minor.width": 1.0,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        # Grid defaults (off by default; individual plots can enable if needed)
        "axes.grid": False,
        # Save defaults
        "savefig.dpi": dpi,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        # Embed TrueType fonts in vector outputs (safer for submission)
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }

    mpl.rcParams.update(rc)

    if use_seaborn:
        try:
            import seaborn as sns  # type: ignore

            sns.set_theme(style=seaborn_style, rc=rc)
        except Exception:
            pass

    return rc


def apply_ax_style(ax, xlabel: str = None, ylabel: str = None, title: str = None,
                   legend: bool = True, grid: bool = True, fontsize: float = 14.0):
    """
    Apply unified axis formatting to a single Axes object.
    Ensures tick labels are visible with proper font.
    """
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=fontsize, fontweight='bold')
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=fontsize, fontweight='bold')
    if title:
        ax.set_title(title, fontsize=fontsize + 2, fontweight='bold')

    ax.tick_params(axis='both', which='major', labelsize=fontsize,
                   width=1.2, length=5, direction='in',
                   top=True, right=True)
    ax.tick_params(axis='both', which='minor', width=1.0, length=3,
                   direction='in', top=True, right=True)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontfamily('serif')

    if grid:
        ax.grid(True, alpha=0.3, linestyle='--')

    if legend and ax.get_legend_handles_labels()[1]:
        ax.legend(fontsize=fontsize, prop={'weight': 'bold', 'family': 'serif'},
                  framealpha=0.9, edgecolor='gray')


