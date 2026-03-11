"""
utils.py - Plot utilities with careful margin/layout handling.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.figure import Figure


def hex_to_rgb_int(hex_code: str) -> tuple[int, int, int]:
    h = hex_code.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def is_dark(hex_code: str) -> bool:
    r, g, b = hex_to_rgb_int(hex_code)
    return (0.299 * r + 0.587 * g + 0.114 * b) < 140


BG   = "#0c0c0c"
BG2  = "#161616"
GRID = "#252525"
TEXT = "#c8c8c0"
DIM  = "#606060"


def _style_ax(ax, fig):
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG2)
    ax.tick_params(colors=DIM, labelsize=8, length=3)
    ax.spines[:].set_visible(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.xaxis.label.set_color(DIM)
    ax.yaxis.label.set_color(DIM)
    ax.title.set_color(TEXT)


def plot_weights(w: np.ndarray, feature_names: list[str]) -> Figure:
    """
    Horizontal bar chart of learned feature weights.
    Fixed: uses constrained_layout + explicit left margin for long labels.
    """
    n = len(feature_names)
    fig, ax = plt.subplots(figsize=(6.5, n * 0.52 + 0.8),
                           layout="constrained")
    _style_ax(ax, fig)

    bar_colors = ["#4ade80" if v >= 0 else "#f87171" for v in w]
    bars = ax.barh(
        feature_names, w,
        color=bar_colors, height=0.58,
        edgecolor="none", zorder=3
    )

    # Zero line
    ax.axvline(0, color=GRID, linewidth=0.8, zorder=2)

    # Grid lines (vertical only, subtle)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(axis="x", color=GRID, linewidth=0.4, zorder=1)

    ax.set_xlabel("Weight value", color=DIM, fontsize=8, labelpad=6)
    ax.set_title("Learned Preference Weights", color=TEXT, fontsize=10,
                 fontweight="bold", pad=10)
    ax.tick_params(axis="y", labelsize=8.5, colors=TEXT)
    ax.tick_params(axis="x", labelsize=7.5, colors=DIM)

    # Value labels — placed just outside each bar
    x_range = max(abs(w.max()), abs(w.min()), 0.01)
    offset = x_range * 0.04
    for bar, val in zip(bars, w):
        xpos = val + (offset if val >= 0 else -offset)
        ha = "left" if val >= 0 else "right"
        ax.text(xpos, bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}", va="center", ha=ha,
                color="#e8e8e0", fontsize=7.5, zorder=4)

    # Expand x-axis slightly so labels don't clip
    xlim = ax.get_xlim()
    pad = (xlim[1] - xlim[0]) * 0.18
    ax.set_xlim(xlim[0] - pad, xlim[1] + pad)

    return fig


def plot_top_colors(
    scores: np.ndarray,
    probs: np.ndarray,
    names: list[str],
    hexes: list[str],
    top_n: int = 10
) -> Figure:
    """Horizontal bars filled with the actual color, labeled with prob %."""
    ranked = np.argsort(scores)[::-1][:top_n]
    top_names  = [names[i]  for i in ranked]
    top_scores = [scores[i] for i in ranked]
    top_probs  = [probs[i]  for i in ranked]
    top_hexes  = [hexes[i]  for i in ranked]

    fig, ax = plt.subplots(figsize=(6.5, top_n * 0.55 + 0.8),
                           layout="constrained")
    _style_ax(ax, fig)

    y_pos = list(range(top_n))[::-1]
    for y, score, prob, hex_c, name in zip(
        y_pos, top_scores, top_probs, top_hexes, top_names
    ):
        ax.barh(y, score, color=hex_c, height=0.62,
                edgecolor="none", zorder=3)
        # Probability annotation
        ax.text(score + 0.005, y, f"{prob*100:.1f}%",
                va="center", ha="left", fontsize=7, color=TEXT, zorder=4)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names, color=TEXT, fontsize=8.5)
    ax.set_xlabel("Preference score  (w · x)", color=DIM, fontsize=8, labelpad=6)
    ax.set_title("Top Predicted Colors", color=TEXT, fontsize=10,
                 fontweight="bold", pad=10)
    ax.axvline(0, color=GRID, linewidth=0.6, zorder=2)
    ax.grid(axis="x", color=GRID, linewidth=0.4, zorder=1)
    ax.tick_params(axis="x", labelsize=7.5, colors=DIM)

    # Pad right for probability labels
    xl = ax.get_xlim()
    ax.set_xlim(xl[0], xl[1] + (xl[1] - xl[0]) * 0.12)

    return fig


def plot_hue_brightness_scatter(
    features: np.ndarray,
    names: list[str],
    hexes: list[str],
    scores: np.ndarray | None = None,
) -> Figure:
    """Scatter: hue (x) vs brightness (y). Dot size ∝ preference score."""
    fig, ax = plt.subplots(figsize=(6.5, 4.2), layout="constrained")
    _style_ax(ax, fig)

    hues   = features[:, 0]
    bright = features[:, 2]

    if scores is not None and scores.max() - scores.min() > 1e-9:
        sizes = 40 + 200 * (scores - scores.min()) / (scores.max() - scores.min())
    else:
        sizes = np.full(len(names), 80)

    for h, b, hex_c, sz in zip(hues, bright, hexes, sizes):
        ax.scatter(h, b, s=sz, color=hex_c,
                   edgecolors="#2a2a2a", linewidths=0.4, zorder=3, alpha=0.9)

    ax.set_xlabel("Hue  (0 = red  ·  0.33 = green  ·  0.67 = blue)",
                  color=DIM, fontsize=7.5, labelpad=6)
    ax.set_ylabel("Brightness", color=DIM, fontsize=7.5, labelpad=6)
    ax.set_title("Color Space Map  —  size reflects preference score",
                 color=TEXT, fontsize=9, fontweight="bold", pad=10)
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(-0.06, 1.06)
    ax.grid(color=GRID, linewidth=0.4, zorder=1)

    return fig