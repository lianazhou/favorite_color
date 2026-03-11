"""
utils.py - Small helper utilities for display, formatting, and plot generation.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure


def hex_to_rgb_int(hex_code: str) -> tuple[int, int, int]:
    """Return (R, G, B) as 0–255 integers."""
    h = hex_code.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def is_dark(hex_code: str) -> bool:
    """Return True if the color is dark (so we should use white text on it)."""
    r, g, b = hex_to_rgb_int(hex_code)
    # Perceived luminance (standard formula)
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return luminance < 140


def format_weight_table(w: np.ndarray, feature_names: list[str]) -> dict:
    """Return a dict {feature: weight_value} for display."""
    return {name: float(w[i]) for i, name in enumerate(feature_names)}


# ── Matplotlib plots ──────────────────────────────────────────────────────────

def plot_weights(w: np.ndarray, feature_names: list[str]) -> Figure:
    """
    Horizontal bar chart of learned feature weights.
    Positive = user likes this feature; negative = dislikes it.
    """
    fig, ax = plt.subplots(figsize=(6, 3))
    fig.patch.set_facecolor("#0f0f0f")
    ax.set_facecolor("#1a1a1a")

    colors = ["#4ade80" if v >= 0 else "#f87171" for v in w]
    bars = ax.barh(feature_names, w, color=colors, height=0.55, edgecolor="none")

    ax.axvline(0, color="#555", linewidth=0.8)
    ax.set_xlabel("Weight", color="#aaa", fontsize=9)
    ax.set_title("Learned Preference Weights", color="#eee", fontsize=10, pad=8)
    ax.tick_params(colors="#bbb", labelsize=8)
    ax.spines[:].set_visible(False)

    # Annotate values
    for bar, val in zip(bars, w):
        xpos = val + (0.01 if val >= 0 else -0.01)
        ha   = "left" if val >= 0 else "right"
        ax.text(xpos, bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}", va="center", ha=ha, color="#ddd", fontsize=7.5)

    fig.tight_layout()
    return fig


def plot_top_colors(
    scores: np.ndarray,
    names: list[str],
    hexes: list[str],
    top_n: int = 8
) -> Figure:
    """
    Horizontal bar chart with actual color fills for the top-N ranked colors.
    """
    ranked = np.argsort(scores)[::-1][:top_n]
    top_names  = [names[i] for i in ranked]
    top_scores = [scores[i] for i in ranked]
    top_hexes  = [hexes[i]  for i in ranked]

    fig, ax = plt.subplots(figsize=(6, top_n * 0.52 + 0.6))
    fig.patch.set_facecolor("#0f0f0f")
    ax.set_facecolor("#1a1a1a")

    y_pos = list(range(top_n))[::-1]
    for y, score, hex_c in zip(y_pos, top_scores, top_hexes):
        ax.barh(y, score, color=hex_c, height=0.65, edgecolor="none")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names, color="#ddd", fontsize=8)
    ax.set_xlabel("Preference Score (w · x)", color="#aaa", fontsize=8)
    ax.set_title("Top Predicted Colors", color="#eee", fontsize=10, pad=8)
    ax.tick_params(axis="x", colors="#888", labelsize=7)
    ax.spines[:].set_visible(False)
    ax.axvline(0, color="#444", linewidth=0.6)

    fig.tight_layout()
    return fig


def plot_hue_brightness_scatter(
    features: np.ndarray,
    names: list[str],
    hexes: list[str],
    scores: np.ndarray | None = None,
) -> Figure:
    """
    Scatter plot: hue (x) vs brightness (y), colored by the actual hex.
    Dot size optionally scaled by preference score.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor("#0f0f0f")
    ax.set_facecolor("#1a1a1a")

    hues   = features[:, 0]
    bright = features[:, 2]

    if scores is not None:
        s_min, s_max = scores.min(), scores.max()
        sizes = 60 + 180 * (scores - s_min) / (s_max - s_min + 1e-9)
    else:
        sizes = np.full(len(names), 100)

    for i, (h, b, hex_c, name, sz) in enumerate(
        zip(hues, bright, hexes, names, sizes)
    ):
        ax.scatter(h, b, s=sz, color=hex_c, edgecolors="#333", linewidths=0.5, zorder=3)

    ax.set_xlabel("Hue  (0=red → 0.33=green → 0.67=blue → 1=red)", color="#aaa", fontsize=7.5)
    ax.set_ylabel("Brightness", color="#aaa", fontsize=7.5)
    ax.set_title("Color Space Map  (size ∝ preference score)", color="#eee", fontsize=9, pad=8)
    ax.tick_params(colors="#777", labelsize=7)
    ax.spines[:].set_visible(False)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.05, 1.05)

    fig.tight_layout()
    return fig
