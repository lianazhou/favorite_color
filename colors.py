"""
colors.py - Color palette and feature extraction for the preference model.

Each color is stored as (name, hex_code). Features are computed from HSV
and RGB values so we can interpret what the learned weights mean.
"""

import colorsys
import numpy as np

# --- Color Palette ---
# ~28 colors with good variety: vivid, pastel, warm, cool, neutral
COLORS = [
    ("Crimson Red",    "#DC143C"),
    ("Tomato",         "#FF6347"),
    ("Tangerine",      "#F28500"),
    ("Amber",          "#FFBF00"),
    ("Lemon Yellow",   "#FFF44F"),
    ("Chartreuse",     "#7FFF00"),
    ("Lime Green",     "#32CD32"),
    ("Forest Green",   "#228B22"),
    ("Teal",           "#008080"),
    ("Aqua",           "#00FFFF"),
    ("Sky Blue",       "#87CEEB"),
    ("Cerulean",       "#2A52BE"),
    ("Royal Blue",     "#4169E1"),
    ("Indigo",         "#4B0082"),
    ("Violet",         "#EE82EE"),
    ("Magenta",        "#FF00FF"),
    ("Hot Pink",       "#FF69B4"),
    ("Blush",          "#FFB6C1"),
    ("Lavender",       "#E6E6FA"),
    ("Mint",           "#98FF98"),
    ("Peach",          "#FFCBA4"),
    ("Coral",          "#FF7F50"),
    ("Slate Gray",     "#708090"),
    ("Charcoal",       "#36454F"),
    ("Ivory",          "#FFFFF0"),
    ("Cream",          "#FFFDD0"),
    ("Chocolate",      "#7B3F00"),
    ("Burgundy",       "#800020"),
]


def hex_to_rgb(hex_code: str) -> tuple[float, float, float]:
    """Convert hex string to normalized RGB floats in [0, 1]."""
    hex_code = hex_code.lstrip("#")
    r = int(hex_code[0:2], 16) / 255.0
    g = int(hex_code[2:4], 16) / 255.0
    b = int(hex_code[4:6], 16) / 255.0
    return r, g, b


def rgb_to_hsv(r: float, g: float, b: float) -> tuple[float, float, float]:
    """Convert normalized RGB to HSV (hue in [0,1], saturation, value)."""
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    return h, s, v


def is_warm(h: float) -> float:
    """
    Warm-vs-cool indicator based on hue.
    Warm hues (red/orange/yellow): roughly h < 0.17 or h > 0.92 → returns 1.0
    Cool hues (green/blue/purple): returns 0.0
    This is a soft, human-intuitive split.
    """
    # Hue wheel: 0=red, 0.083=orange, 0.167=yellow, 0.333=green,
    #            0.5=cyan, 0.667=blue, 0.833=magenta, 1=red again
    if h < 0.17 or h > 0.87:
        return 1.0   # warm
    elif 0.25 < h < 0.75:
        return 0.0   # cool
    else:
        return 0.5   # neutral transition


def extract_features(hex_code: str) -> np.ndarray:
    """
    Extract a 5-dimensional feature vector from a hex color.

    Features (all normalized to roughly [-1, 1] or [0, 1]):
      0: hue           (0 = red, 0.33 = green, 0.67 = blue)
      1: saturation    (0 = gray/white, 1 = fully vivid)
      2: brightness    (0 = black, 1 = white)
      3: warmth        (1 = warm/red/orange/yellow, 0 = cool/blue/green)
      4: colorfulness  (combination of saturation and mid-brightness — pure
                        grays and near-white/black score low)

    Keeping features interpretable makes the learned weights easy to explain:
      - positive hue weight → preference shifts toward green/blue (higher hue)
      - positive saturation weight → likes vivid colors
      - positive brightness weight → likes lighter colors
      - positive warmth weight → prefers warm over cool colors
      - positive colorfulness weight → prefers rich, non-neutral colors
    """
    r, g, b = hex_to_rgb(hex_code)
    h, s, v = rgb_to_hsv(r, g, b)
    warm = is_warm(h)
    # colorfulness: high when saturation is high AND brightness is mid-range
    colorfulness = s * (1 - abs(v - 0.5) * 2)
    return np.array([h, s, v, warm, colorfulness], dtype=np.float64)


# Precompute feature matrix for the whole palette — shape (N, 5)
COLOR_NAMES   = [c[0] for c in COLORS]
COLOR_HEXES   = [c[1] for c in COLORS]
FEATURE_NAMES = ["hue", "saturation", "brightness", "warmth", "colorfulness"]

COLOR_FEATURES = np.stack([extract_features(h) for h in COLOR_HEXES])  # (N, 5)
