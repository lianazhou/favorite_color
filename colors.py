"""
colors.py - Extended color palette (110+ colors) with improved feature extraction.

Feature set expanded to 8 dimensions for better discriminative power:
  hue, saturation, brightness, warmth, colorfulness,
  chroma (perceptual), cool_blue_bias, green_bias
"""

import colorsys
import numpy as np

# 110+ colors spanning vivid, pastel, deep, neutral, earth, neon, skin tones
COLORS = [
    # ── Reds ──
    ("Crimson",          "#DC143C"),
    ("Scarlet",          "#FF2400"),
    ("Vermilion",        "#E34234"),
    ("Rose Red",         "#C21E56"),
    ("Ruby",             "#9B111E"),
    ("Carmine",          "#960018"),
    ("Raspberry",        "#E30B5C"),
    ("Tomato",           "#FF6347"),
    ("Salmon",           "#FA8072"),
    ("Light Coral",      "#F08080"),

    # ── Oranges ──
    ("Tangerine",        "#F28500"),
    ("Burnt Orange",     "#CC5500"),
    ("Amber",            "#FFBF00"),
    ("Apricot",          "#FBCEB1"),
    ("Pumpkin",          "#FF7518"),
    ("Copper",           "#B87333"),
    ("Sienna",           "#A0522D"),

    # ── Yellows ──
    ("Lemon Yellow",     "#FFF44F"),
    ("Golden Yellow",    "#FFC200"),
    ("Canary",           "#FFFF99"),
    ("Saffron",          "#F4C430"),
    ("Mustard",          "#FFDB58"),
    ("Straw",            "#E4D96F"),
    ("Corn",             "#FBEC5D"),

    # ── Yellow-Greens ──
    ("Chartreuse",       "#7FFF00"),
    ("Lime Green",       "#32CD32"),
    ("Yellow-Green",     "#9ACD32"),
    ("Olive",            "#808000"),
    ("Moss",             "#8A9A5B"),
    ("Pear",             "#D1E231"),

    # ── Greens ──
    ("Emerald",          "#50C878"),
    ("Forest Green",     "#228B22"),
    ("Jade",             "#00A86B"),
    ("Hunter Green",     "#355E3B"),
    ("Sage",             "#B2AC88"),
    ("Fern",             "#4F7942"),
    ("Mint Green",       "#98FF98"),
    ("Spring Green",     "#00FF7F"),
    ("Sea Green",        "#2E8B57"),
    ("Pistachio",        "#93C572"),
    ("Avocado",          "#568203"),
    ("Basil",            "#4A5240"),

    # ── Teals & Cyans ──
    ("Teal",             "#008080"),
    ("Dark Teal",        "#003E4A"),
    ("Aqua",             "#00FFFF"),
    ("Turquoise",        "#40E0D0"),
    ("Cyan",             "#00BCD4"),
    ("Cadet Blue",       "#5F9EA0"),
    ("Steel Teal",       "#5F8A8B"),
    ("Verdigris",        "#43B3AE"),

    # ── Blues ──
    ("Sky Blue",         "#87CEEB"),
    ("Baby Blue",        "#89CFF0"),
    ("Powder Blue",      "#B0E0E6"),
    ("Cornflower Blue",  "#6495ED"),
    ("Cerulean",         "#2A52BE"),
    ("Cobalt",           "#0047AB"),
    ("Royal Blue",       "#4169E1"),
    ("Sapphire",         "#0F52BA"),
    ("Navy",             "#001F5B"),
    ("Midnight Blue",    "#191970"),
    ("Denim",            "#1560BD"),
    ("Periwinkle",       "#CCCCFF"),
    ("Steel Blue",       "#4682B4"),
    ("Slate Blue",       "#6A5ACD"),

    # ── Purples & Violets ──
    ("Indigo",           "#4B0082"),
    ("Violet",           "#EE82EE"),
    ("Purple",           "#800080"),
    ("Plum",             "#DDA0DD"),
    ("Orchid",           "#DA70D6"),
    ("Amethyst",         "#9966CC"),
    ("Mauve",            "#E0B0FF"),
    ("Grape",            "#6F2DA8"),
    ("Byzantium",        "#702963"),
    ("Thistle",          "#D8BFD8"),
    ("Wisteria",         "#C9A0DC"),
    ("Lavender",         "#E6E6FA"),
    ("Lilac",            "#C8A2C8"),

    # ── Pinks & Magentas ──
    ("Magenta",          "#FF00FF"),
    ("Hot Pink",         "#FF69B4"),
    ("Deep Pink",        "#FF1493"),
    ("Fuchsia",          "#FF00FF"),
    ("Rose",             "#FF007F"),
    ("Blush",            "#FFB6C1"),
    ("Petal Pink",       "#FFDDE1"),
    ("Bubble Gum",       "#FFC1CC"),
    ("Coral Pink",       "#F88379"),
    ("Flamingo",         "#FC8EAC"),
    ("Carnation",        "#FFA6C9"),

    # ── Browns & Earth Tones ──
    ("Chocolate",        "#7B3F00"),
    ("Mahogany",         "#C04000"),
    ("Chestnut",         "#954535"),
    ("Tan",              "#D2B48C"),
    ("Khaki",            "#C3B091"),
    ("Camel",            "#C19A6B"),
    ("Sand",             "#C2B280"),
    ("Taupe",            "#483C32"),
    ("Mocha",            "#967969"),
    ("Walnut",           "#5C3317"),
    ("Terracotta",       "#E2725B"),
    ("Clay",             "#CC7357"),

    # ── Neutrals & Grays ──
    ("Slate Gray",       "#708090"),
    ("Charcoal",         "#36454F"),
    ("Gunmetal",         "#2A3439"),
    ("Silver",           "#C0C0C0"),
    ("Ash",              "#B2BEB5"),
    ("Gainsboro",        "#DCDCDC"),
    ("Warm Gray",        "#808069"),
    ("Cool Gray",        "#8C92AC"),

    # ── Near-blacks & near-whites ──
    ("Ivory",            "#FFFFF0"),
    ("Cream",            "#FFFDD0"),
    ("Linen",            "#FAF0E6"),
    ("Pearl",            "#F5F5F0"),
    ("Smoke",            "#F5F5F5"),
    ("Off White",        "#FAF9F6"),
    ("Jet Black",        "#0A0A0A"),
    ("Rich Black",       "#010B13"),

    # ── Neons & Vivids ──
    ("Neon Green",       "#39FF14"),
    ("Neon Pink",        "#FF10F0"),
    ("Neon Orange",      "#FF6600"),
    ("Electric Blue",    "#7DF9FF"),
    ("Electric Purple",  "#BF00FF"),
]


def hex_to_rgb(hex_code: str) -> tuple[float, float, float]:
    h = hex_code.lstrip("#")
    return int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0, int(h[4:6], 16) / 255.0


def extract_features(hex_code: str) -> np.ndarray:
    """
    8-dimensional feature vector for more accurate preference modeling.

    Features:
      0  hue            0=red, 0.33=green, 0.67=blue (circular, unwrapped)
      1  saturation     0=gray, 1=fully vivid
      2  brightness     0=black, 1=white (HSV value)
      3  warmth         continuous: 1=warm (red/orange/yellow), 0=cool
      4  colorfulness   s * mid-brightness bell — peaks at vivid mid tones
      5  chroma         perceptual color intensity: sqrt(sum((R-G)^2+(G-B)^2+(B-R)^2)/3)
      6  blue_bias      how much the color leans blue/cyan (cool axis)
      7  red_bias       how much the color leans red/pink (warm axis)

    All features are in [0, 1] or close to it so gradient descent is stable.
    """
    r, g, b = hex_to_rgb(hex_code)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)

    # Warmth: smooth function of hue
    # Red=0, Yellow=0.167, Green=0.333, Cyan=0.5, Blue=0.667, Magenta=0.833
    # Warm: reds, oranges, yellows (h < 0.2 or h > 0.85 → warm)
    # Cool: greens, cyans, blues (h in 0.3–0.75)
    # Use a cosine to make it smooth around the hue circle
    hue_rad = h * 2 * np.pi
    # Warm axis aligns with red (h=0): cos(hue_rad) is +1 at red, -1 at cyan
    warmth = (np.cos(hue_rad) + 1) / 2  # [0, 1]

    # Colorfulness: vivid AND mid-brightness colors feel most colorful
    colorfulness = s * (1.0 - abs(v - 0.55) * 1.6)
    colorfulness = float(np.clip(colorfulness, 0, 1))

    # Perceptual chroma: how far from gray the RGB values are
    chroma = float(np.sqrt(((r - g) ** 2 + (g - b) ** 2 + (b - r) ** 2) / 3.0))
    chroma = np.clip(chroma, 0, 1)

    # Blue bias: blue channel dominance, weighted by saturation
    blue_bias = float(np.clip(b * s * 1.5, 0, 1))

    # Red bias: red channel dominance, weighted by saturation
    red_bias = float(np.clip(r * s * 1.5, 0, 1))

    return np.array([h, s, v, warmth, colorfulness, chroma, blue_bias, red_bias],
                    dtype=np.float64)


COLOR_NAMES    = [c[0] for c in COLORS]
COLOR_HEXES    = [c[1] for c in COLORS]
FEATURE_NAMES  = [
    "hue", "saturation", "brightness",
    "warmth", "colorfulness", "chroma",
    "blue_bias", "red_bias"
]

COLOR_FEATURES = np.stack([extract_features(h) for h in COLOR_HEXES])