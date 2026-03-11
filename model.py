"""
model.py - Probabilistic pairwise preference model.

Math overview (CS109 level):
  - Each time the user picks color A over color B, we treat that as a
    Bernoulli trial with success probability p = sigmoid(w · (x_A - x_B)).
  - w is a weight vector we want to learn; x_A, x_B are feature vectors.
  - We collect many such (A, B, outcome) observations and find the weights w
    that maximize the likelihood of all observed choices — that's MLE.
  - Maximizing likelihood = minimizing negative log-likelihood (NLL):
      L(w) = -∑ [ y·log(p) + (1-y)·log(1-p) ]
    where y=1 if A was chosen, y=0 if B was chosen.
  - We minimize L(w) with plain gradient descent.
"""

import numpy as np


# ── Sigmoid ──────────────────────────────────────────────────────────────────

def sigmoid(z: np.ndarray | float) -> np.ndarray | float:
    """
    sigmoid(z) = 1 / (1 + exp(-z))

    Numerically stable version: clamp z to avoid exp overflow.
    """
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


# ── Preference probability ────────────────────────────────────────────────────

def pref_prob(w: np.ndarray, x_A: np.ndarray, x_B: np.ndarray) -> float:
    """
    P(user chooses A over B | w) = sigmoid(w · (x_A - x_B))

    Interpretation:
      - If w·(x_A - x_B) >> 0, model is very confident the user picks A.
      - If w·(x_A - x_B) ≈ 0, it's a coin flip.
      - If w·(x_A - x_B) << 0, model expects the user picks B.
    """
    score = np.dot(w, x_A - x_B)
    return float(sigmoid(score))


# ── Negative log-likelihood ───────────────────────────────────────────────────

def neg_log_likelihood(
    w: np.ndarray,
    comparisons: list[tuple[np.ndarray, np.ndarray, int]]
) -> float:
    """
    NLL(w) = -∑ [ y·log(p) + (1-y)·log(1-p) ]

    Each comparison is (x_A, x_B, y):
      y = 1  → user chose A
      y = 0  → user chose B (i.e., chose "not A")
    """
    eps = 1e-9  # prevents log(0)
    total = 0.0
    for x_A, x_B, y in comparisons:
        p = pref_prob(w, x_A, x_B)
        p = np.clip(p, eps, 1 - eps)
        total -= y * np.log(p) + (1 - y) * np.log(1 - p)
    return total


# ── Gradient of NLL ──────────────────────────────────────────────────────────

def grad_nll(
    w: np.ndarray,
    comparisons: list[tuple[np.ndarray, np.ndarray, int]]
) -> np.ndarray:
    """
    Gradient of NLL with respect to w.

    For each comparison (x_A, x_B, y):
      p     = sigmoid(w · (x_A - x_B))
      error = p - y          (prediction minus truth)
      grad  = error · (x_A - x_B)   (chain rule through sigmoid)

    So the full gradient is:
      ∇L(w) = ∑ (p_i - y_i) · (x_A_i - x_B_i)

    Intuition: if the model over-predicted (p > y), we push w in the direction
    that lowers scores for the chosen color's features.
    """
    grad = np.zeros_like(w)
    for x_A, x_B, y in comparisons:
        p = pref_prob(w, x_A, x_B)
        error = p - y
        grad += error * (x_A - x_B)
    return grad


# ── Training loop (MLE via gradient descent) ─────────────────────────────────

def train(
    comparisons: list[tuple[np.ndarray, np.ndarray, int]],
    n_steps: int = 200,
    learning_rate: float = 0.1,
    w_init: np.ndarray | None = None,
    n_features: int = 5,
) -> np.ndarray:
    """
    Find weights w that maximize likelihood of observed comparisons.

    This is MLE: we want w* = argmax P(data | w)
                            = argmin NLL(w)

    Algorithm (gradient descent):
      1. Start with w = 0 (or small random values).
      2. For each step:
           a. Compute gradient ∇L(w)  — how NLL changes w.r.t. each weight
           b. Move w in the downhill direction: w ← w - η · ∇L(w)
      3. After n_steps, return w.

    With enough data and steps, w converges to the MLE solution.

    Parameters:
      comparisons   : list of (x_A, x_B, y) tuples
      n_steps       : number of gradient descent iterations
      learning_rate : step size η (too large → oscillates; too small → slow)
      w_init        : optional starting weights (warm-start from previous fit)
      n_features    : dimension of w

    Returns:
      w : learned weight vector (shape [n_features])
    """
    if not comparisons:
        return np.zeros(n_features)

    # Initialize weights
    if w_init is not None:
        w = w_init.copy()
    else:
        w = np.zeros(n_features)

    # Gradient descent loop
    for step in range(n_steps):
        grad = grad_nll(w, comparisons)
        w = w - learning_rate * grad   # take a downhill step

        # Optional: small L2 regularization to keep weights from exploding
        # when data is sparse (prevents overconfident predictions early on)
        w = w * 0.999

    return w


# ── Scoring and prediction ────────────────────────────────────────────────────

def score_colors(w: np.ndarray, color_features: np.ndarray) -> np.ndarray:
    """
    Score every candidate color with the learned weight vector.

    score(color_i) = w · x_i

    Higher score = model predicts user prefers this color.
    This is the linear "utility" the model assigns to each color.
    """
    return color_features @ w   # shape (N,)


def top_k_colors(
    w: np.ndarray,
    color_features: np.ndarray,
    color_names: list[str],
    k: int = 5
) -> list[tuple[str, float]]:
    """Return the top-k colors by score as [(name, score), ...]."""
    scores = score_colors(w, color_features)
    ranked = np.argsort(scores)[::-1]   # descending
    return [(color_names[i], float(scores[i])) for i in ranked[:k]]


# ── Natural-language preference summary ──────────────────────────────────────

def preference_summary(w: np.ndarray, feature_names: list[str]) -> list[str]:
    """
    Turn the learned weight vector into plain-English insights.

    Positive weight for a feature → user likes colors high in that feature.
    Negative weight             → user dislikes colors high in that feature.

    We only report the two or three features with the largest |weight|
    so the summary stays concise.
    """
    # Map feature name → human-readable interpretation pair
    interpretations = {
        "hue": (
            "You lean toward cooler hues (greens, blues, purples)",
            "You lean toward warmer hues (reds, oranges, yellows)"
        ),
        "saturation": (
            "You prefer vivid, highly saturated colors",
            "You prefer muted, desaturated tones"
        ),
        "brightness": (
            "You like bright, light colors",
            "You prefer darker, deeper shades"
        ),
        "warmth": (
            "You gravitate toward warm colors (reds, oranges, yellows)",
            "You gravitate toward cool colors (blues, greens, purples)"
        ),
        "colorfulness": (
            "You love rich, colorful tones over neutrals",
            "You prefer neutral or near-achromatic colors"
        ),
    }

    # Sort features by absolute weight magnitude (most influential first)
    order = np.argsort(np.abs(w))[::-1]
    lines = []
    for i in order[:3]:   # top 3 most influential features
        name = feature_names[i]
        val  = w[i]
        if abs(val) < 0.05:
            continue   # weight too small to say anything meaningful
        pos_text, neg_text = interpretations.get(name, (name, name))
        if val > 0:
            lines.append(f"✦ {pos_text}")
        else:
            lines.append(f"✦ {neg_text}")

    if not lines:
        lines.append("✦ Your preferences are still a bit mysterious — try more rounds!")
    return lines
