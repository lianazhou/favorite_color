"""
model.py - Probabilistic pairwise preference model with improved accuracy.

Improvements over v1:
  - Adam optimizer for faster, more stable convergence
  - Proper L2 regularization (not ad-hoc weight decay)
  - 8-feature vector for richer discrimination
  - Informativeness-based pair selection helper
  - Softmax-based probability scores for final ranking
"""

import numpy as np


# ── Sigmoid ───────────────────────────────────────────────────────────────────

def sigmoid(z):
    """Numerically stable sigmoid."""
    z = np.clip(z, -50, 50)
    return 1.0 / (1.0 + np.exp(-z))


# ── Preference probability ────────────────────────────────────────────────────

def pref_prob(w: np.ndarray, x_A: np.ndarray, x_B: np.ndarray) -> float:
    """P(user chooses A over B | w) = sigmoid(w · (x_A - x_B))"""
    return float(sigmoid(np.dot(w, x_A - x_B)))


# ── NLL and gradient ─────────────────────────────────────────────────────────

def neg_log_likelihood(
    w: np.ndarray,
    comparisons: list,
    l2_lambda: float = 0.01
) -> float:
    """
    NLL with L2 regularization:
      L(w) = -∑[y log p + (1-y) log(1-p)] + λ||w||²

    L2 regularization prevents weights from growing unboundedly with few
    observations, which would make predictions overconfident.
    """
    eps = 1e-9
    total = 0.0
    for x_A, x_B, y in comparisons:
        p = float(np.clip(pref_prob(w, x_A, x_B), eps, 1 - eps))
        total -= y * np.log(p) + (1 - y) * np.log(1 - p)
    total += l2_lambda * float(np.dot(w, w))
    return total


def grad_nll(
    w: np.ndarray,
    comparisons: list,
    l2_lambda: float = 0.01
) -> np.ndarray:
    """
    Gradient of regularized NLL:
      ∇L(w) = ∑(pᵢ - yᵢ)(x_Aᵢ - x_Bᵢ) + 2λw
    """
    grad = np.zeros_like(w, dtype=np.float64)
    for x_A, x_B, y in comparisons:
        p = pref_prob(w, x_A, x_B)
        grad += (p - y) * (x_A - x_B)
    grad += 2.0 * l2_lambda * w   # L2 regularization gradient
    return grad


# ── Adam optimizer ────────────────────────────────────────────────────────────

def train(
    comparisons: list,
    n_steps: int = 500,
    learning_rate: float = 0.05,
    l2_lambda: float = 0.02,
    w_init: np.ndarray | None = None,
    n_features: int = 8,
) -> np.ndarray:
    """
    Maximize likelihood of observed pairwise comparisons using Adam optimizer.

    Adam (Adaptive Moment Estimation) is gradient descent with:
      - m: running average of gradients (momentum)
      - v: running average of squared gradients (adapts per-feature lr)

    Update rule:
      m ← β₁m + (1-β₁)∇       # momentum
      v ← β₂v + (1-β₂)∇²      # RMS scaling
      w ← w - α · m̂ / (√v̂ + ε) # bias-corrected update

    This converges faster and more reliably than vanilla gradient descent,
    especially with a small number of training examples.

    Parameters:
      l2_lambda  : regularization strength (higher = more conservative weights)
    """
    if not comparisons:
        return np.zeros(n_features)

    w = w_init.copy() if w_init is not None else np.zeros(n_features)

    # Adam hyperparameters
    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8
    m = np.zeros_like(w)
    v = np.zeros_like(w)

    n = len(comparisons)

    for t in range(1, n_steps + 1):
        grad = grad_nll(w, comparisons, l2_lambda) / n  # normalize by n

        # Adam moment updates
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad ** 2

        # Bias correction
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        w = w - learning_rate * m_hat / (np.sqrt(v_hat) + eps_adam)

    return w


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_colors(w: np.ndarray, color_features: np.ndarray) -> np.ndarray:
    """
    Raw linear scores: score(c) = w · x_c.
    Higher = model predicts user prefers this color.
    """
    return color_features @ w


def softmax_scores(w: np.ndarray, color_features: np.ndarray) -> np.ndarray:
    """
    Convert raw scores to a proper probability distribution via softmax.
    Useful for displaying "% likelihood this is your favorite."

    softmax(s)ᵢ = exp(sᵢ) / ∑ exp(sⱼ)
    """
    s = score_colors(w, color_features)
    s = s - s.max()   # numerical stability
    exp_s = np.exp(s)
    return exp_s / exp_s.sum()


def top_k_colors(
    w: np.ndarray,
    color_features: np.ndarray,
    color_names: list[str],
    k: int = 5
) -> list[tuple[str, float, float]]:
    """
    Return top-k colors as [(name, raw_score, softmax_prob), ...].
    """
    raw = score_colors(w, color_features)
    probs = softmax_scores(w, color_features)
    ranked = np.argsort(raw)[::-1]
    return [(color_names[i], float(raw[i]), float(probs[i])) for i in ranked[:k]]


# ── Informativeness (for pair selection) ─────────────────────────────────────

def pair_informativeness(
    w: np.ndarray,
    x_A: np.ndarray,
    x_B: np.ndarray
) -> float:
    """
    A comparison is most informative when the model is uncertain (p ≈ 0.5).
    Informativeness = 1 - |2p - 1|  →  max 1.0 when p=0.5, min 0.0 when p=0 or 1.

    Used for active learning: prefer pairs where the model is confused.
    """
    p = pref_prob(w, x_A, x_B)
    return 1.0 - abs(2 * p - 1)


# ── Natural-language preference summary ──────────────────────────────────────

def preference_summary(w: np.ndarray, feature_names: list[str]) -> list[str]:
    """
    Translate learned weights into plain English.
    Reports the 3 most influential features (by |weight|).
    """
    interpretations = {
        "hue": (
            "You lean toward cooler hues — greens, teals, blues",
            "You lean toward warmer hues — reds, oranges, yellows"
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
        "chroma": (
            "You're drawn to high-chroma, perceptually intense colors",
            "You prefer low-chroma, subtle colors"
        ),
        "blue_bias": (
            "Blues and cyans have a strong pull on you",
            "You're not particularly drawn to blue tones"
        ),
        "red_bias": (
            "Reds, pinks, and warm saturated tones appeal to you",
            "You tend away from red-dominant colors"
        ),
    }

    order = np.argsort(np.abs(w))[::-1]
    lines = []
    seen_axes = set()

    for i in order:
        if len(lines) >= 3:
            break
        name = feature_names[i]
        val = w[i]
        if abs(val) < 0.03:
            continue

        # Avoid redundant statements (warmth and hue overlap; blue/red and warmth overlap)
        axis = "warm_cool" if name in ("hue", "warmth", "blue_bias", "red_bias") else name
        if axis in seen_axes:
            continue
        seen_axes.add(axis)

        pos_text, neg_text = interpretations.get(name, (f"high {name}", f"low {name}"))
        lines.append(pos_text if val > 0 else neg_text)

    if not lines:
        lines.append("Your preferences are still forming — try a few more rounds.")
    return lines