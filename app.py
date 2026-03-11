"""
app.py - "Can Probability Guess Your Favorite Color?"

A Streamlit app that learns your color preferences through pairwise comparisons
using a logistic regression model trained with MLE (gradient descent).

Run with:
    streamlit run app.py
"""

import random
import numpy as np
import streamlit as st

from colors import (
    COLOR_NAMES, COLOR_HEXES, COLOR_FEATURES, FEATURE_NAMES
)
from model import train, score_colors, top_k_colors, preference_summary
from utils import (
    is_dark, plot_weights, plot_top_colors,
    plot_hue_brightness_scatter, format_weight_table
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Probability Reads Your Mind",
    page_icon="🎨",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

  html, body, [class*="css"] {
      background-color: #0a0a0a !important;
      color: #e8e8e0;
  }
  h1, h2, h3 { font-family: 'Syne', sans-serif; }
  p, li, label, div { font-family: 'Space Mono', monospace; font-size: 0.82rem; }
  .stButton > button {
      width: 100%;
      height: 160px;
      border: 2px solid #333;
      border-radius: 12px;
      font-family: 'Syne', sans-serif;
      font-size: 1rem;
      font-weight: 700;
      cursor: pointer;
      transition: transform 0.1s ease, border-color 0.2s ease;
  }
  .stButton > button:hover {
      border-color: #fff;
      transform: scale(1.03);
  }
  .stButton > button:active { transform: scale(0.97); }
  .block-container { max-width: 760px; padding-top: 2rem; }
  hr { border-color: #2a2a2a; margin: 1.5rem 0; }
  .metric-box {
      background: #161616;
      border: 1px solid #2a2a2a;
      border-radius: 10px;
      padding: 1rem;
      text-align: center;
  }
  .big-color-name {
      font-family: 'Syne', sans-serif;
      font-size: 2rem;
      font-weight: 800;
      letter-spacing: -0.5px;
  }
</style>
""", unsafe_allow_html=True)

# ── Session state init ────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "comparisons": [],       # list of (x_A, x_B, y) training examples
        "pairs_shown": [],       # list of (idx_A, idx_B) for deduplication
        "weights": np.zeros(len(FEATURE_NAMES)),
        "round": 0,
        "phase": "intro",        # intro | playing | results
        "current_pair": None,    # (idx_A, idx_B)
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

N_ROUNDS_TO_FINISH = 20   # number of comparisons before showing results
MIN_ROUNDS_FOR_LIVE = 3   # minimum comparisons before showing live weights

# ── Helper: pick a fresh pair ─────────────────────────────────────────────────
def pick_pair() -> tuple[int, int]:
    """
    Choose a pair of colors we haven't shown yet (or reuse if exhausted).
    Slightly biased toward colors with similar preference scores to make
    comparisons more informative (the model is most uncertain near ties).
    """
    n = len(COLOR_NAMES)
    scores = score_colors(st.session_state.weights, COLOR_FEATURES)
    seen = set(map(tuple, st.session_state.pairs_shown))

    # Prefer pairs that are "close" in score (informative comparisons)
    candidates = []
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) not in seen:
                diff = abs(scores[i] - scores[j])
                candidates.append((diff, i, j))

    if not candidates:
        # All pairs exhausted — just pick randomly
        i, j = random.sample(range(n), 2)
        return i, j

    # With probability 0.5 pick a close pair; otherwise random (exploration)
    candidates.sort()
    if random.random() < 0.5:
        # pick from the closest third
        pool = candidates[: max(1, len(candidates) // 3)]
    else:
        pool = candidates

    _, i, j = random.choice(pool)
    return i, j


# ── Color swatch HTML ─────────────────────────────────────────────────────────
def color_swatch_html(hex_code: str, name: str, size: int = 120) -> str:
    text_color = "#fff" if is_dark(hex_code) else "#111"
    return f"""
    <div style="
        background:{hex_code};
        width:100%;
        height:{size}px;
        border-radius:10px;
        display:flex;
        align-items:center;
        justify-content:center;
        font-family:'Syne',sans-serif;
        font-weight:700;
        font-size:0.95rem;
        color:{text_color};
        letter-spacing:0.3px;
        box-shadow: 0 4px 20px {hex_code}55;
        user-select:none;
    ">{name}</div>
    """


# ══════════════════════════════════════════════════════════════════════════════
# PHASE: INTRO
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.phase == "intro":
    st.markdown("## 🎨 Can Probability Guess Your Favorite Color?")
    st.markdown("""
A probabilistic model is about to read your mind — using nothing but math.

**How it works:**

Each time you pick one color over another, the app records a **Bernoulli outcome** —  
a 0-or-1 observation. Under the hood, a **logistic regression model** uses your  
clicks as training data and runs **maximum likelihood estimation (MLE)** to learn  
a *preference weight vector*.

After **20 comparisons**, the model will predict your favorite color and explain  
exactly what it learned about your taste.

> *"Every click is a data point. Every data point updates the model.*  
> *That's probability in action."*
""")
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("▶  Start the Experiment", use_container_width=True):
            st.session_state.phase = "playing"
            st.session_state.current_pair = pick_pair()
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PHASE: PLAYING
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.phase == "playing":
    n_done = st.session_state.round

    # Header
    st.markdown(f"### Round {n_done + 1} of {N_ROUNDS_TO_FINISH}")
    progress = n_done / N_ROUNDS_TO_FINISH
    st.progress(progress)
    st.markdown("**Which color do you prefer?**")
    st.markdown("")

    # Ensure a pair is selected
    if st.session_state.current_pair is None:
        st.session_state.current_pair = pick_pair()

    idx_A, idx_B = st.session_state.current_pair
    name_A, hex_A = COLOR_NAMES[idx_A], COLOR_HEXES[idx_A]
    name_B, hex_B = COLOR_NAMES[idx_B], COLOR_HEXES[idx_B]

    # ── Swatch display ──────────────────────────────────────────────────────
    col_a, spacer, col_b = st.columns([5, 1, 5])
    with col_a:
        st.markdown(color_swatch_html(hex_A, name_A), unsafe_allow_html=True)
    with spacer:
        st.markdown("<div style='text-align:center;padding-top:45px;color:#555;font-size:1.3rem;'>or</div>", unsafe_allow_html=True)
    with col_b:
        st.markdown(color_swatch_html(hex_B, name_B), unsafe_allow_html=True)

    st.markdown("")

    # ── Choice buttons ──────────────────────────────────────────────────────
    btn_a, btn_b = st.columns(2)

    def record_choice(chosen: int, other: int, y: int):
        """Record the user's choice and retrain the model."""
        x_A = COLOR_FEATURES[chosen]
        x_B = COLOR_FEATURES[other]
        # y=1 means 'chosen' was preferred; y=0 would mean 'other' was preferred
        st.session_state.comparisons.append((x_A, x_B, y))
        st.session_state.pairs_shown.append((min(chosen, other), max(chosen, other)))

        # Retrain on all data collected so far (batch gradient descent)
        # Warm-start from the previous weights to speed up convergence
        st.session_state.weights = train(
            st.session_state.comparisons,
            n_steps=300,
            learning_rate=0.08,
            w_init=st.session_state.weights,
            n_features=len(FEATURE_NAMES),
        )
        st.session_state.round += 1

        if st.session_state.round >= N_ROUNDS_TO_FINISH:
            st.session_state.phase = "results"
            st.session_state.current_pair = None
        else:
            st.session_state.current_pair = pick_pair()

    with btn_a:
        if st.button(f"← I prefer  {name_A}", key="btn_a"):
            record_choice(idx_A, idx_B, y=1)
            st.rerun()
    with btn_b:
        if st.button(f"I prefer  {name_B}  →", key="btn_b"):
            record_choice(idx_B, idx_A, y=1)
            st.rerun()

    # ── Live model status ───────────────────────────────────────────────────
    if n_done >= MIN_ROUNDS_FOR_LIVE:
        st.markdown("---")
        st.markdown("#### 📡 Live Model Status")
        w = st.session_state.weights

        c1, c2, c3 = st.columns(3)
        c1.metric("Comparisons", n_done)
        best_idx = int(np.argmax(score_colors(w, COLOR_FEATURES)))
        c2.metric("Current Best Guess", COLOR_NAMES[best_idx])
        c3.metric("Rounds Left", N_ROUNDS_TO_FINISH - n_done)

        st.markdown("**Learned Weights**  *(positive = user likes this feature)*")
        weight_dict = format_weight_table(w, FEATURE_NAMES)
        cols = st.columns(len(FEATURE_NAMES))
        for col, (feat, val) in zip(cols, weight_dict.items()):
            delta_color = "normal" if val >= 0 else "inverse"
            col.metric(label=feat, value=f"{val:+.3f}")

        fig_w = plot_weights(w, FEATURE_NAMES)
        st.pyplot(fig_w, use_container_width=True)
    else:
        st.markdown("")
        st.info(f"Model will show live updates after {MIN_ROUNDS_FOR_LIVE} comparisons.")

    # Skip / restart
    st.markdown("---")
    if st.button("🔄  Start Over"):
        for key in ["comparisons", "pairs_shown", "weights", "round", "phase", "current_pair"]:
            del st.session_state[key]
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PHASE: RESULTS
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.phase == "results":
    w = st.session_state.weights
    scores = score_colors(w, COLOR_FEATURES)
    best_idx = int(np.argmax(scores))
    best_name = COLOR_NAMES[best_idx]
    best_hex  = COLOR_HEXES[best_idx]
    top5 = top_k_colors(w, COLOR_FEATURES, COLOR_NAMES, k=5)
    summary = preference_summary(w, FEATURE_NAMES)

    # ── Predicted favorite ──────────────────────────────────────────────────
    text_col = "#fff" if is_dark(best_hex) else "#111"
    st.markdown(f"""
    <div style="
        background: {best_hex};
        border-radius: 16px;
        padding: 2.5rem;
        text-align: center;
        box-shadow: 0 8px 40px {best_hex}88;
        margin-bottom: 1.5rem;
    ">
      <div style="font-family:'Space Mono',monospace;font-size:0.75rem;
                  letter-spacing:3px;text-transform:uppercase;color:{text_col}aa;
                  margin-bottom:0.4rem;">
        The model predicts your favorite color is
      </div>
      <div style="font-family:'Syne',sans-serif;font-size:2.8rem;font-weight:800;
                  color:{text_col};letter-spacing:-1px;">
        {best_name}
      </div>
      <div style="font-family:'Space Mono',monospace;font-size:0.7rem;
                  color:{text_col}99;margin-top:0.3rem;">
        {best_hex}
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Preference summary ──────────────────────────────────────────────────
    st.markdown("#### 🧠 What the Model Learned About You")
    for line in summary:
        st.markdown(f"&nbsp;&nbsp;{line}")

    st.markdown("---")

    # ── Top 5 ───────────────────────────────────────────────────────────────
    st.markdown("#### 🏆 Your Top 5 Colors")
    cols = st.columns(5)
    for col, (name, score) in zip(cols, top5):
        hex_c = COLOR_HEXES[COLOR_NAMES.index(name)]
        col.markdown(color_swatch_html(hex_c, name, size=80), unsafe_allow_html=True)
        col.markdown(f"<div style='text-align:center;font-size:0.7rem;color:#888;margin-top:3px;'>score: {score:.2f}</div>", unsafe_allow_html=True)

    st.markdown("---")

    # ── Charts ──────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["📊 Feature Weights", "🎨 Top Colors", "🗺 Color Map"])
    with tab1:
        st.markdown("Each bar shows how much the model weights that color feature.")
        st.markdown("**Positive** = you prefer this quality. **Negative** = you avoid it.")
        st.pyplot(plot_weights(w, FEATURE_NAMES), use_container_width=True)

    with tab2:
        st.markdown("Bars are filled with the actual color — size reflects preference score.")
        st.pyplot(plot_top_colors(scores, COLOR_NAMES, COLOR_HEXES, top_n=10), use_container_width=True)

    with tab3:
        st.markdown("Each dot is a color placed by hue (x) and brightness (y). Larger = higher preference score.")
        st.pyplot(plot_hue_brightness_scatter(COLOR_FEATURES, COLOR_NAMES, COLOR_HEXES, scores), use_container_width=True)

    st.markdown("---")

    # ── Math explainer ──────────────────────────────────────────────────────
    with st.expander("📐 Show the Math Behind the Prediction"):
        st.markdown("""
**Probabilistic Model**

Each comparison is modeled as a Bernoulli random variable:

> *P(choose A over B | w) = σ(w · (x_A − x_B))*

where σ is the sigmoid function and **w** is our learned weight vector.

**Maximum Likelihood Estimation**

After collecting all your comparisons, we find **w** that maximizes the joint
probability of every observed choice:

> *ŵ = argmax∏ P(yᵢ | xₐᵢ, x_bᵢ, w)*

In practice we minimize the **negative log-likelihood**:

> *L(w) = −∑ [ yᵢ log(pᵢ) + (1−yᵢ) log(1−pᵢ) ]*

using **gradient descent** with explicit gradient updates:

> *w ← w − η · ∇L(w)*

The gradient is: **∇L(w) = ∑ (pᵢ − yᵢ) · (x_Aᵢ − x_Bᵢ)**

After training, we score all 28 colors as **score(c) = w · x_c** and pick the highest.
""")

    # ── Restart ─────────────────────────────────────────────────────────────
    st.markdown("")
    if st.button("🔄  Try Again"):
        for key in ["comparisons", "pairs_shown", "weights", "round", "phase", "current_pair"]:
            del st.session_state[key]
        st.rerun()
