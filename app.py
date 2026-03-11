"""
app.py  —  "Can Probability Guess Your Favorite Color?"

Streamlit app using pairwise logistic preference model + MLE.

Controls:
  Left arrow  →  choose left color
  Right arrow →  choose right color
  R key       →  restart (on results screen)
"""

import random
import numpy as np
import streamlit as st

from colors import (
    COLOR_NAMES, COLOR_HEXES, COLOR_FEATURES, FEATURE_NAMES
)
from model import (
    train, score_colors, softmax_scores,
    top_k_colors, preference_summary, pair_informativeness
)
from utils import (
    is_dark, plot_weights, plot_top_colors, plot_hue_brightness_scatter
)
import streamlit.components.v1 as _components

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Color Preference Lab",
    page_icon=None,
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── CSS & JS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,300;0,400;0,500;1,300&family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,600;0,9..144,800;1,9..144,300&display=swap');

  :root {
    --bg:      #0d0b12;
    --bg2:     #16121e;
    --bg3:     #1f1928;
    --border:  #2e2a38;
    --border2: #3d3850;
    --text:    #ede8f5;
    --dim:     #7a6f90;
    --purple:  #a78bfa;
    --teal:    #5eead4;
    --rose:    #fb7185;
    --amber:   #fbbf24;
  }

  html, body, [class*="css"], .stApp {
    background-color: var(--bg) !important;
    color: var(--text) !important;
  }

  /* Colorful muted radial background */
  .stApp {
    background:
      radial-gradient(ellipse 90% 55% at 15% 0%,  #2d1060 0%, transparent 55%),
      radial-gradient(ellipse 70% 45% at 85% 90%, #0b2545 0%, transparent 55%),
      radial-gradient(ellipse 50% 35% at 75% 15%, #1a0a30 0%, transparent 45%),
      #0d0b12 !important;
  }

  /* Typography */
  h1, h2, h3, h4 {
    font-family: 'Fraunces', serif !important;
    letter-spacing: -0.02em;
    color: var(--text) !important;
  }
  p, li, label, div, span, small {
    font-family: 'DM Mono', monospace !important;
  }

  /* Streamlit block container */
  .block-container {
    max-width: 800px !important;
    padding: 3rem 2rem 4rem !important;
  }

  /* Hide Streamlit chrome */
  #MainMenu, footer, header { visibility: hidden; }
  .stDeployButton { display: none; }
  [data-testid="stToolbar"] { display: none; }

  /* Progress bar */
  [data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, var(--teal), var(--purple)) !important;
    border-radius: 2px !important;
  }
  [data-testid="stProgressBar"] > div {
    background: var(--border) !important;
    border-radius: 2px !important;
    height: 2px !important;
  }

  /* Buttons */
  .stButton > button {
    background: var(--bg2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    padding: 0.55rem 1.4rem !important;
    border-radius: 4px !important;
    transition: all 0.15s ease !important;
    width: auto !important;
  }
  .stButton > button:hover {
    background: var(--bg3) !important;
    border-color: #555 !important;
    color: #fff !important;
  }

  /* Wide choice buttons */
  .choice-btn > .stButton > button {
    width: 100% !important;
    padding: 0.6rem !important;
    font-size: 0.7rem !important;
    border-radius: 0 0 6px 6px !important;
    border-top: none !important;
  }

  /* Metric cards */
  [data-testid="stMetric"] {
    background: var(--bg2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    padding: 0.75rem 1rem !important;
  }
  [data-testid="stMetricLabel"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.65rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    color: var(--dim) !important;
  }
  [data-testid="stMetricValue"] {
    font-family: 'Fraunces', serif !important;
    font-size: 1.4rem !important;
    color: var(--text) !important;
  }

  /* Tabs */
  [data-testid="stTabs"] button {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.72rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    color: var(--dim) !important;
  }
  [data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--text) !important;
    border-bottom-color: var(--text) !important;
  }

  /* Expander — fix arrow/text overlap */
  [data-testid="stExpander"] {
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    background: var(--bg2) !important;
    overflow: hidden !important;
  }
  [data-testid="stExpander"] summary {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.72rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    color: var(--dim) !important;
    padding: 0.9rem 1.1rem !important;
    list-style: none !important;
    display: flex !important;
    align-items: center !important;
    gap: 0.5rem !important;
  }
  [data-testid="stExpander"] summary::marker,
  [data-testid="stExpander"] summary::-webkit-details-marker {
    display: none !important;
  }
  [data-testid="stExpander"] summary > div {
    display: flex !important;
    align-items: center !important;
    gap: 0.5rem !important;
    width: 100% !important;
  }
  [data-testid="stExpander"] summary svg {
    width: 14px !important;
    height: 14px !important;
    flex-shrink: 0 !important;
    color: var(--dim) !important;
  }
  [data-testid="stExpanderDetails"] {
    padding: 0.2rem 1.1rem 1.1rem !important;
  }

  /* Divider */
  hr {
    border: none !important;
    border-top: 1px solid var(--border) !important;
    margin: 2rem 0 !important;
  }

  /* Key hint badge */
  .key-hint {
    display: inline-block;
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: 3px;
    padding: 1px 6px;
    font-size: 0.7rem;
    font-family: 'DM Mono', monospace;
    color: var(--dim);
    vertical-align: middle;
  }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "comparisons":   [],
        "pairs_shown":   set(),
        "weights":       np.zeros(len(FEATURE_NAMES)),
        "adam_m":        np.zeros(len(FEATURE_NAMES)),
        "adam_v":        np.zeros(len(FEATURE_NAMES)),
        "round":         0,
        "phase":         "intro",
        "current_pair":  None,
        "key_press":     "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

N_ROUNDS   = 30   # comparisons before showing results
LIVE_AFTER = 4    # show live weights after this many rounds
N_FEATURES = len(FEATURE_NAMES)
N_COLORS   = len(COLOR_NAMES)


# ── Pair selection ────────────────────────────────────────────────────────────

def pick_pair() -> tuple[int, int]:
    """
    Active learning: with probability 0.6 pick the most informative unseen pair
    (one where the model is most uncertain), otherwise pick randomly.
    Always avoid exact repeats.
    """
    w = st.session_state.weights
    seen = st.session_state.pairs_shown

    # Build candidate pool (unseen pairs only)
    candidates = []
    for i in range(N_COLORS):
        for j in range(i + 1, N_COLORS):
            key = (i, j)
            if key not in seen:
                info = pair_informativeness(w, COLOR_FEATURES[i], COLOR_FEATURES[j])
                candidates.append((info, i, j))

    if not candidates:
        # Exhausted: allow repeats
        i, j = random.sample(range(N_COLORS), 2)
        return i, j

    if random.random() < 0.6:
        # Sort by informativeness descending, pick from top 15%
        candidates.sort(reverse=True)
        pool = candidates[:max(1, len(candidates) // 7)]
    else:
        pool = candidates

    _, i, j = random.choice(pool)
    return i, j


# ── Color swatch HTML ─────────────────────────────────────────────────────────

def swatch_html(hex_code: str, name: str, key_label: str = "") -> str:
    text_color = "#ffffff" if is_dark(hex_code) else "#111111"
    key_badge = (
        f'<div style="position:absolute;top:10px;right:12px;'
        f'background:rgba(0,0,0,0.25);border-radius:3px;'
        f'padding:2px 7px;font-size:0.62rem;font-family:\'DM Mono\',monospace;'
        f'color:{text_color}88;letter-spacing:0.05em;">{key_label}</div>'
        if key_label else ""
    )
    return f"""
    <div style="
        position:relative;
        background:{hex_code};
        width:100%;
        height:200px;
        border-radius:8px 8px 0 0;
        display:flex;
        align-items:center;
        justify-content:center;
        box-shadow: 0 6px 30px {hex_code}40;
        user-select:none;
    ">
      {key_badge}
      <div style="
        font-family:'Fraunces',serif;
        font-size:1.05rem;
        font-weight:600;
        color:{text_color};
        text-align:center;
        padding:0 12px;
        text-shadow: 0 1px 4px rgba(0,0,0,0.2);
        letter-spacing:-0.01em;
      ">{name}</div>
    </div>
    """


# ── Record a choice ───────────────────────────────────────────────────────────

def record_choice(chosen_idx: int, other_idx: int):
    """Store the comparison and retrain."""
    x_A = COLOR_FEATURES[chosen_idx]
    x_B = COLOR_FEATURES[other_idx]
    st.session_state.comparisons.append((x_A, x_B, 1))

    pair_key = (min(chosen_idx, other_idx), max(chosen_idx, other_idx))
    st.session_state.pairs_shown.add(pair_key)

    # Retrain from scratch (warm-started) — Adam handles the momentum state
    st.session_state.weights = train(
        st.session_state.comparisons,
        n_steps=600,
        learning_rate=0.04,
        l2_lambda=0.015,
        w_init=st.session_state.weights,
        n_features=N_FEATURES,
    )

    st.session_state.just_picked = chosen_idx   # feedback banner on next render
    st.session_state.round += 1

    if st.session_state.round >= N_ROUNDS:
        st.session_state.phase = "results"
        st.session_state.current_pair = None
    else:
        st.session_state.current_pair = pick_pair()


def do_restart():
    for key in ["comparisons", "pairs_shown", "weights", "adam_m", "adam_v",
                "round", "phase", "current_pair", "just_picked"]:
        if key in st.session_state:
            del st.session_state[key]


# ══════════════════════════════════════════════════════════════════════════════
#  INTRO
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.phase == "intro":

    st.markdown("""
    <div style="margin-bottom: 0.4rem;">
      <div style="font-family:'DM Mono',monospace;font-size:0.65rem;letter-spacing:0.14em;
                  text-transform:uppercase;color:var(--dim);margin-bottom:0.6rem;">
        CS109 &nbsp;·&nbsp; Probability
      </div>
      <div style="font-family:'Fraunces',serif;font-size:2.8rem;font-weight:800;
                  letter-spacing:-0.03em;line-height:1.1;color:var(--text);">
        Favorite Color<br>Predictor
      </div>
      <div style="font-family:'DM Mono',monospace;font-size:0.72rem;color:var(--dim);
                  margin-top:0.7rem;letter-spacing:0.04em;">
        Liana Zhou &nbsp;·&nbsp; 2026
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("""
    <div style="font-family:'DM Mono',monospace;font-size:0.8rem;line-height:1.85;
                color:#b8b0cc;max-width:600px;margin-bottom:1rem;">
      Each color pair you judge is modeled as a
      <span style="color:var(--teal);font-weight:500;">Bernoulli trial</span>
      — a binary outcome with probability defined by a logistic function:
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:var(--bg2);border:1px solid var(--border);border-radius:8px;
                padding:1.1rem 1.4rem;margin:0 0 1.2rem;font-family:'DM Mono',monospace;
                font-size:0.78rem;color:#c4bedd;line-height:2.2;">
      <span style="color:var(--teal);">P</span>(prefer A over B &nbsp;|&nbsp; <b>w</b>)
      &nbsp;=&nbsp;
      <span style="color:var(--purple);">&sigma;</span>(
        <b>w</b> &middot; (x<sub>A</sub> &minus; x<sub>B</sub>) )
      <br>
      <span style="color:var(--dim);font-size:0.67rem;">
        x encodes hue, saturation, brightness, warmth, chroma + 3 more features
      </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="font-family:'DM Mono',monospace;font-size:0.8rem;line-height:1.85;
                color:#b8b0cc;max-width:600px;margin-bottom:1rem;">
      Your 30 choices define a dataset of i.i.d. Bernoulli observations.
      The weight vector <b>w</b> is fit by
      <span style="color:var(--amber);font-weight:500;">Maximum Likelihood Estimation</span>
      — minimizing the negative log-likelihood via
      <span style="color:var(--rose);font-weight:500;">Adam</span>
      (adaptive gradient descent with per-parameter momentum):
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:var(--bg2);border:1px solid var(--border);border-radius:8px;
                padding:1.1rem 1.4rem;margin:0 0 1.2rem;font-family:'DM Mono',monospace;
                font-size:0.78rem;color:#c4bedd;line-height:2.3;">
      <b>&#373;</b> &nbsp;=&nbsp; argmin &nbsp;&minus;&sum;
        [y<sub>i</sub> log p<sub>i</sub> + (1&minus;y<sub>i</sub>) log(1&minus;p<sub>i</sub>)]
        &nbsp;+&nbsp; &lambda;&#8214;<b>w</b>&#8214;&sup2;
      <br>
      &nabla;L(<b>w</b>) &nbsp;=&nbsp;
        &sum; (p<sub>i</sub> &minus; y<sub>i</sub>)
              (x<sub>A<sub>i</sub></sub> &minus; x<sub>B<sub>i</sub></sub>)
        &nbsp;+&nbsp; 2&lambda;<b>w</b>
      <br>
      <span style="color:var(--dim);font-size:0.67rem;">
        L2 regularization prevents weight explosion under sparse observations
      </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="font-family:'DM Mono',monospace;font-size:0.8rem;line-height:1.85;
                color:#b8b0cc;max-width:600px;margin-bottom:1.6rem;">
      Final prediction: score all 121 colors as <b>w</b> &middot; x, pick argmax.
      Check out the <b>math analysis</b> dropdown after the test to inspect
      learned weights and the full Adam update derivation.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:var(--bg2);border:1px solid var(--border);border-radius:8px;
                padding:0.9rem 1.2rem;margin-bottom:1.8rem;
                font-size:0.72rem;color:var(--dim);">
      Press
      <span style="background:var(--bg3);border:1px solid var(--border2);
                   border-radius:3px;padding:1px 7px;color:#aaa;">&larr;</span>
      and
      <span style="background:var(--bg3);border:1px solid var(--border2);
                   border-radius:3px;padding:1px 7px;color:#aaa;">&rarr;</span>
      to choose, or click the buttons below each color.
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.6, 1])
    with col2:
        if st.button("Begin  —  30 rounds", use_container_width=True):
            st.session_state.phase = "playing"
            st.session_state.current_pair = pick_pair()
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  PLAYING
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.phase == "playing":
    n_done = st.session_state.round

    # Ensure a pair is ready
    if st.session_state.current_pair is None:
        st.session_state.current_pair = pick_pair()

    idx_A, idx_B = st.session_state.current_pair
    name_A, hex_A = COLOR_NAMES[idx_A], COLOR_HEXES[idx_A]
    name_B, hex_B = COLOR_NAMES[idx_B], COLOR_HEXES[idx_B]

    # Arrow keys: find visible .stButton buttons in parent DOM, click [0] or [1]
    _components.html("""
<script>
(function() {
  var par = window.parent.document;
  if (par.__arrowKeysReady) return;
  par.__arrowKeysReady = true;
  par.addEventListener('keydown', function(e) {
    if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
    e.preventDefault();
    var btns = Array.from(par.querySelectorAll('.stButton > button'))
                    .filter(function(b) { return b.offsetParent !== null; });
    var idx = e.key === 'ArrowLeft' ? 0 : 1;
    if (btns[idx]) btns[idx].click();
  });
})();
</script>""", height=0)

    # Selection feedback banner (shows color chosen in previous round)
    if st.session_state.just_picked is not None:
        jp_idx  = st.session_state.just_picked
        jp_name = COLOR_NAMES[jp_idx]
        jp_hex  = COLOR_HEXES[jp_idx]
        jp_tc   = "#fff" if is_dark(jp_hex) else "#111"
        st.markdown(f"""
        <div style="background:{jp_hex}1a;border:1px solid {jp_hex}55;border-radius:6px;
                    padding:0.55rem 1rem;margin-bottom:0.6rem;
                    font-family:'DM Mono',monospace;font-size:0.7rem;color:{jp_hex};
                    display:flex;align-items:center;gap:0.6rem;">
          <span style="background:{jp_hex};color:{jp_tc};border-radius:50%;
                       width:18px;height:18px;display:inline-flex;align-items:center;
                       justify-content:center;font-size:11px;flex-shrink:0;">&#10003;</span>
          You chose &nbsp;<b>{jp_name}</b>
        </div>""", unsafe_allow_html=True)
        st.session_state.just_picked = None

    # Header row
    left_head, right_head = st.columns([3, 1])
    with left_head:
        st.markdown(
            f'<p style="font-family:\'DM Mono\',monospace;font-size:0.7rem;'
            f'letter-spacing:0.08em;text-transform:uppercase;color:#555;">'
            f'Round {n_done + 1} of {N_ROUNDS}</p>',
            unsafe_allow_html=True
        )
    with right_head:
        pct = int(n_done / N_ROUNDS * 100)
        st.markdown(
            f'<p style="font-family:\'DM Mono\',monospace;font-size:0.7rem;'
            f'color:#444;text-align:right;">{pct}%</p>',
            unsafe_allow_html=True
        )

    st.progress(n_done / N_ROUNDS)

    st.markdown(
        '<p style="font-family:\'Fraunces\',serif;font-size:1.25rem;'
        'font-weight:600;margin:1.4rem 0 1rem;letter-spacing:-0.01em;">'
        'Which color do you prefer?</p>',
        unsafe_allow_html=True
    )

    # Swatch columns
    col_a, gap, col_b = st.columns([10, 1, 10])

    with col_a:
        st.markdown(swatch_html(hex_A, name_A, key_label="← left"), unsafe_allow_html=True)
        with st.container():
            st.markdown('<div class="choice-btn">', unsafe_allow_html=True)
            if st.button(name_A, key="btn_a", use_container_width=True):
                record_choice(idx_A, idx_B)
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    with gap:
        st.markdown(
            '<div style="display:flex;align-items:center;justify-content:center;'
            'height:200px;color:#333;font-size:0.75rem;font-family:\'DM Mono\','
            'monospace;">or</div>',
            unsafe_allow_html=True
        )

    with col_b:
        st.markdown(swatch_html(hex_B, name_B, key_label="right →"), unsafe_allow_html=True)
        with st.container():
            st.markdown('<div class="choice-btn">', unsafe_allow_html=True)
            if st.button(name_B, key="btn_b", use_container_width=True):
                record_choice(idx_B, idx_A)
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    # Live model status
    if n_done >= LIVE_AFTER:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(
            '<p style="font-family:\'DM Mono\',monospace;font-size:0.65rem;'
            'letter-spacing:0.1em;text-transform:uppercase;color:#555;margin-bottom:1rem;">'
            'Live model status</p>',
            unsafe_allow_html=True
        )

        w = st.session_state.weights
        scores = score_colors(w, COLOR_FEATURES)
        best_idx = int(np.argmax(scores))

        c1, c2, c3 = st.columns(3)
        c1.metric("Comparisons", n_done)
        c2.metric("Current best guess", COLOR_NAMES[best_idx])
        c3.metric("Rounds remaining", N_ROUNDS - n_done)

        # Inline weight bars (compact)
        st.markdown("<br>", unsafe_allow_html=True)
        fig_w = plot_weights(w, FEATURE_NAMES)
        st.pyplot(fig_w, use_container_width=True)
        plt_close = __import__("matplotlib.pyplot", fromlist=["close"])
        plt_close.close(fig_w)

    # Restart link
    st.markdown("<hr>", unsafe_allow_html=True)
    if st.button("Restart", key="restart_playing"):
        do_restart()
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  RESULTS
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.phase == "results":
    w      = st.session_state.weights
    scores = score_colors(w, COLOR_FEATURES)
    probs  = softmax_scores(w, COLOR_FEATURES)
    best_idx  = int(np.argmax(scores))
    best_name = COLOR_NAMES[best_idx]
    best_hex  = COLOR_HEXES[best_idx]
    best_prob = float(probs[best_idx])
    top5      = top_k_colors(w, COLOR_FEATURES, COLOR_NAMES, k=5)
    summary   = preference_summary(w, FEATURE_NAMES)

    # ── Result header ────────────────────────────────────────────────────────
    txt = "#ffffff" if is_dark(best_hex) else "#111111"
    st.markdown(f"""
    <div style="
        background:{best_hex};
        border-radius:10px;
        padding:2.8rem 2rem;
        text-align:center;
        box-shadow:0 12px 50px {best_hex}50;
        margin-bottom:2rem;
    ">
      <div style="font-family:'DM Mono',monospace;font-size:0.62rem;
                  letter-spacing:0.14em;text-transform:uppercase;
                  color:{txt}88;margin-bottom:0.6rem;">
        Predicted favorite color
      </div>
      <div style="font-family:'Fraunces',serif;font-size:3rem;font-weight:800;
                  color:{txt};letter-spacing:-0.03em;line-height:1.1;">
        {best_name}
      </div>
      <div style="font-family:'DM Mono',monospace;font-size:0.7rem;
                  color:{txt}77;margin-top:0.5rem;">
        {best_hex} &nbsp;·&nbsp; {best_prob*100:.1f}% model confidence
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Preference summary ───────────────────────────────────────────────────
    st.markdown(
        '<p style="font-family:\'DM Mono\',monospace;font-size:0.65rem;'
        'letter-spacing:0.1em;text-transform:uppercase;color:#555;margin-bottom:0.8rem;">'
        'What the model learned</p>',
        unsafe_allow_html=True
    )
    for line in summary:
        st.markdown(
            f'<p style="font-family:\'Fraunces\',serif;font-size:1rem;'
            f'font-weight:300;font-style:italic;color:#c0c0b8;margin:0.3rem 0;">'
            f'{line}</p>',
            unsafe_allow_html=True
        )

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Top 5 swatches ───────────────────────────────────────────────────────
    st.markdown(
        '<p style="font-family:\'DM Mono\',monospace;font-size:0.65rem;'
        'letter-spacing:0.1em;text-transform:uppercase;color:#555;margin-bottom:1rem;">'
        'Top 5 colors</p>',
        unsafe_allow_html=True
    )
    cols = st.columns(5)
    for col, (name, raw_score, prob) in zip(cols, top5):
        hex_c = COLOR_HEXES[COLOR_NAMES.index(name)]
        tc = "#fff" if is_dark(hex_c) else "#111"
        col.markdown(f"""
        <div style="background:{hex_c};height:80px;border-radius:6px;
                    display:flex;flex-direction:column;align-items:center;
                    justify-content:center;gap:3px;
                    box-shadow:0 4px 16px {hex_c}44;">
          <div style="font-family:'Fraunces',serif;font-size:0.7rem;
                      font-weight:600;color:{tc};text-align:center;
                      padding:0 4px;">{name}</div>
          <div style="font-family:'DM Mono',monospace;font-size:0.58rem;
                      color:{tc}99;">{prob*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Charts ───────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["Feature Weights", "Color Rankings", "Color Map"])

    with tab1:
        st.markdown(
            '<p style="font-size:0.75rem;color:#666;margin-bottom:0.5rem;">'
            'Positive weight = you prefer this quality. '
            'Negative = you avoid it.</p>',
            unsafe_allow_html=True
        )
        fig = plot_weights(w, FEATURE_NAMES)
        st.pyplot(fig, use_container_width=True)

    with tab2:
        st.markdown(
            '<p style="font-size:0.75rem;color:#666;margin-bottom:0.5rem;">'
            'Bars are filled with the actual color. '
            'Labels show softmax probability.</p>',
            unsafe_allow_html=True
        )
        fig = plot_top_colors(scores, probs, COLOR_NAMES, COLOR_HEXES, top_n=12)
        st.pyplot(fig, use_container_width=True)

    with tab3:
        st.markdown(
            '<p style="font-size:0.75rem;color:#666;margin-bottom:0.5rem;">'
            'Each dot is a color in hue-brightness space. '
            'Larger dot = higher preference score.</p>',
            unsafe_allow_html=True
        )
        fig = plot_hue_brightness_scatter(COLOR_FEATURES, COLOR_NAMES, COLOR_HEXES, scores)
        st.pyplot(fig, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Math expander ────────────────────────────────────────────────────────
    with st.expander("Math analysis — how the prediction was made"):
        st.markdown("""
**Pairwise preference model**

Each comparison is a Bernoulli r.v. with probability set by logistic regression:

```
P(choose A | w) = σ( w · (x_A − x_B) )     σ(z) = 1 / (1 + e^{−z})
```

**MLE objective (regularized negative log-likelihood)**

```
ŵ = argmin  −∑ᵢ [yᵢ log pᵢ + (1−yᵢ) log(1−pᵢ)]  +  λ‖w‖²
```

**Gradient used in each Adam step**

```
∇L(w) = (1/n) ∑ᵢ (pᵢ − yᵢ)(x_Aᵢ − x_Bᵢ)  +  2λw
```

**Adam update rule** (adaptive moment estimation, 600 steps per round)

```
m ← β₁m + (1−β₁)∇         first moment  (momentum)
v ← β₂v + (1−β₂)∇²        second moment (per-feature learning rate)
w ← w − α · m̂ / (√v̂ + ε)  bias-corrected parameter update
```

**Scoring & prediction:** `score(c) = ŵ · x_c` for all 121 colors → argmax.  
Softmax over raw scores yields the probability distribution shown above.

**Feature vector x** (8 dims, all normalized to [0, 1])  
hue · saturation · brightness · warmth · colorfulness · chroma · blue_bias · red_bias
""")

    # ── Restart ──────────────────────────────────────────────────────────────
    st.markdown("")
    col1, col2, col3 = st.columns([1, 1.4, 1])
    with col2:
        if st.button("Run again", use_container_width=True):
            do_restart()
            st.rerun()