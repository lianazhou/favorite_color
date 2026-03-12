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
    --bg:      #faf8f5;
    --bg2:     #ffffff;
    --bg3:     #f0ece6;
    --border:  #e0d8ce;
    --border2: #c8bdb0;
    --text:    #1a1612;
    --dim:     #8a7f74;
    --purple:  #7c3aed;
    --teal:    #0d9488;
    --rose:    #e11d48;
    --amber:   #d97706;
  }

  /* Soft multicolored gradient */
  html, body, [class*="css"], .stApp {
    background-color: var(--bg) !important;
    color: var(--text) !important;
  }
  .stApp {
    background: linear-gradient(
      135deg,
      #f9c5d1 0%,
      #fddcaa 20%,
      #b8f0d8 38%,
      #b8d8f8 56%,
      #d8b8f8 76%,
      #f9c5d1 100%
    ) !important;
    background-attachment: fixed !important;
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
    height: 3px !important;
  }

  /* Buttons */
  .stButton > button {
    background: white !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.76rem !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    padding: 0.55rem 1.4rem !important;
    border-radius: 6px !important;
    transition: all 0.15s ease !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important;
  }
  .stButton > button:hover {
    background: var(--bg3) !important;
    border-color: var(--border2) !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.10) !important;
  }

  /* Wide choice buttons */
  .choice-btn > .stButton > button {
    width: 100% !important;
    padding: 0.6rem !important;
    font-size: 0.7rem !important;
    border-radius: 0 0 8px 8px !important;
    border-top: none !important;
  }

  /* Metric cards */
  [data-testid="stMetric"] {
    background: white !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 0.75rem 1rem !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05) !important;
  }
  [data-testid="stMetricLabel"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.62rem !important;
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
    font-size: 0.7rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    color: var(--dim) !important;
  }
  [data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--purple) !important;
    border-bottom-color: var(--purple) !important;
  }

  /* Expander */
  [data-testid="stExpander"] {
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    background: white !important;
    overflow: hidden !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05) !important;
  }
  [data-testid="stExpander"] summary {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.7rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    color: var(--dim) !important;
    padding: 0.9rem 1.1rem !important;
    list-style: none !important;
    display: flex !important;
    align-items: center !important;
    gap: 0.5rem !important;
    cursor: pointer !important;
  }
  [data-testid="stExpander"] summary::marker,
  [data-testid="stExpander"] summary::-webkit-details-marker {
    display: none !important;
  }
  /* Chevron floated right so it never overlaps the label text */
  [data-testid="stExpander"] summary > div {
    display: flex !important;
    flex-direction: row !important;
    align-items: center !important;
    justify-content: space-between !important;
    width: 100% !important;
  }
  [data-testid="stExpander"] summary svg {
    width: 14px !important;
    height: 14px !important;
    flex-shrink: 0 !important;
    order: 2 !important;
  }
  [data-testid="stExpander"] summary > div > span,
  [data-testid="stExpander"] summary > div > p {
    order: 1 !important;
  }
  [data-testid="stExpanderDetails"] {
    padding: 0.2rem 1.1rem 1.1rem !important;
  }

  /* Swatch selected highlight ring */
  .swatch-wrap { position: relative; transition: transform 0.15s ease; }
  .swatch-wrap.selected { transform: scale(1.02); }
  .swatch-wrap.selected::after {
    content: '';
    position: absolute;
    inset: -4px;
    border-radius: 12px 12px 0 0;
    border: 3px solid rgba(0,0,0,0.85);
    pointer-events: none;
    z-index: 10;
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
        "just_picked":   None,
        "pending_choice": None,  # (chosen_idx, other_idx) waiting for highlight pause
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
                "round", "phase", "current_pair", "just_picked", "pending_choice"]:
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
    <div style="font-family:'DM Mono',monospace;font-size:0.82rem;line-height:1.9;
                color:#4a4035;max-width:580px;margin-bottom:1.4rem;">
      Pick between 30 pairs of colors. Each choice is a
      <span style="color:var(--teal);font-weight:500;background:#d1fae540;
                   padding:1px 5px;border-radius:3px;">Bernoulli trial</span>
      &mdash; the model treats it as a binary outcome driven by your latent preferences.<br><br>
      After all 30 rounds, a
      <span style="color:var(--purple);font-weight:500;background:#ede9fe40;
                   padding:1px 5px;border-radius:3px;">logistic regression</span>
      weight vector is fit using
      <span style="color:var(--amber);font-weight:500;background:#fef3c740;
                   padding:1px 5px;border-radius:3px;">Maximum Likelihood Estimation</span>
      via
      <span style="color:var(--rose);font-weight:500;background:#fce7f340;
                   padding:1px 5px;border-radius:3px;">gradient ascent</span>
      (Adam optimizer, 600 steps/round, L2-regularized).<br><br>
      The learned weights score all 121 candidate colors &mdash; argmax wins.
      Check out the <b>math analysis</b> dropdown after the test to see
      the full MLE objective, gradient derivation, and Adam update rule.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:white;border:1px solid var(--border);border-radius:8px;
                padding:0.9rem 1.2rem;margin-bottom:1.8rem;
                font-size:0.72rem;color:var(--dim);
                box-shadow:0 1px 4px rgba(0,0,0,0.05);">
      Press
      <span style="background:var(--bg3);border:1px solid var(--border2);
                   border-radius:3px;padding:1px 7px;color:#555;">&larr;</span>
      and
      <span style="background:var(--bg3);border:1px solid var(--border2);
                   border-radius:3px;padding:1px 7px;color:#555;">&rarr;</span>
      to choose with keyboard, or click the buttons below each color.
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

    # Arrow keys: highlight chosen swatch for 200ms, then click button
    _components.html("""
<script>
(function() {
  var par = window.parent.document;
  if (par.__arrowKeysReady) return;
  par.__arrowKeysReady = true;
  par.addEventListener('keydown', function(e) {
    if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
    e.preventDefault();
    var isLeft = e.key === 'ArrowLeft';

    // Highlight chosen swatch immediately
    var swatches = par.querySelectorAll('.swatch-wrap');
    swatches.forEach(function(s) { s.classList.remove('selected'); });
    if (swatches.length >= 2) {
      swatches[isLeft ? 0 : 1].classList.add('selected');
    }

    // Click the button after 200ms
    setTimeout(function() {
      var btns = Array.from(par.querySelectorAll('.stButton > button'))
                      .filter(function(b) { return b.offsetParent !== null; });
      var idx = isLeft ? 0 : 1;
      if (btns[idx]) btns[idx].click();
    }, 200);
  });
})();
</script>""", height=0)

    # Header row
    left_head, right_head = st.columns([3, 1])
    with left_head:
        st.markdown(
            f'<p style="font-family:\'DM Mono\',monospace;font-size:0.7rem;'
            f'letter-spacing:0.08em;text-transform:uppercase;color:#888;">'
            f'Round {n_done + 1} of {N_ROUNDS}</p>',
            unsafe_allow_html=True
        )
    with right_head:
        pct = int(n_done / N_ROUNDS * 100)
        st.markdown(
            f'<p style="font-family:\'DM Mono\',monospace;font-size:0.7rem;'
            f'color:#999;text-align:right;">{pct}%</p>',
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
        st.markdown('<div class="swatch-wrap">', unsafe_allow_html=True)
        st.markdown(swatch_html(hex_A, name_A, key_label="← left"), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        with st.container():
            st.markdown('<div class="choice-btn">', unsafe_allow_html=True)
            if st.button(name_A, key="btn_a", use_container_width=True):
                record_choice(idx_A, idx_B)
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    with gap:
        st.markdown(
            '<div style="display:flex;align-items:center;justify-content:center;'
            'height:200px;color:#bbb;font-size:0.75rem;font-family:\'DM Mono\','
            'monospace;">or</div>',
            unsafe_allow_html=True
        )

    with col_b:
        st.markdown('<div class="swatch-wrap">', unsafe_allow_html=True)
        st.markdown(swatch_html(hex_B, name_B, key_label="right →"), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
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

    # ── Math explainer ───────────────────────────────────────────────────────
    with st.expander("How it works — the math"):
        st.components.v1.html(r"""
<!DOCTYPE html>
<html>
<head>
<script>
window.MathJax = {
  tex: { inlineMath: [['$','$']], displayMath: [['$$','$$']] },
  options: { skipHtmlTags: ['script','noscript','style','textarea'] }
};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Georgia', serif;
    font-size: 13.5px;
    color: #1a1612;
    line-height: 1.7;
    padding: 4px 2px 12px 2px;
  }
  .section { margin-bottom: 20px; }
  .section-title {
    font-family: 'Arial', sans-serif;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #820034;
    border-bottom: 1.5px solid #e8c8d4;
    padding-bottom: 4px;
    margin-bottom: 10px;
  }
  p { margin-bottom: 8px; }
  .mathbox {
    background: #fff8fc;
    border-left: 3px solid #820034;
    border-radius: 0 6px 6px 0;
    padding: 10px 16px;
    margin: 8px 0;
    overflow-x: auto;
  }
  .note {
    font-family: 'Arial', sans-serif;
    font-size: 11.5px;
    color: #888;
    font-style: italic;
    margin-top: 4px;
  }
  .pill-row {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 6px;
  }
  .pill {
    font-family: 'Arial', sans-serif;
    font-size: 11px;
    background: #f5eef2;
    color: #820034;
    border-radius: 20px;
    padding: 3px 10px;
  }
</style>
</head>
<body>

<div class="section">
  <div class="section-title">1 · Bernoulli Trial Model</div>
  <p>
    Each round presents two colors $A$ and $B$. The choice is modeled as a
    Bernoulli trial — $Y_i = 1$ if you chose $A$, $Y_i = 0$ if you chose $B$.
    The probability depends on the dot product of a learned weight vector
    $\mathbf{w} \in \mathbb{R}^8$ with the feature difference:
  </p>
  <div class="mathbox">
    $$p_i = P(Y_i = 1 \mid \mathbf{w}) = \frac{1}{1 + e^{-\mathbf{w}^\top (\mathbf{x}_A - \mathbf{x}_B)}}$$
  </div>
  <p>So each observation follows $Y_i \sim \text{Bernoulli}(p_i)$.</p>
</div>

<div class="section">
  <div class="section-title">2 · Maximum Likelihood Estimation</div>
  <p>
    Assuming the 30 comparisons are independent, the joint likelihood is a
    product of Bernoulli terms. Taking the log turns it into a tractable sum:
  </p>
  <div class="mathbox">
    $$\ell(\mathbf{w}) = \sum_{i=1}^n \bigl[ y_i \log p_i + (1 - y_i)\log(1 - p_i) \bigr]$$
  </div>
  <p>
    We maximize $\ell(\mathbf{w})$, which is equivalent to finding the weights
    most consistent with every choice you made.
  </p>
</div>

<div class="section">
  <div class="section-title">3 · Gradient Ascent</div>
  <p>
    There's no closed-form solution, so we optimize iteratively.
    The gradient of the log-likelihood has a clean form:
  </p>
  <div class="mathbox">
    $$\frac{\partial \ell}{\partial w_k} = \sum_{i=1}^n (y_i - p_i)\,\Delta x_{ik}$$
  </div>
  <p>
    The term $(y_i - p_i)$ is the prediction error. The update rule
    nudges each weight in the direction that better explains your choices:
  </p>
  <div class="mathbox">
    $$\mathbf{w} \;\leftarrow\; \mathbf{w} + \alpha \,\nabla_\mathbf{w}\,\ell(\mathbf{w})$$
  </div>
  <p class="note">
    In practice Adam is used — it adapts the learning rate per feature using
    running estimates of gradient momentum and variance.
  </p>
</div>

<div class="section">
  <div class="section-title">4 · Prediction</div>
  <p>
    After training, every color $c$ in a pool of 121 candidates is scored
    by its alignment with $\hat{\mathbf{w}}$:
  </p>
  <div class="mathbox">
    $$\text{score}(c) = \hat{\mathbf{w}}^\top \mathbf{x}_c \qquad
    \text{favorite} = \operatorname*{arg\,max}_c\; \hat{\mathbf{w}}^\top \mathbf{x}_c$$
  </div>
  <p>
    Softmax converts scores into a probability distribution shown in the results.
    Each color $\mathbf{x}_c \in \mathbb{R}^8$ encodes:
  </p>
  <div class="pill-row">
    <span class="pill">hue</span>
    <span class="pill">saturation</span>
    <span class="pill">brightness</span>
    <span class="pill">warmth</span>
    <span class="pill">colorfulness</span>
    <span class="pill">chroma</span>
    <span class="pill">blue bias</span>
    <span class="pill">red bias</span>
  </div>
</div>

<script>
(function() {
  function resize() {
    var h = document.documentElement.scrollHeight;
    if (window.frameElement) window.frameElement.style.height = (h + 20) + 'px';
  }
  if (window.MathJax && window.MathJax.startup) {
    window.MathJax.startup.promise.then(resize);
  } else {
    setTimeout(resize, 800);
  }
})();
</script>
</body>
</html>
""", height=880, scrolling=True)

    # ── Restart ──────────────────────────────────────────────────────────────
    st.markdown("")
    col1, col2, col3 = st.columns([1, 1.4, 1])
    with col2:
        if st.button("Run again", use_container_width=True):
            do_restart()
            st.rerun()