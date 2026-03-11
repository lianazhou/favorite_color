# 🎨 Can Probability Guess Your Favorite Color?

A CS109 probability challenge project — an interactive app that learns your color
preferences through pairwise comparisons using **maximum likelihood estimation**.

---

## 1. Project Overview

You click through 20 pairs of colors, picking the one you prefer each round.
Behind each click, a logistic regression model treats your choice as a **Bernoulli
random variable** and uses **gradient descent on the negative log-likelihood** to
learn a *preference weight vector*. After 20 comparisons, the model predicts your
favorite color and explains what it learned.

---

## 2. Installation

```bash
# Clone or unzip the project folder, then:
pip install -r requirements.txt
```

Requires Python 3.11+.

---

## 3. Running the App

```bash
streamlit run app.py
```

Open the URL Streamlit prints (usually http://localhost:8501).

---

## 4. The Probabilistic Model

Each color is represented by a **5-dimensional feature vector**:

| Feature | Meaning |
|---|---|
| hue | Position on the color wheel (0=red, 0.33=green, 0.67=blue) |
| saturation | Vividness (0=gray, 1=fully saturated) |
| brightness | Lightness (0=black, 1=white) |
| warmth | 1 for warm colors (red/orange/yellow), 0 for cool |
| colorfulness | High for vivid mid-brightness colors, low for neutrals |

Given two colors A and B with feature vectors **x_A** and **x_B**, the model
defines:

```
P(user prefers A over B | w) = σ( w · (x_A − x_B) )
```

where σ is the sigmoid function and **w** is a learned weight vector.

---

## 5. Maximum Likelihood Estimation (CS109 Language)

Each comparison gives us one **observed Bernoulli outcome**:
- y = 1 if the user chose A
- y = 0 if the user chose B

The **likelihood** of a single observation is:

```
P(y | x_A, x_B, w) = p^y · (1−p)^(1−y)
```

For all N comparisons, the **joint likelihood** assumes independence:

```
L(w) = ∏ᵢ P(yᵢ | w)
```

MLE finds the weights **ŵ** that maximize this product — equivalently, that
minimize the **negative log-likelihood**:

```
NLL(w) = −∑ᵢ [ yᵢ log(pᵢ) + (1−yᵢ) log(1−pᵢ) ]
```

We minimize NLL using **gradient descent**:

```
w ← w − η · ∇NLL(w)
```

The gradient (derived by chain rule through the sigmoid) is:

```
∇NLL(w) = ∑ᵢ (pᵢ − yᵢ) · (x_Aᵢ − x_Bᵢ)
```

---

## 6. Interpreting the Learned Weights

| Weight direction | Meaning |
|---|---|
| w[hue] > 0 | Prefers cooler hues (greens, blues) |
| w[saturation] > 0 | Prefers vivid, saturated colors |
| w[brightness] > 0 | Prefers lighter colors |
| w[warmth] > 0 | Prefers warm colors (reds, oranges) |
| w[colorfulness] > 0 | Prefers rich, non-neutral colors |

Negative weights indicate the opposite preference.

After training, each candidate color is scored as **score(c) = w · x_c** and the
highest-scoring color is the model's prediction.

---

## 7. CS109 Connections

| Concept | Where it appears |
|---|---|
| **Probability** | P(user prefers A over B) defined via sigmoid |
| **Conditional probability** | P(y=1 \| x_A, x_B, w) — outcome conditioned on features and weights |
| **Random variables** | Each user click is a Bernoulli(p) random variable |
| **Likelihood / MLE** | Fitting w by maximizing the joint probability of observed clicks |
| **Logistic regression** | The logistic (sigmoid) function links linear scores to probabilities |
| **Inference** | Given data (clicks), infer hidden parameters (w) |
| **Gradient descent** | Numerical optimization to find the MLE solution |

---

## File Structure

```
app.py          – Streamlit UI (phases: intro → playing → results)
model.py        – sigmoid, NLL, gradient, training loop, scoring, summaries
colors.py       – 28-color palette with HSV/feature extraction
utils.py        – Matplotlib plots and display helpers
requirements.txt
README.md
writeup_notes.md
```
