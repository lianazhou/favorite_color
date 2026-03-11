# Writeup Notes — "Can Probability Guess Your Favorite Color?"

*Draft for CS109 Probability Challenge submission.*

---

## What the Project Does

This project is an interactive app that tries to learn your hidden color
preferences through a sequence of pairwise comparisons. You click through
20 rounds, each time choosing which of two colors you like more. The app
records every choice, trains a probabilistic model in real time, and at the
end predicts your favorite color and provides a plain-English explanation of
your taste — "You prefer vivid, saturated colors" or "You lean toward cooler
hues."

The key idea is that a human's aesthetic preferences are hidden — we can't
observe them directly. But each pairwise click is a noisy signal about those
preferences, and probability theory gives us a rigorous way to extract a
consistent picture from many such signals.

---

## Why This is a Probability-Driven Model

The core insight is that we never *directly* observe someone's favorite color.
What we *do* observe is a sequence of binary choices. Probability is the right
language for this setting because:

1. Each choice is uncertain and noisy — someone might prefer red over blue most
   of the time, but not always.
2. We want to *infer* an unobserved quantity (preferences) from observed data
   (clicks).
3. We need to quantify our uncertainty and update it as new evidence arrives.

This is classical Bayesian-flavored thinking: treat the user's preferences as
latent parameters and use observed data to pin them down.

---

## The Logistic Probability Model

We represent each color as a feature vector **x** in ℝ⁵:
- hue, saturation, brightness, warmth, colorfulness

We posit that the user has a latent preference weight vector **w** ∈ ℝ⁵, where
a large positive weight on "saturation" means "this person loves vivid colors."

Given two colors A and B, the model defines:

```
P(user prefers A over B | w) = σ( w · (x_A − x_B) )
```

where σ(z) = 1 / (1 + e^{−z}) is the sigmoid function.

This model is elegant for three reasons:
- The difference **x_A − x_B** captures *relative* features — what distinguishes A from B.
- The dot product **w · (x_A − x_B)** is a scalar "preference score" for A over B.
- The sigmoid squashes that score to a valid probability in (0, 1).

When **w · (x_A − x_B) >> 0**, the model is nearly certain the user picks A.
When the dot product ≈ 0, it's a coin flip. This is the Bradley–Terry model of
pairwise preference, reframed through logistic regression.

---

## How MLE Is Used

After collecting N comparisons {(x_Aᵢ, x_Bᵢ, yᵢ)}, we want to find the weight
vector **w** that makes the observed choices as probable as possible. This is
**maximum likelihood estimation**.

The likelihood of the full dataset (assuming independence across rounds) is:

```
L(w) = ∏ᵢ P(yᵢ | xₐᵢ, x_bᵢ, w)
      = ∏ᵢ pᵢ^{yᵢ} (1 − pᵢ)^{1 − yᵢ}
```

Taking the log (for numerical stability and to convert products to sums):

```
log L(w) = ∑ᵢ [ yᵢ log(pᵢ) + (1−yᵢ) log(1−pᵢ) ]
```

We maximize this — equivalently minimize the **negative log-likelihood** (NLL) —
using plain gradient descent. The gradient of NLL with respect to **w** is:

```
∇NLL(w) = ∑ᵢ (pᵢ − yᵢ) · (x_Aᵢ − x_Bᵢ)
```

Interpretation: if the model *over-predicted* (pᵢ > yᵢ, i.e., predicted A would
win but B was chosen), the gradient pushes **w** to reduce the score gap between
A and B for that feature pattern.

---

## Why Each User Click Is a Bernoulli Outcome

A **Bernoulli random variable** takes value 1 with probability p and 0 with
probability 1 − p. Each user click fits this exactly:

- The user either clicks A (y = 1) or B (y = 0).
- The underlying probability p = σ(w · (x_A − x_B)) is determined by the model.
- The observed click is one realization of this Bernoulli trial.

The key assumption is that different rounds are *independent* Bernoulli trials —
the outcome of round 3 doesn't depend on round 7. This lets us multiply individual
probabilities to get the joint likelihood, which is what makes MLE tractable.

---

## How the Final Prediction Is Made

After gradient descent converges, we have a weight vector **ŵ** that maximizes
the likelihood of all observed clicks. We then score every candidate color:

```
score(c) = ŵ · x_c
```

This is the linear "utility" the model assigns to color c under the learned
preferences. The color with the highest score is the prediction. The top-5 are
the ranking. The signs of individual weights yield the plain-English summary:
a positive saturation weight → "You like vivid colors"; a negative brightness
weight → "You prefer darker shades."

The entire pipeline — feature extraction, likelihood definition, MLE via
gradient descent, and inference — is explicit, interpretable, and grounded in
the CS109 probability toolkit.

---

## Connection to Broader CS109 Topics

| Topic | Role in this project |
|---|---|
| Probability | Defines P(A preferred | w) via sigmoid |
| Conditional probability | P(y=1 \| features, weights) conditions outcome on parameters |
| Bernoulli random variables | Each click is an independent Bernoulli trial |
| Joint likelihood | Product of per-round Bernoullis = likelihood of full dataset |
| MLE | Find w that maximizes the joint likelihood |
| Logistic regression | The specific probabilistic model linking features to P(y=1) |
| Inference | Use observed data to estimate latent parameters (preferences) |
| Gradient descent | Numerical method to find the MLE solution |
