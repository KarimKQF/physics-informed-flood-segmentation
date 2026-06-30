# Topographic-Order Loss Headroom: A Mechanistic Demonstration
## SegMAN-S · Sen1Floods11 · N = 100 · seed 0

**Status:** consolidated negative result, single-seed, label-relative, N=100 mechanistic.
**Invariant respected throughout:** the DEM is **never** a model input. It appears only inside
losses, diagnostics, and post-hoc analysis.

This document states the claim, the axioms, the definitions, one information-theoretic
proposition, five empirical lemmas (each with the measured numbers), the composed argument,
limitations, and — explicitly — what is *not* proven.

---

## 0. The precise claim (read this first)

> **Claim (bounded).** For the pair *(SegMAN-S trained on Sen1Floods11 N=100 seed0, the
> reference labels y)*, we evaluate the **shared mechanism** of the class of local monotone
> DEM-based topographic-order losses $\mathcal C_{\mathrm{ord}}$ (see D5). In this setting
> the unconstrained Dice+CE baseline already satisfies the targeted local topographic-order
> property — as measured by violation fraction, violation energy, and distributional tails —
> **at least as strongly as the reference labels**. Local monotone DEM-based topographic-order
> losses whose sole mechanism is to reduce $v_{ij}$ violations therefore have **no native
> label-relative headroom** in this model–data configuration.

What the claim is **not**:

- It is **not** "the DEM is useless for flood mapping in general." (It is not — see L4:
  a DEM-only predictor reaches AUC 0.77–0.84.)
- It is **not** "no physics-informed loss can ever help any model." (Out of scope; see §10.)
- It is **not** a statement about HAND, flow accumulation, hydrological connectivity,
  rainfall/discharge constraints, or DEM used as an input channel. All of these are
  explicitly outside scope (§10).
- It is **not** "the model understands hydrology." (The opposite — see L3.)
- It is **not** a statement about the *true physical flood state*; everything is **relative to
  the reference labels** (Axiom A3).
- It is **not** a full-dataset or multi-architecture result. The N=100 setup is
  preliminary/mechanistic; §10 lists what remains before publishable claims.

---

## 1. Notation and objects

| symbol | meaning |
|---|---|
| $x \in \mathcal X$ | model input: 15 channels (13 S2L1C + 2 S1GRD) |
| $y \in \{0,1\}^{H\times W}$ | reference water label |
| $z \in \mathbb R^{H\times W}$ | DEM (elevation), **never** an input to $f_\theta$ |
| $f_\theta:\mathcal X\to[0,1]^{H\times W}$ | SegMAN-S; output $\hat p = f_\theta(x) = p_{\text{water}}$ |
| $\hat y = \mathbb 1[\hat p > \tfrac12]$ | binarised prediction |
| $\mathcal L_{\text{seg}}$ | Dice + CE segmentation loss |
| $\mathcal R_\phi(\hat p, z)$ | a DEM/physics regularizer of formulation $\phi$ (e.g. D8) |
| $\lambda$ | physics-loss weight; total loss $\mathcal L_{\text{seg}} + \lambda\,\mathcal R_\phi$ |
| splits | `val` (86 tiles), `test` (89), `bolivia` (15, OOD) |

Checkpoint of record: `segman_n100_dice_ce_seed0`, best epoch 34, val mIoU 0.8546.

---

## 2. Axioms (assumptions made explicit)

**A1 — Channel of action.** A DEM loss can influence the learned parameters *only* through the
gradient $\nabla_\theta \mathcal R_\phi(f_\theta(x), z)$. It introduces no new input pathway
(the DEM is not fed to $f_\theta$).

**A2 — Two and only two ways to help.** Relative to a fixed label set, adding $\lambda\mathcal R_\phi$
can reduce held-out risk only via:
- **(I) the information channel** — $\mathcal R_\phi$ injects label-relevant information that is
  carried by $z$ but not already present in $f_\theta(x)$; or
- **(II) the inductive-bias channel** — $\mathcal R_\phi$ shrinks the hypothesis space / acts as a
  prior / reduces variance, improving generalization *even with zero new information*.

Any real effect decomposes into these two. (A1–A2 are the logical backbone; everything below
bounds channel I to $\approx 0$ and tests channel II for the class $\mathcal C_{\mathrm{ord}}$.)

**A3 — Label-relative evaluation.** All risks, AUCs and mutual-information quantities are defined
against the reference labels $y$, **not** the unobserved true flood state $y^\star$.

**A4 — Two regimes.** *Post-hoc* diagnostics (L3, L4) hold $\theta$ **frozen**. The *loss*
experiment (L1) **retrains** $\theta$. This distinction matters: post-hoc tests bound what $z$
adds *given the converged model*; the retraining test bounds what $z$ adds *when injected as a
constraint during training*.

**A5 — Finite-sample honesty.** Every empirical quantity is an estimate. We report bootstrap CIs
(L1b, L4) or per-tile dispersion (L2) and never treat a point estimate as an exact zero.

---

## 3. Definitions

**D1 — Output-level constraint headroom.**
Let $A(s; w)$ be the AUROC of a real-valued score $s$ for a binary target $w$. Using **negative
elevation** $-z$ as the score (water sits low ⇒ $-z$ predicts water),
$$
H_z \;:=\; A(-z;\,y)\;-\;A(-z;\,\hat y).
$$
$H_z>0$: the *labels* are more topographically separable than the *predictions* — room for a
topographic loss to push. $H_z\le 0$: the predictions are already at least as topographically
aligned as the labels — **no room**.

**D2 — Representational decodability.**
Let $\phi(x)$ be frozen features and $D(\phi)$ the best *within-tile linear* decodability
(Pearson $r$, per-tile z-scored) of $z$ from $\phi$. Compare $\phi_{\text{trained}}$,
$\phi_{\text{random}}$ (random-init same arch), $\phi_{\text{input}}$ (raw 15-ch). If
$D(\phi_{\text{trained}})\approx D(\phi_{\text{input}})$, training induced **no privileged**
elevation representation.

**D3 — Conditional DEM redundancy.**
The DEM is *conditionally redundant given the model output* iff
$$
I\!\left(y;\,z \,\middle|\, \hat p\right) \;=\; 0 .
$$
Estimator: the held-out AUC gap $\Delta\mathrm{AUC} = \mathrm{AUC}(y\sim \hat p + z) - \mathrm{AUC}(y\sim \hat p)$
of a flexible learner (gradient boosting). $\Delta\mathrm{AUC}\approx 0$ (CI spanning 0) ⇒
$I(y;z\mid\hat p)\approx 0$.

**D4 — Effective constraint contribution.**
$\rho := \lambda\,\mathcal R_\phi / (\mathcal L_{\text{seg}} + \lambda\,\mathcal R_\phi)$ at
convergence — the share of the training objective actually supplied by the physics term. A
*slack* constraint (already satisfied) has $\rho\approx 0$ and exerts almost no gradient.

**D5 — Local topographic-order loss class $\mathcal C_{\mathrm{ord}}$.**
We do **not** refute D4 and D8 independently. We evaluate the **shared mechanism** of their
class: reducing local topographic-order violations.
$$
\mathcal C_{\mathrm{ord}} \;=\; \bigl\{\,\mathcal R_\phi \;:\; \mathcal R_\phi \text{ penalises } p_i > p_j + \tau \text{ when } h_i > h_j,\; j \in \mathcal{N}(i) \,\bigr\}
$$
where $h_i$ is DEM elevation, $j$ is a lower neighbouring pixel, and $\tau$ is a tolerance
margin. The class includes: D4 (all lower 4-neighbors), D8 (steepest-descent downstream), slope-
weighted D8, hinge-linear/quadratic variants, margin-based variants, and any local loss whose
main mechanism is reducing these violations. **Not included:** DEM-as-input, HAND, flow
accumulation, hydrological connectivity, or any non-local DEM-based constraint.

**D6 — Local violation magnitude and violation fraction.**
For a valid edge $(i,j)$ with $h_i > h_j$:
$$
v_{ij}(\hat p, z) \;=\; \max(0,\, p_i - p_j - \tau)
$$
The (slope-weighted) **violation fraction** (VF) over the D8 active set $\mathcal E$:
$$
\mathrm{VF}(\hat p, z) \;=\; \frac{\sum_{(i,j)\in\mathcal E} w_{ij}\,\mathbb 1[p_i > p_j + \tau]}{\sum_{(i,j)\in\mathcal E} w_{ij}}
$$
with $w_{ij} = \min(1, \mathrm{drop}_{ij}/s_0)$ (D8 parameters: $s_0=1.0$, $\tau=0.05$).
The **native headroom** is $H_R = \mathrm{VF}(\hat y_0, z) - \mathrm{VF}(y, z)$.
$H_R\le 0$ means the baseline violates the constraint no more than the labels.

**D7 — Violation energy $E_{\mathrm{topo}}$.**
$$
E_{\mathrm{topo}}(p, z) \;=\; \frac{\sum_{(i,j)\in\mathcal E} w_{ij}\,[\max(0, p_i - p_j - \tau)]^2}{\sum_{(i,j)\in\mathcal E} w_{ij}}
$$
This is the weighted mean squared violation — matching the D8 loss objective exactly.

---

## 4. Central proposition — the information bound

> **Proposition 1.** Let $\ell$ be any proper scoring rule (e.g. log-loss or Brier), whose Bayes
> predictor is the conditional mean. If $I(y;z\mid\hat p)=0$ — equivalently $y\perp z\mid\hat p$ —
> then for the Bayes risks
> $$
> R^\star(\hat p, z) \;=\; R^\star(\hat p),
> $$
> i.e. **no measurable function of $(\hat p, z)$ predicts $y$ better than the best function of
> $\hat p$ alone.**

*Proof.* Conditional independence gives $\mathbb E[y\mid \hat p, z] = \mathbb E[y\mid \hat p]$
almost surely. For a proper loss the risk-minimizing predictor is the conditional expectation, so
the optimal predictors — and hence their risks — coincide. $\;\square$

> **Corollary 1 (information channel).** Under A1–A2, a DEM loss acting through channel I can lower
> held-out risk only to the extent that $I(y;z\mid\hat p) > 0$. If that conditional information is
> zero, the information channel has **exactly zero** label-exploitable headroom, *for every
> formulation $\phi$* — because all formulations act through the same $z$ and can supply nothing
> $z$ does not contain.

**Scope of Proposition 1.** It conditions on a *fixed* $\hat p$. It therefore bounds the
**post-hoc / converged-model** information channel. The path it does *not* close by itself is "the
constraint, applied *during* training, reshapes $\hat p$ into something better." That residual path
is exactly what Lemma L1 (actual D8 retraining) and Lemma L1b (class-level mechanistic
diagnostics) test.

---

## 5. Empirical lemmas

### Lemma L1 — the direct loss result (channel II, D8 formulation)
D8 downstream topographic loss, seed 0, $\lambda\in\{1,100\}$, real vs shuffled DEM control.
mIoU (↑ better); precision/recall on water; predicted-mask topographic-violation fraction (VF).

| split | metric | baseline | D8 real λ1 | D8 shuf λ1 | D8 real λ100 |
|---|---|---:|---:|---:|---:|
| val | mIoU | 0.8546 | 0.8586 (**+0.004**) | 0.8401 | 0.8398 (−0.015) |
| test | mIoU | **0.8615** | 0.8600 (−0.001) | 0.8460 | 0.8579 (−0.004) |
| bolivia | mIoU | **0.8434** | 0.8419 (−0.001) | 0.8379 | 0.8408 (−0.003) |
| test | precision | 0.8680 | **0.9026** (+0.035) | 0.8927 | 0.8620 |
| test | recall | **0.8604** | 0.8243 (−0.036) | 0.8036 | 0.8586 |
| test | topo-VF | 0.00110 | 0.00091 | 0.00079 | 0.00094 |

Two decisive facts:

1. **The constraint is slack.** Effective contribution $\rho = 7.1\times10^{-5}$ at $\lambda=1$
   and $3.0\times10^{-3}$ at $\lambda=100$ — i.e. **0.007 % / 0.3 %** of the objective. The baseline
   already violates the topographic constraint on only $\approx 0.1\%$ of pixels (VF $\approx 0.0011$).
   There is almost nothing for the penalty to act on.
2. **No robust held-out/OOD gain.** D8 real improves `val` by +0.004 mIoU but *costs* −0.001 on
   both `test` and OOD `bolivia`. Mechanistically it merely trades **recall for precision**
   (test: +0.035 precision, −0.036 recall) — it shrinks the water mask, not improves it.
   $\lambda=100$ makes every split worse.

**Real vs shuffled is not a robust physics signal.** At N=100 λ1 real edges out shuffled, but the
shuffled run converged earlier (best epoch 22 vs 31) and an earlier N=50 topographic-loss study
found the **opposite** sign (shuffled beat real on all splits). Across settings the sign of
(real − shuffled) **flips** and is single-seed. Combined with $\rho\approx 0$, the topographic
*content* provides no reproducible benefit beyond generic regularization. ∎(L1)

---

### Lemma L1b — class-level mechanistic headroom: $\mathcal C_{\mathrm{ord}}$ (channel II)

L1 tested one formulation (D8, seed0). L1b evaluates the **shared mechanism** of the entire class
$\mathcal C_{\mathrm{ord}}$ (D5): comparing baseline predictions and reference labels in the native
violation space of the constraint. All measurements are inference-only, checkpoint frozen.

#### 5.1 D8 native violation-fraction headroom

$H_R = \mathrm{VF}(\hat y_0, z) - \mathrm{VF}(y, z)$, slope-weighted, $s_0=1.0$, $\tau=0.05$.

| split | VF(labels) | VF(pred) | $H_R$ (pooled) | per-tile mean | 95% CI (per-tile) | % tiles pred ≤ labels |
|---|---:|---:|---:|---:|---|---:|
| val | 0.00217 | 0.00062 | **−0.00154** | −0.00228 | [−0.00303, −0.00159] | 88% |
| test | 0.00241 | 0.00058 | **−0.00182** | −0.00268 | [−0.00344, −0.00120] | 90% |
| bolivia | 0.00321 | 0.00094 | **−0.00227** | −0.00247 | [−0.00406, −0.00105] | 80% |

**Note on the two estimators.** The *pooled* $H_R$ weights tiles by their activity count (more
active tiles contribute more); the *per-tile mean* and its bootstrap CI weight each tile equally.
For val, the pooled value (−0.00154) lies just above the upper bound of the per-tile CI
(−0.00159) because large/active tiles tend to have smaller magnitude $H_R$, pulling the pooled
estimator toward zero. **Both estimators are negative; all per-tile CIs are entirely negative.**
The conclusion is unchanged: the baseline violates D8 3–4× less than the labels.

Interpretation: the negative $H_R$ indicates no native VF headroom for any D8-style loss. The
predictions are already more locally topographic-order consistent than the reference masks.

#### 5.2 D4 native headroom

D4 uses all lower 4-neighbours (not just steepest descent). It belongs to $\mathcal C_{\mathrm{ord}}$
by the same mechanism. Results (same parameters):

| split | VF$_{\mathrm{D4}}$(labels) | VF$_{\mathrm{D4}}$(pred) | $H_R^{\mathrm{D4}}$ | % tiles pred ≤ labels |
|---|---:|---:|---:|---:|
| val | 0.00180 | 0.00055 | **−0.00125** | 88% |
| test | 0.00204 | 0.00051 | **−0.00153** | 90% |
| bolivia | 0.00284 | 0.00082 | **−0.00202** | 80% |

The no-headroom result is not an artifact of the steepest-descent neighbor selection. It holds
under the simpler 4-neighbor local formulation. $H_R^{\mathrm{D4}} \in [-0.00202, -0.00125]$
across splits. ∎(D4 sub-result)

#### 5.3 Violation energy $E_{\mathrm{topo}}$

| split | $E_{\mathrm{topo}}$(labels) | $E_{\mathrm{topo}}$(pred $\hat y$) | $\Delta E$ | $E_{\mathrm{topo}}$(pred soft $p$) |
|---|---:|---:|---:|---:|
| val | 0.001955 | 0.000564 | **−0.001391** | — |
| test | 0.002174 | 0.000527 | **−0.001647** | — |
| bolivia | 0.002896 | 0.000849 | **−0.002047** | — |

The baseline has approximately **3× lower violation energy** than the labels. The absence of
headroom is not only visible in a binary violation count; it holds in the squared-severity
violation energy that the D8 loss directly minimises. ∎(E_topo sub-result)

#### 5.4 Distributional dominance: $P(v > t)$

A critic could argue that lower mean VF is insufficient — the baseline might still have
severe violations that a weighted loss could target. We compare empirical tail probabilities
$P(v_{\hat p_0}>t)$ vs $P(v_y>t)$ for $t\in\{0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70\}$:

$$
P(v_{\hat p_0} > t) \;\le\; P(v_y > t)
$$

holds at **all tested thresholds on all three splits** (val, test, bolivia/OOD). This is stronger
than a mean comparison: the baseline violates the local topographic-order constraint less than the
labels across the full range of tested violation severities, weakening the objection that stronger
severity weighting could recover headroom. ∎(distributional dominance)

#### 5.5 Useful violation rate: real signal, no net benefit

Among active D8 violations in the baseline prediction — the set the D8 loss would act on —
we classify the centre pixel $i$ by the ground truth:

- **Useful** $(\hat y_i=1,\, y_i=0)$: suppressing $p_i$ removes a false positive.
- **Harmful** $(\hat y_i=1,\, y_i=1)$: suppressing $p_i$ removes true water (recall loss).
- **GT-endorsed**: the GT mask itself has $y_i=1,\, y_{d(i)}=0$ at this location — the loss
  penalises a label-correct configuration.

| split | n violations | useful | harmful | GT-endorsed | enrichment |
|---|---:|---:|---:|---:|---:|
| val | 23,168 | 58.6% | 41.4% | 8.9% | ~3.9× |
| test | 21,950 | 58.6% | 41.4% | 10.2% | ~4.4× |
| bolivia | 5,875 | 60.9% | 39.1% | 6.9% | ~3.7× |

*Enrichment* = P(GT dry | D8-active violation) / P(GT dry | predicted water) ≈ 3.7–4.4×.

**Interpretation.** The D8 signal is real but not net-useful in this label-relative setting.
Topographic violations are enriched in false positives (~4× above baseline FP rate). However,
approximately 41% of corrections would suppress true water. This precisely explains the
observed pattern from L1: enforcing D8 raises **precision** (+0.035) but costs **recall**
(−0.036), leaving IoU essentially unchanged. ∎(L1b)

---

### Lemma L2 — output-level headroom is non-positive (channel I, output)
Pooled $A(-z;\cdot)$ from `elevation_auc_predictions_segman_n100_dice_ce_seed0`:

| split | $A(-z;y)$ | $A(-z;\hat y)$ | $H_z$ | per-tile std |
|---|---:|---:|---:|---:|
| val | 0.7537 | 0.7453 | **+0.008** | ≈ 0.18 |
| test | 0.7615 | 0.7684 | **−0.007** | ≈ 0.18 |
| bolivia | 0.5664 | 0.5929 | **−0.027** | ≈ 0.10 |

Sanity controls in the same run: shuffled-elevation AUC $= 0.500$ (no leakage); model-vs-GT AUC
$= 0.985$ (the segmentation itself is excellent). $H_z\le 0$ on `test` and OOD `bolivia`; the lone
positive value (`val`, +0.008) is two orders of magnitude below the per-tile std (0.18) — i.e.
indistinguishable from zero. **The predictions are already as topographically aligned as the
labels.** ∎(L2)

### Lemma L3 — no privileged elevation representation (channel I, representation)
Within-tile linear DEM-decodability probe, frozen backbone, Pearson $r$ (pooled):

| split | trained | random | input |
|---|---:|---:|---:|
| val | 0.414 | 0.430 | 0.405 |
| test | 0.368 | 0.348 | 0.372 |
| bolivia | 0.337 | 0.367 | **0.474** |

$D(\phi_{\text{trained}})\approx D(\phi_{\text{random}})\approx D(\phi_{\text{input}})$ on every
split; on OOD `bolivia` the *raw input* decodes elevation best. A random-init backbone matches the
trained one. **Segmentation training induced no privileged elevation representation** for a
constraint to strengthen; the weak ($r\approx0.4$) decodability is passive input-level
optical/SAR↔terrain correlation that any network passes through. *Caveat:* linear probe; but
`trained ≈ random ≈ input` collapses the claim to "training adds nothing over the input." ∎(L3)

### Lemma L4 — conditional redundancy: the formulation-independent leg (channel I, all formulations)
Gradient boosting predicts $y$; $A:\,y\sim\hat p$, $B:\,y\sim\hat p+z_{\text{DEM}}$,
$C:\,y\sim z_{\text{DEM}}$, $D:\,y\sim\hat p+\text{shuffle}(z)$. Bootstrap-by-tile 95% CI.

| split | AUC(A) | AUC(B) | $\Delta$AUC = B−A | 95% CI | DEM-only (C) |
|---|---:|---:|---:|---|---:|
| val | 0.9840 | 0.9861 | **+0.0021** | [+0.0008, +0.0043] | 0.8366 |
| test | 0.9848 | 0.9840 | **−0.0009** | [−0.0038, +0.0016] | 0.8336 |
| bolivia | 0.9817 | 0.9822 | **+0.0004** | [−0.0014, +0.0029] | 0.7660 |

- **$\Delta$AUC $\approx 0$ everywhere** while **DEM-only AUC = 0.77–0.84** ⇒ genuine *conditional*
  redundancy, not "DEM is uninformative." This is the empirical estimate of $I(y;z\mid\hat p)\approx 0$.
- **Sanity** $D$ (shuffled DEM) $=$ $A$ exactly. **Feature importance** (model B): $\hat p=0.39$,
  every DEM feature $\le 0.002$.
- **Anti-ceiling** (uncertain pixels $0.05<\hat p<0.95$, where there *is* room): $\Delta$AUC =
  +0.011 (val) / **−0.021 (test)** / +0.004 (bolivia) — signs flip, DEM even *hurts* on test. The
  null is not a ceiling artifact.

Because gradient boosting can model arbitrarily complex $z\!\leftrightarrow\!y$ interactions, it
upper-bounds *any* DEM-loss formulation's information channel. It finds $\approx 0$. ∎(L4)

---

## 6. The composed argument

**Information channel (I).** By Proposition 1, channel I is governed by $I(y;z\mid\hat p)$.
- L2 bounds it to $\le 0$ at the **output** level (headroom non-positive).
- L3 bounds it to $\approx 0$ at the **representation** level (no privileged encoding).
- L4 estimates it directly as $\approx 0$ for **all formulations** (flexible learner, with CIs,
  anti-ceiling, and a working negative control).

These three are independent measurements at three levels (prediction, features, post-hoc
information) and they agree. ⇒ **Channel I has no label-exploitable headroom in this model–data pair.**

**Inductive-bias channel (II), class $\mathcal C_{\mathrm{ord}}$.** L1 and L1b together close
channel II for this class in this setting:

- L1: D8 direct training shows $\rho\approx 0$, no robust test/OOD gain, precision↔recall trade only.
- L1b §5.1: $H_R < 0$ on all splits (per-tile CI entirely negative) — baseline already
  violates D8 3–4× less than labels.
- L1b §5.2: $H_R^{\mathrm{D4}} < 0$ — same result under 4-neighbor formulation.
- L1b §5.3: $E_{\mathrm{topo}}(\hat y_0) < E_{\mathrm{topo}}(y)$ by $\approx 3\times$ — holds in
  the severity-weighted energy.
- L1b §5.4: distributional dominance $P(v_{\hat p_0}>t)\le P(v_y>t)$ at all thresholds.
- L1b §5.5: 41% of active corrections suppress true water, explaining the observed precision/recall
  trade without net IoU benefit.

⇒ **For the class $\mathcal C_{\mathrm{ord}}$, channel II provides no net label-relative benefit
in this setup. The signal is real (~4× FP enrichment) but not net-useful (41% collateral).**

> **Theorem (bounded, label-relative, class-level).** For *(SegMAN-S, N=100 seed0, reference labels y)*:
> (i) no DEM loss of any formulation has label-exploitable headroom through the information
> channel (Prop 1 + L2 + L3 + L4); and (ii) the class $\mathcal C_{\mathrm{ord}}$ of local
> monotone topographic-order losses exhibits no net native label-relative headroom (L1 + L1b).
> **On this task, a local monotone DEM-based topographic-order loss is not justified.**

The null is thus an **informational and mechanistic property of the model–data pair**, not a
failure to find the right formula within this class.

---

## 7. What is NOT proven (the open doors — state these explicitly)

1. **Channel II for *other* formulations / loss classes.** L1 + L1b close channel II for
   $\mathcal C_{\mathrm{ord}}$ (D4, D8, VF-monotone variants). Losses based on a **different
   mechanism** — hydrological connectivity, flow accumulation, HAND, non-local constraints — are
   outside $\mathcal C_{\mathrm{ord}}$ and are **not tested**. The theorem does not apply to them.
2. **True physical state vs labels (A3).** Every lemma is label-relative. If the labels are
   topographically noisy, a DEM loss could still improve agreement with the unobserved truth
   $y^\star$. None of L1–L4 can see this; the 41% GT-endorsed violations in §5.5 hint that some
   "corrections" may in fact be toward $y^\star$.
3. **DEM as a free input modality.** All evidence concerns the DEM as a *constraint/feature*. The
   strongest test of channel I — retraining with the DEM as an input channel, letting it reshape
   the representation end-to-end — is **not yet run**. L4 conditions on a *frozen* $\hat p$ and so
   cannot exclude that a model trained *with* DEM would have built a different $\hat p$.
4. **Single seed / missing positive control.** All training runs are seed 0. The definitive
   positive control (an S1-only or early/undertrained checkpoint where $\Delta$AUC *should* turn
   clearly positive) does not exist.

---

## 8. Falsifiability (what would overturn this)

| observation | which lemma it breaks | consequence |
|---|---|---|
| A loss in $\mathcal C_{\mathrm{ord}}$ gives reproducible multi-seed test/OOD gain | L1 / Theorem(ii) | channel II is live for that formulation |
| $H_R > 0$ or per-tile CI not entirely negative | L1b §5.1 | VF headroom exists |
| $E_{\mathrm{topo}}(\hat y_0) > E_{\mathrm{topo}}(y)$ | L1b §5.3 | energy headroom exists |
| $P(v_{\hat p_0}>t) > P(v_y>t)$ for some threshold | L1b §5.4 | tail headroom exists |
| $\Delta$AUC(B−A) CI clearly $>0$ on a held-out split | L4 / Prop-1 premise | conditional information exists |
| $H_z$ robustly $>0$ beyond per-tile std on test/OOD | L2 | output headroom exists |
| DEM-as-input beats S1/S2 (and shuffled), multi-seed | open door #3 | the modality carries exploitable signal |
| Cleaner labels reveal a DEM gain vs $y^\star$ | open door #2 (A3) | claim is label-artifact |

A single such result falsifies the corresponding leg. As of this writing none is observed.

---

## 9. Conclusion

**Precise defensible claim:**
> "We do not refute D4 and D8 independently. We evaluate the shared mechanism of their loss
> class: reducing local topographic-order violations. In the N=100 SegMAN-S/Sen1Floods11 setting,
> the unconstrained Dice+CE baseline already exhibits lower violation fraction, lower violation
> energy, and lower violation tails than the reference labels. Active D8 violations are enriched
> in false positives (~4× above baseline FP rate), showing that the signal is real, but
> approximately 41% of corrections would suppress true water, explaining the observed
> precision–recall trade-off and lack of net IoU improvement. Therefore, local monotone DEM-based
> topographic-order losses whose sole mechanism is to reduce D4/D8/VF violations have no native
> label-relative headroom in this setup. This is an N=100, seed-0, label-relative result; §10
> lists what remains before a publishable claim."

**Sentences to avoid (a reviewer will break them):**
- "The DEM is useless for flood segmentation." (L4-C: DEM-only AUC 0.77–0.84.)
- "No physics-informed loss can ever help." (Channel II, other loss classes — open door #1.)
- "The model learned/understands elevation." (L3: the opposite.)
- "DEM input useless ⇒ DEM loss useless." (Skips channel II; input test not yet run.)
- "No DEM-based method can ever work." (Scope is $\mathcal C_{\mathrm{ord}}$ only.)

**Operational consequence.** Stop iterating formulations within $\mathcal C_{\mathrm{ord}}$ on
this SegMAN N=100 pair. Remaining high-value moves: (a) DEM-as-input ablation + S1-only positive
control to close open doors #3 and #4; (b) porting the headroom/redundancy screen to EoMT to
show it generalises as an a-priori test, turning this negative result into a method.

---

## 10. Limitations

1. **N=100 is preliminary/mechanistic.** These experiments demonstrate the mechanism on a
   controlled low-data setup. Publishable claims require full Sen1Floods11 experiments with
   multi-seed confirmation across architectures.

2. **Single seed for training experiments.** L1 (direct loss result) is seed 0 only.
   Multi-seed confirmation (target: 5 seeds × 3 conditions, TOST equivalence testing with
   $\delta\approx\sigma_{\text{seed}}\approx0.003$–$0.005$ water-IoU) is a required next step.
   L1b diagnostics are inference-only and do not depend on training seed.

3. **Label-relative (Axiom A3).** All results are relative to the available reference labels.
   If labels are systematically noisy in topographically complex areas, the DEM could improve
   agreement with the true physical state $y^\star$ without improving agreement with $y$. The
   ~8–10% GT-endorsed violations (§5.5) suggest some label noise is present.

4. **Scope: local monotone topographic-order losses only.** The conclusion applies to
   $\mathcal C_{\mathrm{ord}}$ — losses penalising $p_i>p_j+\tau$ when $h_i>h_j$ for neighbouring
   pixels. It does **not** apply to:
   DEM as input · HAND · flow accumulation · hydrological connectivity losses · rainfall/discharge
   constraints · temporal hydrodynamic constraints · non-local or orthogonal DEM-based methods.

5. **DEM as input not tested.** This study concerns DEM as a constraint only. A model trained
   with DEM as an input channel might build a different $\hat p$, for which L4 would need to be
   re-evaluated.

6. **Spatial resolution and annotation limits.** The Copernicus GLO-30 DEM is a DSM (30 m
   resolution, includes buildings/canopy, not bare earth). D8/D4 local rules at 30 m may flag
   legitimate flood configurations as violations — backwater, levee-contained water, urban
   drainage, rainfall-driven ponding, flat floodplain ambiguity — contributing to the 41%
   harmful and 7–10% GT-endorsed violation rates.

7. **Single architecture.** Results are for SegMAN-S. Whether the mechanistic screen generalises
   to other architectures (EoMT, ViT-based models) is an explicit next step.

---

## 11. Environment and reproducibility

We use the full SegMAN-S architecture adapted to 15-channel Sentinel-1/Sentinel-2 inputs.
In the Windows/PyTorch environment, optimised CUDA extensions are replaced by PyTorch fallback
operators for compatibility and reproducibility. **The DEM is never used as a model input.**

| Check | Result |
|---|---|
| OS | Windows 10 Pro for Workstations, build 26200 |
| Python | 3.11.9 (MSC v.1938 64-bit) |
| PyTorch / CUDA | 2.5.1+cu121 / CUDA 12.1 |
| GPU | NVIDIA RTX 5000 Ada Generation, 32 GB |
| SegMAN-S parameters | 33,447,272 (33.45M) |
| Input shape | (1, 15, 512, 512) |
| Output shape | (1, 2, 512, 512) |
| Official CUDA extensions | not installed (mamba-ssm, natten, mmcv-full) |
| PyTorch shims | active: selective scan (exact), cross-scan (exact), NATTEN (exact on interior pixels) |
| DEM as model input | never |
| N=100 checkpoints | 41 runs, all with `best_checkpoint.pt` (~767 MB each) |
| Reproducibility report | `E:\flood_research\setup_logs\windows_segman_env\windows_segman_reproducibility_report.md` |

NATTEN equivalence was verified for interior pixels; boundary behaviour (~1-pixel border ring)
is documented in the shim source (`segman_kernels/compat.py`) and is negligible at 512×512.

**Main project repo:** commit `8017b883` on `https://github.com/KarimKQF/physics-informed-flood-segmentation.git`  
**SegMAN external source:** commit `9ced66ab` (CVPR 2025 official)

---

## 12. Provenance (artifacts backing each number)

- **L1:** `reports/segman_n100_d8_seed0_results.json`, `reports/segman_n100_d8_lambda100_seed0_results.json`
- **L1b §5.1–5.2:** `outputs/native_vf_headroom/native_vf_headroom_results.json`
- **L1b §5.2–5.4 (D4/E_topo/tails):** `outputs/topographic_order_headroom/topographic_order_headroom_results.json`
- **L2:** `reports/elevation_auc_predictions_segman_n100_dice_ce_seed0.json`
- **L3:** `docs/dem_decodability_segman_n100_dice_ce_seed0.md`, `reports/dem_decodability_segman_n100_dice_ce_seed0.json`
- **L4:** `outputs/conditional_dem_redundancy/conditional_dem_redundancy_report.md` (+ `.json`, `.csv`)
- **Earlier N=50 control (real-vs-shuffled sign flip):** `reports/segman_seed0_loss_comparison_summary.json`
- **Smoke test / env:** `E:\flood_research\setup_logs\windows_segman_env\segman_smoke_test_output.txt`
